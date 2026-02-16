"""Cycle-accurate measurement using PTX clock() instructions.

The validate_model.py showed that wall-clock timing (CUDA events) cannot
distinguish schedule orderings for small kernels because launch overhead
(~5 us) drowns out cycle-level differences (~0.5-2 us).

Solution: inject PTX `mov.u32 %r, %clock;` at kernel start and end,
compute delta on-device, and write to output. This gives SM-level
cycle counts with zero launch overhead contamination.

This is the ground truth for pipeline model calibration:
  - Pipeline model predicts: critical_path=1346, random_worst=4773
  - If clock() delta shows e.g. 2000 vs 5500, the model is directionally correct
  - If clock() shows no difference, the scheduling axis is invalid for this kernel

Usage:
    from pipeline.harness.cycle_counter import measure_schedule_cycles
    cycles = measure_schedule_cycles(instructions, schedule_order)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from ..env.instruction import Instruction


def build_cycle_counted_ptx(
    instructions: Dict[int, Instruction],
    schedule_order: List[int],
    kernel_name: str = 'gemm_tile',
    reg_decls: str = '',
) -> str:
    """Build PTX with clock() timing instrumentation.

    Inserts `mov.u32 %rclock_start, %clock;` before the first instruction
    and `mov.u32 %rclock_end, %clock;` after the last, then writes the
    delta to an output buffer.

    The output buffer is an additional .param .u64 ptr_cycles that points
    to a uint32 array where thread 0 writes its cycle count.
    """
    # Emit instructions in schedule order
    body_lines = []
    for inst_id in schedule_order:
        inst = instructions[inst_id]
        body_lines.append(f"    {inst.raw_text}")
    body = "\n".join(body_lines)

    ptx = f"""\
.version 7.8
.target sm_89
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 ptr_a,
    .param .u64 ptr_b,
    .param .u64 ptr_c,
    .param .u64 ptr_cycles
) {{
{reg_decls}
    .reg .b32 %rclock_start;
    .reg .b32 %rclock_end;
    .reg .b32 %rclock_delta;
    .reg .b64 %rd_cycles;

    // Load cycle counter output pointer
    ld.param.u64 %rd_cycles, [ptr_cycles];

    // Read cycle counter BEFORE computation
    mov.u32 %rclock_start, %clock;

{body}

    // Read cycle counter AFTER computation
    mov.u32 %rclock_end, %clock;

    // Compute delta and store
    sub.u32 %rclock_delta, %rclock_end, %rclock_start;
    st.global.u32 [%rd_cycles], %rclock_delta;

    ret;
}}
"""
    return ptx


def measure_schedule_cycles(
    instructions: List[Instruction],
    schedule_order: List[int],
    kernel_name: str = 'gemm_tile',
    n_warmup: int = 50,
    n_runs: int = 200,
    reg_decls: str = '',
) -> Dict:
    """Measure true SM cycles for a schedule ordering using clock() counter.

    Returns cycle counts without kernel launch overhead contamination.

    Args:
        instructions: Instruction objects.
        schedule_order: Schedule as list of instruction IDs.
        kernel_name: Kernel entry point name.
        n_warmup: Warmup iterations.
        n_runs: Measurement iterations.
        reg_decls: Register declarations from the original PTX.

    Returns:
        Dict with median_cycles, mean_cycles, std_cycles, all_cycles, n_runs.
    """
    import cupy as cp
    from ..harness.ptx_compiler import compile_ptx, load_kernel
    from ..harness.ptx_templates import TEMPLATES

    inst_map = {inst.id: inst for inst in instructions}

    # Build cycle-counted PTX
    ptx = build_cycle_counted_ptx(
        inst_map, schedule_order,
        kernel_name=kernel_name,
        reg_decls=reg_decls,
    )

    # Compile
    result = compile_ptx(ptx)
    if not result.success:
        return {'error': result.error}

    # Load kernel
    kernel = load_kernel(result.cubin_path, kernel_name)

    # Get original kernel args + cycles output buffer
    spec = TEMPLATES[kernel_name]()
    base_args = spec.args_factory()
    cycles_buf = cp.zeros(1, dtype=cp.uint32)

    # Combined args: original args + cycles pointer
    args = (*base_args, cycles_buf)

    # Warmup
    for _ in range(n_warmup):
        kernel(spec.grid, spec.block, args)
    cp.cuda.Device().synchronize()

    # Measure
    all_cycles = []
    for _ in range(n_runs):
        cycles_buf[0] = 0  # Reset
        kernel(spec.grid, spec.block, args)
        cp.cuda.Device().synchronize()
        all_cycles.append(int(cycles_buf[0]))

    all_cycles = np.array(all_cycles)

    return {
        'median_cycles': int(np.median(all_cycles)),
        'mean_cycles': float(np.mean(all_cycles)),
        'std_cycles': float(np.std(all_cycles)),
        'min_cycles': int(np.min(all_cycles)),
        'max_cycles': int(np.max(all_cycles)),
        'all_cycles': all_cycles,
        'n_runs': n_runs,
    }


def compare_schedules_by_cycles(
    instructions: List[Instruction],
    schedule_variants: Dict[str, List[int]],
    kernel_name: str = 'gemm_tile',
    n_warmup: int = 50,
    n_runs: int = 200,
    reg_decls: str = '',
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Compare multiple schedule orderings using SM cycle counter.

    This is the corrected version of validate_model.py that avoids
    kernel launch overhead contamination.

    Args:
        instructions: Instruction objects.
        schedule_variants: {name: schedule_order} dict.
        kernel_name: Kernel entry point name.
        n_warmup: Warmup iterations.
        n_runs: Measurement iterations.
        reg_decls: Register declarations from the original PTX.
        verbose: Print comparison table.

    Returns:
        Dict mapping variant name -> cycle measurement results.
    """
    results = {}

    for name, order in schedule_variants.items():
        result = measure_schedule_cycles(
            instructions, order,
            kernel_name=kernel_name,
            n_warmup=n_warmup, n_runs=n_runs,
            reg_decls=reg_decls,
        )
        results[name] = result

    if verbose:
        print(f"\n{'='*65}")
        print(f"SM Cycle Counter Comparison (no launch overhead)")
        print(f"{'='*65}")
        print(f"{'Schedule':<18} {'Median':>8} {'Mean':>10} {'Std':>8} "
              f"{'Min':>8} {'Max':>8}")
        print(f"{'-'*18} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

        for name in sorted(results.keys(),
                          key=lambda n: results[n].get('median_cycles', 0)):
            r = results[name]
            if 'error' in r:
                print(f"{name:<18} ERROR: {r['error']}")
            else:
                print(f"{name:<18} {r['median_cycles']:>8} "
                      f"{r['mean_cycles']:>10.1f} {r['std_cycles']:>8.1f} "
                      f"{r['min_cycles']:>8} {r['max_cycles']:>8}")

        print(f"{'='*65}")

        # Compute correlation with pipeline model if available
        valid = {n: r for n, r in results.items() if 'error' not in r}
        if len(valid) >= 3:
            names = sorted(valid.keys())
            hw_cycles = [valid[n]['median_cycles'] for n in names]

            if len(set(hw_cycles)) > 1:
                from scipy.stats import spearmanr
                # We'd need model predictions here too, but at least show
                # that different schedules produce different cycle counts
                print(f"\nCycle count range: {min(hw_cycles)} - {max(hw_cycles)}")
                spread = (max(hw_cycles) - min(hw_cycles)) / min(hw_cycles) * 100
                print(f"Spread: {spread:.1f}%")
            else:
                print(f"\nAll schedules produce identical cycle counts: {hw_cycles[0]}")
                print("Instruction scheduling has no effect for this kernel.")

    return results
