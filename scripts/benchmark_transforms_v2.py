#!/usr/bin/env python3
"""B1: Benchmark expanded transform set (20 actions) on gemm_tile kernels.

Tests each transform independently on a sample of kernels, measuring
hardware cycles. Reports which transforms produce valid PTX, which
change cycles, and the magnitude of each transform's effect.

This validates the expanded action space before using it for trajectory
collection or RLVR training.

Output: exp-assembly/data/transform_benchmark_v2.json

Usage:
    source SAWO/experiments/chronos/setup_env.sh
    python3 exp-assembly/scripts/benchmark_transforms_v2.py
"""

import sys
import os
import json
import re
import subprocess
import tempfile
import time
import shutil

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from experiments.chronos.harness.ptx_templates import gemm_tile
from experiments.chronos.harness.ptx_compiler import compile_ptx
from experiments.chronos.transform.parsed_kernel import (
    parse_kernel, emit, deep_copy_kernel, BodyLine,
)
from experiments.chronos.transform.base import TransformResult

# Existing transforms
from experiments.chronos.transform.register_budget import RegisterBudgetTransform
from experiments.chronos.transform.cache_hints import (
    CacheHintTransform, _find_unhinted_loads, _LD_GLOBAL_PATTERN,
)
from experiments.chronos.transform.vectorize_loads import VectorizeLoadsTransform
from experiments.chronos.transform.reorder import ReorderTransform

# New transforms
from experiments.chronos.transform.vectorize_stores import VectorizeStoresTransform
from experiments.chronos.transform.prefetch import PrefetchTransform
from experiments.chronos.transform.store_cache_hints import StoreCacheHintTransform
from experiments.chronos.transform.split_vectors import SplitVectorLoadsTransform

DATA_DIR = os.path.join(REPO_ROOT, "data")


# ---------------------------------------------------------------------------
# All 20 transforms
# ---------------------------------------------------------------------------

def apply_cache_hints_all(kernel, policy):
    """Apply cache hint to all unhinted loads."""
    indices = _find_unhinted_loads(kernel)
    if not indices:
        return TransformResult(kernel=kernel, changed=False)
    current = deep_copy_kernel(kernel)
    count = 0
    for idx in indices:
        bl = current.body[idx]
        old_text = bl.raw_text
        new_text = _LD_GLOBAL_PATTERN.sub(rf'\1.{policy}\3', old_text)
        if new_text != old_text:
            current.body[idx] = BodyLine(
                tag="instruction", raw_text=new_text,
                instruction=bl.instruction,
            )
            count += 1
    return TransformResult(kernel=current, changed=count > 0,
                          stats={"loads_hinted": count, "policy": policy})


def apply_transform(ptx_str, transform_name, params=None):
    """Apply a single transform to PTX string. Returns (new_ptx, changed)."""
    parsed = parse_kernel(ptx_str)

    if transform_name == "cache_hints":
        result = apply_cache_hints_all(parsed, params["policy"])
    elif transform_name == "register_budget":
        result = RegisterBudgetTransform().apply(parsed, {"max_regs": params["max_regs"]})
    elif transform_name == "vectorize_loads":
        result = VectorizeLoadsTransform().apply_all(parsed)
    elif transform_name == "vectorize_stores":
        result = VectorizeStoresTransform().apply_all(parsed)
    elif transform_name == "prefetch":
        result = PrefetchTransform().apply(parsed, {"level": params["level"]})
    elif transform_name == "store_cache_hints":
        result = StoreCacheHintTransform().apply_all_with_policy(parsed, params["policy"])
    elif transform_name == "split_vector_loads":
        result = SplitVectorLoadsTransform().apply_all(parsed)
    elif transform_name == "reorder":
        # Reorder needs special handling through ScheduleEnv
        result = ReorderTransform().apply(parsed, {"strategy": params["strategy"]})
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    return emit(result.kernel), result.changed


# Full action space: 20 transforms
TRANSFORMS = [
    # Original 9
    ("vectorize_loads", {}, "vec_ld"),
    ("cache_hints", {"policy": "cs"}, "cache_cs"),
    ("cache_hints", {"policy": "cg"}, "cache_cg"),
    ("register_budget", {"max_regs": 32}, "maxnreg_32"),
    ("register_budget", {"max_regs": 64}, "maxnreg_64"),
    ("register_budget", {"max_regs": 128}, "maxnreg_128"),
    ("register_budget", {"max_regs": 255}, "maxnreg_255"),
    ("reorder", {"strategy": "critical_path"}, "reorder_cp"),
    ("reorder", {"strategy": "interleave"}, "reorder_il"),
    # New 11
    ("vectorize_stores", {}, "vec_st"),
    ("prefetch", {"level": "L1"}, "prefetch_L1"),
    ("prefetch", {"level": "L2"}, "prefetch_L2"),
    ("store_cache_hints", {"policy": "cs"}, "st_cache_cs"),
    ("store_cache_hints", {"policy": "wt"}, "st_cache_wt"),
    ("store_cache_hints", {"policy": "wb"}, "st_cache_wb"),
    ("split_vector_loads", {}, "split_ld"),
    ("cache_hints", {"policy": "ca"}, "cache_ca"),
    ("cache_hints", {"policy": "cv"}, "cache_cv"),
    ("reorder", {"strategy": "loads_first"}, "reorder_lf"),
    ("reorder", {"strategy": "stores_last"}, "reorder_sl"),
]


# ---------------------------------------------------------------------------
# Cycle measurement (reused from greedy_search.py)
# ---------------------------------------------------------------------------

def inject_cycle_counter(ptx_source):
    lines = ptx_source.split('\n')
    last_param_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('.param') and not stripped.startswith('//'):
            last_param_idx = i
    if last_param_idx < 0:
        raise ValueError("No .param found in PTX")

    new_lines = []
    for i, line in enumerate(lines):
        if i == last_param_idx:
            stripped = line.rstrip()
            if not stripped.endswith(','):
                new_lines.append(stripped + ',')
            else:
                new_lines.append(line)
            new_lines.append('    .param .u64 ptr_cycles')
        else:
            new_lines.append(line)
    lines = new_lines

    brace_idx = -1
    last_reg_idx = -1
    first_inst_idx = -1
    ret_idx = -1

    for i, line in enumerate(lines):
        s = line.strip()
        if brace_idx < 0 and '{' in s:
            brace_idx = i
        if brace_idx >= 0 and s.startswith('.reg '):
            last_reg_idx = i

    search_start = max(last_reg_idx + 1, brace_idx + 1)
    for i in range(search_start, len(lines)):
        s = lines[i].strip()
        if not s or s.startswith('//') or s.startswith('.') or s.endswith(':') or s in ('{', '}'):
            continue
        first_inst_idx = i
        break

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() in ('ret;', 'exit;'):
            ret_idx = i
            break

    if brace_idx < 0 or first_inst_idx < 0 or ret_idx < 0:
        raise ValueError("Could not find injection points")

    insert_at = last_reg_idx + 1 if last_reg_idx >= 0 else brace_idx + 1
    clock_regs = [
        "    .reg .b32 %rclock_start;",
        "    .reg .b32 %rclock_end;",
        "    .reg .b32 %rclock_delta;",
        "    .reg .b64 %rd_cycles;",
    ]
    for j, r in enumerate(clock_regs):
        lines.insert(insert_at + j, r)

    offset = len(clock_regs)
    first_inst_idx += offset
    ret_idx += offset

    clock_setup = [
        "    ld.param.u64 %rd_cycles, [ptr_cycles];",
        "    mov.u32 %rclock_start, %clock;",
    ]
    for j, s in enumerate(clock_setup):
        lines.insert(first_inst_idx + j, s)
    ret_idx += len(clock_setup)

    clock_teardown = [
        "    mov.u32 %rclock_end, %clock;",
        "    sub.u32 %rclock_delta, %rclock_end, %rclock_start;",
        "    st.global.u32 [%rd_cycles], %rclock_delta;",
    ]
    for j, t in enumerate(clock_teardown):
        lines.insert(ret_idx + j, t)

    return '\n'.join(lines)


_MEASURE_SCRIPT = '''
import sys, os, json
sys.path.insert(0, {repo_root!r})
import numpy as np

try:
    import cupy as cp
    from experiments.chronos.harness.ptx_compiler import compile_ptx, load_kernel
    from experiments.chronos.harness.ptx_templates import gemm_tile

    with open({ptx_path!r}) as f:
        ptx = f.read()

    result = compile_ptx(ptx)
    if not result.success:
        print(json.dumps({{"error": "compile failed: " + result.error}}))
        sys.exit(0)

    kernel = load_kernel(result.cubin_path, {kernel_name!r})
    spec = gemm_tile(m={m}, n={n}, k={k})

    # Correctness check
    with open({untimed_ptx_path!r}) as f:
        untimed_ptx = f.read()
    uresult = compile_ptx(untimed_ptx)
    if uresult.success:
        ukernel = load_kernel(uresult.cubin_path, spec.kernel_name)
        uargs = spec.args_factory()
        ukernel(spec.grid, spec.block, uargs)
        cp.cuda.Device().synchronize()
        try:
            correct = spec.validator(uargs)
        except Exception:
            correct = False
    else:
        correct = False

    if not correct:
        print(json.dumps({{"error": "incorrect results"}}))
        sys.exit(0)

    # Measure cycles
    base_args = spec.args_factory()
    cycles_buf = cp.zeros(1, dtype=cp.uint32)
    args = (*base_args, cycles_buf)

    for _ in range({n_warmup}):
        kernel(spec.grid, spec.block, args)
    cp.cuda.Device().synchronize()

    all_cycles = []
    for _ in range({n_runs}):
        cycles_buf[0] = 0
        kernel(spec.grid, spec.block, args)
        cp.cuda.Device().synchronize()
        all_cycles.append(int(cycles_buf[0]))

    arr = np.array(all_cycles)
    out = {{
        "cycles": {{
            "median": int(np.median(arr)),
            "mean": round(float(np.mean(arr)), 1),
            "std": round(float(np.std(arr)), 1),
        }},
        "correct": True,
        "ptxas_info": result.ptxas_info or "",
    }}
    print(json.dumps(out))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''


def measure_cycles(ptx_str, m, n, k, n_warmup=50, n_runs=200):
    spec = gemm_tile(m=m, n=n, k=k)

    try:
        timed_ptx = inject_cycle_counter(ptx_str)
    except Exception as e:
        return {"error": f"inject failed: {e}"}

    tmpdir = tempfile.mkdtemp(prefix="bench_")
    ptx_path = os.path.join(tmpdir, "timed.ptx")
    untimed_path = os.path.join(tmpdir, "untimed.ptx")

    with open(ptx_path, "w") as f:
        f.write(timed_ptx)
    with open(untimed_path, "w") as f:
        f.write(ptx_str)

    script = _MEASURE_SCRIPT.format(
        repo_root=REPO_ROOT,
        ptx_path=ptx_path,
        untimed_ptx_path=untimed_path,
        kernel_name=spec.kernel_name,
        m=m, n=n, k=k,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
            env=os.environ.copy(),
        )

        if proc.returncode != 0:
            stderr_lines = [l for l in proc.stderr.strip().split('\n') if l.strip()]
            err = stderr_lines[-1] if stderr_lines else "unknown"
            return {"error": err}

        stdout = proc.stdout.strip()
        if not stdout:
            return {"error": "no output"}

        return json.loads(stdout)

    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Compile-only check (fast, no GPU needed for initial validation)
# ---------------------------------------------------------------------------

def check_compiles(ptx_str):
    """Check if PTX compiles without errors. Returns (success, info_str)."""
    result = compile_ptx(ptx_str)
    return result.success, result.ptxas_info if result.success else result.error


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark_kernel(m, n, k):
    """Benchmark all 20 transforms on a single kernel."""
    spec = gemm_tile(m=m, n=n, k=k)
    baseline_ptx = spec.ptx_source

    # Measure baseline
    baseline_result = measure_cycles(baseline_ptx, m, n, k)
    if "error" in baseline_result:
        return {"error": baseline_result["error"]}

    baseline_cycles = baseline_result["cycles"]["median"]

    results = []
    for t_name, t_params, label in TRANSFORMS:
        entry = {"transform": label, "name": t_name, "params": t_params}

        try:
            new_ptx, changed = apply_transform(baseline_ptx, t_name, t_params)
            entry["changed"] = changed

            if not changed:
                entry["status"] = "no_change"
                results.append(entry)
                continue

            # Check compilation
            compiles, info = check_compiles(new_ptx)
            entry["compiles"] = compiles

            if not compiles:
                entry["status"] = "compile_error"
                entry["error"] = info[:200] if info else "unknown"
                results.append(entry)
                continue

            # Measure cycles
            meas = measure_cycles(new_ptx, m, n, k)
            if "error" in meas:
                entry["status"] = "measure_error"
                entry["error"] = meas["error"][:200]
                results.append(entry)
                continue

            cycles = meas["cycles"]["median"]
            delta_pct = (cycles - baseline_cycles) / baseline_cycles * 100
            entry["status"] = "ok"
            entry["cycles"] = meas["cycles"]
            entry["delta_pct"] = round(delta_pct, 1)
            entry["ptxas_info"] = meas.get("ptxas_info", "")

        except Exception as e:
            entry["status"] = "exception"
            entry["error"] = str(e)[:200]

        results.append(entry)

    return {
        "baseline_cycles": baseline_cycles,
        "baseline_ptxas": baseline_result.get("ptxas_info", ""),
        "transforms": results,
    }


def main():
    # Sample kernels: representative sizes (small, medium, large)
    sample_kernels = [
        (2, 2, 2), (2, 4, 4), (4, 4, 4),
        (4, 6, 4), (4, 8, 8), (6, 6, 6),
        (6, 8, 8), (8, 8, 4), (8, 8, 8),
    ]

    print(f"Benchmarking {len(TRANSFORMS)} transforms on {len(sample_kernels)} kernels")
    print(f"Transforms: {[t[2] for t in TRANSFORMS]}")
    print()

    all_results = []
    t0 = time.time()

    for i, (m, n, k) in enumerate(sample_kernels):
        label = f"gemm_tile({m},{n},{k})"
        print(f"[{i+1}/{len(sample_kernels)}] {label}...", flush=True)

        t1 = time.time()
        result = benchmark_kernel(m, n, k)
        dt = time.time() - t1

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            baseline = result["baseline_cycles"]
            # Count results by status
            statuses = {}
            for tr in result["transforms"]:
                s = tr.get("status", "unknown")
                statuses[s] = statuses.get(s, 0) + 1

            # Best improvement
            improvements = [tr["delta_pct"] for tr in result["transforms"]
                          if tr.get("status") == "ok" and "delta_pct" in tr]
            best = min(improvements) if improvements else 0
            print(f"  baseline={baseline}, statuses={statuses}, "
                  f"best={best:+.1f}% ({dt:.1f}s)")

        all_results.append({
            "kernel": label,
            "m": m, "n": n, "k": k,
            **result,
        })

    wall_time = time.time() - t0

    # Save results
    output = {
        "gpu": "NVIDIA L4",
        "n_transforms": len(TRANSFORMS),
        "n_kernels": len(sample_kernels),
        "transform_labels": [t[2] for t in TRANSFORMS],
        "wall_time_s": round(wall_time, 1),
        "results": all_results,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "transform_benchmark_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone in {wall_time:.1f}s")
    print(f"Saved to {out_path}")

    # Summary: which transforms work?
    print(f"\n{'Transform':<16} {'OK':>4} {'NoChg':>6} {'CErr':>5} {'MErr':>5} {'Exc':>4} {'MeanDelta':>10}")
    print("-" * 60)
    for t_name, t_params, label in TRANSFORMS:
        ok_deltas = []
        counts = {"ok": 0, "no_change": 0, "compile_error": 0, "measure_error": 0, "exception": 0}
        for kr in all_results:
            if "transforms" not in kr:
                continue
            for tr in kr["transforms"]:
                if tr["transform"] == label:
                    s = tr.get("status", "unknown")
                    counts[s] = counts.get(s, 0) + 1
                    if s == "ok" and "delta_pct" in tr:
                        ok_deltas.append(tr["delta_pct"])

        mean_d = f"{np.mean(ok_deltas):+.1f}%" if ok_deltas else "n/a"
        print(f"{label:<16} {counts['ok']:>4} {counts['no_change']:>6} "
              f"{counts['compile_error']:>5} {counts['measure_error']:>5} "
              f"{counts.get('exception', 0):>4} {mean_d:>10}")


if __name__ == "__main__":
    main()
