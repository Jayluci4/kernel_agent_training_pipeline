#!/usr/bin/env python3
"""B1: Greedy search with expanded 20-transform action space.

Compares against the original 9-transform greedy to measure the value
of the new transforms, especially vectorize_stores.

Output: data/greedy_search_v2_results.json

Usage:
    python scripts/greedy_search_v2.py
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

from pipeline.harness.ptx_templates import gemm_tile
from pipeline.harness.ptx_compiler import compile_ptx
from pipeline.transform.parsed_kernel import (
    parse_kernel, emit, deep_copy_kernel, BodyLine,
)
from pipeline.transform.base import TransformResult
from pipeline.transform.register_budget import RegisterBudgetTransform
from pipeline.transform.cache_hints import (
    CacheHintTransform, _find_unhinted_loads, _LD_GLOBAL_PATTERN,
)
from pipeline.transform.vectorize_loads import VectorizeLoadsTransform
from pipeline.transform.vectorize_stores import VectorizeStoresTransform
from pipeline.transform.prefetch import PrefetchTransform
from pipeline.transform.store_cache_hints import StoreCacheHintTransform
from pipeline.transform.split_vectors import SplitVectorLoadsTransform
from pipeline.transform.reorder import ReorderTransform

DATA_DIR = os.path.join(REPO_ROOT, "data")


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def apply_cache_hints_all(kernel, policy):
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
        result = ReorderTransform().apply(parsed, {"strategy": params["strategy"]})
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    return emit(result.kernel), result.changed


# Expanded 20-action set
TRANSFORMS = [
    # Vectorization
    ("vectorize_loads", {}, "vec_ld"),
    ("vectorize_stores", {}, "vec_st"),
    # Load cache hints
    ("cache_hints", {"policy": "cs"}, "cache_cs"),
    ("cache_hints", {"policy": "cg"}, "cache_cg"),
    ("cache_hints", {"policy": "ca"}, "cache_ca"),
    ("cache_hints", {"policy": "cv"}, "cache_cv"),
    # Store cache hints
    ("store_cache_hints", {"policy": "cs"}, "st_cache_cs"),
    ("store_cache_hints", {"policy": "wt"}, "st_cache_wt"),
    ("store_cache_hints", {"policy": "wb"}, "st_cache_wb"),
    # Register budget
    ("register_budget", {"max_regs": 32}, "maxnreg_32"),
    ("register_budget", {"max_regs": 64}, "maxnreg_64"),
    ("register_budget", {"max_regs": 128}, "maxnreg_128"),
    ("register_budget", {"max_regs": 255}, "maxnreg_255"),
    # Reorder strategies
    ("reorder", {"strategy": "critical_path"}, "reorder_cp"),
    ("reorder", {"strategy": "interleave"}, "reorder_il"),
    ("reorder", {"strategy": "loads_first"}, "reorder_lf"),
    ("reorder", {"strategy": "stores_last"}, "reorder_sl"),
    # Prefetch
    ("prefetch", {"level": "L1"}, "prefetch_L1"),
    ("prefetch", {"level": "L2"}, "prefetch_L2"),
    # Split (reverse vectorize)
    ("split_vector_loads", {}, "split_ld"),
]

# Conflict rules: only one from each group can be active
CONFLICT_GROUPS = {
    "cache_hints": {"cache_cs", "cache_cg", "cache_ca", "cache_cv"},
    "store_cache_hints": {"st_cache_cs", "st_cache_wt", "st_cache_wb"},
    "register_budget": {"maxnreg_32", "maxnreg_64", "maxnreg_128", "maxnreg_255"},
    "prefetch": {"prefetch_L1", "prefetch_L2"},
}


def conflicts_with_applied(label, applied_labels):
    """Check if a transform conflicts with already-applied transforms."""
    for group_labels in CONFLICT_GROUPS.values():
        if label in group_labels:
            if applied_labels & group_labels:
                return True
    return False


# ---------------------------------------------------------------------------
# Cycle measurement (same as greedy_search.py)
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
    from pipeline.harness.ptx_compiler import compile_ptx, load_kernel
    from pipeline.harness.ptx_templates import gemm_tile

    with open({ptx_path!r}) as f:
        ptx = f.read()

    result = compile_ptx(ptx)
    if not result.success:
        print(json.dumps({{"error": "compile failed: " + result.error}}))
        sys.exit(0)

    kernel = load_kernel(result.cubin_path, {kernel_name!r})
    spec = gemm_tile(m={m}, n={n}, k={k})

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

    tmpdir = tempfile.mkdtemp(prefix="greedy2_")
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
# Greedy search
# ---------------------------------------------------------------------------

def greedy_search_kernel(m, n, k, max_steps=6):
    spec = gemm_tile(m=m, n=n, k=k)
    current_ptx = spec.ptx_source

    baseline = measure_cycles(current_ptx, m, n, k)
    if "error" in baseline:
        return {"error": baseline["error"], "steps": []}

    baseline_cycles = baseline["cycles"]["median"]

    trace = [{
        "step": 0,
        "transform": "baseline",
        "cycles": baseline["cycles"],
        "delta_vs_baseline": "0.0%",
    }]

    current_cycles = baseline_cycles
    applied_labels = set()

    for step in range(1, max_steps + 1):
        best_transform = None
        best_cycles = current_cycles
        best_ptx = None
        best_full = None

        for t_name, t_params, label in TRANSFORMS:
            # Skip conflicting transforms
            if conflicts_with_applied(label, applied_labels):
                continue
            # Skip if same exact transform already applied
            if label in applied_labels:
                continue

            try:
                new_ptx, changed = apply_transform(current_ptx, t_name, t_params)
                if not changed:
                    continue
            except Exception:
                continue

            result = measure_cycles(new_ptx, m, n, k)
            if "error" in result:
                continue

            median = result["cycles"]["median"]
            if median < best_cycles:
                best_cycles = median
                best_transform = (t_name, t_params, label)
                best_ptx = new_ptx
                best_full = result["cycles"]

        if best_transform is None:
            break

        t_name, t_params, label = best_transform
        improvement = (best_cycles - baseline_cycles) / baseline_cycles * 100

        trace.append({
            "step": step,
            "transform": label,
            "cycles": best_full,
            "delta_vs_baseline": f"{improvement:+.1f}%",
        })

        current_ptx = best_ptx
        current_cycles = best_cycles
        applied_labels.add(label)

    total_improvement = (current_cycles - baseline_cycles) / baseline_cycles * 100

    return {
        "baseline_cycles": baseline_cycles,
        "final_cycles": current_cycles,
        "total_improvement": f"{total_improvement:+.1f}%",
        "n_steps": len(trace) - 1,
        "transforms_applied": sorted(applied_labels),
        "trace": trace,
    }


def main():
    tile_values = [2, 4, 6, 8]
    kernels = [(m, n, k) for m in tile_values for n in tile_values for k in tile_values]

    print(f"Greedy search v2 on {len(kernels)} kernels")
    print(f"Transforms: {len(TRANSFORMS)}")
    print(f"Max steps per kernel: 6")
    print()

    results = []
    t0 = time.time()

    for i, (m, n, k) in enumerate(kernels):
        label = f"gemm_tile({m},{n},{k})"
        print(f"[{i+1}/{len(kernels)}] {label}...", flush=True)

        t1 = time.time()
        result = greedy_search_kernel(m, n, k)
        dt = time.time() - t1

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  baseline={result['baseline_cycles']}, "
                  f"final={result['final_cycles']}, "
                  f"improvement={result['total_improvement']}, "
                  f"steps={result['n_steps']}, "
                  f"transforms={result['transforms_applied']} "
                  f"({dt:.1f}s)")

        entry = {"kernel": label, "m": m, "n": n, "k": k, **result}
        results.append(entry)

        if (i + 1) % 8 == 0:
            _save(results, time.time() - t0)

    wall_time = time.time() - t0
    _save(results, wall_time)

    improvements = [float(r["total_improvement"].rstrip("%"))
                    for r in results if "total_improvement" in r]
    if improvements:
        arr = np.array(improvements)
        print(f"\nSummary (v2, 20 transforms):")
        print(f"  Kernels: {len(improvements)}")
        print(f"  Mean improvement: {np.mean(arr):+.1f}%")
        print(f"  Median improvement: {np.median(arr):+.1f}%")
        print(f"  Best: {np.min(arr):+.1f}%")
        print(f"  Worst: {np.max(arr):+.1f}%")
        print(f"  Wall time: {wall_time:.1f}s")

    # Compare with v1 if available
    v1_path = os.path.join(DATA_DIR, "greedy_search_results.json")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        v1_imps = [float(r["total_improvement"].rstrip("%"))
                   for r in v1.get("results", []) if "total_improvement" in r]
        if v1_imps:
            v1_arr = np.array(v1_imps)
            print(f"\n  Comparison with v1 (9 transforms):")
            print(f"    v1 mean: {np.mean(v1_arr):+.1f}%")
            print(f"    v2 mean: {np.mean(arr):+.1f}%")
            print(f"    v2 - v1: {np.mean(arr) - np.mean(v1_arr):+.1f}pp")

    # Transform frequency
    all_transforms = []
    for r in results:
        if "transforms_applied" in r:
            all_transforms.extend(r["transforms_applied"])
    if all_transforms:
        from collections import Counter
        counts = Counter(all_transforms)
        print(f"\n  Transform frequency (top 10):")
        for t, c in counts.most_common(10):
            print(f"    {t}: {c}/{len(improvements)} ({100*c/len(improvements):.0f}%)")


def _save(results, wall_time):
    output = {
        "gpu": "NVIDIA L4",
        "n_warmup": 50,
        "n_runs": 200,
        "n_transforms": len(TRANSFORMS),
        "wall_time_s": round(wall_time, 1),
        "n_kernels": len(results),
        "results": results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "greedy_search_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
