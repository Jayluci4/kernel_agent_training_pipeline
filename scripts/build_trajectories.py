#!/usr/bin/env python3
"""Build RLVR training trajectories from greedy search v2 results.

Replays each greedy search trace step-by-step:
  1. Generate base PTX from gemm_tile(m, n, k)
  2. Apply transforms in trace order
  3. Extract features at each intermediate state
  4. Package into (state, action, reward, next_state, done) tuples

No GPU required — this is pure PTX parsing and transform replay.

Output: exp-assembly/data/trajectories_v2.jsonl
        exp-assembly/data/trajectory_summary.json

Usage:
    python3 exp-assembly/scripts/build_trajectories.py
"""

import sys
import os
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from pipeline.harness.ptx_templates import gemm_tile
from pipeline.transform.parsed_kernel import parse_kernel, emit
from pipeline.features.kernel_features import (
    extract_features, features_to_array, FEATURE_NAMES,
)

# Reuse transform application logic from greedy_search_v2
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
from pipeline.transform.parsed_kernel import deep_copy_kernel, BodyLine

DATA_DIR = os.path.join(REPO_ROOT, "data")


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ACTION_NAMES = [
    "vec_ld", "vec_st",
    "cache_cs", "cache_cg", "cache_ca", "cache_cv",
    "st_cache_cs", "st_cache_wt", "st_cache_wb",
    "maxnreg_32", "maxnreg_64", "maxnreg_128", "maxnreg_255",
    "reorder_cp", "reorder_il", "reorder_lf", "reorder_sl",
    "prefetch_L1", "prefetch_L2",
    "split_ld",
    "stop",
]

ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}

# Map label → (transform_name, params) for replay
LABEL_TO_TRANSFORM = {
    "vec_ld": ("vectorize_loads", {}),
    "vec_st": ("vectorize_stores", {}),
    "cache_cs": ("cache_hints", {"policy": "cs"}),
    "cache_cg": ("cache_hints", {"policy": "cg"}),
    "cache_ca": ("cache_hints", {"policy": "ca"}),
    "cache_cv": ("cache_hints", {"policy": "cv"}),
    "st_cache_cs": ("store_cache_hints", {"policy": "cs"}),
    "st_cache_wt": ("store_cache_hints", {"policy": "wt"}),
    "st_cache_wb": ("store_cache_hints", {"policy": "wb"}),
    "maxnreg_32": ("register_budget", {"max_regs": 32}),
    "maxnreg_64": ("register_budget", {"max_regs": 64}),
    "maxnreg_128": ("register_budget", {"max_regs": 128}),
    "maxnreg_255": ("register_budget", {"max_regs": 255}),
    "reorder_cp": ("reorder", {"strategy": "critical_path"}),
    "reorder_il": ("reorder", {"strategy": "interleave"}),
    "reorder_lf": ("reorder", {"strategy": "loads_first"}),
    "reorder_sl": ("reorder", {"strategy": "stores_last"}),
    "prefetch_L1": ("prefetch", {"level": "L1"}),
    "prefetch_L2": ("prefetch", {"level": "L2"}),
    "split_ld": ("split_vector_loads", {}),
}

# Conflict groups (from greedy_search_v2.py)
CONFLICT_GROUPS = {
    "cache_hints": {"cache_cs", "cache_cg", "cache_ca", "cache_cv"},
    "store_cache_hints": {"st_cache_cs", "st_cache_wt", "st_cache_wb"},
    "register_budget": {"maxnreg_32", "maxnreg_64", "maxnreg_128", "maxnreg_255"},
    "prefetch": {"prefetch_L1", "prefetch_L2"},
}


def get_available_actions(applied_labels: set) -> list:
    """Return list of action labels available given already-applied transforms."""
    available = []
    for label in ACTION_NAMES:
        if label == "stop":
            available.append(label)
            continue
        if label in applied_labels:
            continue
        # Check conflict groups
        conflict = False
        for group_labels in CONFLICT_GROUPS.values():
            if label in group_labels and applied_labels & group_labels:
                conflict = True
                break
        if not conflict:
            available.append(label)
    return available


# ---------------------------------------------------------------------------
# Transform application (copied from greedy_search_v2 for standalone use)
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
    return TransformResult(kernel=current, changed=count > 0)


def apply_transform(ptx_str, label):
    """Apply a named transform to PTX source. Returns (new_ptx, changed)."""
    t_name, params = LABEL_TO_TRANSFORM[label]
    parsed = parse_kernel(ptx_str)

    if t_name == "cache_hints":
        result = apply_cache_hints_all(parsed, params["policy"])
    elif t_name == "register_budget":
        result = RegisterBudgetTransform().apply(parsed, {"max_regs": params["max_regs"]})
    elif t_name == "vectorize_loads":
        result = VectorizeLoadsTransform().apply_all(parsed)
    elif t_name == "vectorize_stores":
        result = VectorizeStoresTransform().apply_all(parsed)
    elif t_name == "prefetch":
        result = PrefetchTransform().apply(parsed, {"level": params["level"]})
    elif t_name == "store_cache_hints":
        result = StoreCacheHintTransform().apply_all_with_policy(parsed, params["policy"])
    elif t_name == "split_vector_loads":
        result = SplitVectorLoadsTransform().apply_all(parsed)
    elif t_name == "reorder":
        result = ReorderTransform().apply(parsed, {"strategy": params["strategy"]})
    else:
        raise ValueError(f"Unknown transform: {t_name}")

    return emit(result.kernel), result.changed


# ---------------------------------------------------------------------------
# Trajectory building
# ---------------------------------------------------------------------------

def build_kernel_trajectory(result_entry):
    """Build trajectory entries for one kernel from greedy search result.

    Returns a list of (state, action, reward, done) dicts.
    """
    m, n, k = result_entry["m"], result_entry["n"], result_entry["k"]
    trace = result_entry.get("trace", [])
    kernel_id = result_entry["kernel"]

    if not trace or len(trace) < 1:
        return []

    # Generate base PTX
    spec = gemm_tile(m=m, n=n, k=k)
    current_ptx = spec.ptx_source

    # Extract baseline features
    parsed = parse_kernel(current_ptx)
    current_features = extract_features(parsed)

    entries = []
    applied_labels = set()
    baseline_cycles = trace[0]["cycles"]["median"]

    # Process each step in the trace
    for i in range(1, len(trace)):
        step_data = trace[i]
        action_label = step_data["transform"]
        cycles_after = step_data["cycles"]["median"]

        # Cycles before this step
        cycles_before = trace[i - 1]["cycles"]["median"]

        # Per-step reward: relative improvement
        reward = (cycles_after - cycles_before) / cycles_before

        # Available actions at this state
        available = get_available_actions(applied_labels)

        entry = {
            "kernel": kernel_id,
            "m": m, "n": n, "k": k,
            "step": i - 1,
            "features": current_features,
            "feature_array": features_to_array(current_features),
            "action": action_label,
            "action_id": ACTION_TO_ID[action_label],
            "available_actions": available,
            "cycles_before": cycles_before,
            "cycles_after": cycles_after,
            "reward": round(reward, 6),
            "cumulative_improvement": round(
                (cycles_after - baseline_cycles) / baseline_cycles, 6
            ),
            "done": False,
        }
        entries.append(entry)

        # Apply the transform to get next state
        try:
            new_ptx, changed = apply_transform(current_ptx, action_label)
            if changed:
                current_ptx = new_ptx
        except Exception as e:
            print(f"  WARNING: transform {action_label} failed on {kernel_id}: {e}")
            # Continue with unchanged PTX
            pass

        # Update state
        parsed = parse_kernel(current_ptx)
        current_features = extract_features(parsed)
        applied_labels.add(action_label)

    # Add terminal "stop" entry
    final_cycles = trace[-1]["cycles"]["median"]
    available = get_available_actions(applied_labels)
    stop_entry = {
        "kernel": kernel_id,
        "m": m, "n": n, "k": k,
        "step": len(trace) - 1,
        "features": current_features,
        "feature_array": features_to_array(current_features),
        "action": "stop",
        "action_id": ACTION_TO_ID["stop"],
        "available_actions": available,
        "cycles_before": final_cycles,
        "cycles_after": final_cycles,
        "reward": 0.0,
        "cumulative_improvement": round(
            (final_cycles - baseline_cycles) / baseline_cycles, 6
        ),
        "done": True,
    }
    entries.append(stop_entry)

    return entries


def main():
    # Load greedy search v2 results
    results_path = os.path.join(DATA_DIR, "greedy_search_v2_results.json")
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Building trajectories from {len(results)} kernels")
    print(f"Action space: {len(ACTION_NAMES)} actions (20 transforms + stop)")
    print(f"Feature dimensions: {len(FEATURE_NAMES)}")
    print()

    all_entries = []
    action_counts = {}
    total_steps = 0
    n_kernels_with_steps = 0

    for i, result in enumerate(results):
        kernel_id = result["kernel"]
        n_steps = result.get("n_steps", 0)

        if "error" in result:
            print(f"  [{i+1}/{len(results)}] {kernel_id}: SKIPPED (error)")
            continue

        entries = build_kernel_trajectory(result)
        all_entries.extend(entries)

        n_actions = len(entries)
        if n_steps > 0:
            n_kernels_with_steps += 1
        total_steps += n_steps

        # Count actions
        for entry in entries:
            a = entry["action"]
            action_counts[a] = action_counts.get(a, 0) + 1

        if (i + 1) % 16 == 0 or i == 0:
            print(f"  [{i+1}/{len(results)}] {kernel_id}: {n_actions} entries "
                  f"({n_steps} transforms + stop)")

    # Write JSONL
    out_path = os.path.join(DATA_DIR, "trajectories_v2.jsonl")
    with open(out_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Write summary
    summary = {
        "n_kernels": len(results),
        "n_kernels_with_transforms": n_kernels_with_steps,
        "n_entries": len(all_entries),
        "n_transform_entries": len(all_entries) - len(results),
        "n_stop_entries": len(results),
        "total_transform_steps": total_steps,
        "mean_steps_per_kernel": round(total_steps / max(len(results), 1), 2),
        "action_space_size": len(ACTION_NAMES),
        "feature_dimensions": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "action_distribution": dict(sorted(
            action_counts.items(), key=lambda x: -x[1]
        )),
        "source": "greedy_search_v2_results.json",
    }

    summary_path = os.path.join(DATA_DIR, "trajectory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTrajectory summary:")
    print(f"  Kernels: {len(results)} ({n_kernels_with_steps} with transforms)")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Transform entries: {len(all_entries) - len(results)}")
    print(f"  Stop entries: {len(results)}")
    print(f"  Mean steps/kernel: {total_steps / max(len(results), 1):.1f}")
    print(f"\n  Action distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {action}: {count}")
    print(f"\n  Saved to: {out_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
