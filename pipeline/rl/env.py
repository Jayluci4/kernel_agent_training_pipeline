"""Transform selection environment for RLVR training.

Wraps PTX generation, transform application, feature extraction,
and hardware cycle measurement into a step-by-step MDP.

State: 25 scalar features from ParsedKernel
Action: one of 21 discrete actions (20 transforms + stop)
Reward: log(cycles_before / cycles_after) - step_cost
Done: when "stop" is selected or max_steps reached
"""

import sys
import os
import math

import numpy as np
import torch

# Compute repo root dynamically (2 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from pipeline.harness.ptx_templates import gemm_tile
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
from pipeline.features.kernel_features import (
    extract_features, features_to_array,
)

from .policy import (
    ACTION_NAMES, ACTION_TO_ID, N_ACTIONS,
    get_action_mask, get_action_history,
)


# Label -> (transform_name, params)
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

STEP_COST = 0.005  # Per-step penalty to encourage efficiency
REWARD_FLOOR = 0.01  # Ignore <1% cycle changes (noise vs real signal)


# ---------------------------------------------------------------------------
# Transform application
# ---------------------------------------------------------------------------

def _apply_cache_hints_all(kernel, policy):
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
    """Apply a named transform. Returns (new_ptx, changed)."""
    t_name, params = LABEL_TO_TRANSFORM[label]
    parsed = parse_kernel(ptx_str)

    if t_name == "cache_hints":
        result = _apply_cache_hints_all(parsed, params["policy"])
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
# Environment
# ---------------------------------------------------------------------------

class TransformEnv:
    """MDP environment for transform selection on a single kernel.

    Uses fast_measure (in-process CuPy) when use_hardware=True.
    Caches baseline cycle counts across resets.
    """

    def __init__(self, m, n, k, max_steps=6, use_hardware=True):
        self.m = m
        self.n = n
        self.k = k
        self.max_steps = max_steps
        self.use_hardware = use_hardware
        self.kernel_id = f"gemm_tile({m},{n},{k})"
        self._baseline_cached = None

    def _measure(self, ptx_str):
        """Measure cycles using in-process fast path."""
        from .fast_measure import measure_cycles_fast
        return measure_cycles_fast(ptx_str, self.m, self.n, self.k)

    def _get_baseline(self):
        """Get cached baseline cycles."""
        if self._baseline_cached is not None:
            return self._baseline_cached
        from .fast_measure import get_baseline_cycles
        self._baseline_cached = get_baseline_cycles(self.m, self.n, self.k)
        return self._baseline_cached

    def reset(self):
        """Reset to baseline PTX. Returns (features_tensor, action_mask, action_history)."""
        spec = gemm_tile(m=self.m, n=self.n, k=self.k)
        self.current_ptx = spec.ptx_source
        self.applied = set()
        self.step_count = 0

        # Use cached baseline
        if self.use_hardware:
            self.current_cycles = self._get_baseline()
        else:
            self.current_cycles = None

        self.baseline_cycles = self.current_cycles

        # Extract features
        parsed = parse_kernel(self.current_ptx)
        features = extract_features(parsed)
        feat_array = features_to_array(features)
        mask = get_action_mask(self.applied)
        history = get_action_history(self.applied)

        return (
            torch.tensor(feat_array, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(history, dtype=torch.float32),
        )

    def step(self, action_id):
        """Take an action. Returns (next_state, reward, done, info).

        next_state: (features, action_mask, action_history) tensors
        reward: float (log cycle ratio - step cost)
        done: bool
        info: dict with cycles, action label, etc.
        """
        action_label = ACTION_NAMES[action_id]
        info = {"action": action_label, "kernel": self.kernel_id}

        # Terminal action
        if action_label == "stop":
            parsed = parse_kernel(self.current_ptx)
            features = extract_features(parsed)
            feat_array = features_to_array(features)
            mask = get_action_mask(self.applied)
            history = get_action_history(self.applied)
            state = (
                torch.tensor(feat_array, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(history, dtype=torch.float32),
            )
            reward = -STEP_COST
            info["stopped"] = True
            return state, reward, True, info

        # Apply transform
        try:
            new_ptx, changed = apply_transform(self.current_ptx, action_label)
        except Exception as e:
            info["error"] = str(e)
            parsed = parse_kernel(self.current_ptx)
            features = extract_features(parsed)
            feat_array = features_to_array(features)
            mask = get_action_mask(self.applied)
            history = get_action_history(self.applied)
            state = (
                torch.tensor(feat_array, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(history, dtype=torch.float32),
            )
            return state, -STEP_COST * 2, True, info

        if not changed:
            info["no_change"] = True
            reward = -STEP_COST
        elif self.use_hardware:
            # Measure new cycle count (in-process, fast)
            new_cycles = self._measure(new_ptx)
            if new_cycles is None:
                info["measure_error"] = True
                reward = -STEP_COST * 2
            else:
                # Log-transformed reward with noise floor
                raw_log_ratio = math.log(self.current_cycles / new_cycles)
                if abs(raw_log_ratio) < REWARD_FLOOR:
                    # <1% change — treat as noise, not real signal
                    reward = -STEP_COST
                    info["below_floor"] = True
                else:
                    reward = raw_log_ratio - STEP_COST
                info["cycles_before"] = self.current_cycles
                info["cycles_after"] = new_cycles
                self.current_cycles = new_cycles
            self.current_ptx = new_ptx
        else:
            # No hardware: reward is unknown, set to 0
            reward = 0.0
            self.current_ptx = new_ptx

        self.applied.add(action_label)
        self.step_count += 1

        # Check if max steps reached
        done = self.step_count >= self.max_steps

        # Extract new state
        parsed = parse_kernel(self.current_ptx)
        features = extract_features(parsed)
        feat_array = features_to_array(features)
        mask = get_action_mask(self.applied)
        history = get_action_history(self.applied)

        # If no actions available (all masked except stop), auto-done
        if sum(mask) <= 1:  # only stop available
            done = True

        state = (
            torch.tensor(feat_array, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(history, dtype=torch.float32),
        )

        return state, reward, done, info
