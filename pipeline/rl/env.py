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
import json
import math
import subprocess
import tempfile
import shutil

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
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


# Label → (transform_name, params)
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
# Cycle measurement (subprocess-isolated)
# ---------------------------------------------------------------------------

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
        print(json.dumps({{"error": "compile: " + result.error}}))
        sys.exit(0)

    kernel = load_kernel(result.cubin_path, {kernel_name!r})
    spec = gemm_tile(m={m}, n={n}, k={k})

    with open({untimed_path!r}) as f:
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
    print(json.dumps({{
        "median": int(np.median(arr)),
        "mean": round(float(np.mean(arr)), 1),
        "std": round(float(np.std(arr)), 1),
    }}))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''


def inject_cycle_counter(ptx_source):
    """Inject SM clock() instrumentation into PTX kernel."""
    lines = ptx_source.split('\n')

    # Find last .param line
    last_param_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('.param') and not stripped.startswith('//'):
            last_param_idx = i
    if last_param_idx < 0:
        raise ValueError("No .param found")

    # Add cycles pointer parameter
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

    # Find injection points
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

    # Insert clock registers
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

    # Insert clock start
    clock_setup = [
        "    ld.param.u64 %rd_cycles, [ptr_cycles];",
        "    mov.u32 %rclock_start, %clock;",
    ]
    for j, s in enumerate(clock_setup):
        lines.insert(first_inst_idx + j, s)
    ret_idx += len(clock_setup)

    # Insert clock end + store
    clock_teardown = [
        "    mov.u32 %rclock_end, %clock;",
        "    sub.u32 %rclock_delta, %rclock_end, %rclock_start;",
        "    st.global.u32 [%rd_cycles], %rclock_delta;",
    ]
    for j, t in enumerate(clock_teardown):
        lines.insert(ret_idx + j, t)

    return '\n'.join(lines)


def measure_cycles(ptx_str, m, n, k, n_warmup=50, n_runs=200):
    """Measure kernel cycles via persistent worker. Returns median cycles or None on error.

    Uses fast_measure module which maintains a long-lived subprocess with CUDA context,
    eliminating the ~3s CuPy import overhead per measurement.
    """
    try:
        from .fast_measure import measure_cycles_fast
        return measure_cycles_fast(ptx_str, m, n, k, n_warmup, n_runs)
    except ImportError:
        # Fallback to subprocess-per-call if fast_measure unavailable
        pass

    # Fallback implementation (slower, spawns subprocess per call)
    spec = gemm_tile(m=m, n=n, k=k)

    try:
        timed_ptx = inject_cycle_counter(ptx_str)
    except Exception:
        return None

    tmpdir = tempfile.mkdtemp(prefix="rlvr_")
    ptx_path = os.path.join(tmpdir, "timed.ptx")
    untimed_path = os.path.join(tmpdir, "untimed.ptx")

    with open(ptx_path, "w") as f:
        f.write(timed_ptx)
    with open(untimed_path, "w") as f:
        f.write(ptx_str)

    script = _MEASURE_SCRIPT.format(
        repo_root=REPO_ROOT,
        ptx_path=ptx_path,
        untimed_path=untimed_path,
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
            return None
        stdout = proc.stdout.strip()
        if not stdout:
            return None
        data = json.loads(stdout)
        if "error" in data:
            return None
        return data["median"]
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TransformEnv:
    """MDP environment for transform selection on a single kernel."""

    def __init__(self, m, n, k, max_steps=6, use_hardware=True):
        self.m = m
        self.n = n
        self.k = k
        self.max_steps = max_steps
        self.use_hardware = use_hardware
        self.kernel_id = f"gemm_tile({m},{n},{k})"

    def reset(self):
        """Reset to baseline PTX. Returns (features_tensor, action_mask, action_history)."""
        spec = gemm_tile(m=self.m, n=self.n, k=self.k)
        self.current_ptx = spec.ptx_source
        self.applied = set()
        self.step_count = 0

        # Measure baseline
        if self.use_hardware:
            self.current_cycles = measure_cycles(
                self.current_ptx, self.m, self.n, self.k
            )
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
            # Return current state, penalty, done
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
            # Measure new cycle count
            new_cycles = measure_cycles(new_ptx, self.m, self.n, self.k)
            if new_cycles is None:
                info["measure_error"] = True
                reward = -STEP_COST * 2
            else:
                # Log-transformed reward
                reward = math.log(self.current_cycles / new_cycles) - STEP_COST
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
