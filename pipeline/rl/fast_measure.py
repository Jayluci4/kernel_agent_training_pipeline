"""Persistent measurement worker for RLVR training.

Spawns a single long-lived subprocess that holds the CuPy/CUDA context.
Measurements are sent via stdin/stdout JSON protocol.

Eliminates the ~3s CuPy import overhead per measurement while keeping
subprocess isolation (bad PTX can't corrupt the main process).
"""

import os
import sys
import json
import subprocess
import logging
import atexit

import numpy as np

logger = logging.getLogger(__name__)

# Compute repo root dynamically (2 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Worker subprocess script (runs in its own process with CuPy)
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r'''
import sys, os, json, tempfile, shutil
sys.path.insert(0, {repo_root!r})
import numpy as np

# One-time import
import cupy as cp
from pipeline.harness.ptx_compiler import compile_ptx, load_kernel
from pipeline.harness.ptx_templates import gemm_tile

# Signal ready
print(json.dumps({{"status": "ready"}}), flush=True)

def inject_cycle_counter(ptx_source):
    lines = ptx_source.split('\n')
    last_param_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('.param') and not stripped.startswith('//'):
            last_param_idx = i
    if last_param_idx < 0:
        raise ValueError("No .param found")

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
        if brace_idx < 0 and '{{' in s:
            brace_idx = i
        if brace_idx >= 0 and s.startswith('.reg '):
            last_reg_idx = i

    search_start = max(last_reg_idx + 1, brace_idx + 1)
    for i in range(search_start, len(lines)):
        s = lines[i].strip()
        if not s or s.startswith('//') or s.startswith('.') or s.endswith(':') or s in ('{{', '}}'):
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


def measure(ptx_str, m, n, k, n_warmup=50, n_runs=200):
    spec = gemm_tile(m=m, n=n, k=k)
    try:
        timed_ptx = inject_cycle_counter(ptx_str)
    except Exception as e:
        return {{"error": "inject: " + str(e)}}

    try:
        result = compile_ptx(timed_ptx)
        if not result.success:
            return {{"error": "compile: " + result.error}}
        kernel = load_kernel(result.cubin_path, spec.kernel_name)

        # Validate untimed version
        uresult = compile_ptx(ptx_str)
        if not uresult.success:
            return {{"error": "compile_untimed: " + uresult.error}}
        ukernel = load_kernel(uresult.cubin_path, spec.kernel_name)
        uargs = spec.args_factory()
        ukernel(spec.grid, spec.block, uargs)
        cp.cuda.Device().synchronize()
        try:
            correct = spec.validator(uargs)
        except Exception:
            correct = False
        if not correct:
            return {{"error": "incorrect results"}}

        # Measure
        base_args = spec.args_factory()
        cycles_buf = cp.zeros(1, dtype=cp.uint32)
        args = (*base_args, cycles_buf)

        for _ in range(n_warmup):
            kernel(spec.grid, spec.block, args)
        cp.cuda.Device().synchronize()

        all_cycles = []
        for _ in range(n_runs):
            cycles_buf[0] = 0
            kernel(spec.grid, spec.block, args)
            cp.cuda.Device().synchronize()
            all_cycles.append(int(cycles_buf[0]))

        arr = np.array(all_cycles)
        return {{"median": int(np.median(arr))}}

    except Exception as e:
        return {{"error": str(e)}}


# Main loop: read JSON requests from stdin, write responses to stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        result = measure(
            req["ptx"], req["m"], req["n"], req["k"],
            req.get("n_warmup", 50), req.get("n_runs", 200),
        )
        print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({{"error": "worker: " + str(e)}}), flush=True)
'''


# ---------------------------------------------------------------------------
# Worker manager
# ---------------------------------------------------------------------------

_worker_proc = None
_baseline_cache = {}


def _start_worker():
    """Start the persistent measurement worker subprocess."""
    global _worker_proc

    if _worker_proc is not None and _worker_proc.poll() is None:
        return True  # already running

    script = _WORKER_SCRIPT.format(repo_root=REPO_ROOT)
    try:
        _worker_proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
            bufsize=1,
        )

        # Wait for ready signal
        ready_line = _worker_proc.stdout.readline().strip()
        if not ready_line:
            logger.error("Worker failed to start (no output)")
            _worker_proc.kill()
            _worker_proc = None
            return False

        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            logger.error("Worker not ready: %s", ready)
            _worker_proc.kill()
            _worker_proc = None
            return False

        logger.info("Measurement worker started (PID %d)", _worker_proc.pid)
        return True

    except Exception as e:
        logger.error("Failed to start worker: %s", e)
        _worker_proc = None
        return False


def _stop_worker():
    """Gracefully stop the worker."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        try:
            _worker_proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            _worker_proc.stdin.flush()
            _worker_proc.wait(timeout=5)
        except Exception:
            _worker_proc.kill()
        logger.info("Measurement worker stopped")
    _worker_proc = None


# Clean up on exit
atexit.register(_stop_worker)


def _send_request(ptx_str, m, n, k, n_warmup=50, n_runs=200):
    """Send a measurement request to the worker. Returns median cycles or None."""
    global _worker_proc

    if not _start_worker():
        return None

    # Check if worker died
    if _worker_proc.poll() is not None:
        logger.warning("Worker died, restarting...")
        _worker_proc = None
        if not _start_worker():
            return None

    request = json.dumps({
        "ptx": ptx_str,
        "m": m, "n": n, "k": k,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    })

    try:
        _worker_proc.stdin.write(request + "\n")
        _worker_proc.stdin.flush()

        response_line = _worker_proc.stdout.readline().strip()
        if not response_line:
            logger.warning("Worker returned empty response, restarting...")
            _worker_proc.kill()
            _worker_proc = None
            return None

        result = json.loads(response_line)
        if "error" in result:
            logger.debug("Measurement error: %s", result["error"])
            return None

        return result["median"]

    except Exception as e:
        logger.warning("Worker communication error: %s, restarting...", e)
        try:
            _worker_proc.kill()
        except Exception:
            pass
        _worker_proc = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_cycles_fast(ptx_str, m, n, k, n_warmup=50, n_runs=200):
    """Measure kernel cycles via persistent worker. Returns median cycles or None."""
    return _send_request(ptx_str, m, n, k, n_warmup, n_runs)


def get_baseline_cycles(m, n, k):
    """Get cached baseline cycles for a kernel. Measures once."""
    key = (m, n, k)
    if key not in _baseline_cache:
        from pipeline.harness.ptx_templates import gemm_tile
        spec = gemm_tile(m=m, n=n, k=k)
        cycles = measure_cycles_fast(spec.ptx_source, m, n, k)
        _baseline_cache[key] = cycles
        if cycles is not None:
            logger.debug("Baseline %s: %d cycles", key, cycles)
    return _baseline_cache[key]


def clear_cache():
    """Clear baseline cache."""
    _baseline_cache.clear()
