"""PTX compilation for L4 GPU (sm_89).

Uses ptxas for PTX -> cubin compilation and CuPy for loading/execution.
Supports SASS dump via cuobjdump for verifying instruction ordering.

Why ptxas over NVRTC:
  NVRTC compiles CUDA C, not raw PTX. For raw PTX strings we need
  ptxas (PTX -> cubin) then CuPy RawModule(path=cubin) to load.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompilationResult:
    """Result of PTX compilation."""
    success: bool
    cubin_path: Optional[str] = None
    error: Optional[str] = None
    ptxas_info: Optional[str] = None


def _find_ptxas() -> str:
    """Find ptxas binary compatible with the target GPU.

    Search order:
      1. CHRONOS_PTXAS environment variable
      2. pip-installed nvidia-cuda-nvcc (CUDA 12.x, supports sm_89)
      3. System PATH (may be older CUDA version)
    """
    custom = os.environ.get('CHRONOS_PTXAS')
    if custom and os.path.isfile(custom):
        return custom

    # pip-installed nvidia-cuda-nvcc has sm_89 support
    try:
        import nvidia.cuda_nvcc
        for base in nvidia.cuda_nvcc.__path__:
            pip_ptxas = os.path.join(base, 'bin', 'ptxas')
            if os.path.isfile(pip_ptxas):
                return pip_ptxas
    except ImportError:
        pass

    # Fallback: check common pip site-packages paths
    import site
    for sp in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) \
            else [site.getusersitepackages()]:
        candidate = os.path.join(sp, 'nvidia', 'cuda_nvcc', 'bin', 'ptxas')
        if os.path.isfile(candidate):
            return candidate

    return 'ptxas'


def detect_gpu_arch() -> str:
    """Detect GPU compute capability via CuPy. Falls back to sm_89."""
    try:
        import cupy as cp
        cc = cp.cuda.Device(0).compute_capability
        return f'sm_{cc}'
    except Exception:
        return 'sm_89'


def compile_ptx(ptx_source: str, arch: Optional[str] = None,
                output_dir: Optional[str] = None) -> CompilationResult:
    """Compile PTX source string to cubin using ptxas.

    Args:
        ptx_source: Full PTX source code string.
        arch: Target GPU architecture (e.g. 'sm_89'). Auto-detected if None.
        output_dir: Directory for output files. Uses temp dir if None.

    Returns:
        CompilationResult with cubin path on success, error message on failure.
    """
    if arch is None:
        arch = detect_gpu_arch()

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='chronos_')

    ptx_path = os.path.join(output_dir, 'kernel.ptx')
    cubin_path = os.path.join(output_dir, 'kernel.cubin')

    with open(ptx_path, 'w') as f:
        f.write(ptx_source)

    ptxas_bin = _find_ptxas()

    try:
        result = subprocess.run(
            [ptxas_bin, f'--gpu-name={arch}', '-v',
             '-o', cubin_path, ptx_path],
            capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        return CompilationResult(
            success=False,
            error=f"ptxas not found at '{ptxas_bin}'. Install CUDA toolkit or set CHRONOS_PTXAS."
        )
    except subprocess.TimeoutExpired:
        return CompilationResult(success=False, error="ptxas timed out (30s).")

    if result.returncode != 0:
        return CompilationResult(
            success=False,
            error=result.stderr.strip()
        )

    return CompilationResult(
        success=True,
        cubin_path=cubin_path,
        # ptxas -v prints register usage and other info to stderr
        ptxas_info=result.stderr.strip() if result.stderr.strip() else None
    )


def load_kernel(cubin_path: str, kernel_name: str):
    """Load a compiled cubin and return the kernel function.

    Args:
        cubin_path: Path to compiled .cubin file.
        kernel_name: Name of the kernel entry point.

    Returns:
        CuPy kernel function that can be launched via kernel(grid, block, args).
    """
    import cupy as cp
    mod = cp.RawModule(path=cubin_path)
    return mod.get_function(kernel_name)


def get_sass(cubin_path: str) -> Optional[str]:
    """Dump SASS assembly from cubin for inspection.

    Useful for verifying that ptxas preserved our instruction ordering
    (or to see how it reordered things).
    """
    try:
        result = subprocess.run(
            ['cuobjdump', '--dump-sass', cubin_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def compile_and_load(ptx_source: str, kernel_name: str,
                     arch: Optional[str] = None):
    """Compile PTX and load kernel in one step.

    Args:
        ptx_source: Full PTX source code string.
        kernel_name: Kernel entry point name.
        arch: Target architecture. Auto-detected if None.

    Returns:
        Tuple of (kernel_function, CompilationResult).

    Raises:
        RuntimeError: If compilation fails.
    """
    result = compile_ptx(ptx_source, arch=arch)
    if not result.success:
        raise RuntimeError(f"PTX compilation failed: {result.error}")

    kernel = load_kernel(result.cubin_path, kernel_name)
    return kernel, result
