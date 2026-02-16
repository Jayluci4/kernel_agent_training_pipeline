#!/bin/bash
# Project Chronos - CUDA environment setup for L4 (sm_89)
#
# Source this before running tests or experiments:
#   source experiments/chronos/setup_env.sh
#
# Required packages:
#   pip install cupy-cuda12x nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12

SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null)
if [ -z "$SITE_PACKAGES" ]; then
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)
fi

# Add NVIDIA library paths for CuPy runtime
NVIDIA_BASE="${SITE_PACKAGES}/nvidia"

if [ -d "${NVIDIA_BASE}/cuda_nvrtc/lib" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_BASE}/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${NVIDIA_BASE}/cuda_runtime/lib" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_BASE}/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${NVIDIA_BASE}/cublas/lib" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_BASE}/cublas/lib:${LD_LIBRARY_PATH:-}"
fi

echo "Chronos env configured. LD_LIBRARY_PATH updated."
echo "  ptxas: $(which ptxas 2>/dev/null || echo 'will use pip-installed version')"
echo "  GPU:   $(python3 -c 'import cupy; d=cupy.cuda.runtime.getDeviceProperties(0); print(d["name"].decode())' 2>/dev/null || echo 'unknown')"
