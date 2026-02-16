"""Project Chronos Phase 0: PTX Compilation and Benchmarking Harness.

Compiles arbitrary PTX instruction orderings and benchmarks them on L4 (sm_89).
Main entry point: compile_and_run(ptx_source, kernel_name, args_factory, grid, block) -> float
"""

from .ptx_compiler import compile_ptx, compile_and_load, load_kernel, get_sass
from .benchmark import benchmark_kernel, compile_and_run, compare_schedules
from .ptx_templates import KernelSpec, vector_add, dot_product, gemm_tile, TEMPLATES
