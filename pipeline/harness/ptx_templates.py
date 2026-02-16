"""Parameterized PTX kernel templates for L4 (sm_89).

Kernels of increasing complexity for harness validation:
  1. vector_add      - element-wise c = a + b (load/store/compute)
  2. dot_product     - warp-level reduction (shuffle instructions)
  3. gemm_tile       - MxNxK matrix multiply (FMA interleaving with loads)
  4. gemm_tile_large - 8x8x8 variant (~771 instructions, richer scheduling)

Each function returns a KernelSpec containing PTX source, launch config,
argument factory, and output validator.
"""

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class KernelSpec:
    """Complete specification for a benchmarkable kernel."""
    name: str
    ptx_source: str
    kernel_name: str
    grid: Tuple[int, ...]
    block: Tuple[int, ...]
    args_factory: Callable   # () -> tuple of cupy arrays/scalars
    validator: Callable      # (args_tuple) -> bool
    description: str = ""


def vector_add(n: int = 1024) -> KernelSpec:
    """Element-wise vector addition: c[i] = a[i] + b[i].

    Tests basic load, store, and arithmetic pipeline.
    Launches n/256 blocks of 256 threads each.
    """
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    ptx_source = """\
.version 7.8
.target sm_89
.address_size 64

.visible .entry vector_add(
    .param .u64 ptr_a,
    .param .u64 ptr_b,
    .param .u64 ptr_c,
    .param .u32 n
) {
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;
    .reg .b32 %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [ptr_a];
    ld.param.u64 %rd1, [ptr_b];
    ld.param.u64 %rd2, [ptr_c];
    ld.param.u32 %r0, [n];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;

    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra $L_EXIT;

    mul.wide.u32 %rd3, %r4, 4;

    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];

    add.f32 %f2, %f0, %f1;

    add.u64 %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f2;

$L_EXIT:
    ret;
}
"""

    def args_factory():
        import cupy as cp
        a = cp.random.randn(n).astype(cp.float32)
        b = cp.random.randn(n).astype(cp.float32)
        c = cp.zeros(n, dtype=cp.float32)
        return (a, b, c, cp.uint32(n))

    def validator(args):
        import cupy as cp
        a, b, c, _ = args
        expected = a + b
        return bool(cp.allclose(c, expected, atol=1e-5))

    return KernelSpec(
        name='vector_add',
        ptx_source=ptx_source,
        kernel_name='vector_add',
        grid=(grid_size,),
        block=(block_size,),
        args_factory=args_factory,
        validator=validator,
        description=f'Element-wise addition of {n}-element vectors',
    )


def dot_product(n: int = 32) -> KernelSpec:
    """Warp-level dot product using shuffle butterfly reduction.

    Single warp (32 threads) computes dot(a[:32], b[:32]).
    Tests warp shuffle instructions (shfl.sync.bfly) which are
    important for L4 scheduling - the MCTS agent must learn when
    to issue shuffles vs compute.
    """
    ptx_source = """\
.version 7.8
.target sm_89
.address_size 64

.visible .entry dot_product(
    .param .u64 ptr_a,
    .param .u64 ptr_b,
    .param .u64 ptr_out
) {
    .reg .f32 %f<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [ptr_a];
    ld.param.u64 %rd1, [ptr_b];
    ld.param.u64 %rd2, [ptr_out];

    mov.u32 %r0, %tid.x;

    // Load a[tid] and b[tid]
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // product = a[tid] * b[tid]
    mul.f32 %f2, %f0, %f1;

    // Butterfly reduction across full warp (32 threads)
    // Stage 1: XOR with 16 (pairs lanes 0-15 with 16-31)
    mov.b32 %r1, %f2;
    shfl.sync.bfly.b32 %r2, %r1, 16, 31, 0xFFFFFFFF;
    mov.b32 %f3, %r2;
    add.f32 %f2, %f2, %f3;

    // Stage 2: XOR with 8
    mov.b32 %r1, %f2;
    shfl.sync.bfly.b32 %r2, %r1, 8, 31, 0xFFFFFFFF;
    mov.b32 %f3, %r2;
    add.f32 %f2, %f2, %f3;

    // Stage 3: XOR with 4
    mov.b32 %r1, %f2;
    shfl.sync.bfly.b32 %r2, %r1, 4, 31, 0xFFFFFFFF;
    mov.b32 %f3, %r2;
    add.f32 %f2, %f2, %f3;

    // Stage 4: XOR with 2
    mov.b32 %r1, %f2;
    shfl.sync.bfly.b32 %r2, %r1, 2, 31, 0xFFFFFFFF;
    mov.b32 %f3, %r2;
    add.f32 %f2, %f2, %f3;

    // Stage 5: XOR with 1
    mov.b32 %r1, %f2;
    shfl.sync.bfly.b32 %r2, %r1, 1, 31, 0xFFFFFFFF;
    mov.b32 %f3, %r2;
    add.f32 %f2, %f2, %f3;

    // Thread 0 writes the final result
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra $L_EXIT;
    st.global.f32 [%rd2], %f2;

$L_EXIT:
    ret;
}
"""

    def args_factory():
        import cupy as cp
        a = cp.random.randn(32).astype(cp.float32)
        b = cp.random.randn(32).astype(cp.float32)
        out = cp.zeros(1, dtype=cp.float32)
        return (a, b, out)

    def validator(args):
        import numpy as np
        a, b, out = args
        # Use numpy on CPU to avoid cublas dependency
        expected = float(np.dot(a.get().astype(np.float64), b.get().astype(np.float64)))
        actual = float(out[0])
        # Relaxed tolerance for warp reduction (float32 associativity)
        return abs(actual - expected) < max(abs(expected) * 1e-3, 1e-2)

    return KernelSpec(
        name='dot_product',
        ptx_source=ptx_source,
        kernel_name='dot_product',
        grid=(1,),
        block=(32,),
        args_factory=args_factory,
        validator=validator,
        description='Warp-level dot product with shuffle butterfly reduction',
    )


def gemm_tile(m: int = 4, n: int = 4, k: int = 4) -> KernelSpec:
    """Matrix multiply tile: C[m,n] = A[m,k] * B[k,n].

    Single thread computes the full tile using scalar FMA instructions.
    This is the primary scheduling target for Chronos.

    Register allocation (generalized for arbitrary m, n, k):
      A: %f0           .. %f{m*k-1}
      B: %f{m*k}       .. %f{m*k+k*n-1}
      C: %f{m*k+k*n}   .. %f{m*k+k*n+m*n-1}

    Instruction counts:
      3 param loads + m*n inits + m*k A loads + k*n B loads + m*n*k FMAs + m*n stores
      For 4x4x4: 3 + 16 + 16 + 16 + 64 + 16 = 131
      For 8x8x8: 3 + 64 + 64 + 64 + 512 + 64 = 771

    Memory layout: row-major flat arrays.
    """
    # Computed register offsets (generalized, not hardcoded)
    b_offset = m * k
    c_offset = m * k + k * n

    # Build FMA body: C[i,j] += A[i,kk] * B[kk,j]
    fma_lines = []
    for i in range(m):
        for j in range(n):
            c_reg = c_offset + i * n + j
            for kk in range(k):
                a_reg = i * k + kk
                b_reg = b_offset + kk * n + j
                fma_lines.append(
                    f"    fma.rn.f32 %f{c_reg}, %f{a_reg}, %f{b_reg}, %f{c_reg};"
                )

    fma_body = "\n".join(fma_lines)

    # Load A
    load_a_lines = []
    for i in range(m * k):
        offset = i * 4
        load_a_lines.append(f"    ld.global.f32 %f{i}, [%rd3+{offset}];")
    load_a = "\n".join(load_a_lines)

    # Load B
    load_b_lines = []
    for i in range(k * n):
        reg = b_offset + i
        offset = i * 4
        load_b_lines.append(f"    ld.global.f32 %f{reg}, [%rd4+{offset}];")
    load_b = "\n".join(load_b_lines)

    # Initialize C accumulators to zero
    init_c_lines = []
    for i in range(m * n):
        reg = c_offset + i
        init_c_lines.append(f"    mov.f32 %f{reg}, 0f00000000;")
    init_c = "\n".join(init_c_lines)

    # Store C
    store_c_lines = []
    for i in range(m * n):
        reg = c_offset + i
        offset = i * 4
        store_c_lines.append(f"    st.global.f32 [%rd5+{offset}], %f{reg};")
    store_c = "\n".join(store_c_lines)

    total_fregs = c_offset + m * n + 4  # A + B + C + headroom

    ptx_source = f"""\
.version 7.8
.target sm_89
.address_size 64

.visible .entry gemm_tile(
    .param .u64 ptr_a,
    .param .u64 ptr_b,
    .param .u64 ptr_c
) {{
    .reg .f32 %f<{total_fregs}>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd3, [ptr_a];
    ld.param.u64 %rd4, [ptr_b];
    ld.param.u64 %rd5, [ptr_c];

    // Initialize C accumulators to zero
{init_c}

    // Load A tile ({m}x{k})
{load_a}

    // Load B tile ({k}x{n})
{load_b}

    // Compute C = A * B ({m*n*k} FMA instructions)
{fma_body}

    // Store C tile ({m}x{n})
{store_c}

    ret;
}}
"""

    def args_factory():
        import cupy as cp
        a = cp.random.randn(m * k).astype(cp.float32)
        b = cp.random.randn(k * n).astype(cp.float32)
        c = cp.zeros(m * n, dtype=cp.float32)
        return (a, b, c)

    def validator(args):
        import numpy as np
        a, b, c = args
        # Use numpy on CPU to avoid cublas dependency
        a_np = a.get().reshape(m, k).astype(np.float64)
        b_np = b.get().reshape(k, n).astype(np.float64)
        c_np = c.get().astype(np.float64)
        expected = (a_np @ b_np).flatten()
        return bool(np.allclose(c_np, expected, atol=1e-3))

    return KernelSpec(
        name='gemm_tile',
        ptx_source=ptx_source,
        kernel_name='gemm_tile',
        grid=(1,),
        block=(1,),  # Single thread computes the full tile
        args_factory=args_factory,
        validator=validator,
        description=f'{m}x{n}x{k} matrix multiply tile ({m*n*k} FMAs)',
    )


# Registry of all templates
TEMPLATES = {
    'vector_add': vector_add,
    'dot_product': dot_product,
    'gemm_tile': gemm_tile,
    'gemm_tile_large': lambda: gemm_tile(m=8, n=8, k=8),
}
