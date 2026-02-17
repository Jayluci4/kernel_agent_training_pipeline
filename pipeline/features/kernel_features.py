"""Extract scalar features from a ParsedKernel for ML training.

Features capture the transform-relevant state of PTX code:
  - Instruction counts by type (loads, stores, FMA, branches)
  - Vectorization state (how many loads/stores are vectorized)
  - Cache hint coverage (how many loads/stores have hints)
  - Register allocation state (declared regs, maxnreg budget)
  - Instruction mix ratios (compute vs memory intensity)
  - Global memory intensity (compute ops per global memory op)
  - Broad vector memory ratio (detects pre-vectorized code like Triton)
  - Vectorizable ratio (fraction of memory ops that can be vectorized)
  - Register pressure index (declared regs vs hardware limit)
  - Dependency chain depth (critical path length in register DAG)

These features change as transforms are applied, making them suitable
for tracking state in an MDP: apply vec_ld -> vec_ld_ratio increases,
apply maxnreg_128 -> maxnreg changes, etc.
"""

import re
from typing import Dict, List

from ..transform.parsed_kernel import ParsedKernel

# L4 (sm_89) hardware register limit per thread
_L4_MAX_REGS_PER_THREAD = 255


# -- Regex patterns for instruction classification --

_LD_GLOBAL = re.compile(r'ld\.global')
_LD_GLOBAL_VEC = re.compile(r'ld\.global(?:\.\w+)*\.v[24]')
_ST_GLOBAL = re.compile(r'st\.global')
_ST_GLOBAL_VEC = re.compile(r'st\.global(?:\.\w+)*\.v[24]')
_FMA = re.compile(r'\bfma\.')
_MUL = re.compile(r'\bmul\.')
_ADD = re.compile(r'\badd\.')
_MOV = re.compile(r'\bmov\.')
_CVT = re.compile(r'\bcvt\.')
_LD_PARAM = re.compile(r'ld\.param')
_PREFETCH = re.compile(r'prefetch\.global')
_CACHE_HINT_LD = re.compile(r'ld\.global\.(?:cs|cg|ca|cv)')
_CACHE_HINT_ST = re.compile(r'st\.global\.(?:wb|wt|cs)')
_MAXNREG = re.compile(r'\.maxnreg\s+(\d+)')

# Broad vector detection: catches Triton's L1::evict_last syntax and
# any other cache policy qualifiers that break the narrow \.\w+ pattern.
# Looks for .v followed by digit(s) anywhere after ld/st.global or ld/st.shared.
_VEC_DOT_V = re.compile(r'\.v\d+')
_MEM_GLOBAL_SHARED = re.compile(r'(?:ld|st)\.(?:global|shared)')

# Triton advanced memory patterns (not ld.global, but equivalent):
#   cp.async.ca.shared.global — async DMA global->shared (replaces ld.global in matmul)
#   ldmatrix.sync.aligned     — warp-level shared->register matrix load
_CP_ASYNC = re.compile(r'cp\.async\.\w+\.shared\.global')
_LDMATRIX = re.compile(r'ldmatrix\.sync')

# Shared memory loads/stores (for total memory accounting)
_LD_SHARED = re.compile(r'ld\.shared')
_ST_SHARED = re.compile(r'st\.shared')


def extract_features(kernel: ParsedKernel) -> Dict[str, float]:
    """Extract scalar features from a ParsedKernel.

    Returns a dict of 30 features. All values are numeric (int or float).
    Designed to be JSON-serializable for trajectory storage.
    """
    n_instr = 0
    n_ld_global = 0
    n_ld_global_vec = 0
    n_st_global = 0
    n_st_global_vec = 0
    n_fma = 0
    n_mul = 0
    n_add = 0
    n_mov = 0
    n_cvt = 0
    n_ld_param = 0
    n_prefetch = 0
    n_cache_hint_ld = 0
    n_cache_hint_st = 0
    n_branch = 0
    n_vec_mem_broad = 0      # any global/shared mem op with .v{N}
    n_mem_global_shared = 0  # any global/shared mem op
    n_cp_async = 0           # cp.async.*.shared.global (Triton DMA)
    n_ldmatrix = 0           # ldmatrix.sync (Triton matrix load)

    for bl in kernel.body:
        if bl.tag == "instruction":
            n_instr += 1
            t = bl.raw_text

            # Global loads
            if _LD_GLOBAL.search(t):
                n_ld_global += 1
                if _LD_GLOBAL_VEC.search(t):
                    n_ld_global_vec += 1
                if _CACHE_HINT_LD.search(t):
                    n_cache_hint_ld += 1

            # Global stores
            if _ST_GLOBAL.search(t):
                n_st_global += 1
                if _ST_GLOBAL_VEC.search(t):
                    n_st_global_vec += 1
                if _CACHE_HINT_ST.search(t):
                    n_cache_hint_st += 1

            # Broad vector memory detection (catches Triton patterns
            # like ld.global.L1::evict_last.v4.b32 that the narrow
            # _LD_GLOBAL_VEC regex misses)
            if _MEM_GLOBAL_SHARED.search(t):
                n_mem_global_shared += 1
                if _VEC_DOT_V.search(t):
                    n_vec_mem_broad += 1

            # Triton advanced memory ops
            if _CP_ASYNC.search(t):
                n_cp_async += 1
            if _LDMATRIX.search(t):
                n_ldmatrix += 1

            # Compute instructions
            if _FMA.search(t):
                n_fma += 1
            if _MUL.search(t):
                n_mul += 1
            if _ADD.search(t):
                n_add += 1
            if _MOV.search(t):
                n_mov += 1
            if _CVT.search(t):
                n_cvt += 1

            # Param loads
            if _LD_PARAM.search(t):
                n_ld_param += 1

            # Prefetch
            if _PREFETCH.search(t):
                n_prefetch += 1

        elif bl.tag == "branch":
            n_branch += 1

    # Check preamble for .maxnreg directive
    maxnreg = 0
    m = _MAXNREG.search(kernel.preamble)
    if m:
        maxnreg = int(m.group(1))

    # Register counts from declarations
    total_regs = sum(kernel.reg_decls.values())
    n_f32_regs = kernel.reg_decls.get('.f32', 0)
    n_b64_regs = kernel.reg_decls.get('.b64', 0)
    n_pred_regs = kernel.reg_decls.get('.pred', 0)

    # Derived values
    n_total = max(n_instr, 1)
    n_compute = n_fma + n_mul + n_add
    n_mem = n_ld_global + n_st_global

    # ── Feature A: Vectorizable Ratio (The Anti-Triton Feature) ──────
    # Fraction of global memory ops sitting in coalescable groups.
    # Includes BOTH loads (via vectorize_loads) AND stores (via vectorize_stores).
    # High: scalar code with consecutive accesses -> vec_ld/vec_st will help.
    # Zero: pre-vectorized (Triton) or irregular access -> skip vectorization.
    vectorizable_ratio = _compute_vectorizable_ratio(
        kernel, n_ld_global, n_ld_global_vec, n_st_global, n_st_global_vec,
    )

    # ── Feature B: Global Memory Intensity (The Cache Guard) ────────
    # Compute ops per global memory op.  FMA = 2 FLOPs.
    # High (>5): compute-bound -> cache_cg keeps data in L1 for reuse.
    # Low  (<1): memory-bound  -> cache_cg pollutes L1, use cache_cs.
    n_flops = n_fma * 2 + n_add + n_mul
    global_memory_intensity = round(
        n_flops / max(n_ld_global + n_st_global, 1), 4
    )

    # ── Feature C: Register Pressure Index (The Budget Selector) ────
    # Declared registers / hardware limit (255 on L4 sm_89).
    # High (>0.5): register-hungry -> maxnreg can force spill trade-off.
    # Low  (<0.2): register-light -> maxnreg has no effect.
    register_pressure_index = round(
        total_regs / _L4_MAX_REGS_PER_THREAD, 4
    )

    # ── Feature D: Dependency Chain Depth (The Reorder Trigger) ─────
    # Longest path in register-dependency DAG / total instructions.
    # High (>0.8): serialized chain -> reorder can overlap independent ops.
    # Low  (<0.3): already parallel -> reorder is pointless.
    dependency_chain_depth = _compute_dependency_chain_depth(kernel, n_instr)

    # ── Advanced Memory Ratio (Triton Detector) ─────────────────────
    # Fraction of memory traffic using advanced patterns:
    #   .v{N} vectors, cp.async DMA, ldmatrix warp loads.
    # High (>0.3): pre-optimized code, scalar transforms irrelevant.
    # Zero: simple ld/st.global, all transforms apply.
    n_advanced_mem = n_vec_mem_broad + n_cp_async + n_ldmatrix
    n_all_mem = n_mem_global_shared + n_cp_async + n_ldmatrix
    vec_mem_ratio = round(
        n_advanced_mem / max(n_all_mem, 1), 4
    )

    return {
        # Raw instruction counts
        "n_instructions": n_instr,
        "n_ld_global": n_ld_global,
        "n_st_global": n_st_global,
        "n_fma": n_fma,
        "n_ld_param": n_ld_param,
        "n_prefetch": n_prefetch,
        "n_branch": n_branch,

        # Vectorization state
        "n_ld_global_vec": n_ld_global_vec,
        "n_st_global_vec": n_st_global_vec,
        "vec_ld_ratio": round(n_ld_global_vec / max(n_ld_global, 1), 4),
        "vec_st_ratio": round(n_st_global_vec / max(n_st_global, 1), 4),

        # Cache hint state
        "n_cache_hint_ld": n_cache_hint_ld,
        "n_cache_hint_st": n_cache_hint_st,
        "hint_ld_ratio": round(n_cache_hint_ld / max(n_ld_global, 1), 4),
        "hint_st_ratio": round(n_cache_hint_st / max(n_st_global, 1), 4),

        # Instruction mix ratios
        "load_ratio": round(n_ld_global / n_total, 4),
        "store_ratio": round(n_st_global / n_total, 4),
        "fma_ratio": round(n_fma / n_total, 4),
        "compute_ratio": round(n_compute / n_total, 4),
        "mem_ratio": round(n_mem / n_total, 4),
        "compute_to_mem": round(n_compute / max(n_mem, 1), 4),

        # Register state
        "total_regs": total_regs,
        "n_f32_regs": n_f32_regs,
        "n_b64_regs": n_b64_regs,
        "maxnreg": maxnreg,

        # Phase C features
        "vectorizable_ratio": vectorizable_ratio,
        "global_memory_intensity": global_memory_intensity,
        "register_pressure_index": register_pressure_index,
        "dependency_chain_depth": dependency_chain_depth,
        "vec_mem_ratio": vec_mem_ratio,
    }


def _compute_vectorizable_ratio(
    kernel: ParsedKernel,
    n_ld_global: int,
    n_ld_global_vec: int,
    n_st_global: int,
    n_st_global_vec: int,
) -> float:
    """Fraction of global memory ops sitting in vectorizable groups.

    Counts both loads (via _find_vectorizable_groups from vectorize_loads.py)
    and stores (via _find_vectorizable_store_groups from vectorize_stores.py).
    Denominator is all scalar global memory ops (loads + stores minus already-vectorized).

    Returns 0.0 if no scalar memory ops exist.
    """
    from ..transform.vectorize_loads import _find_vectorizable_groups
    from ..transform.vectorize_stores import _find_vectorizable_store_groups

    n_scalar_ld = n_ld_global - n_ld_global_vec
    n_scalar_st = n_st_global - n_st_global_vec
    n_scalar_mem = n_scalar_ld + n_scalar_st
    if n_scalar_mem <= 0:
        return 0.0

    n_coalescable = 0
    try:
        ld_groups = _find_vectorizable_groups(kernel)
        n_coalescable += sum(g["group_size"] for g in ld_groups)
    except Exception:
        pass

    try:
        st_groups = _find_vectorizable_store_groups(kernel)
        n_coalescable += sum(g["group_size"] for g in st_groups)
    except Exception:
        pass

    return round(n_coalescable / n_scalar_mem, 4)


def _compute_dependency_chain_depth(
    kernel: ParsedKernel,
    n_instr: int,
) -> float:
    """Longest path in register-dependency DAG normalized by instruction count.

    Builds a DAG from register def-use chains: instruction B depends on
    instruction A if B reads a register that A wrote. The longest path
    (critical path) determines the minimum serial execution length.

    High (>0.8): serialized chain, reorder can overlap independent work.
    Low  (<0.3): already parallel, reorder has no effect.
    """
    if n_instr <= 1:
        return 0.0

    # last_writer[reg_name] = index into instruction sequence that last wrote it
    last_writer: dict = {}
    # longest_path[i] = longest path ending at instruction i
    longest_path: List[int] = []

    for bl in kernel.body:
        if bl.tag != "instruction" or bl.instruction is None:
            continue

        inst = bl.instruction
        # Find max path length among all producers of our source regs
        max_dep = 0
        for reg in inst.src_regs:
            if reg in last_writer:
                max_dep = max(max_dep, longest_path[last_writer[reg]])

        path_len = max_dep + 1
        idx = len(longest_path)
        longest_path.append(path_len)

        # Update last writer for each destination register
        for reg in inst.dest_regs:
            last_writer[reg] = idx

    if not longest_path:
        return 0.0

    return round(max(longest_path) / max(n_instr, 1), 4)


# Feature names in stable order (for array conversion)
FEATURE_NAMES = [
    "n_instructions", "n_ld_global", "n_st_global", "n_fma",
    "n_ld_param", "n_prefetch", "n_branch",
    "n_ld_global_vec", "n_st_global_vec", "vec_ld_ratio", "vec_st_ratio",
    "n_cache_hint_ld", "n_cache_hint_st", "hint_ld_ratio", "hint_st_ratio",
    "load_ratio", "store_ratio", "fma_ratio", "compute_ratio",
    "mem_ratio", "compute_to_mem",
    "total_regs", "n_f32_regs", "n_b64_regs", "maxnreg",
    "vectorizable_ratio", "global_memory_intensity",
    "register_pressure_index", "dependency_chain_depth",
    "vec_mem_ratio",
]


def features_to_array(features: Dict[str, float]) -> list:
    """Convert feature dict to a fixed-order list for ML models."""
    return [features[name] for name in FEATURE_NAMES]
