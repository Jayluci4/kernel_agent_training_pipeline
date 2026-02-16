"""Extract scalar features from a ParsedKernel for ML training.

Features capture the transform-relevant state of PTX code:
  - Instruction counts by type (loads, stores, FMA, branches)
  - Vectorization state (how many loads/stores are vectorized)
  - Cache hint coverage (how many loads/stores have hints)
  - Register allocation state (declared regs, maxnreg budget)
  - Instruction mix ratios (compute vs memory intensity)

These features change as transforms are applied, making them suitable
for tracking state in an MDP: apply vec_ld → vec_ld_ratio increases,
apply maxnreg_128 → maxnreg changes, etc.
"""

import re
from typing import Dict

from ..transform.parsed_kernel import ParsedKernel


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


def extract_features(kernel: ParsedKernel) -> Dict[str, float]:
    """Extract scalar features from a ParsedKernel.

    Returns a dict of ~25 features. All values are numeric (int or float).
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
    }


# Feature names in stable order (for array conversion)
FEATURE_NAMES = [
    "n_instructions", "n_ld_global", "n_st_global", "n_fma",
    "n_ld_param", "n_prefetch", "n_branch",
    "n_ld_global_vec", "n_st_global_vec", "vec_ld_ratio", "vec_st_ratio",
    "n_cache_hint_ld", "n_cache_hint_st", "hint_ld_ratio", "hint_st_ratio",
    "load_ratio", "store_ratio", "fma_ratio", "compute_ratio",
    "mem_ratio", "compute_to_mem",
    "total_regs", "n_f32_regs", "n_b64_regs", "maxnreg",
]


def features_to_array(features: Dict[str, float]) -> list:
    """Convert feature dict to a fixed-order list for ML models."""
    return [features[name] for name in FEATURE_NAMES]
