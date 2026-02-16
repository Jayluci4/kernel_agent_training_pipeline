"""Store cache hint transform: add cache policies to global stores.

PTX global store cache policies for L4 (sm_89):
  wb - write-back (default), write to L2, dirty line written back later
  wt - write-through, write to both L1 and L2 simultaneously
  cs - cache streaming / evict-first (good for one-time writes)

Syntax: st.global.f32 -> st.global.wb.f32
The policy modifier goes between the state space and the data type.
"""

import re
from typing import Dict, List

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel
from .base import PtxTransform, TransformResult


STORE_CACHE_POLICIES = ("wb", "wt", "cs")

# Matches st.global with optional existing hint
# Group 1: "st.global"
# Group 2: existing hint if any
# Group 3: data type suffix (.f32, .v2.f32, etc.)
_ST_GLOBAL_PATTERN = re.compile(
    r'(st\.global)(?:\.(wb|wt|cs))?(\.(?:f32|f64|b32|b64|u32|u64|s32|s64|b16|u16|s16|f16|v2|v4)\S*)'
)


def _has_store_hint(opcode: str) -> bool:
    """Check if a store opcode already has a cache hint."""
    parts = opcode.split('.')
    if len(parts) >= 3:
        return parts[2] in STORE_CACHE_POLICIES
    return False


def _find_unhinted_stores(kernel: ParsedKernel) -> List[int]:
    """Return body indices of st.global instructions without cache hints."""
    indices = []
    for i, bl in enumerate(kernel.body):
        if bl.instruction is None:
            continue
        op = bl.instruction.opcode
        if op.startswith('st.global') and not _has_store_hint(op):
            indices.append(i)
    return indices


class StoreCacheHintTransform(PtxTransform):
    """Add cache policy hints to global store instructions."""

    name = "store_cache_hints"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        indices = _find_unhinted_stores(kernel)
        if not indices:
            return []
        options = []
        for idx in indices:
            for policy in STORE_CACHE_POLICIES:
                options.append({"target_idx": idx, "policy": policy})
        return options

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        target_idx = params["target_idx"]
        policy = params["policy"]

        new_kernel = deep_copy_kernel(kernel)
        bl = new_kernel.body[target_idx]

        if bl.instruction is None or not bl.instruction.opcode.startswith('st.global'):
            return TransformResult(kernel=new_kernel, changed=False)

        old_text = bl.raw_text
        new_text = _ST_GLOBAL_PATTERN.sub(
            rf'\1.{policy}\3', old_text
        )

        if new_text == old_text:
            return TransformResult(kernel=new_kernel, changed=False)

        new_kernel.body[target_idx] = BodyLine(
            tag="instruction",
            raw_text=new_text,
            instruction=bl.instruction,
        )

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={"target_idx": target_idx, "policy": policy},
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Hint all unhinted global stores with 'cs' (evict-first)."""
        return self.apply_all_with_policy(kernel, "cs")

    def apply_all_with_policy(self, kernel: ParsedKernel, policy: str) -> TransformResult:
        """Hint all unhinted stores with a specific policy."""
        indices = _find_unhinted_stores(kernel)
        if not indices:
            return TransformResult(kernel=kernel, changed=False)

        current = deep_copy_kernel(kernel)
        count = 0

        for idx in indices:
            bl = current.body[idx]
            old_text = bl.raw_text
            new_text = _ST_GLOBAL_PATTERN.sub(rf'\1.{policy}\3', old_text)

            if new_text != old_text:
                current.body[idx] = BodyLine(
                    tag="instruction",
                    raw_text=new_text,
                    instruction=bl.instruction,
                )
                count += 1

        return TransformResult(
            kernel=current,
            changed=count > 0,
            stats={"stores_hinted": count, "policy": policy},
        )
