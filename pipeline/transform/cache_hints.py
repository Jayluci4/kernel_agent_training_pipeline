"""Cache hint transform: add cache policies to global loads.

PTX global load cache policies for L4 (sm_89):
  ca - cache at all levels (L1 + L2), default behavior
  cg - cache in L2, bypass L1 (good for streaming reads)
  cs - cache streaming / evict-first (good for one-time reads)
  cv - volatile, bypass all caches (for coherency)

Syntax: ld.global.f32 -> ld.global.ca.f32
The policy modifier goes between the state space and the data type.
"""

import re
from typing import Dict, List

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel, get_instructions
from .base import PtxTransform, TransformResult


CACHE_POLICIES = ("ca", "cg", "cs", "cv")

# Matches ld.global with optional existing hint, capturing the parts
# Group 1: everything before the data type (e.g., "ld.global" or "ld.global.ca")
# Group 2: existing hint if any
_LD_GLOBAL_PATTERN = re.compile(
    r'(ld\.global)(?:\.(ca|cg|cs|cv))?(\.(?:f32|f64|b32|b64|u32|u64|s32|s64|b16|u16|s16|f16|v2|v4)\S*)'
)


def _has_cache_hint(opcode: str) -> bool:
    """Check if a load opcode already has a cache hint."""
    parts = opcode.split('.')
    # ld.global.f32 -> no hint
    # ld.global.ca.f32 -> has hint
    if len(parts) >= 3:
        return parts[2] in CACHE_POLICIES
    return False


def _find_unhinted_loads(kernel: ParsedKernel) -> List[int]:
    """Return body indices of ld.global instructions without cache hints."""
    indices = []
    for i, bl in enumerate(kernel.body):
        if bl.instruction is None:
            continue
        op = bl.instruction.opcode
        if op.startswith('ld.global') and not _has_cache_hint(op):
            indices.append(i)
    return indices


class CacheHintTransform(PtxTransform):
    """Add cache policy hints to global load instructions."""

    name = "cache_hints"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        indices = _find_unhinted_loads(kernel)
        if not indices:
            return []
        # Return one option per (load_index, policy) pair
        options = []
        for idx in indices:
            for policy in CACHE_POLICIES:
                options.append({"target_idx": idx, "policy": policy})
        return options

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        target_idx = params["target_idx"]
        policy = params["policy"]

        new_kernel = deep_copy_kernel(kernel)
        bl = new_kernel.body[target_idx]

        if bl.instruction is None or not bl.instruction.opcode.startswith('ld.global'):
            return TransformResult(kernel=new_kernel, changed=False)

        # Insert policy into raw_text: ld.global.f32 -> ld.global.ca.f32
        old_text = bl.raw_text
        new_text = _LD_GLOBAL_PATTERN.sub(
            rf'\1.{policy}\3', old_text
        )

        if new_text == old_text:
            return TransformResult(kernel=new_kernel, changed=False)

        # Update the body line
        new_kernel.body[target_idx] = BodyLine(
            tag="instruction",
            raw_text=new_text,
            instruction=bl.instruction,  # opcode metadata unchanged for scheduling
        )

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={"target_idx": target_idx, "policy": policy},
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Hint all unhinted global loads with 'ca' (cache all levels)."""
        indices = _find_unhinted_loads(kernel)
        if not indices:
            return TransformResult(kernel=kernel, changed=False)

        current = deep_copy_kernel(kernel)
        count = 0

        for idx in indices:
            bl = current.body[idx]
            old_text = bl.raw_text
            new_text = _LD_GLOBAL_PATTERN.sub(r'\1.ca\3', old_text)

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
            stats={"loads_hinted": count, "policy": "ca"},
        )
