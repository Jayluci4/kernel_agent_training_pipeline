"""Vectorize stores transform: merge scalar stores into vector stores.

Finds groups of 2 or 4 consecutive st.global.f32 instructions storing
to sequential addresses (same base register, offsets +0, +4, +8, +12)
and merges them into a single st.global.v2.f32 or st.global.v4.f32.

PTX syntax:
  Before: st.global.f32 [%rd5+0], %f32;
          st.global.f32 [%rd5+4], %f33;

  After:  st.global.v2.f32 [%rd5+0], {%f32, %f33};
"""

import re
from typing import Dict, List, Optional, Tuple

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel
from .base import PtxTransform, TransformResult


# Parse a scalar global store: st.global[.hint].f32 [%base+offset], %src;
_SCALAR_STORE_PATTERN = re.compile(
    r'st\.global(?:\.\w+)?\.f32\s+\[(%\w+)(?:\+(\d+))?\],\s*(%\w+)'
)


def _parse_store(raw_text: str) -> Optional[Tuple[str, int, str]]:
    """Parse a scalar global f32 store.

    Returns (base_reg, byte_offset, src_reg) or None.
    """
    m = _SCALAR_STORE_PATTERN.search(raw_text)
    if m:
        base = m.group(1)
        offset = int(m.group(2)) if m.group(2) else 0
        src = m.group(3)
        return base, offset, src
    return None


def _find_vectorizable_store_groups(kernel: ParsedKernel) -> List[Dict]:
    """Find groups of consecutive scalar stores that can be vectorized.

    Returns list of dicts with:
      body_indices: list of body line indices
      src_regs: list of source registers
      base_reg: shared base register
      first_offset: byte offset of first store
      group_size: 2 or 4
    """
    stores = []
    for i, bl in enumerate(kernel.body):
        if bl.instruction is None:
            continue
        if not bl.instruction.opcode.startswith('st.global'):
            continue
        parsed = _parse_store(bl.raw_text)
        if parsed:
            base, offset, src = parsed
            stores.append({
                "body_idx": i,
                "src_reg": src,
                "base_reg": base,
                "offset": offset,
            })

    if len(stores) < 2:
        return []

    groups = []
    used = set()

    for group_size in (4, 2):
        for start in range(len(stores)):
            if start in used:
                continue
            if start + group_size > len(stores):
                continue

            candidates = stores[start:start + group_size]

            # Same base register
            base = candidates[0]["base_reg"]
            if not all(c["base_reg"] == base for c in candidates):
                continue

            # Consecutive body indices (no intervening instructions)
            body_indices = [c["body_idx"] for c in candidates]
            inst_slots = []
            for idx in range(body_indices[0], body_indices[-1] + 1):
                if kernel.body[idx].instruction is not None:
                    inst_slots.append(idx)
            if inst_slots != body_indices:
                continue

            # Sequential offsets (+4 bytes apart for f32)
            first_offset = candidates[0]["offset"]
            expected = [first_offset + 4 * j for j in range(group_size)]
            actual = [c["offset"] for c in candidates]
            if actual != expected:
                continue

            # None already used
            indices_set = set(range(start, start + group_size))
            if indices_set & used:
                continue

            groups.append({
                "body_indices": body_indices,
                "src_regs": [c["src_reg"] for c in candidates],
                "base_reg": base,
                "first_offset": first_offset,
                "group_size": group_size,
            })
            used |= indices_set

    return groups


class VectorizeStoresTransform(PtxTransform):
    """Merge consecutive scalar stores into vector stores."""

    name = "vectorize_stores"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        groups = _find_vectorizable_store_groups(kernel)
        return [
            {
                "group_idx": i,
                "group_size": g["group_size"],
                "base_reg": g["base_reg"],
                "first_offset": g["first_offset"],
            }
            for i, g in enumerate(groups)
        ]

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        group_idx = params["group_idx"]
        groups = _find_vectorizable_store_groups(kernel)

        if group_idx >= len(groups):
            return TransformResult(kernel=kernel, changed=False)

        group = groups[group_idx]
        return self._apply_group(kernel, group)

    def _apply_group(self, kernel: ParsedKernel, group: Dict) -> TransformResult:
        new_kernel = deep_copy_kernel(kernel)

        body_indices = group["body_indices"]
        src_regs = group["src_regs"]
        base_reg = group["base_reg"]
        first_offset = group["first_offset"]
        group_size = group["group_size"]

        # Build vector store: st.global.v2.f32 [%rd5+0], {%f32, %f33};
        regs_str = ", ".join(src_regs)
        offset_str = f"+{first_offset}" if first_offset > 0 else ""
        vec_text = f"    st.global.v{group_size}.f32 [{base_reg}{offset_str}], {{{regs_str}}};"

        # Replace first store with vector store
        new_kernel.body[body_indices[0]] = BodyLine(
            tag="instruction",
            raw_text=vec_text,
            instruction=None,
        )

        # Remove subsequent stores (reverse order to keep indices valid)
        for idx in reversed(body_indices[1:]):
            del new_kernel.body[idx]

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={
                "group_size": group_size,
                "src_regs": src_regs,
                "base_reg": base_reg,
            },
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Vectorize all store groups (largest first)."""
        groups = _find_vectorizable_store_groups(kernel)
        if not groups:
            return TransformResult(kernel=kernel, changed=False)

        current = deep_copy_kernel(kernel)
        total_vectorized = 0

        sorted_groups = sorted(groups, key=lambda g: g["body_indices"][0], reverse=True)

        for group in sorted_groups:
            body_indices = group["body_indices"]
            src_regs = group["src_regs"]
            base_reg = group["base_reg"]
            first_offset = group["first_offset"]
            group_size = group["group_size"]

            regs_str = ", ".join(src_regs)
            offset_str = f"+{first_offset}" if first_offset > 0 else ""
            vec_text = f"    st.global.v{group_size}.f32 [{base_reg}{offset_str}], {{{regs_str}}};"

            current.body[body_indices[0]] = BodyLine(
                tag="instruction",
                raw_text=vec_text,
                instruction=None,
            )
            for idx in reversed(body_indices[1:]):
                del current.body[idx]

            total_vectorized += 1

        return TransformResult(
            kernel=current,
            changed=total_vectorized > 0,
            stats={"groups_vectorized": total_vectorized},
        )
