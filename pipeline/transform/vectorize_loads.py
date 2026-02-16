"""Vectorize loads transform: merge scalar loads into vector loads.

Finds groups of 2 or 4 consecutive ld.global.f32 instructions loading
from sequential addresses (same base register, offsets +0, +4, +8, +12)
and merges them into a single ld.global.v2.f32 or ld.global.v4.f32.

PTX syntax:
  Before: ld.global.f32 %f0, [%rd3+0];
          ld.global.f32 %f1, [%rd3+4];
          ld.global.f32 %f2, [%rd3+8];
          ld.global.f32 %f3, [%rd3+12];

  After:  ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3+0];
"""

import re
from typing import Dict, List, Optional, Tuple

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel
from .base import PtxTransform, TransformResult


# Parse a scalar global load: ld.global[.hint].f32 %dest, [%base+offset];
_SCALAR_LOAD_PATTERN = re.compile(
    r'ld\.global(?:\.\w+)?\.f32\s+(%\w+),\s*\[(%\w+)(?:\+(\d+))?\]'
)


def _parse_load(raw_text: str) -> Optional[Tuple[str, str, int]]:
    """Parse a scalar global f32 load.

    Returns (dest_reg, base_reg, byte_offset) or None.
    """
    m = _SCALAR_LOAD_PATTERN.search(raw_text)
    if m:
        dest = m.group(1)
        base = m.group(2)
        offset = int(m.group(3)) if m.group(3) else 0
        return dest, base, offset
    return None


def _find_vectorizable_groups(kernel: ParsedKernel) -> List[Dict]:
    """Find groups of consecutive scalar loads that can be vectorized.

    Returns list of dicts with:
      body_indices: list of body line indices
      dest_regs: list of destination registers
      base_reg: shared base register
      first_offset: byte offset of first load
      group_size: 2 or 4
    """
    # Collect all scalar f32 global loads with their body indices
    loads = []
    for i, bl in enumerate(kernel.body):
        if bl.instruction is None:
            continue
        if not bl.instruction.opcode.startswith('ld.global'):
            continue
        parsed = _parse_load(bl.raw_text)
        if parsed:
            dest, base, offset = parsed
            loads.append({
                "body_idx": i,
                "dest_reg": dest,
                "base_reg": base,
                "offset": offset,
            })

    if len(loads) < 2:
        return []

    groups = []
    used = set()

    # Try groups of 4 first, then 2
    for group_size in (4, 2):
        for start in range(len(loads)):
            if start in used:
                continue
            if start + group_size > len(loads):
                continue

            candidates = loads[start:start + group_size]

            # Check: same base register
            base = candidates[0]["base_reg"]
            if not all(c["base_reg"] == base for c in candidates):
                continue

            # Check: body indices are consecutive instruction slots
            # (no intervening instructions between them)
            body_indices = [c["body_idx"] for c in candidates]
            inst_slots_between = []
            for idx in range(body_indices[0], body_indices[-1] + 1):
                if kernel.body[idx].instruction is not None:
                    inst_slots_between.append(idx)
            if inst_slots_between != body_indices:
                continue

            # Check: sequential offsets (+4 bytes apart for f32)
            first_offset = candidates[0]["offset"]
            expected = [first_offset + 4 * j for j in range(group_size)]
            actual = [c["offset"] for c in candidates]
            if actual != expected:
                continue

            # Check: none already used
            indices_set = set(range(start, start + group_size))
            if indices_set & used:
                continue

            groups.append({
                "body_indices": body_indices,
                "dest_regs": [c["dest_reg"] for c in candidates],
                "base_reg": base,
                "first_offset": first_offset,
                "group_size": group_size,
            })
            used |= indices_set

    return groups


class VectorizeLoadsTransform(PtxTransform):
    """Merge consecutive scalar loads into vector loads."""

    name = "vectorize_loads"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        groups = _find_vectorizable_groups(kernel)
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
        groups = _find_vectorizable_groups(kernel)

        if group_idx >= len(groups):
            return TransformResult(kernel=kernel, changed=False)

        group = groups[group_idx]
        return self._apply_group(kernel, group)

    def _apply_group(self, kernel: ParsedKernel, group: Dict) -> TransformResult:
        """Apply vectorization to a single group of loads."""
        new_kernel = deep_copy_kernel(kernel)

        body_indices = group["body_indices"]
        dest_regs = group["dest_regs"]
        base_reg = group["base_reg"]
        first_offset = group["first_offset"]
        group_size = group["group_size"]

        # Build vector load instruction
        regs_str = ", ".join(dest_regs)
        offset_str = f"+{first_offset}" if first_offset > 0 else ""
        vec_text = f"    ld.global.v{group_size}.f32 {{{regs_str}}}, [{base_reg}{offset_str}];"

        # Replace first load with vector load
        new_kernel.body[body_indices[0]] = BodyLine(
            tag="instruction",
            raw_text=vec_text,
            instruction=None,  # complex instruction, skip detailed parse
        )

        # Remove subsequent loads (mark as blank to preserve body structure)
        # We remove from end to start to keep indices valid
        for idx in reversed(body_indices[1:]):
            del new_kernel.body[idx]

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={
                "group_size": group_size,
                "dest_regs": dest_regs,
                "base_reg": base_reg,
            },
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Vectorize all groups (largest first)."""
        groups = _find_vectorizable_groups(kernel)
        if not groups:
            return TransformResult(kernel=kernel, changed=False)

        # Apply groups from last to first (so body indices stay valid)
        current = deep_copy_kernel(kernel)
        total_vectorized = 0

        # Process groups in reverse body order to keep indices stable
        sorted_groups = sorted(groups, key=lambda g: g["body_indices"][0], reverse=True)

        for group in sorted_groups:
            body_indices = group["body_indices"]
            dest_regs = group["dest_regs"]
            base_reg = group["base_reg"]
            first_offset = group["first_offset"]
            group_size = group["group_size"]

            regs_str = ", ".join(dest_regs)
            offset_str = f"+{first_offset}" if first_offset > 0 else ""
            vec_text = f"    ld.global.v{group_size}.f32 {{{regs_str}}}, [{base_reg}{offset_str}];"

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
