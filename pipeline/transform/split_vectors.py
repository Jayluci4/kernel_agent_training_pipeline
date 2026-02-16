"""Split vector loads transform: expand vector loads back to scalar loads.

Reverses vectorization by splitting ld.global.v2.f32 or ld.global.v4.f32
back into individual scalar ld.global.f32 instructions.

This gives the instruction scheduler more flexibility to interleave loads
with compute, at the cost of more instructions.

PTX syntax:
  Before: ld.global.v2.f32 {%f0, %f1}, [%rd3+0];
  After:  ld.global.f32 %f0, [%rd3+0];
          ld.global.f32 %f1, [%rd3+4];

  Before: ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd3+0];
  After:  ld.global.f32 %f0, [%rd3+0];
          ld.global.f32 %f1, [%rd3+4];
          ld.global.f32 %f2, [%rd3+8];
          ld.global.f32 %f3, [%rd3+12];
"""

import re
from typing import Dict, List, Optional, Tuple

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel
from .base import PtxTransform, TransformResult


# Match vector loads: ld.global[.hints].v{2,4}.f32 {regs}, [base+offset]
_VECTOR_LOAD_PATTERN = re.compile(
    r'ld\.global((?:\.\w+)*)\.v([24])\.f32\s*\{([^}]+)\},\s*\[(%\w+)(?:\+(\d+))?\]'
)


def _parse_vector_load(raw_text: str) -> Optional[Dict]:
    """Parse a vector global f32 load.

    Returns dict with hints, width, regs, base_reg, offset, or None.
    """
    m = _VECTOR_LOAD_PATTERN.search(raw_text)
    if not m:
        return None

    hints = m.group(1)  # e.g., ".ca" or ""
    width = int(m.group(2))
    regs_str = m.group(3)
    base = m.group(4)
    offset = int(m.group(5)) if m.group(5) else 0

    regs = [r.strip() for r in regs_str.split(',')]
    if len(regs) != width:
        return None

    return {
        "hints": hints,
        "width": width,
        "regs": regs,
        "base_reg": base,
        "offset": offset,
    }


def _find_vector_loads(kernel: ParsedKernel) -> List[Tuple[int, Dict]]:
    """Find all vector load instructions in the kernel body.

    Returns list of (body_index, parsed_info) tuples.
    """
    results = []
    for i, bl in enumerate(kernel.body):
        if bl.tag != "instruction":
            continue
        parsed = _parse_vector_load(bl.raw_text)
        if parsed:
            results.append((i, parsed))
    return results


class SplitVectorLoadsTransform(PtxTransform):
    """Split vector loads back into scalar loads for scheduling flexibility."""

    name = "split_vector_loads"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        vloads = _find_vector_loads(kernel)
        return [
            {
                "body_idx": idx,
                "width": info["width"],
                "base_reg": info["base_reg"],
            }
            for idx, info in vloads
        ]

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        body_idx = params["body_idx"]
        new_kernel = deep_copy_kernel(kernel)

        bl = new_kernel.body[body_idx]
        parsed = _parse_vector_load(bl.raw_text)
        if not parsed:
            return TransformResult(kernel=new_kernel, changed=False)

        hints = parsed["hints"]
        regs = parsed["regs"]
        base = parsed["base_reg"]
        offset = parsed["offset"]

        # Build scalar load instructions
        scalar_lines = []
        for j, reg in enumerate(regs):
            byte_offset = offset + j * 4
            offset_str = f"+{byte_offset}" if byte_offset > 0 else ""
            text = f"    ld.global{hints}.f32 {reg}, [{base}{offset_str}];"
            scalar_lines.append(BodyLine(
                tag="instruction",
                raw_text=text,
                instruction=None,
            ))

        # Replace the vector load with scalar loads
        new_kernel.body[body_idx:body_idx + 1] = scalar_lines

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={
                "width": parsed["width"],
                "base_reg": base,
                "n_scalar_loads": len(scalar_lines),
            },
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Split all vector loads into scalar loads."""
        vloads = _find_vector_loads(kernel)
        if not vloads:
            return TransformResult(kernel=kernel, changed=False)

        current = deep_copy_kernel(kernel)
        total_split = 0

        # Process in reverse order to keep body indices valid
        for body_idx, info in reversed(vloads):
            hints = info["hints"]
            regs = info["regs"]
            base = info["base_reg"]
            offset = info["offset"]

            scalar_lines = []
            for j, reg in enumerate(regs):
                byte_offset = offset + j * 4
                offset_str = f"+{byte_offset}" if byte_offset > 0 else ""
                text = f"    ld.global{hints}.f32 {reg}, [{base}{offset_str}];"
                scalar_lines.append(BodyLine(
                    tag="instruction",
                    raw_text=text,
                    instruction=None,
                ))

            current.body[body_idx:body_idx + 1] = scalar_lines
            total_split += 1

        return TransformResult(
            kernel=current,
            changed=total_split > 0,
            stats={"loads_split": total_split},
        )
