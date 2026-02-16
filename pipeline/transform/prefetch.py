"""Prefetch transform: insert prefetch instructions for global loads.

Inserts prefetch.global.L1 or prefetch.global.L2 instructions early in the
kernel body to pre-warm the cache before the actual loads execute.

For gemm_tile kernels, this means prefetching B tile addresses while
A tile loads are happening, or prefetching all addresses at kernel start
so data flows from DRAM while the kernel does initialization.

PTX syntax:
  prefetch.global.L1 [%rd3+0];
  prefetch.global.L2 [%rd4+16];
"""

import re
from typing import Dict, List

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel
from .base import PtxTransform, TransformResult


PREFETCH_LEVELS = ("L1", "L2")

# Parse global load addresses: ld.global[.hints].f32 %dest, [%base+offset]
_GLOBAL_LOAD_ADDR = re.compile(
    r'ld\.global(?:\.\w+)*\.(?:f32|f64|b32|b64|u32|u64)\s+%\w+,\s*\[(%\w+)(?:\+(\d+))?\]'
)

# Match ld.param lines (to find insertion point after all param loads)
_PARAM_LOAD = re.compile(r'ld\.param\.')


def _collect_load_addresses(kernel: ParsedKernel, per_cacheline: bool = True) -> List[Dict]:
    """Collect load addresses from global loads for prefetching.

    If per_cacheline=True (default), only keeps one address per 128-byte
    cache line per base register. This avoids redundant prefetches since
    one prefetch covers the whole cache line.
    """
    seen = set()
    addresses = []
    for bl in kernel.body:
        if bl.instruction is None:
            continue
        m = _GLOBAL_LOAD_ADDR.search(bl.raw_text)
        if m:
            base = m.group(1)
            offset = int(m.group(2)) if m.group(2) else 0

            if per_cacheline:
                # L4 cache line = 128 bytes. One prefetch per line.
                cache_line = offset // 128
                key = (base, cache_line)
                aligned_offset = cache_line * 128
            else:
                key = (base, offset)
                aligned_offset = offset

            if key not in seen:
                seen.add(key)
                addresses.append({"base_reg": base, "offset": aligned_offset})
    return addresses


def _find_insertion_point(kernel: ParsedKernel) -> int:
    """Find the body index right after the last param load.

    Prefetch must go after param loads (which set up the base addresses)
    but before the actual global loads.
    """
    last_param_idx = -1
    for i, bl in enumerate(kernel.body):
        if bl.instruction is None:
            continue
        if _PARAM_LOAD.search(bl.raw_text):
            last_param_idx = i

    # Insert after the last param load, or at the start of the body
    return last_param_idx + 1 if last_param_idx >= 0 else 0


class PrefetchTransform(PtxTransform):
    """Insert prefetch instructions for global load addresses."""

    name = "prefetch"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        addresses = _collect_load_addresses(kernel)
        if not addresses:
            return []
        # One option per cache level
        return [{"level": level} for level in PREFETCH_LEVELS]

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        level = params["level"]
        addresses = _collect_load_addresses(kernel)
        if not addresses:
            return TransformResult(kernel=kernel, changed=False)

        new_kernel = deep_copy_kernel(kernel)
        insert_at = _find_insertion_point(new_kernel)

        # Build prefetch instructions
        prefetch_lines = []
        for addr in addresses:
            offset_str = f"+{addr['offset']}" if addr['offset'] > 0 else ""
            text = f"    prefetch.global.{level} [{addr['base_reg']}{offset_str}];"
            prefetch_lines.append(BodyLine(
                tag="instruction",
                raw_text=text,
                instruction=None,  # prefetch has no dest register
            ))

        # Insert all prefetch lines at the insertion point
        for j, pf_line in enumerate(prefetch_lines):
            new_kernel.body.insert(insert_at + j, pf_line)

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={
                "level": level,
                "n_prefetches": len(prefetch_lines),
                "addresses": [(a["base_reg"], a["offset"]) for a in addresses],
            },
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Insert L2 prefetch for all global load addresses."""
        return self.apply(kernel, {"level": "L2"})
