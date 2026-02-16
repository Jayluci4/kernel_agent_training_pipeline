"""Register budget transform: insert/modify .maxnreg directive.

Controls ptxas register allocation. Lower register count = higher occupancy
(more warps per SM) but potential spills. Higher count = fewer warps but
no spills.

L4 (sm_89) register budgets:
  255 regs/thread -> 1 warp/SM
  128 regs/thread -> 2 warps/SM
   64 regs/thread -> 4 warps/SM
   32 regs/thread -> 8 warps/SM

PTX syntax: .maxnreg goes between the parameter list and the opening brace:
  .visible .entry kernel(
      .param .u64 ptr
  )
  .maxnreg 64
  {
      ...
  }
"""

import re
from typing import Dict, List

from .parsed_kernel import ParsedKernel, deep_copy_kernel
from .base import PtxTransform, TransformResult


BUDGET_OPTIONS = [32, 64, 128, 255]

_MAXNREG_PATTERN = re.compile(r'\.maxnreg\s+\d+')


class RegisterBudgetTransform(PtxTransform):
    """Insert or modify .maxnreg N directive in the kernel preamble."""

    name = "register_budget"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        return [{"max_regs": n} for n in BUDGET_OPTIONS]

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        max_regs = params["max_regs"]
        new_kernel = deep_copy_kernel(kernel)
        directive = f".maxnreg {max_regs}"

        preamble = new_kernel.preamble

        # Check if .maxnreg already exists in preamble
        if _MAXNREG_PATTERN.search(preamble):
            new_preamble = _MAXNREG_PATTERN.sub(directive, preamble)
            changed = new_preamble != preamble
            new_kernel.preamble = new_preamble
            return TransformResult(
                kernel=new_kernel,
                changed=changed,
                stats={"max_regs": max_regs, "action": "replaced"},
            )

        # Insert .maxnreg between ') {' — split on the opening brace
        # Preamble ends with something like "...\n) {"
        # We need: "...\n)\n.maxnreg N\n{"
        if ') {' in preamble:
            new_preamble = preamble.replace(') {', f')\n{directive}\n{{', 1)
        elif ')\n{' in preamble:
            new_preamble = preamble.replace(')\n{', f')\n{directive}\n{{', 1)
        else:
            # Fallback: insert before the last '{'
            idx = preamble.rfind('{')
            if idx >= 0:
                new_preamble = preamble[:idx] + f'{directive}\n' + preamble[idx:]
            else:
                # Can't find insertion point
                return TransformResult(
                    kernel=new_kernel, changed=False,
                    stats={"error": "no insertion point found"},
                )

        new_kernel.preamble = new_preamble

        return TransformResult(
            kernel=new_kernel,
            changed=True,
            stats={"max_regs": max_regs, "action": "inserted"},
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        # Default: 64 registers (4 warps/SM on L4)
        return self.apply(kernel, {"max_regs": 64})
