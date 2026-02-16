"""PTX kernel representation with round-trip parse/emit.

Parses full PTX source into a structured representation that preserves
ALL lines (instructions, labels, directives, branches, comments, etc.)
for program transformation.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..env.instruction import Instruction, _parse_single_instruction, REG_PATTERN


@dataclass
class BodyLine:
    """A single line in the kernel body (between { and })."""
    tag: str  # "instruction" | "label" | "directive" | "comment" | "branch" | "ret" | "blank"
    raw_text: str
    instruction: Optional[Instruction] = None  # set only when tag == "instruction"


@dataclass
class ParsedKernel:
    """Full PTX kernel representation for transformation."""
    name: str
    preamble: str          # everything from start through the '{' line
    body: List[BodyLine]   # all lines inside { ... }
    reg_decls: Dict[str, int] = field(default_factory=dict)  # register type -> count


# Pattern for .reg declarations: .reg .f32 %f<48>;
_REG_DECL_PATTERN = re.compile(
    r'\.reg\s+(\.\w+)\s+%\w+<(\d+)>\s*;'
)


def _classify_body_line(line: str) -> str:
    """Classify a kernel body line by type."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("//"):
        return "comment"
    if stripped.endswith(":"):
        return "label"
    if stripped.startswith("."):
        return "directive"
    if stripped in ("ret;", "exit;"):
        return "ret"

    # Check for branch (with or without predication)
    text = stripped.rstrip(";").strip()
    if text.startswith("@"):
        m = re.match(r'@[!]?%\w+\s+(.*)', text)
        if m and m.group(1).startswith("bra"):
            return "branch"
    if text.startswith("bra"):
        return "branch"

    return "instruction"


def parse_kernel(ptx_source: str) -> ParsedKernel:
    """Parse PTX source into a ParsedKernel.

    Preserves ALL lines for round-trip fidelity.
    parse_kernel -> emit must produce compilable PTX identical to the input.
    """
    lines = ptx_source.split('\n')

    # Find kernel name
    name = ""
    for line in lines:
        m = re.search(r'\.entry\s+(\w+)', line)
        if m:
            name = m.group(1)
            break

    # Find opening brace — everything up to and including that line is the preamble
    preamble_end = 0
    for i, line in enumerate(lines):
        if '{' in line:
            preamble_end = i
            break

    preamble = '\n'.join(lines[:preamble_end + 1])
    body_start = preamble_end + 1

    # Find closing brace (search from end)
    body_end = len(lines) - 1
    for i in range(len(lines) - 1, body_start - 1, -1):
        if lines[i].strip() == '}':
            body_end = i
            break

    # Parse body lines
    body = []
    reg_decls = {}
    inst_id = 0

    for i in range(body_start, body_end):
        line = lines[i]
        tag = _classify_body_line(line)

        # Extract register declaration metadata
        if tag == "directive":
            m = _REG_DECL_PATTERN.search(line)
            if m:
                reg_decls[m.group(1)] = int(m.group(2))

        # Parse instruction if applicable
        instruction = None
        if tag == "instruction":
            instruction = _parse_single_instruction(line, inst_id)
            if instruction is not None:
                inst_id += 1
            else:
                # _parse_single_instruction rejected it (e.g. bare '{')
                tag = "blank"

        body.append(BodyLine(tag=tag, raw_text=line, instruction=instruction))

    return ParsedKernel(
        name=name,
        preamble=preamble,
        body=body,
        reg_decls=reg_decls,
    )


def emit(kernel: ParsedKernel) -> str:
    """Reconstruct valid PTX source from a ParsedKernel."""
    parts = [kernel.preamble]
    for bl in kernel.body:
        parts.append(bl.raw_text)
    parts.append("}")
    return '\n'.join(parts) + '\n'


def get_instructions(kernel: ParsedKernel) -> List[Instruction]:
    """Extract schedulable instructions from a ParsedKernel."""
    return [bl.instruction for bl in kernel.body if bl.instruction is not None]


def deep_copy_kernel(kernel: ParsedKernel) -> ParsedKernel:
    """Create a deep copy of a ParsedKernel for pure transforms."""
    return copy.deepcopy(kernel)
