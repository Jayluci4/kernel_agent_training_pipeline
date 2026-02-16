"""PTX instruction representation and parsing.

Parses PTX kernel bodies into Instruction objects with register
def-use information for dependency analysis.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Matches PTX registers: %r0, %rd1, %f2, %p3, etc.
REG_PATTERN = re.compile(r'%[a-z]+\d+')


@dataclass
class Instruction:
    """A single PTX instruction with metadata for scheduling."""
    id: int
    opcode: str
    raw_text: str
    dest_regs: List[str] = field(default_factory=list)
    src_regs: List[str] = field(default_factory=list)
    pipeline: str = 'FP32'
    latency: int = 4

    def __repr__(self):
        return f"Inst({self.id}: {self.opcode} -> {self.dest_regs})"


def classify_instruction(opcode: str) -> tuple:
    """Classify PTX opcode into (pipeline, latency_cycles).

    Pipeline types: FP32, INT32, LSU, SFU, TENSOR, SYNC, CTRL
    Latencies are approximate for L4 (sm_89).
    """
    op = opcode.lower()

    # Memory operations
    if op.startswith('ld.global') or op.startswith('st.global'):
        return 'LSU', 571
    if op.startswith('ld.shared') or op.startswith('st.shared'):
        return 'LSU', 30
    if op.startswith('ld.param') or op.startswith('ld.const'):
        return 'LSU', 4
    if op.startswith(('ld.', 'st.', 'cp.async')):
        return 'LSU', 30

    # Floating-point arithmetic
    if op.startswith(('fma.', 'add.f', 'sub.f', 'mul.f', 'mad.f',
                      'neg.f', 'abs.f', 'min.f', 'max.f')):
        return 'FP32', 4

    # Integer arithmetic
    if op.startswith(('add.s', 'add.u', 'sub.s', 'sub.u',
                      'mul.lo', 'mul.hi', 'mul.wide',
                      'mad.lo', 'mad.hi', 'mad.wide',
                      'and.', 'or.', 'xor.', 'not.', 'shl.', 'shr.',
                      'min.s', 'min.u', 'max.s', 'max.u')):
        return 'INT32', 4

    # Special function unit
    if op.startswith(('sin.', 'cos.', 'rsqrt.', 'rcp.', 'sqrt.',
                      'lg2.', 'ex2.', 'tanh.')):
        return 'SFU', 24

    # Tensor core
    if op.startswith(('mma.', 'wmma.')):
        return 'TENSOR', 32

    # Moves, conversions, comparisons, predicates
    if op.startswith(('mov.', 'cvt.', 'selp.', 'setp.', 'set.',
                      'shfl.', 'prmt.')):
        return 'FP32', 4

    # Synchronization
    if op.startswith(('bar.', 'membar.', 'fence.', 'atom.', 'red.')):
        return 'SYNC', 4

    # Control flow
    if op in ('ret', 'exit') or op.startswith('bra'):
        return 'CTRL', 1

    return 'FP32', 4


def _parse_single_instruction(line: str, inst_id: int) -> Optional[Instruction]:
    """Parse one PTX instruction line into an Instruction object.

    Returns None for non-instruction lines (comments, labels, directives, ret).
    """
    stripped = line.strip()

    # Skip empty, comments, labels, register declarations
    if not stripped:
        return None
    if stripped.startswith('//'):
        return None
    if stripped.endswith(':'):
        return None
    if stripped.startswith('.reg ') or stripped.startswith('.param '):
        return None
    if stripped in ('ret;', 'exit;', '{', '}'):
        return None

    # Remove trailing semicolon and whitespace
    text = stripped.rstrip(';').strip()

    # Handle predication: @%p0 opcode ...
    pred_reg = None
    if text.startswith('@'):
        match = re.match(r'@(%\w+)\s+(.+)', text)
        if match:
            pred_reg = match.group(1)
            text = match.group(2)
        else:
            return None

    # Split into opcode and operand string
    parts = text.split(None, 1)
    if not parts:
        return None
    opcode = parts[0]
    operand_str = parts[1] if len(parts) > 1 else ''

    # Skip branch instructions (not schedulable in our game)
    if opcode.startswith('bra'):
        return None

    # Extract all register tokens from operand string
    all_regs = REG_PATTERN.findall(operand_str)

    # Determine dest vs src based on instruction type
    if opcode.startswith('st.') or opcode.startswith('red.'):
        # Stores: all register operands are sources (including address reg)
        dest_regs = []
        src_regs = list(all_regs)
    elif opcode.startswith('setp.'):
        # setp writes predicate (first operand), reads the rest
        dest_regs = all_regs[:1] if all_regs else []
        src_regs = all_regs[1:]
    else:
        # General: first register is destination, rest are sources
        if all_regs:
            dest_regs = [all_regs[0]]
            src_regs = all_regs[1:]
        else:
            dest_regs = []
            src_regs = []

    # Add predicate register as source
    if pred_reg and REG_PATTERN.match(pred_reg):
        src_regs = [pred_reg] + src_regs

    pipeline, latency = classify_instruction(opcode)

    return Instruction(
        id=inst_id,
        opcode=opcode,
        raw_text=stripped,
        dest_regs=dest_regs,
        src_regs=src_regs,
        pipeline=pipeline,
        latency=latency,
    )


def parse_ptx_body(ptx_source: str, kernel_name: Optional[str] = None) -> List[Instruction]:
    """Parse PTX source and extract schedulable instructions.

    Args:
        ptx_source: Full PTX source string or just the kernel body.
        kernel_name: If provided, extracts body of this kernel.
                     If None, parses all lines as instructions.

    Returns:
        List of Instruction objects with sequential IDs starting from 0.
    """
    lines = ptx_source.split('\n')

    # If kernel_name specified, extract only that kernel's body
    if kernel_name:
        in_kernel = False
        found_open_brace = False
        brace_depth = 0
        body_lines = []
        for line in lines:
            if not in_kernel:
                if f'.entry {kernel_name}' in line:
                    in_kernel = True
                    if '{' in line:
                        found_open_brace = True
                        brace_depth = 1
                continue
            # in_kernel is True: scan for opening brace if not yet found
            brace_depth += line.count('{') - line.count('}')
            if not found_open_brace:
                if brace_depth > 0:
                    found_open_brace = True
                continue
            # Inside the kernel body
            if brace_depth <= 0:
                break
            body_lines.append(line)
        lines = body_lines

    instructions = []
    inst_id = 0
    for line in lines:
        inst = _parse_single_instruction(line, inst_id)
        if inst is not None:
            instructions.append(inst)
            inst_id += 1

    return instructions
