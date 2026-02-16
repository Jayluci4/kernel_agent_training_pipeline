"""Instruction reorder transform: dependency-safe reordering.

Reorders schedulable instructions within the kernel body while preserving
data dependencies (RAW/WAR/WAW). Non-instruction lines (labels, directives,
branches, ret, comments) stay at their original positions.

Uses InstructionDAG from env/dag.py for dependency analysis and scheduling
heuristics from env/schedule_env.py.
"""

import copy
from typing import Dict, List

from .parsed_kernel import ParsedKernel, BodyLine, deep_copy_kernel, get_instructions
from .base import PtxTransform, TransformResult
from ..env.dag import InstructionDAG
from ..env.instruction import Instruction
from ..env.schedule_env import ScheduleEnv


STRATEGIES = ("critical_path", "interleave", "program_order", "loads_first", "stores_last")


def _reindex_instructions(instructions: List[Instruction]):
    """Re-index instructions to have contiguous IDs 0..N-1.

    Returns (reindexed_instructions, id_map) where id_map maps
    new_id -> original_id for recovering the original ordering.
    """
    id_map = {}
    reindexed = []
    for new_id, inst in enumerate(instructions):
        new_inst = copy.copy(inst)
        id_map[new_id] = inst.id
        new_inst.id = new_id
        reindexed.append(new_inst)
    return reindexed, id_map


def _reorder_body(kernel: ParsedKernel, new_order: List[int]) -> ParsedKernel:
    """Rewrite kernel body with instructions in new_order.

    Non-instruction body lines stay at their original positions.
    Instruction slots are filled in new_order sequence.
    """
    new_kernel = deep_copy_kernel(kernel)

    # Collect instruction body lines by their instruction ID
    inst_by_id = {}
    for bl in kernel.body:
        if bl.instruction is not None:
            inst_by_id[bl.instruction.id] = bl

    # Find instruction slot positions in the body
    inst_slots = [i for i, bl in enumerate(new_kernel.body) if bl.instruction is not None]

    if len(inst_slots) != len(new_order):
        raise ValueError(
            f"new_order has {len(new_order)} instructions but body has {len(inst_slots)} slots"
        )

    # Fill slots with reordered instructions
    for slot_idx, inst_id in zip(inst_slots, new_order):
        bl = inst_by_id[inst_id]
        new_kernel.body[slot_idx] = BodyLine(
            tag="instruction",
            raw_text=bl.raw_text,
            instruction=bl.instruction,
        )

    return new_kernel


class ReorderTransform(PtxTransform):
    """Dependency-safe instruction reordering using scheduling heuristics."""

    name = "reorder"

    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        instructions = get_instructions(kernel)
        if len(instructions) < 2:
            return []
        return [{"strategy": s} for s in STRATEGIES]

    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        strategy = params["strategy"]
        instructions = get_instructions(kernel)

        if len(instructions) < 2:
            return TransformResult(kernel=kernel, changed=False)

        # Re-index to contiguous 0..N-1 (required by DAG/ScheduleEnv)
        reindexed, id_map = _reindex_instructions(instructions)

        env = ScheduleEnv()
        env.reset(reindexed)

        # Apply scheduling heuristic
        if strategy == "critical_path":
            while True:
                legal = env.legal_actions()
                if not legal:
                    break
                action = max(legal, key=lambda i: (env.dag.height.get(i, 0), -i))
                _, _, done, info = env.step(action)
                if done:
                    break
        elif strategy == "interleave":
            from ..env.pipeline_model import MATH_PIPELINES, MEMORY_PIPELINES
            last_was_math = True
            while True:
                legal = env.legal_actions()
                if not legal:
                    break
                math_ready = [i for i in legal if env.dag.instructions[i].pipeline in MATH_PIPELINES]
                mem_ready = [i for i in legal if env.dag.instructions[i].pipeline in MEMORY_PIPELINES]
                other = [i for i in legal if i not in math_ready and i not in mem_ready]

                if last_was_math and mem_ready:
                    candidates = mem_ready
                    last_was_math = False
                elif not last_was_math and math_ready:
                    candidates = math_ready
                    last_was_math = True
                elif math_ready:
                    candidates = math_ready
                    last_was_math = True
                elif mem_ready:
                    candidates = mem_ready
                    last_was_math = False
                else:
                    candidates = other

                action = max(candidates, key=lambda i: (env.dag.height.get(i, 0), -i))
                _, _, done, info = env.step(action)
                if done:
                    break
        elif strategy == "program_order":
            while True:
                legal = env.legal_actions()
                if not legal:
                    break
                action = min(legal)
                _, _, done, info = env.step(action)
                if done:
                    break
        elif strategy == "loads_first":
            # Prioritize memory loads to maximize memory-level parallelism.
            # Issues all available loads before math, then math before stores.
            while True:
                legal = env.legal_actions()
                if not legal:
                    break
                loads = [i for i in legal
                         if env.dag.instructions[i].opcode.startswith('ld.')]
                stores = [i for i in legal
                          if env.dag.instructions[i].opcode.startswith('st.')]
                math = [i for i in legal if i not in loads and i not in stores]

                if loads:
                    candidates = loads
                elif math:
                    candidates = math
                else:
                    candidates = stores if stores else legal

                action = max(candidates, key=lambda i: (env.dag.height.get(i, 0), -i))
                _, _, done, info = env.step(action)
                if done:
                    break
        elif strategy == "stores_last":
            # Push stores as late as possible. Prioritize loads and math,
            # only issue a store when nothing else is ready.
            while True:
                legal = env.legal_actions()
                if not legal:
                    break
                non_stores = [i for i in legal
                              if not env.dag.instructions[i].opcode.startswith('st.')]

                if non_stores:
                    candidates = non_stores
                else:
                    candidates = legal

                action = max(candidates, key=lambda i: (env.dag.height.get(i, 0), -i))
                _, _, done, info = env.step(action)
                if done:
                    break
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Map re-indexed order back to original instruction IDs
        reindexed_order = env.get_schedule_order()
        new_order = [id_map[rid] for rid in reindexed_order]
        original_order = [inst.id for inst in instructions]
        changed = new_order != original_order

        new_kernel = _reorder_body(kernel, new_order)
        total_cycles = env.get_total_cycles()

        return TransformResult(
            kernel=new_kernel,
            changed=changed,
            stats={
                "strategy": strategy,
                "total_cycles": total_cycles,
                "changed": changed,
            },
        )

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        return self.apply(kernel, {"strategy": "critical_path"})
