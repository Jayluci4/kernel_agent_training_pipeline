"""Project Chronos Phase 1: Scheduling Game Environment.

The "game": given a DAG of PTX instructions, place them one by one
into a schedule. Legal moves = instructions whose dependencies are satisfied.
Reward = -cycles (minimize execution latency).

Key insight for L4: dual-issue Math+Memory ops in the same cycle.
The MCTS agent must learn to keep a balanced ready queue mix.

Two hardware constraints the schedule must respect:
  1. Dual-issue: interleave Math+Memory for 2x throughput
  2. Register pressure: stay within budget to avoid spills to DRAM
"""

from .instruction import Instruction, parse_ptx_body, classify_instruction
from .dag import InstructionDAG, DepType
from .pipeline_model import L4PipelineModel, analyze_ready_queue_mix
from .register_tracker import RegisterPressureTracker, PressureSnapshot
from .schedule_env import (
    ScheduleEnv, Observation,
    random_rollout, critical_path_heuristic,
    program_order_schedule, interleave_heuristic,
    compare_all_heuristics,
)
