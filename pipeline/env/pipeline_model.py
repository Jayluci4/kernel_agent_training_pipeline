"""Approximate cycle model for L4 (Ada Lovelace sm_89) sub-core.

Models the two constraints that determine single-warp instruction throughput:
  1. Pipeline throughput: each functional unit has a max issue rate
  2. Data dependencies: register readiness from prior writes

Ada Lovelace (like Volta, Turing, Ampere before it) has a SINGLE dispatch
unit per sub-core partition. Each warp scheduler issues at most 1 instruction
per cycle. There is NO dual-issue from the same warp.

The scheduling lever is DATA DEPENDENCY DISTANCE: placing high-latency
producers (loads, SFU ops) far enough ahead of their consumers so that
the register is ready by the time the consumer issues. When the distance
is too short, the warp stalls.

    Stall = max(0, producer_latency - distance_in_schedule)

Ground truth comes from hardware measurement. This model is the fast
approximation for MCTS rollouts during training.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .instruction import Instruction


# Pipeline throughput: minimum cycles between consecutive issues on the same pipeline.
# Derived from L4 sub-core functional unit counts and warp width (32 threads).
PIPELINE_ISSUE_RATE = {
    'FP32': 1,    # 32 dedicated FP32 units -> 1 cycle per warp instruction
    'INT32': 1,   # 32 flex FP32/INT32 units -> 1 cycle per warp instruction
    'LSU': 4,     # 8 LSU units -> 32/8 = 4 cycles per warp instruction
    'SFU': 8,     # 4 SFU units -> 32/4 = 8 cycles per warp instruction
    'TENSOR': 1,  # 1 tensor core -> 1 cycle issue (but 32-cycle latency)
    'SYNC': 1,    # barrier/fence
    'CTRL': 1,    # control flow
}

# Pipeline categories for ready queue analysis.
MATH_PIPELINES = {'FP32', 'INT32', 'SFU', 'TENSOR'}
MEMORY_PIPELINES = {'LSU'}

# Single dispatch unit per sub-core partition on Ada Lovelace.
# Volta removed dual-issue (Pascal had MAX_ISSUE_WIDTH=2).
# Turing, Ampere, Ada all inherited single-dispatch.
MAX_ISSUE_WIDTH = 1

# Kept for backward compatibility in imports but empty — no dual-issue on sm_89.
DUAL_ISSUE_PAIRS: Set[frozenset] = set()


@dataclass
class PipelineState:
    """Snapshot of pipeline occupancy and register readiness.

    Designed for efficient copy/restore (used heavily in MCTS tree search).
    """
    # Cycle at which each pipeline next becomes free to accept an instruction
    pipeline_free_at: Dict[str, int] = field(default_factory=lambda: {
        'FP32': 0, 'INT32': 0, 'LSU': 0, 'SFU': 0,
        'TENSOR': 0, 'SYNC': 0, 'CTRL': 0,
    })

    # Cycle at which each virtual register becomes readable
    register_ready_at: Dict[str, int] = field(default_factory=dict)

    # Current cycle counter (monotonically increasing)
    current_cycle: int = 0

    # Cumulative statistics
    stall_cycles: int = 0       # Total cycles where warp was idle waiting
    issue_count: int = 0        # Number of instructions issued

    def copy(self) -> 'PipelineState':
        """Create an independent copy for MCTS branching."""
        new = PipelineState()
        new.pipeline_free_at = dict(self.pipeline_free_at)
        new.register_ready_at = dict(self.register_ready_at)
        new.current_cycle = self.current_cycle
        new.stall_cycles = self.stall_cycles
        new.issue_count = self.issue_count
        return new

    def to_vector(self) -> List[float]:
        """Fixed-size feature vector for neural network input.

        Returns 7 values: cycles until each pipeline is free (relative to current_cycle).
        Clamped to [0, inf). Zero means the pipeline is available now.
        """
        pipelines = ['FP32', 'INT32', 'LSU', 'SFU', 'TENSOR', 'SYNC', 'CTRL']
        return [
            max(0.0, float(self.pipeline_free_at[p] - self.current_cycle))
            for p in pipelines
        ]


class L4PipelineModel:
    """Cycle estimator for L4 sub-core with single-dispatch warp scheduler.

    The L4 warp scheduler dispatches exactly 1 instruction per cycle
    per warp, subject to:
      (a) The target pipeline must be free (throughput constraint)
      (b) All source registers must be ready (data dependency)

    The scheduling game's goal: order instructions so that high-latency
    producers (loads, SFU) are placed far enough before their consumers
    to avoid register-readiness stalls.

    Does NOT model:
      - Multi-warp interleaving (TLP hiding latency)
      - Cache hit/miss variation (uses worst-case DRAM latency for globals)
      - Shared memory bank conflicts
      - Instruction fetch/decode costs
      - Register bank conflicts
    """

    def __init__(self):
        self.state = PipelineState()
        self._timeline: List[Tuple[int, int, str]] = []  # (inst_id, issue_cycle, pipeline)

    def reset(self):
        """Reset to fresh pipeline state."""
        self.state = PipelineState()
        self._timeline = []

    def copy_state(self) -> PipelineState:
        """Return an independent copy of the current pipeline state."""
        return self.state.copy()

    def restore_state(self, state: PipelineState):
        """Restore pipeline state from a copy."""
        self.state = state.copy()

    def issue(self, inst: Instruction) -> int:
        """Issue an instruction, returning the cycle it begins execution.

        Single-dispatch model:
        1. Pipeline must be free (throughput)
        2. Source registers must be ready (data dependency)
        3. Must wait until at least current_cycle + 1 (single-issue per cycle)
        """
        pipeline = inst.pipeline
        issue_rate = PIPELINE_ISSUE_RATE.get(pipeline, 1)
        s = self.state

        # Earliest cycle this instruction can issue:
        # 1. Pipeline throughput constraint
        earliest = s.pipeline_free_at.get(pipeline, 0)

        # 2. Data dependency: all source registers must be ready
        for reg in inst.src_regs:
            ready = s.register_ready_at.get(reg, 0)
            if ready > earliest:
                earliest = ready

        # 3. Single-issue: can't issue in the same cycle as previous instruction
        if s.issue_count > 0:
            earliest = max(earliest, s.current_cycle + 1)

        # Count stall cycles (gap where warp was idle)
        if s.issue_count > 0:
            gap = earliest - s.current_cycle - 1
            if gap > 0:
                s.stall_cycles += gap

        issue_cycle = earliest

        # --- Update state ---
        # Pipeline throughput: block this pipeline for issue_rate cycles
        s.pipeline_free_at[pipeline] = issue_cycle + issue_rate

        # Register readiness: dest registers available after latency
        completion_cycle = issue_cycle + inst.latency
        for reg in inst.dest_regs:
            s.register_ready_at[reg] = completion_cycle

        # Advance current cycle
        s.current_cycle = issue_cycle
        s.issue_count += 1

        self._timeline.append((inst.id, issue_cycle, pipeline))
        return issue_cycle

    def estimate_cycles(self, schedule: List[Instruction]) -> int:
        """Estimate total execution cycles for a complete schedule.

        Returns the cycle at which the last instruction's result
        becomes available (issue_cycle + latency of the latest-completing
        instruction).
        """
        self.reset()
        if not schedule:
            return 0

        last_completion = 0
        for inst in schedule:
            issue_cycle = self.issue(inst)
            completion = issue_cycle + inst.latency
            if completion > last_completion:
                last_completion = completion

        return last_completion

    def estimate_with_breakdown(self, schedule: List[Instruction]) -> Dict:
        """Estimate cycles with detailed per-instruction timing.

        Returns:
          - total_cycles: makespan
          - timeline: [(inst_id, opcode, pipeline, issue_cycle, completion_cycle)]
          - pipeline_utilization: {pipeline: fraction_of_total_busy}
          - stall_cycles: total idle cycles (no instruction could issue)
        """
        self.reset()
        if not schedule:
            return {
                'total_cycles': 0, 'timeline': [],
                'pipeline_utilization': {}, 'stall_cycles': 0,
            }

        timeline = []
        pipeline_busy = defaultdict(int)

        for inst in schedule:
            issue_cycle = self.issue(inst)
            completion = issue_cycle + inst.latency
            timeline.append((inst.id, inst.opcode, inst.pipeline, issue_cycle, completion))
            pipeline_busy[inst.pipeline] += PIPELINE_ISSUE_RATE.get(inst.pipeline, 1)

        last_completion = max(c for _, _, _, _, c in timeline)

        utilization = {}
        if last_completion > 0:
            for p, busy in pipeline_busy.items():
                utilization[p] = busy / last_completion

        return {
            'total_cycles': last_completion,
            'timeline': timeline,
            'pipeline_utilization': utilization,
            'stall_cycles': self.state.stall_cycles,
            # Backward compatibility — these are always 0 now
            'dual_issue_count': 0,
            'single_issue_count': self.state.issue_count,
            'dual_issue_rate': 0.0,
        }

    def step_cost(self, inst: Instruction) -> Tuple[int, PipelineState]:
        """Issue one instruction. Returns (stall_cycles, new_state).

        stall_cycles: how many cycles the pipeline was idle before this
        instruction could issue. A good schedule minimizes this.
        """
        prev_cycle = self.state.current_cycle
        issue_cycle = self.issue(inst)
        stall = max(0, issue_cycle - prev_cycle - 1) if self.state.issue_count > 1 else 0
        return stall, self.state.copy()


def analyze_ready_queue_mix(ready_instructions: List[Instruction]) -> Dict[str, float]:
    """Analyze the pipeline composition of the ready queue.

    Returns a dict with pipeline type counts and fractions.
    The balance_score measures diversity of pipeline types in the ready queue.
    With single-dispatch, balance helps avoid structural hazards
    (e.g., all LSU instructions would bottleneck on 4-cycle issue rate).
    """
    if not ready_instructions:
        return {
            'math_count': 0, 'memory_count': 0, 'other_count': 0,
            'total': 0, 'math_fraction': 0.0, 'memory_fraction': 0.0,
            'balance_score': 0.0,
        }

    math_count = sum(1 for i in ready_instructions if i.pipeline in MATH_PIPELINES)
    memory_count = sum(1 for i in ready_instructions if i.pipeline in MEMORY_PIPELINES)
    other_count = len(ready_instructions) - math_count - memory_count
    total = len(ready_instructions)

    math_frac = math_count / total
    memory_frac = memory_count / total

    balance = 2.0 * min(math_frac, memory_frac)

    return {
        'math_count': math_count,
        'memory_count': memory_count,
        'other_count': other_count,
        'total': total,
        'math_fraction': math_frac,
        'memory_fraction': memory_frac,
        'balance_score': balance,
    }
