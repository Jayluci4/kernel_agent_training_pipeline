"""Register pressure tracker for instruction scheduling.

Tracks the number of live registers at each point in a partial schedule.
A register is "live" if:
  1. Some scheduled instruction wrote it
  2. Some unscheduled instruction will read it

This models the hardware constraint: the L4 SM has 64K 32-bit registers
shared across all active warps. Each warp can use up to 255 registers,
but higher usage reduces occupancy (fewer concurrent warps to hide latency).

Target register budgets for L4 (sm_89, 64K registers):
  255 regs/thread -> 1 warp/SM   (min occupancy)
  128 regs/thread -> 2 warps/SM
   64 regs/thread -> 4 warps/SM
   32 regs/thread -> 8 warps/SM  (good occupancy)

If the MCTS schedule exceeds the register budget, ptxas will spill
variables to Local Memory (DRAM latency ~500 cycles), destroying any
latency-hiding gains.

Usage:
    tracker = RegisterPressureTracker(instructions)
    for inst_id in schedule_order:
        pressure = tracker.schedule(inst_id, instructions[inst_id])
        if pressure > threshold:
            reward -= penalty
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .instruction import Instruction


@dataclass
class PressureSnapshot:
    """Register pressure at one point in the schedule."""
    live_count: int          # Number of currently live registers
    max_live: int            # Maximum live count seen so far
    just_freed: List[str]    # Registers freed by the last scheduled instruction
    just_allocated: List[str]  # Registers allocated by the last scheduled instruction


class RegisterPressureTracker:
    """Tracks live register count during instruction scheduling.

    For each register, maintains:
      - Who wrote it last (in schedule order)
      - How many unscheduled consumers remain

    When an instruction is scheduled:
      - Its dest registers become live (allocated)
      - Its src registers: decrement remaining consumer count.
        If count reaches 0, register is freed.

    The consumer count comes from the instruction list directly:
    for each register, count how many instructions read it AFTER the
    last instruction that writes it (in program order). This gives an
    upper bound on live pressure that matches what ptxas computes.
    """

    def __init__(self, instructions: List[Instruction]):
        self._n = len(instructions)
        self._instructions = {inst.id: inst for inst in instructions}

        # Build register consumer map.
        # For each register, track the last writer (in program order) and
        # all readers after that writer. This gives the def-use chain that
        # determines when a register can be freed.
        #
        # register -> (last_writer_id, set of reader_ids after last_writer)
        self._reg_def_consumers: Dict[str, Tuple[int, Set[int]]] = {}

        # Process instructions in program order (by ID)
        last_writer: Dict[str, int] = {}
        readers_since_write: Dict[str, Set[int]] = defaultdict(set)

        for inst in instructions:
            # Process reads first
            for reg in inst.src_regs:
                readers_since_write[reg].add(inst.id)

            # Process writes
            for reg in inst.dest_regs:
                # Save previous def-use chain if it existed
                if reg in last_writer:
                    old_writer = last_writer[reg]
                    old_readers = readers_since_write.get(reg, set()) - {inst.id}
                    self._reg_def_consumers[f"{reg}@{old_writer}"] = (
                        old_writer, frozenset(old_readers)
                    )

                # Start new def-use chain for this write
                last_writer[reg] = inst.id
                readers_since_write[reg] = set()

        # Finalize remaining def-use chains
        for reg, writer_id in last_writer.items():
            readers = readers_since_write.get(reg, set())
            self._reg_def_consumers[f"{reg}@{writer_id}"] = (
                writer_id, frozenset(readers)
            )

        # Runtime state
        self._scheduled: Set[int] = set()
        self._live_defs: Set[str] = set()  # Keys into _reg_def_consumers
        self._remaining: Dict[str, int] = {}  # def_key -> remaining consumer count
        self._live_count: int = 0
        self._max_live: int = 0
        self._history: List[int] = []  # live count after each scheduled instruction

        # Initialize remaining consumer counts
        for key, (writer_id, consumers) in self._reg_def_consumers.items():
            self._remaining[key] = len(consumers)

    def schedule(self, inst_id: int) -> PressureSnapshot:
        """Schedule an instruction and update register pressure.

        Args:
            inst_id: The instruction ID being scheduled.

        Returns:
            PressureSnapshot with current register pressure info.
        """
        inst = self._instructions[inst_id]
        self._scheduled.add(inst_id)

        just_freed = []
        just_allocated = []

        # 1. This instruction WRITES registers -> they become live
        for reg in inst.dest_regs:
            key = f"{reg}@{inst_id}"
            if key in self._reg_def_consumers:
                _, consumers = self._reg_def_consumers[key]
                if consumers:  # Only live if there are consumers
                    # Check how many consumers are still unscheduled
                    unscheduled = len(consumers - self._scheduled)
                    if unscheduled > 0:
                        self._live_defs.add(key)
                        self._remaining[key] = unscheduled
                        just_allocated.append(reg)

        # 2. This instruction READS registers -> decrement consumer counts
        #    Check ALL definitions where this instruction is a consumer
        for key, (writer_id, consumers) in self._reg_def_consumers.items():
            if inst_id in consumers and writer_id in self._scheduled:
                # This instruction consumed this definition
                if key in self._remaining:
                    self._remaining[key] -= 1
                    if self._remaining[key] <= 0 and key in self._live_defs:
                        self._live_defs.discard(key)
                        reg_name = key.split('@')[0]
                        just_freed.append(reg_name)

        # Recount live (authoritative)
        self._live_count = len(self._live_defs)
        self._max_live = max(self._max_live, self._live_count)
        self._history.append(self._live_count)

        return PressureSnapshot(
            live_count=self._live_count,
            max_live=self._max_live,
            just_freed=just_freed,
            just_allocated=just_allocated,
        )

    def copy(self) -> 'RegisterPressureTracker':
        """Create an independent copy for MCTS branching."""
        new = RegisterPressureTracker.__new__(RegisterPressureTracker)
        new._n = self._n
        new._instructions = self._instructions  # Shared (immutable)
        new._reg_def_consumers = self._reg_def_consumers  # Shared (immutable)
        new._scheduled = set(self._scheduled)
        new._live_defs = set(self._live_defs)
        new._remaining = dict(self._remaining)
        new._live_count = self._live_count
        new._max_live = self._max_live
        new._history = list(self._history)
        return new

    @property
    def live_count(self) -> int:
        """Current number of live registers."""
        return self._live_count

    @property
    def max_live(self) -> int:
        """Maximum live registers seen so far in this schedule."""
        return self._max_live

    @property
    def history(self) -> List[int]:
        """Live count after each scheduled instruction."""
        return self._history

    def pressure_summary(self) -> Dict:
        """Return a summary of register pressure metrics."""
        return {
            'current_live': self._live_count,
            'max_live': self._max_live,
            'history': list(self._history),
            'mean_live': sum(self._history) / len(self._history) if self._history else 0,
        }
