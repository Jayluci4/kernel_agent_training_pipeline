"""Scheduling game environment for PTX instruction ordering.

Gym-like interface: the agent picks one instruction from the ready
queue at each step. Episode ends when all instructions are scheduled.

The key scheduling insight for L4 (Ada Lovelace):
  The sub-core can dual-issue Math+Memory in the same cycle.
  A schedule that interleaves FP32/INT32 ops with LD/ST ops enables
  dual-issue and doubles effective throughput.
  A schedule that clusters all Math or all Memory ops forces
  single-issue and halves throughput.

The MCTS agent must learn to maintain a balanced ready queue mix.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .instruction import Instruction, parse_ptx_body
from .dag import InstructionDAG, DepType
from .pipeline_model import (
    L4PipelineModel, analyze_ready_queue_mix,
    MATH_PIPELINES, MEMORY_PIPELINES,
    DUAL_ISSUE_PAIRS, MAX_ISSUE_WIDTH,
)
from .register_tracker import RegisterPressureTracker


# Instruction type categories for one-hot encoding
OPCODE_CATEGORIES = [
    'ld.global', 'ld.shared', 'ld.param', 'ld.const',
    'st.global', 'st.shared',
    'fma', 'add.f', 'mul.f', 'sub.f',
    'add.s', 'add.u', 'mul.lo', 'mad',
    'mov', 'cvt', 'setp', 'selp',
    'sin', 'cos', 'rsqrt', 'rcp', 'ex2',
    'mma', 'wmma',
    'bar', 'membar', 'atom',
    'shfl',
    'other',
]

PIPELINE_NAMES = ['FP32', 'INT32', 'LSU', 'SFU', 'TENSOR', 'SYNC', 'CTRL']


def _classify_opcode_category(opcode: str) -> int:
    """Map opcode to category index for one-hot encoding."""
    op = opcode.lower()
    for i, cat in enumerate(OPCODE_CATEGORIES[:-1]):
        if op.startswith(cat):
            return i
    return len(OPCODE_CATEGORIES) - 1


@dataclass
class Observation:
    """State observation for the scheduling agent.

    All arrays are numpy for compatibility with both PyTorch and
    direct MCTS consumption.
    """
    # Per-instruction features: [N, feat_dim]
    node_features: np.ndarray

    # DAG edges in COO format: [2, E]
    edge_index: np.ndarray

    # Edge type features: [E, edge_feat_dim]
    edge_attr: np.ndarray

    # Boolean mask: which instructions can be scheduled next [N]
    ready_mask: np.ndarray

    # Fraction of instructions already scheduled
    schedule_progress: float

    # Pipeline state vector [7] (cycles remaining per pipeline)
    pipeline_state: np.ndarray

    # Ready queue pipeline mix [3]: (math_fraction, memory_fraction, balance_score)
    # balance_score = 2*min(math_frac, memory_frac), in [0, 1]
    # 1.0 = every Math op can pair with a Memory op for dual-issue
    # 0.0 = all same pipeline type, no dual-issue possible
    ready_queue_mix: np.ndarray

    # Register pressure [2]: (current_live_count, max_live_count)
    # Normalized by register_budget. Values >1.0 mean spill territory.
    register_pressure: np.ndarray


class ScheduleEnv:
    """PTX instruction scheduling as a sequential decision game.

    State: (DAG, scheduled set, ready queue, pipeline occupancy)
    Action: pick one instruction from ready queue
    Transition: mark instruction as scheduled, update ready queue and pipelines
    Reward: configurable -- see reward_mode

    Usage:
        env = ScheduleEnv()
        obs = env.reset(instructions)
        while not done:
            actions = env.legal_actions()
            obs, reward, done, info = env.step(actions[0])
        ptx = env.render_ptx()
    """

    # Default register budget: 64 registers per thread allows 4 warps/SM
    # on L4 (32 threads/warp * 4 warps * 64 regs = 8192 regs out of 64K).
    # Exceeding this triggers spills to Local Memory (~500 cycle penalty).
    DEFAULT_REGISTER_BUDGET = 64

    def __init__(self, reward_mode: str = 'dense',
                 register_budget: int = DEFAULT_REGISTER_BUDGET):
        """
        Args:
            reward_mode:
              'dense'  -- per-step: -stall_cycles (penalizes pipeline bubbles)
              'sparse' -- terminal only: -total_cycles
              'dual'   -- per-step: -stall + bonus for dual-issue achieved
            register_budget: Maximum live registers before spill penalty.
                             Default 64 (allows 4 warps/SM on L4).
        """
        self.reward_mode = reward_mode
        self.register_budget = register_budget
        self.dag: Optional[InstructionDAG] = None
        self.pipeline = L4PipelineModel()
        self.reg_tracker: Optional[RegisterPressureTracker] = None
        self._scheduled_ids: List[int] = []
        self._scheduled_set: frozenset = frozenset()
        self._done: bool = True
        self._total_stalls: int = 0
        # Per-instruction scoreboard history: recorded at issue time
        self._scoreboard_history: Dict[int, Dict] = {}

    def reset(self, instructions: List[Instruction]) -> Observation:
        """Initialize a new scheduling episode.

        Args:
            instructions: List of Instruction objects (from parse_ptx_body).

        Returns:
            Initial observation.
        """
        self.dag = InstructionDAG(instructions)
        self.pipeline = L4PipelineModel()
        self.reg_tracker = RegisterPressureTracker(instructions)
        self._scheduled_ids = []
        self._scheduled_set = frozenset()
        self._done = False
        self._total_stalls = 0
        self._scoreboard_history = {}
        return self._observe()

    def reset_from_ptx(self, ptx_source: str,
                       kernel_name: Optional[str] = None) -> Observation:
        """Convenience: parse PTX and reset."""
        instructions = parse_ptx_body(ptx_source, kernel_name)
        return self.reset(instructions)

    def step(self, action: int,
             observe: bool = True) -> Tuple[Observation, float, bool, Dict]:
        """Schedule instruction `action` from the ready queue.

        Args:
            action: Instruction ID to schedule next.
            observe: If False, skip observation computation (returns None for obs).
                     Use for bulk replay where only the final observation is needed.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")
        if self.dag is None:
            raise RuntimeError("No DAG loaded. Call reset().")

        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(
                f"Instruction {action} is not in ready queue. "
                f"Legal actions: {legal}"
            )

        # Snapshot scoreboard BEFORE issue (what the model should see)
        inst = self.dag.instructions[action]
        s = self.pipeline.state
        pre_cycle = s.current_cycle
        fu_wait = max(0, s.pipeline_free_at.get(inst.pipeline, 0) - pre_cycle)
        reg_wait = 0
        for reg in inst.src_regs:
            rw = max(0, s.register_ready_at.get(reg, 0) - pre_cycle)
            if rw > reg_wait:
                reg_wait = rw

        # Issue the instruction through the pipeline model
        stall, _ = self.pipeline.step_cost(inst)
        self._total_stalls += stall

        # Record scoreboard state at issue time
        # dual_issued is always 0 — no dual-issue on Ada Lovelace (single dispatch)
        self._scoreboard_history[action] = {
            'stall': stall,
            'fu_wait': fu_wait,
            'reg_wait': reg_wait,
            'dual_issued': 0.0,
            'issue_cycle': self.pipeline.state.current_cycle,
        }

        # Track register pressure
        pressure = self.reg_tracker.schedule(action)
        reg_over_budget = max(0, pressure.live_count - self.register_budget)

        # Update scheduled set
        self._scheduled_ids.append(action)
        self._scheduled_set = frozenset(self._scheduled_ids)

        # Check if all instructions are scheduled
        self._done = len(self._scheduled_ids) == self.dag.n

        # Compute reward (includes register pressure penalty)
        reward = self._compute_reward(stall, reg_over_budget)

        info = {
            'stall_cycles': stall,
            'instruction': inst.opcode,
            'pipeline': inst.pipeline,
            'scheduled_count': len(self._scheduled_ids),
            'total_stalls': self._total_stalls,
            'live_registers': pressure.live_count,
            'max_live_registers': pressure.max_live,
            'register_over_budget': reg_over_budget,
        }

        if self._done:
            schedule = [self.dag.instructions[i] for i in self._scheduled_ids]
            breakdown = L4PipelineModel().estimate_with_breakdown(schedule)
            info['total_cycles'] = breakdown['total_cycles']
            info['schedule_order'] = list(self._scheduled_ids)
            info['dual_issue_count'] = breakdown['dual_issue_count']
            info['single_issue_count'] = breakdown['single_issue_count']
            info['dual_issue_rate'] = breakdown['dual_issue_rate']
            info['pipeline_utilization'] = breakdown['pipeline_utilization']
            # Register pressure summary
            reg_summary = self.reg_tracker.pressure_summary()
            info['max_live_registers'] = reg_summary['max_live']
            info['mean_live_registers'] = reg_summary['mean_live']
            info['register_budget'] = self.register_budget
            info['register_spill_risk'] = reg_summary['max_live'] > self.register_budget

        obs = self._observe() if observe else None
        return obs, reward, self._done, info

    def _compute_reward(self, stall: int,
                        reg_over_budget: int = 0) -> float:
        """Compute step reward based on reward_mode.

        All modes include a register pressure penalty when live registers
        exceed the budget. The penalty is linear in the overshoot:
        each register over budget costs 10 cycles.
        """
        reg_penalty = -float(reg_over_budget) * 10.0

        if self.reward_mode == 'dense':
            return -float(stall) + reg_penalty
        elif self.reward_mode == 'sparse':
            if self._done:
                schedule = [self.dag.instructions[i] for i in self._scheduled_ids]
                total = L4PipelineModel().estimate_cycles(schedule)
                return -float(total) + reg_penalty
            return reg_penalty
        elif self.reward_mode == 'dual':
            # With single-dispatch, 'dual' mode is same as 'dense'
            return -float(stall) + reg_penalty
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def legal_actions(self) -> List[int]:
        """Return instruction IDs that can be scheduled next."""
        if self.dag is None or self._done:
            return []
        return self.dag.get_ready_set(self._scheduled_set)

    def render_ptx(self) -> str:
        """Emit the scheduled instructions as a PTX body string.

        Returns instructions in schedule order, which can be wrapped
        in a kernel entry point for compilation.
        """
        if self.dag is None:
            return ""
        lines = []
        for inst_id in self._scheduled_ids:
            inst = self.dag.instructions[inst_id]
            lines.append(f"    {inst.raw_text}")
        return "\n".join(lines)

    def get_schedule_order(self) -> List[int]:
        """Return the current schedule as a list of instruction IDs."""
        return list(self._scheduled_ids)

    def get_total_cycles(self) -> int:
        """Estimate total cycles for the current (possibly partial) schedule."""
        if not self._scheduled_ids or self.dag is None:
            return 0
        schedule = [self.dag.instructions[i] for i in self._scheduled_ids]
        return L4PipelineModel().estimate_cycles(schedule)

    def _observe(self) -> Observation:
        """Build the current observation."""
        if self.dag is None:
            raise RuntimeError("No DAG loaded.")

        n = self.dag.n

        # --- Node features ---
        # [opcode_category_onehot(30), pipeline_onehot(7), latency(1),
        #  depth(1), height(1), status_onehot(3),
        #  scoreboard: stall(1), fu_wait(1), reg_wait(1), dual_issue(1), issue_cycle(1)]
        n_opcode = len(OPCODE_CATEGORIES)
        n_pipeline = len(PIPELINE_NAMES)
        feat_dim = n_opcode + n_pipeline + 1 + 1 + 1 + 3 + 5  # 48
        node_features = np.zeros((n, feat_dim), dtype=np.float32)

        ready_set = set(self.legal_actions())

        max_latency = 571.0
        max_depth = max(self.dag.depth.values()) if self.dag.depth else 1.0
        max_height = max(self.dag.height.values()) if self.dag.height else 1.0

        # Normalization constants for scoreboard features
        # Use max_latency as scale (stalls/waits are bounded by instruction latencies)
        stall_scale = max(max_latency, 1.0)
        # For issue_cycle normalization: use n * average_latency as rough total
        cycle_scale = max(float(self.pipeline.state.current_cycle), 1.0) if self._scheduled_ids else 1.0

        s = self.pipeline.state

        for inst_id in range(n):
            inst = self.dag.instructions[inst_id]
            offset = 0

            # Opcode category one-hot
            cat = _classify_opcode_category(inst.opcode)
            node_features[inst_id, offset + cat] = 1.0
            offset += n_opcode

            # Pipeline one-hot
            if inst.pipeline in PIPELINE_NAMES:
                pidx = PIPELINE_NAMES.index(inst.pipeline)
                node_features[inst_id, offset + pidx] = 1.0
            offset += n_pipeline

            # Latency (normalized)
            node_features[inst_id, offset] = inst.latency / max_latency
            offset += 1

            # Depth (normalized)
            node_features[inst_id, offset] = self.dag.depth.get(inst_id, 0) / max(max_depth, 1.0)
            offset += 1

            # Height (normalized)
            node_features[inst_id, offset] = self.dag.height.get(inst_id, 0) / max(max_height, 1.0)
            offset += 1

            # Status one-hot: [scheduled, ready, blocked]
            if inst_id in self._scheduled_set:
                node_features[inst_id, offset] = 1.0
            elif inst_id in ready_set:
                node_features[inst_id, offset + 1] = 1.0
            else:
                node_features[inst_id, offset + 2] = 1.0
            offset += 3

            # --- Scoreboard features (5 dims) ---
            if inst_id in self._scoreboard_history:
                # SCHEDULED: use historical data from when this instruction was issued
                hist = self._scoreboard_history[inst_id]
                node_features[inst_id, offset] = hist['stall'] / stall_scale
                node_features[inst_id, offset + 1] = hist['fu_wait'] / stall_scale
                node_features[inst_id, offset + 2] = hist['reg_wait'] / stall_scale
                node_features[inst_id, offset + 3] = hist['dual_issued']
                node_features[inst_id, offset + 4] = hist['issue_cycle'] / max(cycle_scale, 1.0)
            elif inst_id in ready_set:
                # READY: compute predicted scoreboard from current pipeline state
                fu_w = max(0, s.pipeline_free_at.get(inst.pipeline, 0) - s.current_cycle)
                reg_w = 0
                for reg in inst.src_regs:
                    rw = max(0, s.register_ready_at.get(reg, 0) - s.current_cycle)
                    if rw > reg_w:
                        reg_w = rw
                predicted_stall = max(fu_w, reg_w)

                # No dual-issue on Ada Lovelace (single dispatch per cycle)
                node_features[inst_id, offset] = predicted_stall / stall_scale
                node_features[inst_id, offset + 1] = fu_w / stall_scale
                node_features[inst_id, offset + 2] = reg_w / stall_scale
                node_features[inst_id, offset + 3] = 0.0  # no dual-issue on sm_89
                node_features[inst_id, offset + 4] = 0.0  # not issued yet
            # else: BLOCKED — all scoreboard features remain 0.0

        # --- Edge index and attributes ---
        edge_pairs = self.dag.get_edge_index()
        edge_types = self.dag.get_edge_types()

        if edge_pairs:
            edge_index = np.array(edge_pairs, dtype=np.int64).T
            edge_attr = np.zeros((len(edge_pairs), 3), dtype=np.float32)
            for i, (src, dst) in enumerate(edge_pairs):
                types = edge_types.get((src, dst), [])
                for dt in types:
                    if dt == DepType.RAW:
                        edge_attr[i, 0] = 1.0
                    elif dt == DepType.WAR:
                        edge_attr[i, 1] = 1.0
                    elif dt == DepType.WAW:
                        edge_attr[i, 2] = 1.0
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 3), dtype=np.float32)

        # --- Ready mask ---
        ready_mask = np.zeros(n, dtype=np.bool_)
        for inst_id in ready_set:
            ready_mask[inst_id] = True

        # --- Progress and pipeline state ---
        progress = len(self._scheduled_ids) / max(n, 1)
        pipeline_vec = np.array(self.pipeline.state.to_vector(), dtype=np.float32)

        # --- Ready queue pipeline mix ---
        ready_instructions = [self.dag.instructions[i] for i in ready_set]
        mix = analyze_ready_queue_mix(ready_instructions)
        ready_queue_mix = np.array([
            mix['math_fraction'],
            mix['memory_fraction'],
            mix['balance_score'],
        ], dtype=np.float32)

        # --- Register pressure ---
        # Normalized by budget. Values > 1.0 = spill territory.
        current_live = self.reg_tracker.live_count if self.reg_tracker else 0
        max_live = self.reg_tracker.max_live if self.reg_tracker else 0
        budget = max(self.register_budget, 1)
        register_pressure = np.array([
            current_live / budget,
            max_live / budget,
        ], dtype=np.float32)

        return Observation(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            ready_mask=ready_mask,
            schedule_progress=progress,
            pipeline_state=pipeline_vec,
            ready_queue_mix=ready_queue_mix,
            register_pressure=register_pressure,
        )


# --- Baseline heuristics ---

def random_rollout(env: ScheduleEnv, instructions: List[Instruction],
                   seed: Optional[int] = None) -> Tuple[List[int], int]:
    """Schedule instructions in random valid order.

    Returns (schedule_order, total_cycles).
    """
    rng = np.random.RandomState(seed)
    env.reset(instructions)
    while True:
        legal = env.legal_actions()
        if not legal:
            break
        action = legal[rng.randint(len(legal))]
        _, _, done, info = env.step(action)
        if done:
            return info['schedule_order'], info['total_cycles']
    return env.get_schedule_order(), env.get_total_cycles()


def critical_path_heuristic(env: ScheduleEnv,
                            instructions: List[Instruction]) -> Tuple[List[int], int]:
    """Greedy scheduling by longest remaining path (height).

    Standard critical-path list scheduling. At each step, picks the
    ready instruction with the highest height (most downstream work).

    Returns (schedule_order, total_cycles).
    """
    env.reset(instructions)
    while True:
        legal = env.legal_actions()
        if not legal:
            break
        action = max(legal, key=lambda i: (env.dag.height.get(i, 0), -i))
        _, _, done, info = env.step(action)
        if done:
            return info['schedule_order'], info['total_cycles']
    return env.get_schedule_order(), env.get_total_cycles()


def program_order_schedule(env: ScheduleEnv,
                           instructions: List[Instruction]) -> Tuple[List[int], int]:
    """Schedule in original program order (baseline "do nothing").

    Returns (schedule_order, total_cycles).
    """
    env.reset(instructions)
    while True:
        legal = env.legal_actions()
        if not legal:
            break
        action = min(legal)
        _, _, done, info = env.step(action)
        if done:
            return info['schedule_order'], info['total_cycles']
    return env.get_schedule_order(), env.get_total_cycles()


def interleave_heuristic(env: ScheduleEnv,
                         instructions: List[Instruction]) -> Tuple[List[int], int]:
    """Dual-issue-aware scheduling: alternate Math and Memory ops.

    This heuristic directly targets the L4 dual-issue optimization:
    it alternates between scheduling a Math pipeline instruction and
    a Memory pipeline instruction, maximizing the chance that consecutive
    instructions land on dual-issuable pipeline pairs.

    Within each category (Math/Memory), ties are broken by critical
    path height (highest first), then by instruction ID.

    When only one category is available in the ready queue, falls back
    to critical-path ordering.

    Returns (schedule_order, total_cycles).
    """
    env.reset(instructions)
    last_was_math = True  # Start by preferring Memory (to issue a Load first)

    while True:
        legal = env.legal_actions()
        if not legal:
            break

        # Partition ready instructions by pipeline type
        math_ready = [i for i in legal
                      if env.dag.instructions[i].pipeline in MATH_PIPELINES]
        memory_ready = [i for i in legal
                        if env.dag.instructions[i].pipeline in MEMORY_PIPELINES]
        other_ready = [i for i in legal
                       if i not in math_ready and i not in memory_ready]

        # Select category: alternate Math/Memory when both available
        if last_was_math and memory_ready:
            candidates = memory_ready
            last_was_math = False
        elif not last_was_math and math_ready:
            candidates = math_ready
            last_was_math = True
        elif math_ready:
            candidates = math_ready
            last_was_math = True
        elif memory_ready:
            candidates = memory_ready
            last_was_math = False
        else:
            candidates = other_ready

        # Within the chosen category, pick by critical path height
        action = max(candidates, key=lambda i: (env.dag.height.get(i, 0), -i))
        _, _, done, info = env.step(action)
        if done:
            return info['schedule_order'], info['total_cycles']

    return env.get_schedule_order(), env.get_total_cycles()


def compare_all_heuristics(instructions: List[Instruction],
                           n_random: int = 20) -> Dict:
    """Run all heuristics on the same instruction set and compare.

    Returns a dict with cycle counts, dual-issue rates, and rankings.
    """
    env = ScheduleEnv()

    # Deterministic heuristics
    _, po_cycles = program_order_schedule(env, instructions)
    _, cp_cycles = critical_path_heuristic(env, instructions)
    _, il_cycles = interleave_heuristic(env, instructions)

    # Get detailed breakdown for each
    def _get_breakdown(heuristic_fn):
        env_tmp = ScheduleEnv()
        order, _ = heuristic_fn(env_tmp, instructions)
        schedule = [env_tmp.dag.instructions[i] for i in order]
        return L4PipelineModel().estimate_with_breakdown(schedule)

    po_bd = _get_breakdown(program_order_schedule)
    cp_bd = _get_breakdown(critical_path_heuristic)
    il_bd = _get_breakdown(interleave_heuristic)

    # Random rollouts
    random_cycles = []
    for seed in range(n_random):
        _, rc = random_rollout(env, instructions, seed=seed)
        random_cycles.append(rc)

    results = {
        'program_order': {
            'cycles': po_cycles,
            'dual_issue_rate': po_bd['dual_issue_rate'],
            'dual_issues': po_bd['dual_issue_count'],
            'stalls': po_bd['stall_cycles'],
        },
        'critical_path': {
            'cycles': cp_cycles,
            'dual_issue_rate': cp_bd['dual_issue_rate'],
            'dual_issues': cp_bd['dual_issue_count'],
            'stalls': cp_bd['stall_cycles'],
        },
        'interleave': {
            'cycles': il_cycles,
            'dual_issue_rate': il_bd['dual_issue_rate'],
            'dual_issues': il_bd['dual_issue_count'],
            'stalls': il_bd['stall_cycles'],
        },
        'random': {
            'cycles_mean': float(np.mean(random_cycles)),
            'cycles_min': min(random_cycles),
            'cycles_max': max(random_cycles),
            'cycles_std': float(np.std(random_cycles)),
        },
        'best_heuristic': min(
            [('program_order', po_cycles),
             ('critical_path', cp_cycles),
             ('interleave', il_cycles)],
            key=lambda x: x[1]
        ),
    }

    return results
