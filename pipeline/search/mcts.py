"""Monte Carlo Tree Search for PTX instruction scheduling.

Pure MCTS with UCB1 (no neural network priors). Uses the pipeline
model for fast rollout evaluation. The search discovers scheduling
patterns from first principles — no hardcoded heuristics.

The search tree represents partial schedules. At each node, legal
actions are instructions whose DAG dependencies are satisfied.
Rollouts complete the schedule randomly and evaluate with the
pipeline cycle model.

This is Phase 2: validate that MCTS finds schedules in the
DAG's search space. Phase 3 replaces random rollouts with
learned value/policy networks (MuZero).
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..env.instruction import Instruction
from ..env.dag import InstructionDAG
from ..env.pipeline_model import L4PipelineModel, PipelineState
from ..env.register_tracker import RegisterPressureTracker
from .node import MCTSNode


@dataclass
class MCTSConfig:
    """MCTS hyperparameters.

    Defaults are calibrated for instruction DAGs with 10-200 nodes
    and 2-15 legal actions per step.
    """
    # Number of simulations (tree traversals) per move
    num_simulations: int = 100

    # UCB exploration constant. Controls exploration vs exploitation.
    # Higher = more exploration of unvisited children.
    # 2.0-4.0 for initial learning (aggressive exploration).
    # 1.0-1.5 once policy improves.
    c_puct: float = 2.5

    # Temperature for action selection after search.
    # 0.0 = deterministic (most-visited)
    # 1.0 = proportional to visit counts (exploratory)
    temperature: float = 0.0

    # Number of random rollouts per leaf evaluation.
    # More = lower variance but slower.
    num_rollouts: int = 1

    # Maximum rollout depth (0 = complete schedule)
    max_rollout_depth: int = 0

    # Random seed for reproducibility
    seed: Optional[int] = None

    # Value normalization: normalize Q-values to [-1, 0] range.
    # Required because raw cycle counts vary hugely across kernels.
    normalize_values: bool = True

    # Register pressure: max live registers before penalty.
    # 64 = 4 warps/SM on L4 (good occupancy).
    # 0 = disable register pressure tracking.
    register_budget: int = 64

    # Penalty per register over budget (subtracted from value).
    # 10 = each excess register costs 10 cycles equivalent.
    register_penalty_per_reg: float = 10.0


@dataclass
class SearchResult:
    """Result of MCTS search at one decision point."""
    # The action (instruction ID) selected by the search
    action: int

    # Visit distribution (MCTS policy) — training target for neural net
    visit_distribution: Dict[int, float]

    # Q-values per action
    action_values: Dict[int, float]

    # Root node visit count
    root_visits: int

    # Search statistics
    max_depth_reached: int = 0
    mean_rollout_value: float = 0.0
    search_time_ms: float = 0.0


class _ScheduleState:
    """Lightweight schedule state for MCTS tree traversal.

    Avoids the overhead of the full ScheduleEnv by tracking only
    what MCTS needs: the scheduled instruction set, pipeline state,
    and register pressure.
    """

    def __init__(self, dag: InstructionDAG, pipeline: L4PipelineModel,
                 reg_tracker: Optional[RegisterPressureTracker] = None,
                 register_budget: int = 64,
                 register_penalty: float = 10.0):
        self.dag = dag
        self.pipeline = pipeline
        self.reg_tracker = reg_tracker
        self.register_budget = register_budget
        self.register_penalty = register_penalty
        self.scheduled_ids: List[int] = []
        self.scheduled_set: frozenset = frozenset()

    def copy(self) -> '_ScheduleState':
        """Create an independent copy for rollouts."""
        new = _ScheduleState.__new__(_ScheduleState)
        new.dag = self.dag  # Shared (immutable during search)
        new.pipeline = L4PipelineModel()
        new.pipeline.restore_state(self.pipeline.copy_state())
        new.reg_tracker = self.reg_tracker.copy() if self.reg_tracker else None
        new.register_budget = self.register_budget
        new.register_penalty = self.register_penalty
        new.scheduled_ids = list(self.scheduled_ids)
        new.scheduled_set = self.scheduled_set
        return new

    def apply_action(self, action: int):
        """Schedule one instruction. Mutates this state."""
        inst = self.dag.instructions[action]
        self.pipeline.issue(inst)
        if self.reg_tracker:
            self.reg_tracker.schedule(action)
        self.scheduled_ids.append(action)
        self.scheduled_set = frozenset(self.scheduled_ids)

    def legal_actions(self) -> List[int]:
        """Instructions whose predecessors are all scheduled."""
        if len(self.scheduled_ids) == self.dag.n:
            return []
        return self.dag.get_ready_set(self.scheduled_set)

    @property
    def is_terminal(self) -> bool:
        return len(self.scheduled_ids) == self.dag.n

    def evaluate(self) -> float:
        """Return the value for the current schedule.

        Value = -(total_cycles + register_pressure_penalty).
        Higher (less negative) is better.
        """
        schedule = [self.dag.instructions[i] for i in self.scheduled_ids]
        model = L4PipelineModel()
        cycles = float(model.estimate_cycles(schedule))

        # Register pressure penalty
        reg_penalty = 0.0
        if self.reg_tracker and self.register_budget > 0:
            max_live = self.reg_tracker.max_live
            overshoot = max(0, max_live - self.register_budget)
            reg_penalty = overshoot * self.register_penalty

        return -(cycles + reg_penalty)


class MCTS:
    """Monte Carlo Tree Search for instruction scheduling.

    Usage:
        mcts = MCTS(config)
        # Search for the best schedule for a set of instructions
        schedule, cycles, stats = mcts.search_full_schedule(instructions)

        # Or step-by-step (for training data collection):
        mcts.set_root(instructions)
        while not mcts.is_terminal():
            result = mcts.search_step()
            mcts.apply_action(result.action)
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Internal state
        self._dag: Optional[InstructionDAG] = None
        self._root: Optional[MCTSNode] = None
        self._state: Optional[_ScheduleState] = None

        # Value normalization (MinMax across all rollouts in this search)
        self._min_value = float('inf')
        self._max_value = float('-inf')

    def set_root(self, instructions: List[Instruction]):
        """Initialize MCTS for a new scheduling problem.

        Args:
            instructions: The instructions to schedule (from parse_ptx_body).
        """
        self._dag = InstructionDAG(instructions)
        self._root = MCTSNode()

        # Create register tracker if budget is set
        reg_tracker = None
        if self.config.register_budget > 0:
            reg_tracker = RegisterPressureTracker(instructions)

        self._state = _ScheduleState(
            self._dag, L4PipelineModel(),
            reg_tracker=reg_tracker,
            register_budget=self.config.register_budget,
            register_penalty=self.config.register_penalty_per_reg,
        )
        self._min_value = float('inf')
        self._max_value = float('-inf')

    def is_terminal(self) -> bool:
        """True if all instructions have been scheduled."""
        if self._state is None:
            return True
        return self._state.is_terminal

    def search_step(self) -> SearchResult:
        """Run MCTS simulations and return the best action for the current state.

        This is the core MCTS loop:
        1. SELECT: traverse tree using UCB until reaching a leaf
        2. EXPAND: create children for the leaf
        3. ROLLOUT: complete the schedule randomly from the leaf
        4. BACKPROPAGATE: update values up the tree

        Returns:
            SearchResult with the chosen action and search statistics.
        """
        if self._state is None or self._dag is None or self._root is None:
            raise RuntimeError("Call set_root() first")

        t0 = time.monotonic()
        max_depth = 0
        rollout_values = []

        for sim in range(self.config.num_simulations):
            # 1. SELECT + EXPAND
            node, leaf_state = self._select_and_expand()

            # Track depth
            depth = node.depth
            if depth > max_depth:
                max_depth = depth

            # 2. ROLLOUT (evaluate leaf)
            if node.is_terminal:
                value = leaf_state.evaluate()
            else:
                value = self._rollout(leaf_state)

            rollout_values.append(value)

            # Update normalization bounds
            if value < self._min_value:
                self._min_value = value
            if value > self._max_value:
                self._max_value = value

            # 3. BACKPROPAGATE
            normalized = self._normalize(value)
            node.backpropagate(normalized)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Choose action
        action = self._root.best_action(self.config.temperature)

        result = SearchResult(
            action=action,
            visit_distribution=self._root.get_visit_distribution(),
            action_values=self._root.get_action_values(),
            root_visits=self._root.visit_count,
            max_depth_reached=max_depth,
            mean_rollout_value=float(np.mean(rollout_values)) if rollout_values else 0.0,
            search_time_ms=elapsed_ms,
        )

        return result

    def apply_action(self, action: int):
        """Commit an action and advance the root to the chosen child.

        Reuses the subtree under the chosen child (tree reuse).
        """
        if self._state is None or self._root is None:
            raise RuntimeError("Call set_root() first")

        # Apply action to the real state
        self._state.apply_action(action)

        # Advance root to the chosen child (tree reuse)
        if action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None  # Detach from old tree (GC)
        else:
            # Child doesn't exist (shouldn't happen, but handle gracefully)
            self._root = MCTSNode()

    def search_full_schedule(
        self,
        instructions: List[Instruction],
        callback: Optional[Callable[[int, SearchResult], None]] = None,
    ) -> Tuple[List[int], int, Dict]:
        """Search for the full schedule from scratch.

        Args:
            instructions: Instructions to schedule.
            callback: Optional function called after each step with
                      (step_number, search_result). Use for logging
                      or collecting training data.

        Returns:
            (schedule_order, total_cycles, stats_dict)
        """
        self.set_root(instructions)

        schedule = []
        step_results = []
        step = 0

        while not self.is_terminal():
            result = self.search_step()
            schedule.append(result.action)
            step_results.append(result)

            if callback is not None:
                callback(step, result)

            self.apply_action(result.action)
            step += 1

        # Evaluate the final schedule
        final_schedule = [self._dag.instructions[i] for i in schedule]
        model = L4PipelineModel()
        breakdown = model.estimate_with_breakdown(final_schedule)

        # Compute register pressure for the final schedule
        if self.config.register_budget > 0:
            reg_tracker = RegisterPressureTracker(list(self._dag.instructions.values()))
            for inst_id in schedule:
                reg_tracker.schedule(inst_id)
            reg_summary = reg_tracker.pressure_summary()
        else:
            reg_summary = {'max_live': 0, 'mean_live': 0}

        stats = {
            'total_cycles': breakdown['total_cycles'],
            'dual_issue_count': breakdown['dual_issue_count'],
            'single_issue_count': breakdown['single_issue_count'],
            'dual_issue_rate': breakdown['dual_issue_rate'],
            'stall_cycles': breakdown['stall_cycles'],
            'total_search_time_ms': sum(r.search_time_ms for r in step_results),
            'mean_simulations_per_step': self.config.num_simulations,
            'total_steps': step,
            'mean_rollout_value': float(np.mean([r.mean_rollout_value for r in step_results])),
            'max_live_registers': reg_summary['max_live'],
            'mean_live_registers': reg_summary['mean_live'],
            'register_budget': self.config.register_budget,
            'register_spill_risk': reg_summary['max_live'] > self.config.register_budget,
        }

        return schedule, breakdown['total_cycles'], stats

    def _select_and_expand(self) -> Tuple[MCTSNode, _ScheduleState]:
        """SELECT phase: traverse tree using UCB until reaching a leaf.
        EXPAND phase: create children at the leaf.

        Returns:
            (leaf_node, leaf_state) where leaf_state is a copy of the
            schedule state at the leaf node.
        """
        node = self._root
        state = self._state.copy()

        # Traverse existing tree using UCB
        while node.is_expanded and not node.is_terminal:
            node = node.select_child(self.config.c_puct)
            state.apply_action(node.action)

        # If not yet expanded, expand
        if not node.is_expanded and not node.is_terminal:
            legal = state.legal_actions()
            if not legal:
                node.is_terminal = True
            else:
                node.expand(legal)  # Uniform priors for pure MCTS

        return node, state

    def _rollout(self, state: _ScheduleState) -> float:
        """ROLLOUT phase: complete the schedule randomly from the given state.

        Uses the pipeline model for fast evaluation (no hardware measurement).

        Returns:
            Negative total cycles for the complete schedule.
        """
        state = state.copy()  # Don't mutate the tree state

        depth = 0
        max_depth = self.config.max_rollout_depth

        while not state.is_terminal:
            legal = state.legal_actions()
            if not legal:
                break

            # Random action selection (pure MCTS rollout policy)
            action = legal[self.rng.randint(len(legal))]
            state.apply_action(action)

            depth += 1
            if max_depth > 0 and depth >= max_depth:
                break

        if state.is_terminal:
            return state.evaluate()
        else:
            # Incomplete rollout: estimate remaining from current state
            # (shouldn't happen with max_rollout_depth=0)
            return state.evaluate()

    def _normalize(self, value: float) -> float:
        """Normalize value to approximately [-1, 0] range.

        Uses MinMax normalization across all rollouts seen so far.
        This is critical for UCB to work correctly when raw cycle
        counts vary from ~100 to ~10000.
        """
        if not self.config.normalize_values:
            return value

        value_range = self._max_value - self._min_value
        if value_range < 1e-8:
            return 0.0

        # Map [min, max] -> [-1, 0]
        # min_value maps to -1 (worst), max_value maps to 0 (best)
        return (value - self._min_value) / value_range - 1.0


def mcts_vs_baselines(
    instructions: List[Instruction],
    config: Optional[MCTSConfig] = None,
    n_random: int = 20,
    verbose: bool = True,
) -> Dict:
    """Run MCTS and compare against all baseline heuristics.

    This is the primary validation function for Phase 2.

    Returns:
        Dict with cycle counts, dual-issue rates, and rankings for all methods.
    """
    from ..env.schedule_env import (
        ScheduleEnv, compare_all_heuristics,
    )

    # Run baseline heuristics
    baseline_results = compare_all_heuristics(instructions, n_random=n_random)

    # Run MCTS
    if config is None:
        config = MCTSConfig()
    mcts = MCTS(config)
    mcts_schedule, mcts_cycles, mcts_stats = mcts.search_full_schedule(instructions)

    results = {
        **baseline_results,
        'mcts': {
            'cycles': mcts_cycles,
            'dual_issue_rate': mcts_stats['dual_issue_rate'],
            'dual_issues': mcts_stats['dual_issue_count'],
            'stalls': mcts_stats['stall_cycles'],
            'search_time_ms': mcts_stats['total_search_time_ms'],
            'max_live_regs': mcts_stats['max_live_registers'],
            'register_spill_risk': mcts_stats['register_spill_risk'],
            'schedule': mcts_schedule,
        },
    }

    # Recompute best including MCTS
    all_methods = [
        ('program_order', baseline_results['program_order']['cycles']),
        ('critical_path', baseline_results['critical_path']['cycles']),
        ('interleave', baseline_results['interleave']['cycles']),
        ('mcts', mcts_cycles),
    ]
    results['best_overall'] = min(all_methods, key=lambda x: x[1])

    if verbose:
        po = baseline_results['program_order']
        cp = baseline_results['critical_path']
        il = baseline_results['interleave']
        rnd = baseline_results['random']
        mc = results['mcts']

        print(f"\n{'='*70}")
        print(f"Scheduling Comparison — {len(instructions)} instructions, "
              f"{config.num_simulations} sims/step")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'Cycles':>8} {'Dual%':>8} {'Duals':>8} "
              f"{'Stalls':>8} {'Time':>10}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        print(f"{'Program order':<20} {po['cycles']:>8} {po['dual_issue_rate']:>7.1%} "
              f"{po['dual_issues']:>8} {po['stalls']:>8} {'--':>10}")
        print(f"{'Critical path':<20} {cp['cycles']:>8} {cp['dual_issue_rate']:>7.1%} "
              f"{cp['dual_issues']:>8} {cp['stalls']:>8} {'--':>10}")
        print(f"{'Interleave':<20} {il['cycles']:>8} {il['dual_issue_rate']:>7.1%} "
              f"{il['dual_issues']:>8} {il['stalls']:>8} {'--':>10}")
        print(f"{'MCTS':<20} {mc['cycles']:>8} {mc['dual_issue_rate']:>7.1%} "
              f"{mc['dual_issues']:>8} {mc['stalls']:>8} "
              f"{mc['search_time_ms']:>8.0f}ms")
        print(f"{'Random (mean)':<20} {rnd['cycles_mean']:>8.0f}")
        print(f"{'Random (min/max)':<20} {rnd['cycles_min']:>8}/{rnd['cycles_max']:>8}")
        print(f"{'='*70}")

        best_name, best_cycles = results['best_overall']
        print(f"Best: {best_name} ({best_cycles} cycles)")
        if po['cycles'] > 0:
            print(f"Speedup vs program order: {po['cycles']/best_cycles:.2f}x")
        if cp['cycles'] > 0:
            print(f"Speedup vs critical path: {cp['cycles']/best_cycles:.2f}x")

    return results


def sweep_simulations(
    instructions: List[Instruction],
    sim_counts: Optional[List[int]] = None,
    seed: int = 0,
    verbose: bool = True,
) -> List[Dict]:
    """Run MCTS with increasing simulation budgets.

    Shows how schedule quality improves with more search.
    This produces the "learning curve" the founder wants to see.

    Args:
        instructions: Instructions to schedule.
        sim_counts: List of simulation counts to try.
                    Default: [1, 5, 10, 25, 50, 100, 200, 500]
        seed: Random seed for reproducibility.
        verbose: Print results table.

    Returns:
        List of dicts with {num_sims, cycles, dual_issue_rate, search_time_ms}.
    """
    if sim_counts is None:
        sim_counts = [1, 5, 10, 25, 50, 100, 200, 500]

    results = []

    for n_sims in sim_counts:
        config = MCTSConfig(
            num_simulations=n_sims,
            c_puct=2.5,
            temperature=0.0,
            seed=seed,
            normalize_values=True,
        )
        mcts = MCTS(config)
        schedule, cycles, stats = mcts.search_full_schedule(instructions)
        results.append({
            'num_sims': n_sims,
            'cycles': cycles,
            'dual_issue_rate': stats['dual_issue_rate'],
            'dual_issues': stats['dual_issue_count'],
            'stalls': stats['stall_cycles'],
            'search_time_ms': stats['total_search_time_ms'],
        })

    if verbose:
        print(f"\n{'='*60}")
        print(f"MCTS Simulation Sweep — {len(instructions)} instructions")
        print(f"{'='*60}")
        print(f"{'Sims':>6} {'Cycles':>8} {'Dual%':>8} {'Stalls':>8} {'Time(ms)':>10}")
        print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for r in results:
            print(f"{r['num_sims']:>6} {r['cycles']:>8} "
                  f"{r['dual_issue_rate']:>7.1%} {r['stalls']:>8} "
                  f"{r['search_time_ms']:>10.1f}")
        print(f"{'='*60}")

        # Show convergence
        if len(results) >= 2:
            worst = max(r['cycles'] for r in results)
            best = min(r['cycles'] for r in results)
            if worst > 0:
                improvement = (worst - best) / worst
                print(f"Improvement from {sim_counts[0]} to {sim_counts[-1]} sims: "
                      f"{improvement:.1%}")

    return results
