"""Generate BC pretraining dataset: 2,000 schedules measured on L4 hardware.

Why: The pipeline model has Spearman=0.462 (rankings are wrong). To calibrate
the learned world model, we need ground truth cycle counts from the L4.

How: Mixed generation strategy to cover the search space:
  - 80% random schedules (uniform over DAG-valid orderings)
  - 10% heuristic schedules (critical_path, interleave, MCTS)
  - 10% perturbed heuristic (swap 2-3 adjacent pairs near optima)

Each schedule is:
  1. Generated as an action sequence (list of instruction IDs)
  2. Replayed through ScheduleEnv to 75% completion (observation checkpoint)
  3. Measured with SM clock() cycle counter on real L4 hardware
  4. Stored with observation, cycles, and metadata

Output: bc_dataset.pkl

Usage:
    source experiments/chronos/setup_env.sh
    python -m experiments.chronos.training.gen_bc_data
"""

import os
import sys
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiments.chronos.env.instruction import Instruction, parse_ptx_body
from experiments.chronos.env.dag import InstructionDAG
from experiments.chronos.env.schedule_env import (
    ScheduleEnv, random_rollout, critical_path_heuristic,
    interleave_heuristic, program_order_schedule,
)
from experiments.chronos.env.pipeline_model import L4PipelineModel
from experiments.chronos.search.mcts import MCTS, MCTSConfig
from experiments.chronos.harness.ptx_templates import gemm_tile, TEMPLATES
from experiments.chronos.harness.cycle_counter import measure_schedule_cycles
from experiments.chronos.harness.validate_model import _extract_reg_declarations


def generate_random_schedule_fast(dag: InstructionDAG,
                                  seed: int) -> List[int]:
    """Generate a random DAG-valid schedule (fast, reuses pre-built DAG)."""
    rng = np.random.RandomState(seed)
    scheduled = set()
    actions = []
    for _ in range(dag.n):
        ready = dag.get_ready_set(frozenset(scheduled))
        if not ready:
            break
        action = ready[rng.randint(len(ready))]
        actions.append(action)
        scheduled.add(action)
    return actions


def generate_heuristic_schedule(instructions: List[Instruction],
                                name: str, seed: int = 0,
                                register_budget: int = 64) -> List[int]:
    """Generate a schedule using a named heuristic."""
    env = ScheduleEnv(register_budget=register_budget)
    if name == 'critical_path':
        order, _ = critical_path_heuristic(env, instructions)
    elif name == 'interleave':
        order, _ = interleave_heuristic(env, instructions)
    elif name == 'program_order':
        order, _ = program_order_schedule(env, instructions)
    elif name == 'mcts':
        config = MCTSConfig(num_simulations=10, seed=seed, c_puct=2.5,
                           register_budget=register_budget)
        mcts = MCTS(config)
        order, _, _ = mcts.search_full_schedule(instructions)
    else:
        raise ValueError(f"Unknown heuristic: {name}")
    return order


def is_valid_schedule(instructions: List[Instruction],
                      schedule: List[int]) -> bool:
    """Check if a schedule respects DAG constraints."""
    dag = InstructionDAG(instructions)
    scheduled = set()
    for action in schedule:
        ready = dag.get_ready_set(frozenset(scheduled))
        if action not in ready:
            return False
        scheduled.add(action)
    return True


def generate_perturbed_schedule(instructions: List[Instruction],
                                base_schedule: List[int],
                                seed: int,
                                n_swaps: int = 3) -> List[int]:
    """Perturb a heuristic schedule by swapping adjacent pairs.

    Swaps are only applied if they maintain DAG validity.
    """
    dag = InstructionDAG(instructions)
    rng = np.random.RandomState(seed)
    schedule = list(base_schedule)

    for _ in range(n_swaps * 3):  # Try more times to get n_swaps actual swaps
        if len(schedule) < 2:
            break
        idx = rng.randint(0, len(schedule) - 1)
        a, b = schedule[idx], schedule[idx + 1]

        # Check if b has a direct dependency on a
        if a in dag.predecessors.get(b, set()):
            continue  # Can't swap: b depends on a directly

        # Check transitive dependency via BFS
        # For adjacent swaps, we only need to check if a is an ancestor of b
        # Quick check: if a's successors reach b, skip
        if _is_ancestor(dag, a, b):
            continue

        schedule[idx], schedule[idx + 1] = b, a

    return schedule


def _is_ancestor(dag: InstructionDAG, ancestor: int, descendant: int) -> bool:
    """Check if ancestor has a path to descendant in the DAG."""
    visited = set()
    stack = [ancestor]
    while stack:
        node = stack.pop()
        if node == descendant:
            return True
        if node in visited:
            continue
        visited.add(node)
        for succ in dag.successors.get(node, set()):
            stack.append(succ)
    return False


def collect_terminal_observation(instructions: List[Instruction],
                                 schedule: List[int],
                                 register_budget: int = 64) -> Dict:
    """Replay full schedule through env and collect TERMINAL observation.

    Uses 100% progress so the GNN sees the COMPLETE scheduling order.
    The observation now includes per-instruction scoreboard features
    (stall_at_issue, fu_wait, reg_wait, dual_issued, issue_cycle)
    computed by the pipeline model at each step. These encode the
    PHYSICS of what happened during execution — the model sees the
    scoreboard, not just positions.

    Also appends schedule_position as the final feature.

    Returns observation dict with node_features shape [N, 49]
    (48 base with scoreboard + 1 position).
    """
    env = ScheduleEnv(register_budget=register_budget)
    env.reset(instructions)
    N = len(instructions)

    obs = None
    for i, action in enumerate(schedule):
        # Skip observation on intermediate steps (770x speedup for 8x8x8)
        is_last = (i == N - 1)
        obs, _, done, info = env.step(action, observe=is_last)

    # Add schedule_position: step/N for each node
    schedule_pos = np.zeros((N, 1), dtype=np.float32)
    for step, inst_id in enumerate(schedule):
        schedule_pos[inst_id, 0] = step / N

    # Append to node_features: [N, 48] -> [N, 49]
    node_features = np.concatenate([obs.node_features, schedule_pos], axis=1)

    return {
        'node_features': node_features,
        'edge_index': obs.edge_index,
        'edge_attr': obs.edge_attr,
        'ready_mask': obs.ready_mask,
        'pipeline_state': obs.pipeline_state,
        'register_pressure': obs.register_pressure,
    }


def get_model_cycles(instructions: List[Instruction],
                     schedule: List[int]) -> int:
    """Get pipeline model cycle prediction for a schedule."""
    inst_map = {inst.id: inst for inst in instructions}
    ordered = [inst_map[i] for i in schedule]
    return L4PipelineModel().estimate_cycles(ordered)


def generate_all_schedules(instructions: List[Instruction],
                           n_total: int = 2000,
                           register_budget: int = 64) -> List[Dict]:
    """Generate schedules with mixed strategy.

    Returns list of dicts with 'schedule' and 'strategy' keys.
    """
    n_random = int(n_total * 0.80)
    n_heuristic = int(n_total * 0.10)
    n_perturbed = n_total - n_random - n_heuristic

    # Build DAG once (expensive) and reuse for all fast generation
    dag = InstructionDAG(instructions)
    samples = []

    # 1. Random schedules (80%)
    print(f"  Generating {n_random} random schedules...")
    t0 = time.monotonic()
    for seed in range(n_random):
        schedule = generate_random_schedule_fast(dag, seed=seed)
        samples.append({'schedule': schedule, 'strategy': 'random'})
        if (seed + 1) % 500 == 0:
            print(f"    [{seed+1}/{n_random}] {time.monotonic()-t0:.1f}s")
    print(f"    Done in {time.monotonic()-t0:.1f}s")

    # 2. Heuristic schedules (10%)
    # 3 deterministic heuristics as calibration anchors,
    # remaining filled with random schedules (different seed range)
    print(f"  Generating {n_heuristic} heuristic schedules...")
    for name in ['critical_path', 'interleave', 'program_order']:
        schedule = generate_heuristic_schedule(instructions, name,
                                               register_budget=register_budget)
        samples.append({'schedule': schedule, 'strategy': f'heuristic_{name}'})

    # Fill remaining heuristic slots with random (different seed range)
    for i in range(n_heuristic - 3):
        schedule = generate_random_schedule_fast(dag, seed=n_random + i)
        samples.append({'schedule': schedule, 'strategy': 'heuristic_random'})

    # 3. Perturbed heuristic schedules (10%)
    print(f"  Generating {n_perturbed} perturbed schedules...")
    base_cp = generate_heuristic_schedule(instructions, 'critical_path',
                                           register_budget=register_budget)
    base_il = generate_heuristic_schedule(instructions, 'interleave',
                                           register_budget=register_budget)

    for i in range(n_perturbed):
        base = base_cp if i % 2 == 0 else base_il
        n_swaps = np.random.randint(2, 6)
        schedule = generate_perturbed_schedule(
            instructions, base, seed=10000 + i, n_swaps=n_swaps)
        samples.append({'schedule': schedule, 'strategy': 'perturbed'})

    return samples


def measure_schedules(instructions: List[Instruction],
                      samples: List[Dict],
                      reg_decls: str,
                      kernel_name: str = 'gemm_tile',
                      n_warmup: int = 50,
                      n_runs: int = 200) -> None:
    """Measure all schedules with SM cycle counter on L4 hardware.

    Deduplicates identical schedules to avoid redundant compilation.
    Updates samples in-place with 'hw_cycles' key.
    """
    # Deduplicate: group by schedule tuple
    schedule_to_indices = {}
    for i, sample in enumerate(samples):
        key = tuple(sample['schedule'])
        if key not in schedule_to_indices:
            schedule_to_indices[key] = []
        schedule_to_indices[key].append(i)

    n_unique = len(schedule_to_indices)
    print(f"  {len(samples)} total schedules, {n_unique} unique")
    print(f"  Measuring {n_unique} unique schedules...")

    t0 = time.monotonic()
    measured = 0
    errors = 0

    for schedule_tuple, indices in schedule_to_indices.items():
        schedule = list(schedule_tuple)
        result = measure_schedule_cycles(
            instructions, schedule,
            kernel_name=kernel_name,
            n_warmup=n_warmup, n_runs=n_runs,
            reg_decls=reg_decls,
        )

        if 'error' in result:
            errors += 1
            for idx in indices:
                samples[idx]['hw_cycles'] = -1
        else:
            for idx in indices:
                samples[idx]['hw_cycles'] = result['median_cycles']

        measured += 1
        if measured % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = measured / elapsed
            remaining = (n_unique - measured) / rate if rate > 0 else 0
            print(f"    [{measured}/{n_unique}] "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                  f"{errors} errors")

    elapsed = time.monotonic() - t0
    print(f"  Measurement complete: {elapsed:.0f}s, {errors} errors")


def build_dataset(instructions: List[Instruction],
                  samples: List[Dict],
                  kernel_name: str = 'gemm_tile',
                  register_budget: int = 64) -> Dict:
    """Build the final dataset with observations and metadata.

    For each sample, replays the full schedule and collects the
    terminal observation (which encodes the complete schedule state).
    """
    print(f"  Collecting terminal observations (100% progress)...")

    valid_samples = [s for s in samples if s.get('hw_cycles', -1) > 0]
    print(f"  {len(valid_samples)} valid samples "
          f"(dropped {len(samples) - len(valid_samples)} with measurement errors)")

    dataset = {
        'observations': [],
        'hw_cycles': [],
        'model_cycles': [],
        'schedules': [],
        'strategies': [],
        'kernel_name': kernel_name,
        'n_instructions': len(instructions),
        'register_budget': register_budget,
    }

    for i, sample in enumerate(valid_samples):
        obs = collect_terminal_observation(instructions, sample['schedule'],
                                           register_budget=register_budget)
        model_cyc = get_model_cycles(instructions, sample['schedule'])

        dataset['observations'].append(obs)
        dataset['hw_cycles'].append(sample['hw_cycles'])
        dataset['model_cycles'].append(model_cyc)
        dataset['schedules'].append(sample['schedule'])
        dataset['strategies'].append(sample['strategy'])

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(valid_samples)}] observations collected")

    return dataset


def print_dataset_stats(dataset: Dict):
    """Print dataset statistics."""
    cycles = np.array(dataset['hw_cycles'])
    model = np.array(dataset['model_cycles'])
    strategies = dataset['strategies']

    print(f"\n{'='*60}")
    print(f"BC Dataset Summary")
    print(f"{'='*60}")
    print(f"Total samples:      {len(cycles)}")
    print(f"Kernel:             {dataset['kernel_name']}")
    print(f"Instructions:       {dataset['n_instructions']}")
    print()

    # Cycle statistics
    print(f"Hardware cycles:    {cycles.min()} - {cycles.max()} "
          f"(mean={cycles.mean():.1f}, std={cycles.std():.1f})")
    print(f"Model cycles:       {model.min()} - {model.max()} "
          f"(mean={model.mean():.1f})")

    # Normalized targets: (baseline - cycles) / scale
    # baseline = median, scale = IQR (kernel-agnostic)
    baseline = float(np.median(cycles))
    q75, q25 = np.percentile(cycles, 75), np.percentile(cycles, 25)
    scale = max(float(q75 - q25), 1.0)
    targets = (baseline - cycles) / scale
    print(f"Normalization:      baseline={baseline:.1f} (median), scale={scale:.1f} (IQR)")
    print(f"Normalized targets: {targets.min():.3f} - {targets.max():.3f} "
          f"(mean={targets.mean():.3f})")

    # Strategy breakdown
    print(f"\nStrategy breakdown:")
    for strat in sorted(set(strategies)):
        count = sum(1 for s in strategies if s == strat)
        strat_cycles = [c for c, s in zip(cycles, strategies) if s == strat]
        print(f"  {strat:<25} {count:>5}  "
              f"cycles: {min(strat_cycles)}-{max(strat_cycles)} "
              f"(mean={np.mean(strat_cycles):.1f})")

    # Spearman: model vs hardware
    from scipy.stats import spearmanr
    sp, p = spearmanr(model, cycles)
    print(f"\nModel vs Hardware Spearman: {sp:.3f} (p={p:.4f})")

    # Unique cycle counts
    unique_cycles = sorted(set(cycles))
    print(f"Unique cycle values: {len(unique_cycles)} ({unique_cycles[:10]}...)")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-total', type=int, default=2000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--kernel', type=str, default='gemm_tile',
                        choices=list(TEMPLATES.keys()),
                        help='Kernel template to use')
    parser.add_argument('--register-budget', type=int, default=None,
                        help='Register budget (auto-detected if not set)')
    args = parser.parse_args()

    n_total = args.n_total
    kernel_name = args.kernel
    print("=" * 60)
    print(f"BC Dataset Generation: {n_total:,} Schedules on L4 Hardware")
    print("=" * 60)

    # Parse kernel
    spec = TEMPLATES[kernel_name]()
    instructions = parse_ptx_body(spec.ptx_source, kernel_name=spec.kernel_name)
    reg_decls = _extract_reg_declarations(spec.ptx_source, spec.kernel_name)

    # Auto-detect register budget from kernel size
    # Count unique float registers to estimate pressure
    all_regs = set()
    for inst in instructions:
        all_regs.update(r for r in inst.dest_regs if r.startswith('%f'))
        all_regs.update(r for r in inst.src_regs if r.startswith('%f'))
    n_float_regs = len(all_regs)
    register_budget = args.register_budget or max(64, n_float_regs + 8)

    print(f"Kernel: {kernel_name}, {len(instructions)} instructions, "
          f"{n_float_regs} float regs, budget={register_budget}")

    # Generate schedules
    print(f"\nStep 1: Generate schedules")
    t0 = time.monotonic()
    samples = generate_all_schedules(instructions, n_total=n_total,
                                     register_budget=register_budget)
    gen_time = time.monotonic() - t0
    print(f"  Generated {len(samples)} schedules in {gen_time:.1f}s")

    # Measure on hardware
    print(f"\nStep 2: Measure SM cycles on L4 hardware")
    measure_schedules(instructions, samples, reg_decls,
                     kernel_name=spec.kernel_name,
                     n_warmup=50, n_runs=200)

    # Build dataset with observations
    print(f"\nStep 3: Collect observations")
    dataset = build_dataset(instructions, samples,
                           kernel_name=kernel_name,
                           register_budget=register_budget)

    # Print statistics
    print_dataset_stats(dataset)

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    default_name = f'bc_dataset_{kernel_name}.pkl' if kernel_name != 'gemm_tile' else 'bc_dataset.pkl'
    output_path = args.output or os.path.join(output_dir, default_name)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
