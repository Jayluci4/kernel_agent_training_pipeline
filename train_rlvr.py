"""RLVR training for PTX transform selection.

Usage:
    # BC warm-start only (CPU, no GPU needed)
    python scripts/train_rlvr.py --bc-only

    # Full GRPO training (requires NVIDIA GPU + CUDA env)
    source experiments/chronos/setup_env.sh
    python scripts/train_rlvr.py

    # Evaluate a saved checkpoint
    python scripts/train_rlvr.py --eval --checkpoint results/rlvr/checkpoint_latest.pt

    # Quick test (2 kernels, 5 epochs)
    python scripts/train_rlvr.py --quick
"""

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
from experiments.chronos.rl.grpo import GRPOTrainer

# All 64 gemm_tile kernels from greedy v2
ALL_KERNELS = [
    (m, n, k)
    for m in [2, 4, 6, 8]
    for n in [2, 4, 6, 8]
    for k in [2, 4, 6, 8]
]

QUICK_KERNELS = [(4, 6, 8), (6, 8, 8)]

logger = logging.getLogger("rlvr")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


def main():
    parser = argparse.ArgumentParser(
        description="RLVR training for PTX transform selection"
    )

    # Data paths
    parser.add_argument(
        "--trajectories",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "trajectories_v2.jsonl"),
        help="Path to BC trajectory JSONL",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "rlvr"),
        help="Directory for checkpoints and logs",
    )

    # Training modes
    parser.add_argument("--bc-only", action="store_true", help="BC warm-start only")
    parser.add_argument("--eval", action="store_true", help="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for --eval")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 kernels)")
    parser.add_argument(
        "--no-hardware", action="store_true",
        help="Disable hardware measurement (dry run)",
    )

    # Hyperparameters
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--grpo-epochs", type=int, default=450)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)

    args = parser.parse_args()

    # Resolve paths
    args.trajectories = os.path.abspath(args.trajectories)
    args.save_dir = os.path.abspath(args.save_dir)

    setup_logging(args.save_dir)

    # Select kernels
    if args.quick:
        kernels = QUICK_KERNELS
        args.bc_epochs = min(args.bc_epochs, 5)
        args.grpo_epochs = min(args.grpo_epochs, 5)
        args.eval_every = 5
        args.group_size = 2
        logger.info("Quick mode: %d kernels, %d BC + %d GRPO epochs",
                     len(kernels), args.bc_epochs, args.grpo_epochs)
    else:
        kernels = ALL_KERNELS
        logger.info("Full mode: %d kernels", len(kernels))

    use_hardware = not args.no_hardware
    device = "cpu"  # MLP is tiny, CPU is fine

    logger.info("Configuration:")
    logger.info("  kernels: %d", len(kernels))
    logger.info("  use_hardware: %s", use_hardware)
    logger.info("  device: %s", device)
    logger.info("  trajectories: %s", args.trajectories)
    logger.info("  save_dir: %s", args.save_dir)

    # Build trainer
    trainer = GRPOTrainer(
        kernels=kernels,
        hidden=args.hidden,
        lr=args.lr,
        bc_lr=args.bc_lr,
        group_size=args.group_size,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        beta=args.beta,
        use_hardware=use_hardware,
        device=device,
    )

    # Evaluation mode
    if args.eval:
        if not args.checkpoint:
            parser.error("--eval requires --checkpoint")
        trainer.load_checkpoint(args.checkpoint)
        if not use_hardware:
            logger.error("Evaluation requires hardware measurement (remove --no-hardware)")
            sys.exit(1)
        result = trainer.evaluate()
        logger.info("Mean improvement: %.1f%% (%d kernels)",
                     result["mean_improvement"] * 100, result["n_kernels"])
        out_path = os.path.join(args.save_dir, "eval_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", out_path)
        return

    # BC-only mode
    if args.bc_only:
        acc = trainer.bc_warmstart(args.trajectories, n_epochs=args.bc_epochs)
        logger.info("BC warm-start accuracy: %.3f", acc)
        trainer._save_checkpoint(args.save_dir, args.bc_epochs)

        # Save BC stats
        stats_path = os.path.join(args.save_dir, "bc_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "bc_loss": trainer.stats["bc_loss"],
                "bc_accuracy": trainer.stats["bc_accuracy"],
                "best_accuracy": acc,
            }, f, indent=2)
        logger.info("Saved BC stats: %s", stats_path)
        return

    # Full training
    result = trainer.train(
        trajectory_path=args.trajectories,
        n_bc_epochs=args.bc_epochs,
        n_grpo_epochs=args.grpo_epochs,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
    )

    # Save final results
    out_path = os.path.join(args.save_dir, "training_result.json")
    with open(out_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(result, f, indent=2, default=convert)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    import numpy as np
    main()
