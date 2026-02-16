"""GRPO trainer for PTX transform selection.

Implements:
1. BC warm-start from stored trajectory JSONL
2. GRPO rollout generation with hardware-in-the-loop measurement
3. MC-GRPO advantage computation (median baseline, global normalization)
4. Clipped surrogate loss with KL penalty against reference policy
"""

import copy
import json
import math
import random
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .policy import (
    TransformPolicy, N_FEATURES, N_ACTIONS,
    ACTION_NAMES, ACTION_TO_ID,
    get_action_mask, get_action_history,
    CONFLICT_GROUPS,
)
from .env import TransformEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BC data loading
# ---------------------------------------------------------------------------

def _reconstruct_history(entries_for_kernel):
    """Given trajectory entries for one kernel (sorted by step), add action history."""
    applied = set()
    for entry in entries_for_kernel:
        entry["_action_history"] = get_action_history(applied)
        entry["_action_mask"] = get_action_mask(applied)
        if entry["action"] != "stop":
            applied.add(entry["action"])


def load_bc_data(path):
    """Load trajectory JSONL and convert to BC training tensors.

    Groups entries by kernel to reconstruct proper action histories.
    Returns dict with features, masks, histories, actions tensors.
    """
    # Group entries by kernel
    by_kernel = defaultdict(list)
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            by_kernel[entry["kernel"]].append(entry)

    # Sort each kernel's entries by step and reconstruct history
    for kernel_name in by_kernel:
        by_kernel[kernel_name].sort(key=lambda e: e["step"])
        _reconstruct_history(by_kernel[kernel_name])

    # Flatten back to list and convert to tensors
    features_list = []
    masks_list = []
    histories_list = []
    actions_list = []

    for entries in by_kernel.values():
        for entry in entries:
            features_list.append(
                torch.tensor(entry["feature_array"], dtype=torch.float32)
            )
            masks_list.append(
                torch.tensor(entry["_action_mask"], dtype=torch.float32)
            )
            histories_list.append(
                torch.tensor(entry["_action_history"], dtype=torch.float32)
            )
            actions_list.append(
                torch.tensor(entry["action_id"], dtype=torch.long)
            )

    return {
        "features": torch.stack(features_list),
        "masks": torch.stack(masks_list),
        "histories": torch.stack(histories_list),
        "actions": torch.stack(actions_list),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """GRPO training loop for PTX transform selection."""

    def __init__(
        self,
        kernels,
        hidden=128,
        lr=3e-4,
        bc_lr=1e-3,
        group_size=8,
        batch_size=32,
        max_steps=6,
        epsilon=0.2,
        beta=0.01,
        use_hardware=True,
        device="cpu",
    ):
        self.kernels = kernels
        self.group_size = group_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.beta = beta
        self.use_hardware = use_hardware
        self.device = device

        # Policy networks
        self.policy = TransformPolicy(hidden=hidden).to(device)
        self.ref_policy = TransformPolicy(hidden=hidden).to(device)

        # Optimizers
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.bc_optimizer = Adam(self.policy.parameters(), lr=bc_lr)

        # Environments (lazy-created per kernel)
        self.envs = {}
        for m, n, k in kernels:
            self.envs[(m, n, k)] = TransformEnv(
                m=m, n=n, k=k,
                max_steps=max_steps,
                use_hardware=use_hardware,
            )

        # Training stats
        self.stats = defaultdict(list)

    # ---- BC warm-start ----

    def bc_epoch(self, data):
        """Run one epoch of behavior cloning. Returns (loss, accuracy)."""
        self.policy.train()
        n = len(data["actions"])
        indices = torch.randperm(n)

        total_loss = 0.0
        n_correct = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]

            features = data["features"][idx].to(self.device)
            masks = data["masks"][idx].to(self.device)
            histories = data["histories"][idx].to(self.device)
            actions = data["actions"][idx].to(self.device)

            logits = self.policy(features, masks, histories)
            loss = F.cross_entropy(logits, actions)

            self.bc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.bc_optimizer.step()

            total_loss += loss.item() * len(idx)
            n_correct += (logits.argmax(dim=-1) == actions).sum().item()

        return total_loss / n, n_correct / n

    def bc_warmstart(self, trajectory_path, n_epochs=50):
        """BC warm-start phase. Returns best accuracy achieved."""
        logger.info("Loading trajectories from %s", trajectory_path)
        data = load_bc_data(trajectory_path)
        logger.info("Loaded %d trajectory entries for BC", len(data["actions"]))

        best_acc = 0.0
        best_state = None

        for epoch in range(n_epochs):
            loss, acc = self.bc_epoch(data)
            self.stats["bc_loss"].append(loss)
            self.stats["bc_accuracy"].append(acc)

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.policy.state_dict())

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "BC epoch %d/%d: loss=%.4f, accuracy=%.3f (best=%.3f)",
                    epoch + 1, n_epochs, loss, acc, best_acc,
                )

        # Restore best BC weights and set as reference
        if best_state is not None:
            self.policy.load_state_dict(best_state)
        self.ref_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        logger.info("BC warm-start complete. Best accuracy: %.3f", best_acc)
        return best_acc

    # ---- GRPO rollouts ----

    def generate_rollout(self, env, temperature=1.0):
        """Generate one episode. Returns list of step dicts."""
        self.policy.eval()
        state = env.reset()
        features, mask, history = state

        trajectory = []
        for _ in range(self.max_steps):
            with torch.no_grad():
                action_id, log_prob = self.policy.get_action(
                    features.to(self.device),
                    mask.to(self.device),
                    history.to(self.device),
                    temperature=temperature,
                )

            next_state, reward, done, info = env.step(action_id)

            trajectory.append({
                "features": features.clone(),
                "mask": mask.clone(),
                "history": history.clone(),
                "action": action_id,
                "reward": reward,
                "log_prob": log_prob,
            })

            if done:
                break
            features, mask, history = next_state

        return trajectory

    def collect_rollouts(self, kernel_batch, temperature=1.0):
        """Collect G rollouts per kernel. Returns dict[kernel -> list of trajectories]."""
        rollouts = defaultdict(list)
        for m, n, k in kernel_batch:
            env = self.envs[(m, n, k)]
            for _ in range(self.group_size):
                traj = self.generate_rollout(env, temperature=temperature)
                rollouts[(m, n, k)].append(traj)
        return rollouts

    # ---- GRPO advantage ----

    def compute_advantages(self, rollouts):
        """MC-GRPO advantages: median baseline per kernel, global normalization.

        Returns list of dicts with (features, mask, history, action, advantage, old_log_prob).
        """
        all_entries = []

        for kernel, trajectories in rollouts.items():
            # Episode returns
            episode_rewards = [
                sum(step["reward"] for step in traj)
                for traj in trajectories
            ]

            # Median baseline (MC-GRPO)
            baseline = float(np.median(episode_rewards))

            for traj, R in zip(trajectories, episode_rewards):
                advantage = R - baseline
                # Outcome-level: same advantage for all steps in the episode
                for step in traj:
                    all_entries.append({
                        "features": step["features"],
                        "mask": step["mask"],
                        "history": step["history"],
                        "action": step["action"],
                        "advantage": advantage,
                        "old_log_prob": step["log_prob"],
                    })

        # Global z-normalization
        if len(all_entries) > 1:
            advs = np.array([e["advantage"] for e in all_entries])
            mean_a = advs.mean()
            std_a = advs.std() + 1e-8
            for e in all_entries:
                e["advantage"] = (e["advantage"] - mean_a) / std_a

        return all_entries

    # ---- Policy gradient update ----

    def grpo_update(self, advantages):
        """One GRPO gradient step. Returns loss dict."""
        self.policy.train()
        random.shuffle(advantages)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0

        for start in range(0, len(advantages), self.batch_size):
            batch = advantages[start : start + self.batch_size]

            features = torch.stack([e["features"] for e in batch]).to(self.device)
            masks = torch.stack([e["mask"] for e in batch]).to(self.device)
            histories = torch.stack([e["history"] for e in batch]).to(self.device)
            actions = torch.tensor(
                [e["action"] for e in batch], dtype=torch.long
            ).to(self.device)
            advs = torch.tensor(
                [e["advantage"] for e in batch], dtype=torch.float32
            ).to(self.device)
            old_lps = torch.tensor(
                [e["old_log_prob"] for e in batch], dtype=torch.float32
            ).to(self.device)

            # Current log probs
            new_lps = self.policy.log_probs(features, masks, histories, actions)

            # Importance ratio
            ratio = torch.exp(new_lps - old_lps)

            # Clipped surrogate
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty against reference
            with torch.no_grad():
                ref_lps = self.ref_policy.log_probs(
                    features, masks, histories, actions
                )
            kl = (ref_lps - new_lps).mean()
            kl_loss = self.beta * kl

            loss = policy_loss + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "kl_loss": total_kl_loss / denom,
            "n_steps": len(advantages),
        }

    # ---- Evaluation ----

    def evaluate(self):
        """Greedy evaluation on all kernels. Returns results dict."""
        self.policy.eval()
        improvements = []
        per_kernel = {}

        for m, n, k in self.kernels:
            env = self.envs[(m, n, k)]
            state = env.reset()
            features, mask, history = state

            if env.baseline_cycles is None:
                continue

            actions_taken = []
            for _ in range(self.max_steps):
                with torch.no_grad():
                    action_id = self.policy.get_greedy_action(
                        features.to(self.device),
                        mask.to(self.device),
                        history.to(self.device),
                    )

                next_state, reward, done, info = env.step(action_id)
                actions_taken.append(ACTION_NAMES[action_id])

                if done:
                    break
                features, mask, history = next_state

            if env.current_cycles and env.baseline_cycles:
                imp = (env.current_cycles - env.baseline_cycles) / env.baseline_cycles
            else:
                imp = 0.0

            improvements.append(imp)
            per_kernel[f"gemm_tile({m},{n},{k})"] = {
                "improvement": round(imp, 4),
                "actions": actions_taken,
                "baseline_cycles": env.baseline_cycles,
                "final_cycles": env.current_cycles,
            }

        mean_imp = float(np.mean(improvements)) if improvements else 0.0
        return {
            "mean_improvement": mean_imp,
            "n_kernels": len(improvements),
            "per_kernel": per_kernel,
        }

    # ---- Full training ----

    def train(
        self,
        trajectory_path,
        n_bc_epochs=50,
        n_grpo_epochs=450,
        eval_every=50,
        save_dir=None,
    ):
        """Full pipeline: BC warm-start followed by GRPO.

        Returns final evaluation result (or stats if use_hardware=False).
        """
        # Phase 1: BC warm-start
        logger.info("=" * 60)
        logger.info("Phase 1: BC Warm-start (%d epochs)", n_bc_epochs)
        logger.info("=" * 60)
        bc_acc = self.bc_warmstart(trajectory_path, n_epochs=n_bc_epochs)

        if self.use_hardware:
            eval_result = self.evaluate()
            logger.info(
                "Post-BC evaluation: mean improvement = %.1f%%",
                eval_result["mean_improvement"] * 100,
            )
            self.stats["eval_improvement"].append(eval_result["mean_improvement"])

        # Phase 2: GRPO
        logger.info("=" * 60)
        logger.info("Phase 2: GRPO Training (%d epochs)", n_grpo_epochs)
        logger.info("=" * 60)

        for epoch in range(n_grpo_epochs):
            global_epoch = n_bc_epochs + epoch

            # Temperature schedule: exploration then exploitation
            temperature = 1.5 if epoch < 150 else 0.5

            # Sample kernel batch
            if len(self.kernels) <= self.batch_size:
                kernel_batch = list(self.kernels)
            else:
                kernel_batch = random.sample(self.kernels, self.batch_size)

            # Collect rollouts
            rollouts = self.collect_rollouts(kernel_batch, temperature=temperature)

            # Compute advantages
            advantages = self.compute_advantages(rollouts)
            if not advantages:
                logger.warning("Epoch %d: empty advantages, skipping", global_epoch)
                continue

            # Policy gradient update
            update = self.grpo_update(advantages)
            self.stats["grpo_loss"].append(update["loss"])
            self.stats["grpo_policy_loss"].append(update["policy_loss"])
            self.stats["grpo_kl_loss"].append(update["kl_loss"])

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                all_rewards = []
                for trajs in rollouts.values():
                    for traj in trajs:
                        all_rewards.append(sum(s["reward"] for s in traj))
                mean_r = np.mean(all_rewards) if all_rewards else 0

                logger.info(
                    "GRPO %d (global %d): loss=%.4f (pol=%.4f kl=%.4f) "
                    "reward=%.4f T=%.1f steps=%d",
                    epoch + 1, global_epoch + 1,
                    update["loss"], update["policy_loss"], update["kl_loss"],
                    mean_r, temperature, update["n_steps"],
                )

            # Periodic evaluation
            if self.use_hardware and (epoch + 1) % eval_every == 0:
                eval_result = self.evaluate()
                self.stats["eval_improvement"].append(eval_result["mean_improvement"])
                logger.info(
                    "Eval at epoch %d: mean improvement = %.1f%% (%d kernels)",
                    global_epoch + 1,
                    eval_result["mean_improvement"] * 100,
                    eval_result["n_kernels"],
                )
                if save_dir:
                    self._save_checkpoint(save_dir, global_epoch + 1, eval_result)

        # Final evaluation
        if self.use_hardware:
            final = self.evaluate()
            logger.info("=" * 60)
            logger.info(
                "Final: mean improvement = %.1f%%", final["mean_improvement"] * 100
            )
            logger.info("=" * 60)
            if save_dir:
                self._save_checkpoint(save_dir, n_bc_epochs + n_grpo_epochs, final)
            return final

        return dict(self.stats)

    # ---- Checkpointing ----

    def _save_checkpoint(self, save_dir, epoch, eval_result=None):
        import os

        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "policy": self.policy.state_dict(),
            "ref_policy": self.ref_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "stats": dict(self.stats),
        }
        if eval_result:
            ckpt["eval_result"] = eval_result

        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(ckpt, path)

        latest = os.path.join(save_dir, "checkpoint_latest.pt")
        torch.save(ckpt, latest)
        logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "stats" in ckpt:
            self.stats = defaultdict(list, ckpt["stats"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
        return ckpt.get("epoch", 0)
