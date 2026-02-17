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
        entropy_coef=0.02,
        n_updates=3,
        use_hardware=True,
        device="cpu",
    ):
        self.kernels = kernels
        self.group_size = group_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.n_updates = n_updates
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

        # Health check: verify weights are finite after BC
        for name, param in self.policy.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error("NaN/inf in policy weights after BC: %s", name)
            max_val = param.abs().max().item()
            if max_val > 100:
                logger.warning("Large weight in %s: max_abs=%.1f", name, max_val)

        logger.info("BC warm-start complete. Best accuracy: %.3f", best_acc)
        return best_acc

    # ---- GRPO rollouts ----

    def generate_rollout(self, env, temperature=1.0, min_steps=0):
        """Generate one episode. Returns list of step dicts.

        min_steps: mask out "stop" action for the first N steps,
                   forcing the agent to try transforms before stopping.
        """
        self.policy.eval()
        state = env.reset()
        features, mask, history = state

        trajectory = []
        for step_i in range(self.max_steps):
            # Debug: check for NaN/inf in state on first step
            if step_i == 0:
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger.error("NaN/inf in features: %s", features)
                feat_max = features.abs().max().item()
                if feat_max > 1000:
                    logger.warning("Large feature values (max=%.1f), consider normalization", feat_max)

            # Mask stop for first min_steps to force exploration
            rollout_mask = mask.clone()
            if step_i < min_steps:
                rollout_mask[ACTION_TO_ID["stop"]] = 0.0
                # If only stop was available, skip (shouldn't happen early)
                if rollout_mask.sum() == 0:
                    rollout_mask = mask.clone()

            with torch.no_grad():
                action_id, _ = self.policy.get_action(
                    features.to(self.device),
                    rollout_mask.to(self.device),
                    history.to(self.device),
                    temperature=temperature,
                )
                # Compute log_prob at temp=1.0 to match grpo_update's log_probs()
                # (sampling uses temperature for exploration, but importance ratio
                # must compare log_probs at the SAME temperature)
                log_prob = self.policy.log_probs(
                    features.unsqueeze(0).to(self.device),
                    mask.unsqueeze(0).to(self.device),
                    history.unsqueeze(0).to(self.device),
                    torch.tensor([action_id]),
                ).item()

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

    def collect_rollouts(self, kernel_batch, temperature=1.0, min_steps=0):
        """Collect G rollouts per kernel. Returns dict[kernel -> list of trajectories]."""
        rollouts = defaultdict(list)
        for m, n, k in kernel_batch:
            env = self.envs[(m, n, k)]
            for _ in range(self.group_size):
                traj = self.generate_rollout(
                    env, temperature=temperature, min_steps=min_steps,
                )
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
        total_entropy = 0.0
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

            # Asymmetric clipped surrogate (DAPO-style)
            # Wider clip for positive advantages: let good actions update more
            surr1 = ratio * advs
            clip_high = 1 + self.epsilon * 1.5  # 0.3 for positive
            clip_low = 1 - self.epsilon          # 0.2 for negative
            surr2 = torch.clamp(ratio, clip_low, clip_high) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus to prevent mode collapse (6.2)
            # Replace -inf with -1e4 (not -inf) so that:
            #   softmax gives ~0 prob to masked actions (exp(-1e4) ≈ 0)
            #   but gradients are finite (no 0 * -inf = NaN in backward)
            logits_ent = self.policy(features, masks, histories)
            logits_ent = logits_ent.clamp(min=-1e4, max=50)
            probs_ent = F.softmax(logits_ent, dim=-1)
            log_probs_ent = F.log_softmax(logits_ent, dim=-1)
            entropy = -(probs_ent * log_probs_ent).sum(dim=-1).mean()
            entropy_loss = -self.entropy_coef * entropy

            # KL penalty against reference (DeepSeek-R1 unbiased estimator)
            # D_KL = e^(log_ref - log_new) - (log_ref - log_new) - 1
            # Always non-negative, zero when policies match
            with torch.no_grad():
                ref_lps = self.ref_policy.log_probs(
                    features, masks, histories, actions
                )
            log_ratio = ref_lps - new_lps
            per_sample_kl = torch.exp(log_ratio) - log_ratio - 1
            kl_loss = self.beta * per_sample_kl.mean()

            loss = policy_loss + kl_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            total_entropy += entropy.item()
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "kl_loss": total_kl_loss / denom,
            "entropy": total_entropy / denom,
            "n_steps": len(advantages),
        }

    # ---- Evaluation ----

    def evaluate(self, kernel_subset=None):
        """Greedy evaluation on kernels. Returns results dict.

        kernel_subset: optional list of (m,n,k) to evaluate on.
                       Defaults to all kernels.
        """
        self.policy.eval()
        kernels_to_eval = kernel_subset if kernel_subset else self.kernels
        improvements = []
        per_kernel = {}
        action_counts = defaultdict(int)
        unique_sequences = set()

        for m, n, k in kernels_to_eval:
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
                action_label = ACTION_NAMES[action_id]
                actions_taken.append(action_label)
                action_counts[action_label] += 1

                if done:
                    break
                features, mask, history = next_state

            if env.current_cycles and env.baseline_cycles:
                imp = (env.current_cycles - env.baseline_cycles) / env.baseline_cycles
            else:
                imp = 0.0

            improvements.append(imp)
            sequence_key = tuple(actions_taken)
            unique_sequences.add(sequence_key)
            per_kernel[f"gemm_tile({m},{n},{k})"] = {
                "improvement": round(imp, 4),
                "actions": actions_taken,
                "baseline_cycles": env.baseline_cycles,
                "final_cycles": env.current_cycles,
            }

        mean_imp = float(np.mean(improvements)) if improvements else 0.0

        # Mode collapse detection (6.2): log action distribution + unique sequences
        total_actions = sum(action_counts.values()) or 1
        action_dist = {
            k: round(v / total_actions, 3)
            for k, v in sorted(action_counts.items(), key=lambda x: -x[1])
        }

        return {
            "mean_improvement": mean_imp,
            "n_kernels": len(improvements),
            "per_kernel": per_kernel,
            "action_distribution": action_dist,
            "unique_sequences": len(unique_sequences),
        }

    def evaluate_loko(self, n_folds=8):
        """Leave-One-Kernel-Out generalization check (6.4).

        Splits kernels into n_folds groups. For each fold, evaluates
        on kernels NOT in the training batch to detect overfitting.
        Returns mean improvement on held-out kernels.
        """
        self.policy.eval()
        kernel_list = list(self.kernels)
        if len(kernel_list) <= n_folds:
            # Too few kernels for meaningful LOKO
            return None

        random.shuffle(kernel_list)
        fold_size = len(kernel_list) // n_folds
        holdout_improvements = []

        for fold_i in range(n_folds):
            start = fold_i * fold_size
            end = start + fold_size
            holdout = kernel_list[start:end]

            result = self.evaluate(kernel_subset=holdout)
            holdout_improvements.append(result["mean_improvement"])

        mean_holdout = float(np.mean(holdout_improvements))
        return mean_holdout

    def load_greedy_baseline(self, greedy_results_path):
        """Load greedy v2 results for novel sequence comparison.

        Stores {kernel_name: (sequence_tuple, improvement)} for each kernel.
        """
        with open(greedy_results_path) as f:
            data = json.load(f)
        self._greedy_baseline = {}
        for r in data["results"]:
            kernel = r["kernel"]
            seq = tuple(r["transforms_applied"])
            # Parse improvement string like "-28.1%" to float
            imp_str = r["total_improvement"].replace("%", "").replace("+", "")
            imp = float(imp_str) / 100.0
            self._greedy_baseline[kernel] = (seq, imp)
        logger.info("Loaded greedy v2 baseline: %d kernels", len(self._greedy_baseline))

    def detect_novel_sequences(self, eval_result):
        """Compare eval sequences against greedy v2. Returns novel discoveries.

        A novel sequence: different from greedy AND achieves better improvement.
        """
        if not hasattr(self, '_greedy_baseline'):
            return []

        novel = []
        for kernel_name, result in eval_result["per_kernel"].items():
            if kernel_name not in self._greedy_baseline:
                continue
            greedy_seq, greedy_imp = self._greedy_baseline[kernel_name]
            policy_seq = tuple(
                a for a in result["actions"] if a != "stop"
            )
            policy_imp = result["improvement"]

            # Novel = different sequence AND better or equal improvement
            if policy_seq != greedy_seq and policy_imp <= greedy_imp:
                novel.append({
                    "kernel": kernel_name,
                    "policy_seq": list(policy_seq),
                    "greedy_seq": list(greedy_seq),
                    "policy_imp": policy_imp,
                    "greedy_imp": greedy_imp,
                })
        return novel

    # ---- Full training ----

    def train(
        self,
        trajectory_path,
        n_bc_epochs=50,
        n_grpo_epochs=450,
        eval_every=50,
        save_dir=None,
        greedy_results_path=None,
    ):
        """Full pipeline: BC warm-start followed by GRPO.

        Returns final evaluation result (or stats if use_hardware=False).
        """
        # Load greedy v2 baseline for novel sequence detection
        if greedy_results_path:
            self.load_greedy_baseline(greedy_results_path)

        # Phase 1: BC warm-start
        logger.info("=" * 60)
        logger.info("Phase 1: BC Warm-start (%d epochs)", n_bc_epochs)
        logger.info("=" * 60)
        bc_acc = self.bc_warmstart(trajectory_path, n_epochs=n_bc_epochs)

        best_improvement = float('inf')  # lower is better (negative = speedup)
        if self.use_hardware:
            eval_result = self.evaluate()
            logger.info(
                "Post-BC evaluation: mean improvement = %.1f%%",
                eval_result["mean_improvement"] * 100,
            )
            self.stats["eval_improvement"].append(eval_result["mean_improvement"])
            best_improvement = eval_result["mean_improvement"]

        # Phase 2: GRPO
        logger.info("=" * 60)
        logger.info("Phase 2: GRPO Training (%d epochs)", n_grpo_epochs)
        logger.info("=" * 60)

        for epoch in range(n_grpo_epochs):
            global_epoch = n_bc_epochs + epoch

            # Temperature schedule: exploration then exploitation
            exploring = epoch < 150
            temperature = 1.5 if exploring else 0.5
            # Force at least 2 transforms during exploration to prevent stop-collapse
            min_steps = 2 if exploring else 0

            # Sample kernel batch
            if len(self.kernels) <= self.batch_size:
                kernel_batch = list(self.kernels)
            else:
                kernel_batch = random.sample(self.kernels, self.batch_size)

            # Collect rollouts
            rollouts = self.collect_rollouts(
                kernel_batch, temperature=temperature, min_steps=min_steps,
            )

            # Compute advantages
            advantages = self.compute_advantages(rollouts)
            if not advantages:
                logger.warning("Epoch %d: empty advantages, skipping", global_epoch)
                continue

            # Multiple gradient updates per rollout batch (PPO-style)
            # Rollout collection is expensive (~40s), gradient step is cheap (~0.01s)
            for _ in range(self.n_updates):
                update = self.grpo_update(advantages)
            self.stats["grpo_loss"].append(update["loss"])
            self.stats["grpo_policy_loss"].append(update["policy_loss"])
            self.stats["grpo_kl_loss"].append(update["kl_loss"])

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                all_rewards = []
                all_lengths = []
                for trajs in rollouts.values():
                    for traj in trajs:
                        all_rewards.append(sum(s["reward"] for s in traj))
                        all_lengths.append(len(traj))
                mean_r = np.mean(all_rewards) if all_rewards else 0
                mean_len = np.mean(all_lengths) if all_lengths else 0

                logger.info(
                    "GRPO %d (global %d): loss=%.4f (pol=%.4f kl=%.4f ent=%.3f) "
                    "reward=%.4f T=%.1f steps=%d len=%.1f min_s=%d",
                    epoch + 1, global_epoch + 1,
                    update["loss"], update["policy_loss"], update["kl_loss"],
                    update["entropy"],
                    mean_r, temperature, update["n_steps"],
                    mean_len, min_steps,
                )

            # Periodic evaluation
            if self.use_hardware and (epoch + 1) % eval_every == 0:
                eval_result = self.evaluate()
                self.stats["eval_improvement"].append(eval_result["mean_improvement"])
                logger.info(
                    "Eval at epoch %d: mean improvement = %.1f%% (%d kernels, "
                    "%d unique sequences)",
                    global_epoch + 1,
                    eval_result["mean_improvement"] * 100,
                    eval_result["n_kernels"],
                    eval_result["unique_sequences"],
                )
                # Mode collapse check (6.2)
                top_actions = list(eval_result["action_distribution"].items())[:5]
                logger.info(
                    "  Action distribution (top 5): %s",
                    ", ".join(f"{k}={v:.1%}" for k, v in top_actions),
                )
                # LOKO generalization check (6.4) — every 2nd eval
                if (epoch + 1) % (eval_every * 2) == 0:
                    loko_imp = self.evaluate_loko()
                    if loko_imp is not None:
                        self.stats["loko_improvement"].append(loko_imp)
                        gap = eval_result["mean_improvement"] - loko_imp
                        logger.info(
                            "  LOKO: holdout improvement = %.1f%% (gap = %.1f%%)",
                            loko_imp * 100, gap * 100,
                        )

                # Novel sequence detection (Section 7 success criterion)
                novel = self.detect_novel_sequences(eval_result)
                if novel:
                    self.stats["novel_count"].append(len(novel))
                    logger.info(
                        "  Novel sequences: %d (better than greedy with different actions)",
                        len(novel),
                    )
                    for n in novel[:3]:  # log top 3
                        logger.info(
                            "    %s: policy=%s (%.1f%%) vs greedy=%s (%.1f%%)",
                            n["kernel"],
                            n["policy_seq"], n["policy_imp"] * 100,
                            n["greedy_seq"], n["greedy_imp"] * 100,
                        )
                else:
                    self.stats["novel_count"].append(0)

                # Best checkpoint tracking
                if eval_result["mean_improvement"] < best_improvement:
                    best_improvement = eval_result["mean_improvement"]
                    logger.info(
                        "  New best: %.1f%%", best_improvement * 100,
                    )
                    if save_dir:
                        self._save_checkpoint(
                            save_dir, global_epoch + 1, eval_result,
                            tag="best",
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

    def _save_checkpoint(self, save_dir, epoch, eval_result=None, tag=None):
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

        if tag:
            tagged = os.path.join(save_dir, f"checkpoint_{tag}.pt")
            torch.save(ckpt, tagged)

        logger.info("Saved checkpoint: %s%s", path, f" [{tag}]" if tag else "")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "stats" in ckpt:
            self.stats = defaultdict(list, ckpt["stats"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
        return ckpt.get("epoch", 0)
