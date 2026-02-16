"""MLP policy for PTX transform selection.

Input: 25 kernel features + 21-dim action mask + 21-dim action history = 67 dims
Output: 21 logits over actions (masked before softmax)

The action history encodes which transforms have been applied so far,
enabling the policy to reason about non-monotone interactions
(e.g., reorder helps AFTER vectorize but not before).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

N_FEATURES = 25
N_ACTIONS = 21

ACTION_NAMES = [
    "vec_ld", "vec_st",
    "cache_cs", "cache_cg", "cache_ca", "cache_cv",
    "st_cache_cs", "st_cache_wt", "st_cache_wb",
    "maxnreg_32", "maxnreg_64", "maxnreg_128", "maxnreg_255",
    "reorder_cp", "reorder_il", "reorder_lf", "reorder_sl",
    "prefetch_L1", "prefetch_L2",
    "split_ld",
    "stop",
]

ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}

# Conflict groups: only one from each group can be active
CONFLICT_GROUPS = {
    "cache_hints": {"cache_cs", "cache_cg", "cache_ca", "cache_cv"},
    "store_cache_hints": {"st_cache_cs", "st_cache_wt", "st_cache_wb"},
    "register_budget": {"maxnreg_32", "maxnreg_64", "maxnreg_128", "maxnreg_255"},
    "prefetch": {"prefetch_L1", "prefetch_L2"},
    "reorder": {"reorder_cp", "reorder_il", "reorder_lf", "reorder_sl"},
}


def get_action_mask(applied_set):
    """Return binary mask of available actions given applied transforms.

    applied_set: set of action label strings already applied.
    Returns: list of 21 ints (0 or 1).
    """
    mask = []
    for label in ACTION_NAMES:
        if label == "stop":
            mask.append(1)
            continue
        if label in applied_set:
            mask.append(0)
            continue
        conflict = False
        for group_labels in CONFLICT_GROUPS.values():
            if label in group_labels and applied_set & group_labels:
                conflict = True
                break
        mask.append(0 if conflict else 1)
    return mask


def get_action_history(applied_set):
    """Return binary vector encoding which transforms have been applied.

    applied_set: set of action label strings already applied.
    Returns: list of 21 ints (0 or 1).
    """
    return [1 if name in applied_set else 0 for name in ACTION_NAMES]


class TransformPolicy(nn.Module):
    """MLP policy for transform selection."""

    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        input_dim = N_FEATURES + N_ACTIONS + N_ACTIONS  # 67
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_ACTIONS),
        )
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, action_mask, action_history):
        """Compute masked logits.

        features: [B, 25]
        action_mask: [B, 21] binary (1=available)
        action_history: [B, 21] binary (1=previously applied)
        Returns: logits [B, 21] with -inf for unavailable actions
        """
        x = torch.cat([features, action_mask, action_history], dim=-1)
        logits = self.net(x)
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        return logits

    def get_distribution(self, features, action_mask, action_history, temperature=1.0):
        """Return Categorical distribution over actions."""
        logits = self.forward(features, action_mask, action_history)
        # Clamp finite logits to prevent softmax overflow
        # (-inf from masking is preserved intentionally)
        finite_mask = logits.isfinite()
        logits = torch.where(finite_mask, logits.clamp(-50, 50), logits)
        probs = F.softmax(logits / temperature, dim=-1)
        # Fallback: if softmax produced NaN (all -inf or numerical issue),
        # use uniform over valid actions
        if probs.isnan().any():
            valid = action_mask > 0
            probs = valid.float() / valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        return Categorical(probs)

    @torch.no_grad()
    def get_action(self, features, action_mask, action_history, temperature=1.0):
        """Sample a single action. Returns (action_id, log_prob)."""
        dist = self.get_distribution(
            features.unsqueeze(0), action_mask.unsqueeze(0),
            action_history.unsqueeze(0), temperature,
        )
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    @torch.no_grad()
    def get_greedy_action(self, features, action_mask, action_history):
        """Return the highest-probability action (no sampling)."""
        logits = self.forward(
            features.unsqueeze(0), action_mask.unsqueeze(0),
            action_history.unsqueeze(0),
        )
        return logits.argmax(dim=-1).item()

    def log_probs(self, features, action_mask, action_history, actions):
        """Compute log probabilities of given actions.

        features: [B, 25]
        action_mask: [B, 21]
        action_history: [B, 21]
        actions: [B] long tensor
        Returns: [B] log probabilities
        """
        logits = self.forward(features, action_mask, action_history)
        # Clamp finite logits to prevent overflow
        finite_mask = logits.isfinite()
        logits = torch.where(finite_mask, logits.clamp(-50, 50), logits)
        log_probs = F.log_softmax(logits, dim=-1)
        result = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        # Clamp result to avoid -inf for numerical stability
        return result.clamp(min=-50)
