"""Prediction network: f(latent_state) -> (policy, value).

Architecture: Cross-Attention (Pointer Network style).

Policy head: Dot-product matching.
  logits = (H @ Q) / sqrt(D)
  The pipeline Query selects instructions whose properties match the
  current hardware needs. No MLP per node, no pooling.

Value head: Multi-head cross-attention.
  Multiple learned queries attend to H from different "perspectives."
  Concatenated contexts feed an MLP to predict the scalar value.
  This preserves per-instruction detail (unlike single-vector pooling).

Both heads receive the LatentState (H = Keys, Q = Query).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .representation import LatentState


class PredictionNetwork(nn.Module):
    """Cross-attention policy + value heads for MCTS guidance.

    f(s) -> (pi, v)
      pi: [N] logits from Q · H^T, masked to legal actions
      v: scalar from multi-head cross-attention over H
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_value_heads: int = 4,
        value_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        # Policy: project Q into "policy query" space
        # logits = H @ policy_query / sqrt(D)
        self.policy_query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Value: multi-head cross-attention
        # Each head projects Q into a different query, attends to H
        self.num_value_heads = num_value_heads
        head_dim = hidden_dim // num_value_heads
        self.value_query_proj = nn.Linear(hidden_dim, hidden_dim)  # Q -> K heads
        self.value_key_proj = nn.Linear(hidden_dim, hidden_dim)    # H -> K heads
        self.value_val_proj = nn.Linear(hidden_dim, hidden_dim)    # H -> K heads
        self.head_dim = head_dim

        # Value MLP: from concatenated cross-attention output + Q
        vlayers = []
        in_dim = hidden_dim * 2  # cross-attn output + Q
        for _ in range(value_layers - 1):
            vlayers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        vlayers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*vlayers)

    def forward(self, state: LatentState,
                action_mask: Optional[torch.Tensor] = None) -> tuple:
        """Compute policy logits and value.

        Args:
            state: LatentState with H (node_embeddings) and Q (pipeline_context).
            action_mask: Optional [N] bool mask. If None, uses ready & ~scheduled.

        Returns:
            (policy_logits: [N], value: scalar Tensor)
        """
        H = state.node_embeddings     # [N, D]
        Q = state.pipeline_context    # [D]

        # --- Policy: dot-product matching ---
        q_policy = self.policy_query_proj(Q)  # [D]
        policy_logits = (H @ q_policy) * self.scale  # [N]

        # Mask non-legal actions
        if action_mask is not None:
            mask = action_mask
        else:
            mask = state.ready_mask & ~state.scheduled_mask
        policy_logits = policy_logits.masked_fill(~mask, float('-inf'))

        # --- Value: multi-head cross-attention ---
        N = H.shape[0]
        K = self.num_value_heads
        d_k = self.head_dim

        # Project Q and H into multi-head space
        q_val = self.value_query_proj(Q).view(K, d_k)      # [K, d_k]
        k_val = self.value_key_proj(H).view(N, K, d_k)     # [N, K, d_k]
        v_val = self.value_val_proj(H).view(N, K, d_k)     # [N, K, d_k]

        # Per-head attention: [K, 1] x [N, K, d_k] -> [K, N]
        attn_scores = torch.einsum('kd,nkd->kn', q_val, k_val) * (d_k ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [K, N]

        # Per-head context: weighted sum of values
        context = torch.einsum('kn,nkd->kd', attn_weights, v_val)  # [K, d_k]
        context = context.reshape(-1)  # [hidden_dim]

        # Value from context + Q
        value_input = torch.cat([context, Q], dim=-1)  # [2*hidden_dim]
        value = self.value_net(value_input).squeeze(-1)

        return policy_logits, value

    def get_policy_probs(self, state: LatentState) -> torch.Tensor:
        """Return action probabilities (softmax over legal actions)."""
        logits, _ = self.forward(state)
        return F.softmax(logits, dim=-1)
