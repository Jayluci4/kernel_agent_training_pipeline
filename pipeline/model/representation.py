"""Representation network: h(observation) -> latent state.

Architecture: Query-Key split (no global pooling).

The latent state has TWO components:
  H (Keys): Instruction embeddings [N, hidden_dim]
    - Encode WHAT each instruction IS (opcode, pipeline, latency, deps)
    - Computed via GNN over the DAG structure
    - Preserve per-instruction identity (no compression)

  Q (Query): Pipeline context [hidden_dim]
    - Encode WHAT the hardware NEEDS (pipeline busy counters, register pressure)
    - Computed via MLP from pipeline_state + register_pressure
    - Changes at each scheduling step

The interaction between H and Q happens in the prediction heads:
  Policy: Q · H^T (dot-product selects matching instructions)
  Value: cross-attention(Q, H) (weighted combination of instruction info)

This replaces attention pooling. No information loss.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from dataclasses import dataclass
from typing import Optional


@dataclass
class LatentState:
    """Per-node latent representation of the scheduling state.

    Split into Keys (instruction identity) and Query (hardware needs).
    No global pooling — per-instruction detail is preserved.
    """
    node_embeddings: torch.Tensor   # [N, hidden_dim] Keys
    pipeline_context: torch.Tensor  # [hidden_dim] Query
    ready_mask: torch.Tensor        # [N] bool
    scheduled_mask: torch.Tensor    # [N] bool
    edge_index: torch.Tensor        # [2, E]
    pipeline_state: torch.Tensor    # [7]
    register_pressure: torch.Tensor # [2]

    def detach(self) -> 'LatentState':
        return LatentState(
            node_embeddings=self.node_embeddings.detach(),
            pipeline_context=self.pipeline_context.detach(),
            ready_mask=self.ready_mask.detach(),
            scheduled_mask=self.scheduled_mask.detach(),
            edge_index=self.edge_index.detach(),
            pipeline_state=self.pipeline_state.detach(),
            register_pressure=self.register_pressure.detach(),
        )


class RepresentationNetwork(nn.Module):
    """GNN encoder with Query-Key split.

    H = GNN(node_features, edge_index) — instruction embeddings (Keys)
    Q = MLP(pipeline_state || register_pressure) — hardware context (Query)

    H and Q are kept SEPARATE (no pipeline injection into H).
    Their interaction is deferred to the prediction heads.
    """

    def __init__(
        self,
        node_feat_dim: int = 48,
        edge_feat_dim: int = 3,
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        global_feat_dim: int = 9,  # pipeline_state(7) + register_pressure(2)
        dropout: float = 0.0,
        use_edge_features: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features

        # H: Instruction identity encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GAT layers with residual connections
        gat_edge_dim = edge_feat_dim if use_edge_features else None
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim, hidden_dim // num_heads,
                    heads=num_heads, concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=gat_edge_dim,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        # Q: Pipeline context encoder (hardware needs)
        self.pipeline_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,    # [N, node_feat_dim]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, edge_feat_dim]
        pipeline_state: torch.Tensor,   # [7]
        register_pressure: torch.Tensor,# [2]
        ready_mask: torch.Tensor,       # [N]
        scheduled_mask: torch.Tensor,   # [N]
    ) -> LatentState:
        """Encode observation into Keys (H) and Query (Q)."""

        # H: Instruction embeddings via GNN
        h = self.node_encoder(node_features)  # [N, hidden_dim]
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_res = h
            if self.use_edge_features:
                h = gat(h, edge_index, edge_attr=edge_attr)
            else:
                h = gat(h, edge_index)  # [N, hidden_dim]
            h = norm(h + h_res)
            h = torch.relu(h)

        # Q: Pipeline context from hardware state
        pipeline_feats = torch.cat([pipeline_state, register_pressure], dim=-1)
        q = self.pipeline_encoder(pipeline_feats)  # [hidden_dim]

        return LatentState(
            node_embeddings=h,
            pipeline_context=q,
            ready_mask=ready_mask,
            scheduled_mask=scheduled_mask,
            edge_index=edge_index,
            pipeline_state=pipeline_state,
            register_pressure=register_pressure,
        )
