"""Phase 3: Learned world model for PTX instruction scheduling.

Architecture (MuZero-inspired, adapted for combinatorial optimization):

Representation: h(obs) -> {node_embeddings, global_context}
  GNN encoder over the instruction DAG. Produces per-node embeddings
  (NOT a single pooled vector) to preserve which-instruction-causes-what.

Dynamics: g(s, action) -> s'
  Updates node embeddings when an instruction is scheduled. The scheduled
  node broadcasts its effect to neighbors via cross-attention.

Prediction: f(s) -> (policy, value, reward)
  Policy: per-node score (node embedding + global context) masked to legal actions.
  Value: attention-weighted pool -> scalar (total remaining cost estimate).
  Reward: predict dense step reward (stall cycles from pipeline model).

Key design decisions from failure mode analysis:
  1. No global pooling in latent state (keeps node-level detail)
  2. Dense reward from pipeline model + terminal hardware correction
  3. Behavior cloning on real hardware data before MCTS self-play
"""

from .representation import RepresentationNetwork
from .dynamics import DynamicsNetwork
from .prediction import PredictionNetwork
from .muzero_net import ChronosNet
