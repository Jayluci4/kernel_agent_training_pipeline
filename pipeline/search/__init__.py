"""Project Chronos Phase 2: MCTS Search Engine.

Pure Monte Carlo Tree Search with UCB1 for instruction scheduling.
No neural network priors — uses random rollouts for value estimation
and the pipeline model for leaf evaluation.

The MCTS discovers scheduling patterns from first principles.
No hardcoded heuristics. The search finds what the hardware rewards.
"""

from .node import MCTSNode
from .mcts import MCTS, MCTSConfig
