"""Phase 3 training infrastructure for the learned world model.

Training pipeline:
  1. Data collection: MCTS self-play generates (obs, action, policy, value, reward) tuples
  2. Replay buffer: stores trajectories for experience replay
  3. Trainer: MuZero-style training loop with K-step unrolling
  4. Behavior cloning: pretraining on real hardware data (sim-to-real anchor)

Key design decisions:
  - Dense rewards from pipeline model at every step (not just terminal)
  - Terminal hardware correction applied as discounted correction factor
  - BC pretraining on 10K random schedules measured on real L4
"""

from .replay_buffer import ReplayBuffer, Trajectory
from .data_collector import DataCollector
from .trainer import Trainer
