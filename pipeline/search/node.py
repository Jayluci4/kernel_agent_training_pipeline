"""MCTS tree node for instruction scheduling.

Each node represents a partial schedule: a sequence of instruction IDs
that have been committed so far. Children are the legal next instructions.

The node stores:
  - Visit count N(s)
  - Total value W(s)  (sum of all rollout returns through this node)
  - Mean value Q(s) = W(s) / N(s)
  - Prior probability P(s, a) (uniform for pure MCTS, learned for MuZero)
  - The action (instruction ID) that led to this node from its parent

Value convention: NEGATIVE cycles. Higher (less negative) = better schedule.
The MCTS maximizes Q(s).
"""

import math
from typing import Dict, List, Optional


class MCTSNode:
    """A node in the MCTS search tree.

    Represents the state after scheduling a specific instruction (or
    the root state before any scheduling decisions).

    Children are lazily expanded: created when first visited during
    expansion phase, not at construction time.
    """

    __slots__ = [
        'action', 'parent', 'children',
        'visit_count', 'total_value', 'prior',
        '_is_terminal', '_legal_actions',
    ]

    def __init__(
        self,
        action: Optional[int] = None,
        parent: Optional['MCTSNode'] = None,
        prior: float = 1.0,
    ):
        """
        Args:
            action: The instruction ID that was scheduled to reach this state.
                    None for the root node.
            parent: Parent node in the tree.
            prior: Prior probability P(parent_state, action).
                   Uniform (1.0) for pure MCTS, learned for MuZero.
        """
        self.action = action
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}  # action -> child node
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior = prior
        self._is_terminal: Optional[bool] = None
        self._legal_actions: Optional[List[int]] = None

    @property
    def q_value(self) -> float:
        """Mean value Q(s) = W(s) / N(s). Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """True if this node has been expanded (children created)."""
        return len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        """True if this node represents a complete schedule."""
        if self._is_terminal is None:
            return False
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value: bool):
        self._is_terminal = value

    def expand(self, legal_actions: List[int], priors: Optional[Dict[int, float]] = None):
        """Create child nodes for each legal action.

        Args:
            legal_actions: Instruction IDs that can be scheduled next.
            priors: Optional prior probabilities per action.
                    If None, uses uniform priors (1/|actions|).
        """
        self._legal_actions = legal_actions
        if not legal_actions:
            self._is_terminal = True
            return

        n_actions = len(legal_actions)
        for action in legal_actions:
            if priors is not None and action in priors:
                p = priors[action]
            else:
                p = 1.0 / n_actions
            self.children[action] = MCTSNode(
                action=action,
                parent=self,
                prior=p,
            )

    def select_child(self, c_puct: float = 1.41) -> 'MCTSNode':
        """Select the child with the highest UCB score.

        UCB1 formula:
            UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(child))

        This is the PUCT variant used by AlphaZero/MuZero:
        - Q(s, a): exploitation (mean value of this child)
        - P(s, a): prior (uniform for pure MCTS)
        - sqrt(N(parent)) / (1 + N(child)): exploration bonus

        Args:
            c_puct: Exploration constant. Higher = more exploration.
                    Standard MuZero uses ~1.25-2.5.
                    For initial learning, use 2.0-4.0 (aggressive exploration).

        Returns:
            The child node with highest UCB score.
        """
        if not self.children:
            raise ValueError("Cannot select child of unexpanded node")

        parent_visits_sqrt = math.sqrt(self.visit_count)
        best_score = float('-inf')
        best_child = None

        for child in self.children.values():
            # Exploitation: mean value
            exploitation = child.q_value

            # Exploration: prior * sqrt(parent_visits) / (1 + child_visits)
            exploration = c_puct * child.prior * parent_visits_sqrt / (1 + child.visit_count)

            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, value: float):
        """Propagate a rollout value up from this node to the root.

        Args:
            value: The value to propagate (negative cycles for a complete schedule).
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def best_action(self, temperature: float = 0.0) -> int:
        """Select the best action based on visit counts.

        Args:
            temperature: Controls stochasticity.
                0.0 = deterministic (most-visited child)
                1.0 = proportional to visit counts
                >1.0 = more uniform / exploratory

        Returns:
            The action (instruction ID) to take.
        """
        if not self.children:
            raise ValueError("No children to select from")

        if temperature == 0.0:
            # Deterministic: pick most-visited, break ties by Q-value.
            # Q-value tiebreaker is critical: when sims < branching factor,
            # many children have equal visit counts (1 each). Without Q-value
            # tiebreaking, max() returns the lowest action key, degenerating
            # to program-order scheduling.
            return max(
                self.children.keys(),
                key=lambda a: (self.children[a].visit_count, self.children[a].q_value)
            )

        # Stochastic: sample proportional to N(s,a)^(1/tau)
        import numpy as np

        actions = list(self.children.keys())
        visits = np.array([self.children[a].visit_count for a in actions], dtype=np.float64)

        if temperature == 1.0:
            probs = visits / visits.sum()
        else:
            # Apply temperature
            log_visits = np.log(visits + 1e-8)
            log_visits = log_visits / temperature
            log_visits -= log_visits.max()  # Numerical stability
            probs = np.exp(log_visits)
            probs = probs / probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def get_visit_distribution(self) -> Dict[int, float]:
        """Return normalized visit counts as a probability distribution.

        Used as the MCTS policy target for training neural network priors.
        """
        if not self.children:
            return {}
        total = sum(c.visit_count for c in self.children.values())
        if total == 0:
            n = len(self.children)
            return {a: 1.0 / n for a in self.children}
        return {a: c.visit_count / total for a, c in self.children.items()}

    def get_action_values(self) -> Dict[int, float]:
        """Return Q-values for each child action."""
        return {a: c.q_value for a, c in self.children.items()}

    @property
    def depth(self) -> int:
        """Depth of this node in the tree (root = 0)."""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def __repr__(self):
        return (
            f"MCTSNode(action={self.action}, "
            f"N={self.visit_count}, "
            f"Q={self.q_value:.1f}, "
            f"children={len(self.children)})"
        )
