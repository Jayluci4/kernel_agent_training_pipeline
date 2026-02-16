"""Base class for PTX transforms.

All transforms are pure: they return a new ParsedKernel and never mutate
the input. Each transform defines:
  - applicable(): what parameter choices exist for this kernel
  - apply(): apply with specific parameters
  - apply_all(): apply with default/maximal settings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from .parsed_kernel import ParsedKernel


@dataclass
class TransformResult:
    """Result of applying a single transform."""
    kernel: ParsedKernel
    changed: bool
    stats: Dict = field(default_factory=dict)


class PtxTransform(ABC):
    """Abstract base class for PTX program transforms."""

    name: str = "base"

    @abstractmethod
    def applicable(self, kernel: ParsedKernel) -> List[Dict]:
        """Return list of parameter dicts for valid applications.

        Each dict can be passed to apply() as params.
        Empty list means the transform cannot apply to this kernel.
        """

    @abstractmethod
    def apply(self, kernel: ParsedKernel, params: Dict) -> TransformResult:
        """Apply transform with specific parameters.

        Must be pure: returns new ParsedKernel, never mutates input.
        """

    def apply_all(self, kernel: ParsedKernel) -> TransformResult:
        """Apply transform with default/maximal settings.

        Default: applies first option from applicable().
        Override for transforms where "apply all" has specific meaning.
        """
        options = self.applicable(kernel)
        if not options:
            return TransformResult(kernel=kernel, changed=False)
        return self.apply(kernel, options[0])
