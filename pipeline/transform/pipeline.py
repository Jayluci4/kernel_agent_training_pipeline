"""Transform pipeline: sequential composition of transforms.

Applies a sequence of (transform, params) pairs to a kernel.
Accumulates stats from each step.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .parsed_kernel import ParsedKernel
from .base import PtxTransform, TransformResult


@dataclass
class PipelineResult:
    """Result of running a multi-step transform pipeline."""
    kernel: ParsedKernel
    steps: List[TransformResult] = field(default_factory=list)
    total_changed: bool = False


class TransformPipeline:
    """Sequential composition of PTX transforms."""

    def __init__(self, transforms: List[Tuple[PtxTransform, Dict]] = None):
        self.transforms: List[Tuple[PtxTransform, Dict]] = transforms or []

    def add(self, transform: PtxTransform, params: Dict = None) -> 'TransformPipeline':
        """Add a transform step. Returns self for chaining."""
        self.transforms.append((transform, params or {}))
        return self

    def run(self, kernel: ParsedKernel) -> PipelineResult:
        """Apply all transforms sequentially."""
        current = kernel
        steps = []
        any_changed = False

        for transform, params in self.transforms:
            result = transform.apply(current, params)
            steps.append(result)
            current = result.kernel
            if result.changed:
                any_changed = True

        return PipelineResult(
            kernel=current,
            steps=steps,
            total_changed=any_changed,
        )
