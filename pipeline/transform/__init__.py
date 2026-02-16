"""PTX Transform Library.

Provides program-level transforms for PTX kernels:
  - RegisterBudgetTransform: control register allocation via .maxnreg
  - CacheHintTransform: add cache policies to global loads
  - StoreCacheHintTransform: add cache policies to global stores
  - ReorderTransform: dependency-safe instruction reordering (5 strategies)
  - VectorizeLoadsTransform: merge scalar loads into vector loads
  - VectorizeStoresTransform: merge scalar stores into vector stores
  - PrefetchTransform: insert prefetch instructions for global loads
  - SplitVectorLoadsTransform: expand vector loads back to scalar

All transforms are pure (return new ParsedKernel, never mutate input).
"""

from .parsed_kernel import ParsedKernel, BodyLine, parse_kernel, emit, get_instructions
from .base import PtxTransform, TransformResult
from .register_budget import RegisterBudgetTransform
from .cache_hints import CacheHintTransform
from .store_cache_hints import StoreCacheHintTransform
from .reorder import ReorderTransform
from .vectorize_loads import VectorizeLoadsTransform
from .vectorize_stores import VectorizeStoresTransform
from .prefetch import PrefetchTransform
from .split_vectors import SplitVectorLoadsTransform
from .pipeline import TransformPipeline, PipelineResult


ALL_TRANSFORMS = [
    RegisterBudgetTransform,
    CacheHintTransform,
    StoreCacheHintTransform,
    ReorderTransform,
    VectorizeLoadsTransform,
    VectorizeStoresTransform,
    PrefetchTransform,
    SplitVectorLoadsTransform,
]

_REGISTRY = {cls.name: cls for cls in ALL_TRANSFORMS}


def get_transform(name: str) -> PtxTransform:
    """Instantiate a transform by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown transform: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()
