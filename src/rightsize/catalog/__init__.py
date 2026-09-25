"""Model catalog: ModelFacts from Hub metadata without downloading (F1).

Plan and TODO: docs/plans/F01-model-catalog.md
"""

from __future__ import annotations

from rightsize.catalog.curated import search
from rightsize.catalog.hub import facts
from rightsize.catalog.variants import variants
from rightsize.catalog.weights import safetensors_header

__all__ = ["facts", "safetensors_header", "search", "variants"]
