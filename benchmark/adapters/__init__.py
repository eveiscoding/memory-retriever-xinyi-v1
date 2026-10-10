"""Adapters from public benchmark formats to the common schema."""

from .locomo import load_locomo
from .synthetic import load_synthetic

__all__ = ["load_locomo", "load_synthetic"]
