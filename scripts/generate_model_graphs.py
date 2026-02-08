#!/usr/bin/env python
"""
Generate model graph images for the hierarchical MMM blog.
Run from project root: python scripts/generate_model_graphs.py

Outputs: posts/hierarchical-mmm/assets/model_*.png
"""

import sys
from pathlib import Path

# Run from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pymc as pm

from src.data import make_synthetic_mmm_data
from src.models import (
    build_unpooled_mmm,
    build_centered_hierarchical_mmm,
    build_noncentered_hierarchical_mmm,
    build_tuned_noncentered_hierarchical_mmm,
)
from src.config import TUNED_MULTIPLIER

OUTPUT_DIR = project_root / "posts" / "hierarchical-mmm" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent landscape size for all model graphs (width x height in inches)
FIG_SIZE = (16, 8)


def main():
    print("Generating data...")
    data, _ = make_synthetic_mmm_data(seed=42)

    print("Generating model graphs...")
    unpooled_model = build_unpooled_mmm(data)
    pm.model_to_graphviz(unpooled_model, save=str(OUTPUT_DIR / "model_unpooled.png"), figsize=FIG_SIZE)
    print("  - model_unpooled.png")

    centered_model = build_centered_hierarchical_mmm(data)
    pm.model_to_graphviz(centered_model, save=str(OUTPUT_DIR / "model_centered.png"), figsize=FIG_SIZE)
    print("  - model_centered.png")

    noncentered_model = build_noncentered_hierarchical_mmm(data)
    pm.model_to_graphviz(noncentered_model, save=str(OUTPUT_DIR / "model_noncentered.png"), figsize=FIG_SIZE)
    print("  - model_noncentered.png")

    tuned_model = build_tuned_noncentered_hierarchical_mmm(data, offset_multiplier=TUNED_MULTIPLIER)
    pm.model_to_graphviz(tuned_model, save=str(OUTPUT_DIR / "model_tuned.png"), figsize=FIG_SIZE)
    print("  - model_tuned.png")

    print("Done.")


if __name__ == "__main__":
    main()
