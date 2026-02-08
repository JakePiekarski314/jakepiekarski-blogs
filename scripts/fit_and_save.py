#!/usr/bin/env python
"""
Fit all four MMM models, sample priors, and save idata + data + truth to disk.
Run once from project root: python scripts/fit_and_save.py

Outputs go to posts/hierarchical-mmm/assets/
"""

import sys
from pathlib import Path

# Run from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pickle
import arviz as az

from src.data import make_synthetic_mmm_data, CHANNEL_NAME
from src.models import (
    build_unpooled_mmm,
    build_centered_hierarchical_mmm,
    build_noncentered_hierarchical_mmm,
    build_tuned_noncentered_hierarchical_mmm,
    fit_model,
    sample_prior,
)
from src.plotting import apply_channel_labels_to_idata
from src.config import (
    DRAWS,
    TUNE,
    CHAINS,
    TARGET_ACCEPT_UNPOOLED,
    TARGET_ACCEPT_CENTERED,
    TARGET_ACCEPT_NONCENTERED,
    PRIOR_SAMPLES,
    TUNED_MULTIPLIER,
)

OUTPUT_DIR = project_root / "posts" / "hierarchical-mmm" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Generating data...")
    data, truth = make_synthetic_mmm_data(seed=42)

    # Unpooled
    print("Fitting unpooled model...")
    unpooled_model = build_unpooled_mmm(data)
    idata_unpooled = fit_model(
        unpooled_model,
        seed=42,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_UNPOOLED,
    )
    prior_unpooled = sample_prior(unpooled_model, samples=PRIOR_SAMPLES, seed=42)
    apply_channel_labels_to_idata(idata_unpooled, CHANNEL_NAME)
    apply_channel_labels_to_idata(prior_unpooled, CHANNEL_NAME)

    # Centered
    print("Fitting centered model...")
    centered_model = build_centered_hierarchical_mmm(data)
    idata_centered_pp = fit_model(
        centered_model,
        seed=42,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_CENTERED,
    )
    prior_centered_pp = sample_prior(centered_model, samples=PRIOR_SAMPLES, seed=42)
    apply_channel_labels_to_idata(idata_centered_pp, CHANNEL_NAME)
    apply_channel_labels_to_idata(prior_centered_pp, CHANNEL_NAME)

    # Non-centered
    print("Fitting non-centered model...")
    noncentered_model = build_noncentered_hierarchical_mmm(data)
    idata_non_centered_pp = fit_model(
        noncentered_model,
        seed=42,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_NONCENTERED,
    )
    prior_non_centered_pp = sample_prior(noncentered_model, samples=PRIOR_SAMPLES, seed=42)
    apply_channel_labels_to_idata(idata_non_centered_pp, CHANNEL_NAME)
    apply_channel_labels_to_idata(prior_non_centered_pp, CHANNEL_NAME)

    # Tuned
    print("Fitting tuned model...")
    tuned_model = build_tuned_noncentered_hierarchical_mmm(data, offset_multiplier=TUNED_MULTIPLIER)
    idata_tuned_non_centered_pp = fit_model(
        tuned_model,
        seed=42,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT_NONCENTERED,
    )
    prior_tuned_non_centered_pp = sample_prior(tuned_model, samples=PRIOR_SAMPLES, seed=42)
    apply_channel_labels_to_idata(idata_tuned_non_centered_pp, CHANNEL_NAME)
    apply_channel_labels_to_idata(prior_tuned_non_centered_pp, CHANNEL_NAME)

    # Save
    print("Saving to", OUTPUT_DIR)
    idata_unpooled.to_netcdf(OUTPUT_DIR / "idata_unpooled.nc")
    idata_centered_pp.to_netcdf(OUTPUT_DIR / "idata_centered.nc")
    idata_non_centered_pp.to_netcdf(OUTPUT_DIR / "idata_noncentered.nc")
    idata_tuned_non_centered_pp.to_netcdf(OUTPUT_DIR / "idata_tuned.nc")

    prior_unpooled.to_netcdf(OUTPUT_DIR / "prior_unpooled.nc")
    prior_centered_pp.to_netcdf(OUTPUT_DIR / "prior_centered.nc")
    prior_non_centered_pp.to_netcdf(OUTPUT_DIR / "prior_noncentered.nc")
    prior_tuned_non_centered_pp.to_netcdf(OUTPUT_DIR / "prior_tuned.nc")

    with open(OUTPUT_DIR / "data.pkl", "wb") as f:
        pickle.dump(data, f)
    with open(OUTPUT_DIR / "truth.pkl", "wb") as f:
        pickle.dump(truth, f)

    print("Done.")


if __name__ == "__main__":
    main()
