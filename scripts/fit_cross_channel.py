#!/usr/bin/env python
"""Fit the cross-channel interaction MMM and save render-time assets.

Run once from the project root (needs requirements-fit.txt):

    python scripts/fit_cross_channel.py

Outputs (to posts/cross-channel-measurement-mmm/assets/):
  - posterior.nc  : thinned posterior of structural parameters (+ y_mean)
  - data.pkl      : inputs and true parameters needed to recompute reports
  - model_graph.png : committed PyMC model graph (graphviz not run in CI)

Nothing here is imported at render time; the post only reads the saved files.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pymc as pm
import arviz as az
import xarray as xr

from src.crosschannel.data import generate_synthetic_data, CHANNEL_LABELS
from src.crosschannel.model import build_interaction_mmm

RANDOM_SEED = 42
DRAWS = 1000
TUNE = 1000
CHAINS = 4
TARGET_ACCEPT = 0.9
THIN_TO = 1000  # total posterior draws to keep after stacking chains

ASSETS = PROJECT_ROOT / "posts" / "cross-channel-measurement-mmm" / "assets"

# Structural parameters kept for render-time recomputation (+ y_mean for the fit plot).
KEEP_VARS = [
    "intercept", "sigma", "beta", "control_beta",
    "gamma_offdiag", "gamma_eff", "adstock_alpha", "saturation_b", "saturation_c",
    "y_mean",
]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data (seed=%d)..." % RANDOM_SEED)
    truth = generate_synthetic_data(seed=RANDOM_SEED)

    print("Building model...")
    model = build_interaction_mmm(truth)
    print("  free RVs:", [v.name for v in model.free_RVs])

    print("Sampling with numpyro NUTS (%d chains x %d draws, target_accept=%.2f)..." % (CHAINS, DRAWS, TARGET_ACCEPT))
    with model:
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED,
            nuts_sampler="numpyro", progressbar=False, return_inferencedata=True,
        )
        idata.extend(pm.sample_posterior_predictive(
            idata, var_names=["y_mean"], random_seed=RANDOM_SEED, progressbar=False,
        ))

    # ------------------------------------------------------- convergence gate
    free_names = [v.name for v in model.free_RVs]
    summ = az.summary(idata, var_names=free_names, round_to=4)
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(summ["ess_bulk"].min())
    n_div = int(idata.sample_stats["diverging"].values.sum())
    print(f"  max R-hat = {max_rhat:.4f} | min ESS(bulk) = {min_ess:.0f} | divergences = {n_div}")
    if max_rhat > 1.05 or n_div > 0:
        print("WARNING: convergence diagnostics look poor; inspect before publishing.")

    # ------------------------------------------------------------- recovery
    post = idata.posterior
    beta_med = post["beta"].median(("chain", "draw")).values
    gamma_med = post["gamma_eff"].median(("chain", "draw")).values
    print("  beta recovery (est vs true):")
    for k, name in enumerate(truth.channel_names):
        print(f"    {name}: {beta_med[k]:.3f} vs {truth.beta_true[k]:.3f}")
    print("  key gamma recovery (affected<-modifier, est vs true):")
    true = truth.gamma_true
    for (i, j) in [(0, 1), (1, 0), (2, 3), (3, 2), (3, 0)]:
        print(f"    {truth.channel_names[i]}<-{truth.channel_names[j]}: "
              f"{gamma_med[i, j]:+.3f} vs {true[i, j]:+.3f}")

    # --------------------------------------------- reduced, thinned posterior
    # Save the model's OWN contribution totals (summed over the window, in £) so
    # the post reports figures that exactly match the fitted model rather than a
    # numpy re-derivation of the media transforms.
    tmean = float(truth.target_mean)
    ref_tot = (post["reference_contribution_scaled"].sum("date") * tmean).rename("reference_contribution_tot")
    obs_tot = (post["realised_mix_contribution_scaled"].sum("date") * tmean).rename("realised_mix_contribution_tot")

    keep = [v for v in KEEP_VARS if v in post]
    ds = xr.merge([post[keep], ref_tot.to_dataset(), obs_tot.to_dataset()])
    ds = ds.stack(sample=("chain", "draw")).reset_index("sample")
    n = ds.sizes["sample"]
    step = max(1, n // THIN_TO)
    ds = ds.isel(sample=slice(None, None, step))
    ds = ds.transpose("sample", ...)
    # drop MultiIndex leftovers so the file writes cleanly
    ds = ds.drop_vars([c for c in ("chain", "draw") if c in ds.coords], errors="ignore")
    out_nc = ASSETS / "posterior.nc"
    ds.to_netcdf(out_nc, engine="h5netcdf")
    print(f"Saved {out_nc} ({ds.sizes['sample']} draws, {out_nc.stat().st_size / 1e6:.2f} MB)")

    # ---------------------------------------------------------------- data
    data = {
        "media_actual": truth.media_actual,
        "media_scaled": truth.media_scaled,
        "controls_scaled": truth.controls_scaled,
        "y_actual": truth.y_actual,
        "y_scaled": truth.y_scaled,
        "target_mean": truth.target_mean,
        "channel_names": truth.channel_names,
        "control_names": truth.control_names,
        "channel_labels": CHANNEL_LABELS,
        "beta_true": truth.beta_true,
        "gamma_true": truth.gamma_true,
        "alpha_true": truth.alpha_true,
        "sat_b_true": truth.sat_b_true,
        "sat_c_true": truth.sat_c_true,
    }
    with open(ASSETS / "data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {ASSETS / 'data.pkl'}")

    # --------------------------------------------------------- model graph
    try:
        graph = pm.model_to_graphviz(model)
        graph.render(filename=str(ASSETS / "model_graph"), format="png", cleanup=True)
        print(f"Saved {ASSETS / 'model_graph.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"Could not render model graph: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
