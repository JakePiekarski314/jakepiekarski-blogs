"""PyMC model builders for hierarchical MMM (unpooled, centered, non-centered, tuned)."""

import numpy as np
import pymc as pm
from pymc_extras.prior import Prior


def build_unpooled_mmm(data, priors_cfg=None):
    """Build unpooled log-linear MMM; each channel beta estimated independently."""
    priors_cfg = priors_cfg or {}
    coords = data["coords"]
    X = data["X"]
    y_obs = data["y_obs"]

    with pm.Model(coords=coords) as model:
        Xd = pm.Data("X", X, dims=("date", "channel"))
        alpha = Prior("Normal", mu=0.5, sigma=0.5).create_variable("alpha")
        beta = Prior("Normal", mu=0.0, sigma=0.1, dims="channel").create_variable("beta")
        sigma = Prior("HalfNormal", sigma=0.5).create_variable("sigma")

        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(Xd, beta),
            dims="date",
        )
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y_obs,
            dims="date",
        )
        _add_predictive_deterministics(mu, sigma)

    return model


def build_centered_hierarchical_mmm(data, priors_cfg=None):
    """Build centered hierarchical MMM; beta_j ~ Normal(mu_beta, sigma_beta)."""
    priors_cfg = priors_cfg or {}
    coords = data["coords"]
    X = data["X"]
    y_obs = data["y_obs"]

    with pm.Model(coords=coords) as model:
        Xd = pm.Data("X", X, dims=("date", "channel"))
        alpha = Prior("Normal", mu=0.5, sigma=0.5).create_variable("alpha")
        beta = Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=0.1),
            sigma=Prior("HalfNormal", sigma=0.1),
            dims="channel",
        ).create_variable("beta")
        sigma = Prior("HalfNormal", sigma=0.5).create_variable("sigma")

        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(Xd, beta),
            dims="date",
        )
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y_obs,
            dims="date",
        )
        _add_predictive_deterministics(mu, sigma)

    return model


def build_noncentered_hierarchical_mmm(data, priors_cfg=None):
    """Build non-centered hierarchical MMM; z_j ~ N(0,1), beta_j = mu + sigma*z_j."""
    priors_cfg = priors_cfg or {}
    coords = data["coords"]
    X = data["X"]
    y_obs = data["y_obs"]

    with pm.Model(coords=coords) as model:
        Xd = pm.Data("X", X, dims=("date", "channel"))
        alpha = Prior("HalfNormal", sigma=0.5).create_variable("alpha")
        beta = Prior(
            "Normal",
            mu=Prior("Normal", mu=0.0, sigma=0.1),
            sigma=Prior("HalfNormal", sigma=0.1),
            dims="channel",
            centered=False,
        ).create_variable("beta")
        sigma = Prior("HalfNormal", sigma=0.5).create_variable("sigma")

        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(Xd, beta),
            dims="date",
        )
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y_obs,
            dims="date",
        )
        _add_predictive_deterministics(mu, sigma)

    return model


def build_tuned_noncentered_hierarchical_mmm(data, priors_cfg=None, offset_multiplier: float = 0.05):
    """Build tuned non-centered MMM; c10 gets stronger pooling via s_mult < 1."""
    priors_cfg = priors_cfg or {}
    coords = data["coords"]
    X = data["X"]
    y_obs = data["y_obs"]
    channel_cols = list(coords["channel"])

    with pm.Model(coords=coords) as model:
        Xd = pm.Data("X", X, dims=("date", "channel"))
        alpha = Prior("HalfNormal", sigma=0.5).create_variable("alpha")

        mu_beta = Prior("Normal", mu=0.0, sigma=0.1).create_variable("mu_beta")
        sigma_beta = Prior("HalfNormal", sigma=0.1).create_variable("sigma_beta")
        z = Prior("Normal", mu=0.0, sigma=1.0, dims="channel").create_variable("z")

        s = np.ones(len(channel_cols), dtype=float)
        s[channel_cols.index("c10")] = offset_multiplier
        s_mult = pm.Data("s_mult", s, dims="channel")

        beta = pm.Deterministic("beta", mu_beta + sigma_beta * s_mult * z, dims="channel")
        sigma = Prior("HalfNormal", sigma=0.5).create_variable("sigma")

        mu = pm.Deterministic(
            "mu",
            alpha + pm.math.dot(Xd, beta),
            dims="date",
        )
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y_obs,
            dims="date",
        )
        _add_predictive_deterministics(mu, sigma)

    return model


def _add_predictive_deterministics(mu, sigma):
    """Add y_mean and y_rep deterministics (shared across all models)."""
    pm.Deterministic("y_mean", pm.math.exp(mu), dims="date")
    pm.Deterministic(
        "y_rep",
        pm.math.exp(mu + pm.Normal.dist(0.0, sigma)),
        dims="date",
    )


def fit_model(
    model,
    seed: int = 42,
    draws: int = 500,
    tune: int = 500,
    chains: int = 4,
    target_accept: float = 0.9,
):
    """Sample posterior and posterior predictive; return idata."""
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        return_inferencedata=True,
        model=model,
        progressbar=False,
    )
    idata = pm.sample_posterior_predictive(
        idata,
        model=model,
        var_names=["y_obs", "y_mean", "y_rep"],
        random_seed=seed,
        extend_inferencedata=True,
    )
    return idata


def sample_prior(model, samples: int = 1000, seed: int = 42):
    """Sample from prior predictive; return InferenceData."""
    return pm.sample_prior_predictive(samples=samples, random_seed=seed, model=model)
