"""PyMC builder for the directional cross-channel interaction MMM.

Offline only: imported by ``scripts/fit_cross_channel.py``. Not used at render
time. Ported from the source notebook with tight, calibrated adstock/saturation
priors so that beta and gamma remain jointly identifiable.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.xtensor.type import as_xtensor

from pymc_marketing.mmm import GeometricAdstock, TanhSaturation
from pymc_marketing.mmm.events import EventEffect, GaussianBasis
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_extras.prior import Prior


def build_interaction_mmm(truth) -> pm.Model:
    """Build the interaction MMM for a ``SyntheticTruth`` instance."""
    n_channels = len(truth.channel_names)

    coords = {
        "date": truth.df.index.to_numpy(),
        "channel": truth.channel_names,
        "modifier_channel": truth.channel_names,
        "control": truth.control_names,
        "holiday": truth.holiday_names,
    }

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------ inputs
        media_data = pm.Data("media_scaled", truth.media_scaled.values, dims=("date", "channel"))
        controls_data = pm.Data("controls_scaled", truth.controls_scaled.values, dims=("date", "control"))
        holiday_distance_data = pm.Data("holiday_distance", truth.holiday_distance.values, dims=("date", "holiday"))
        dayofyear = pm.Data("dayofyear", truth.df.index.dayofyear.to_numpy().astype(float), dims="date")
        target_mean = float(truth.target_mean)

        # ------------------------------------------------------------ priors
        intercept = Prior("Normal", mu=0.8, sigma=0.15).create_variable("intercept")
        beta = Prior("HalfNormal", sigma=0.15, dims="channel").create_variable("beta")
        # gamma is a log-multiplier. With transformed media in (0, 1), gamma ~ 0.5
        # is a ~65% effectiveness lift at full intensity. sigma=0.40 regularises
        # the many near-zero entries while letting genuine effects escape; in
        # production prefer a sparsity-inducing prior (horseshoe / regularised
        # Laplace).
        gamma = Prior("Normal", mu=0.0, sigma=0.40, dims=("channel", "modifier_channel")).create_variable("gamma")
        control_beta = Prior("Normal", mu=0.0, sigma=0.10, dims="control").create_variable("control_beta")
        sigma = Prior("HalfNormal", sigma=0.03).create_variable("sigma")

        # mask the diagonal so a channel cannot modify itself
        gamma_eff = pm.Deterministic(
            "gamma_eff", gamma * (1.0 - pt.eye(n_channels)), dims=("channel", "modifier_channel")
        )

        # -------------------------------------------- adstock + saturation
        # Calibrated (as if from lift tests / geo experiments) and pinned tightly
        # so the saturation curve cannot absorb the multiplicative gamma signal.
        calibrated_alpha = np.array([0.20, 0.10, 0.25, 0.05])
        calibrated_b = np.array([0.9, 1.0, 0.8, 1.1])
        calibrated_c = np.array([1.0, 0.9, 1.2, 1.0])

        # normalize=False so the adstock matches the (non-normalised) data-generating
        # process and the calibrated alpha priors are on the same scale.
        adstock = GeometricAdstock(
            l_max=12,
            normalize=False,
            priors={
                "alpha": Prior("TruncatedNormal", mu=calibrated_alpha, sigma=0.05, lower=0.0, upper=1.0, dims="channel")
            },
        )
        saturation = TanhSaturation(
            priors={
                "b": Prior("TruncatedNormal", mu=calibrated_b, sigma=0.05, lower=0.0, dims="channel"),
                "c": Prior("TruncatedNormal", mu=calibrated_c, sigma=0.05, lower=0.0, dims="channel"),
            }
        )
        media_x = as_xtensor(media_data, dims=("date", "channel"))
        adstocked_media_x = adstock.apply(media_x, core_dim="date")
        transformed_media_x = saturation.apply(adstocked_media_x, core_dim="date")
        transformed_media = pm.Deterministic(
            "transformed_media", transformed_media_x.transpose("date", "channel").values, dims=("date", "channel")
        )

        # ---------------------------------------- contribution decomposition
        # Dimensions:
        #   date:             observation week
        #   channel:          affected channel (i)
        #   modifier_channel: channel modifying the affected channel (j)
        reference_contribution_scaled = pm.Deterministic(
            "reference_contribution_scaled", transformed_media * beta, dims=("date", "channel")
        )

        # log_multiplier[t, i] = sum_j transformed_media[t, j] * gamma_eff[i, j]
        log_multiplier = pt.dot(transformed_media, gamma_eff.T)
        interaction_multiplier = pm.Deterministic(
            "interaction_multiplier", pt.exp(log_multiplier), dims=("date", "channel")
        )

        observed_contribution_scaled = pm.Deterministic(
            "observed_contribution_scaled",
            reference_contribution_scaled * interaction_multiplier,
            dims=("date", "channel"),
        )
        pm.Deterministic(
            "interaction_contribution_scaled",
            observed_contribution_scaled - reference_contribution_scaled,
            dims=("date", "channel"),
        )

        # ------------------------------------------------ controls / fourier
        control_contribution_scaled = pm.Deterministic(
            "control_contribution_scaled", pt.dot(controls_data - 1.0, control_beta), dims="date"
        )

        fourier = YearlyFourier(n_order=2, prior=Prior("Normal", mu=0.0, sigma=0.05, dims="fourier"))
        dayofyear_x = as_xtensor(dayofyear, dims=("date",))
        fourier_contribution_scaled = pm.Deterministic(
            "fourier_contribution_scaled", fourier.apply(dayofperiod=dayofyear_x).values, dims="date"
        )

        # ----------------------------------------------- holidays / events
        gaussian_basis = GaussianBasis(priors={"sigma": Prior("Gamma", mu=7.0, sigma=1.0, dims="holiday")})
        holiday_effect_size = Prior("Normal", mu=0.0, sigma=0.05, dims="holiday")
        event_effect = EventEffect(basis=gaussian_basis, effect_size=holiday_effect_size, dims=("holiday",))
        holiday_distance_x = as_xtensor(holiday_distance_data, dims=("date", "holiday"))
        holiday_contribution_scaled = pm.Deterministic(
            "holiday_contribution_scaled",
            event_effect.apply(holiday_distance_x, name="holiday").transpose("date", "holiday").values,
            dims=("date", "holiday"),
        )

        # ---------------------------------------------------------------- mu
        contribution = pm.Deterministic(
            "contribution",
            intercept
            + observed_contribution_scaled.sum(axis=-1)
            + control_contribution_scaled
            + fourier_contribution_scaled
            + holiday_contribution_scaled.sum(axis=-1),
            dims="date",
        )
        pm.Deterministic("y_mean", contribution * target_mean, dims="date")

        pm.Normal("y_obs", mu=contribution, sigma=sigma, observed=truth.y_scaled.values, dims="date")

    return model
