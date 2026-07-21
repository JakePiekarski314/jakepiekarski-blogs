"""Synthetic data generation for the cross-channel interaction MMM.

Ported from the source notebook. The known ("true") directional gamma matrix
lets us check recovery. gamma[i, j] is the effect of modifier channel j on the
effectiveness of affected channel i (rows = affected, columns = modifier).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .transforms import geometric_adstock_1d, tanh_saturation_1d

RANDOM_SEED = 42

# Reader-friendly labels for the four synthetic channels. The interaction design
# makes channel_2 (video) the headline modifier of channel_1 (search), matching
# the worked example in the post.
CHANNEL_LABELS = {
    "channel_1": "Paid search",
    "channel_2": "Online video",
    "channel_3": "Paid social",
    "channel_4": "Display",
}


@dataclass
class SyntheticTruth:
    df: pd.DataFrame
    media_actual: pd.DataFrame
    media_scaled: pd.DataFrame
    controls_actual: pd.DataFrame
    controls_scaled: pd.DataFrame
    holiday_distance: pd.DataFrame
    fourier_features: pd.DataFrame
    y_actual: pd.Series
    y_scaled: pd.Series
    target_mean: float
    holiday_dates: dict
    channel_names: list
    control_names: list
    holiday_names: list
    # True parameters (for recovery overlays)
    beta_true: np.ndarray
    gamma_true: np.ndarray
    alpha_true: np.ndarray
    sat_b_true: np.ndarray
    sat_c_true: np.ndarray


def _difference_in_days(model_dates, event_dates) -> np.ndarray:
    one_day = np.timedelta64(1, "D")
    return (model_dates.to_numpy()[:, None] - event_dates.to_numpy()) / one_day


def nearest_event_distance(model_dates, holiday_dates: dict) -> np.ndarray:
    """Signed distance (days) from each date to the nearest occurrence of each holiday."""
    out = np.zeros((len(model_dates), len(holiday_dates)))
    for j, (_, centers) in enumerate(holiday_dates.items()):
        diffs = _difference_in_days(model_dates, pd.DatetimeIndex(centers))
        nearest = np.argmin(np.abs(diffs), axis=1)
        out[:, j] = diffs[np.arange(len(model_dates)), nearest]
    return out


def _gaussian_bump(distance_days: np.ndarray, sigma_days: float) -> np.ndarray:
    return (1.0 / (sigma_days * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (distance_days / sigma_days) ** 2
    )


def generate_synthetic_data(
    n_weeks: int = 208,
    start_date: str = "2021-01-04",
    seed: int = RANDOM_SEED,
) -> SyntheticTruth:
    """Generate three years of weekly data for four media channels with a known,
    sparse, directional interaction (gamma) matrix."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    channel_names = [f"channel_{i + 1}" for i in range(4)]
    control_names = ["control_1", "control_2"]
    holiday_names = ["christmas", "easter"]
    n_channels = len(channel_names)

    # --- media spend (actual units, GBP) --------------------------------
    # Independent temporal variation plus ~30% dark weeks per channel give the
    # model the contrast it needs to separate gamma_ij from beta_i.
    base_levels = np.array([50_000.0, 35_000.0, 25_000.0, 15_000.0])
    phase_shift_weeks = np.array([0, 13, 26, 39])
    media_actual = np.zeros((n_weeks, n_channels))
    week_idx = np.arange(n_weeks)
    campaign_active = rng.binomial(1, 0.7, size=(n_weeks, n_channels))
    for i, base in enumerate(base_levels):
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (week_idx + phase_shift_weeks[i]) / 52)
        noise = rng.lognormal(mean=0.0, sigma=0.30, size=n_weeks)
        burst_weeks = rng.choice(n_weeks, size=20, replace=False)
        burst = np.ones(n_weeks)
        burst[burst_weeks] *= rng.uniform(1.8, 2.6, size=burst_weeks.size)
        media_actual[:, i] = base * seasonal * noise * burst * campaign_active[:, i]
    media_actual = pd.DataFrame(media_actual, index=dates, columns=channel_names)

    media_scaled = media_actual / media_actual.max()

    # --- true adstock + saturation transforms ---------------------------
    alpha_true = np.array([0.20, 0.10, 0.25, 0.05])
    sat_b_true = np.array([0.9, 1.0, 0.8, 1.1])
    sat_c_true = np.array([1.0, 0.9, 1.2, 1.0])
    transformed_media = np.zeros_like(media_scaled.values)
    for i in range(n_channels):
        adstocked = geometric_adstock_1d(media_scaled.values[:, i], alpha=alpha_true[i], l_max=12)
        transformed_media[:, i] = tanh_saturation_1d(adstocked, b=sat_b_true[i], c=sat_c_true[i])

    # --- directional gamma matrix (gamma[i, j] = effect of j on i) ------
    beta_true = np.array([0.12, 0.10, 0.09, 0.08])
    gamma_true = np.zeros((n_channels, n_channels))
    gamma_true[0, 1] = 0.50   # channel_2 strongly amplifies channel_1 (headline)
    gamma_true[1, 0] = 0.20   # the reverse direction is much smaller (asymmetry)
    gamma_true[2, 3] = 0.80   # channel_4 strongly amplifies channel_3
    gamma_true[3, 2] = 0.25   # channel_3 amplifies channel_4 weakly
    gamma_true[3, 0] = -0.60  # channel_1 cannibalises channel_4 (negative interaction)
    gamma_eff = gamma_true * (1 - np.eye(n_channels))

    log_mult = transformed_media @ gamma_eff.T
    interaction_multiplier = np.exp(log_mult)
    silo_contribution = transformed_media * beta_true
    observed_contribution = silo_contribution * interaction_multiplier
    media_signal = observed_contribution.sum(axis=1)

    # --- controls -------------------------------------------------------
    raw_controls = np.column_stack([
        1.0 + 0.25 * np.sin(2 * np.pi * week_idx / 52 + 0.4) + rng.normal(0.0, 0.15, size=n_weeks),
        1.0 + 0.10 * np.cos(2 * np.pi * week_idx / 26) + rng.normal(0.0, 0.10, size=n_weeks),
    ])
    controls_actual = pd.DataFrame(raw_controls, index=dates, columns=control_names)
    controls_scaled = controls_actual / controls_actual.mean()
    control_beta_true = np.array([0.06, -0.04])
    control_signal = (controls_scaled.values - 1.0) @ control_beta_true

    # --- Fourier seasonality --------------------------------------------
    dayofyear = dates.dayofyear.to_numpy()
    period = 365.25
    fourier_features = pd.DataFrame(
        {
            "sin_1": np.sin(2 * np.pi * 1 * dayofyear / period),
            "cos_1": np.cos(2 * np.pi * 1 * dayofyear / period),
            "sin_2": np.sin(2 * np.pi * 2 * dayofyear / period),
            "cos_2": np.cos(2 * np.pi * 2 * dayofyear / period),
        },
        index=dates,
    )
    fourier_beta_true = np.array([0.04, 0.02, -0.02, 0.015])
    fourier_signal = fourier_features.values @ fourier_beta_true

    # --- holidays (Gaussian bumps mirroring GaussianBasis) --------------
    years = sorted(set(dates.year))
    holiday_dates = {
        "christmas": [pd.Timestamp(f"{y}-12-25") for y in years],
        "easter": [pd.Timestamp(easter(y)) for y in years],
    }
    holiday_distance_arr = nearest_event_distance(dates, holiday_dates)
    holiday_amp_true = np.array([0.10, 0.05])
    holiday_sigma_true = np.array([7.0, 7.0])
    holiday_signal = np.zeros(n_weeks)
    for j in range(len(holiday_names)):
        bump = _gaussian_bump(holiday_distance_arr[:, j], holiday_sigma_true[j])
        scale = holiday_sigma_true[j] * np.sqrt(2 * np.pi)
        holiday_signal += holiday_amp_true[j] * bump * scale
    holiday_distance = pd.DataFrame(holiday_distance_arr, index=dates, columns=holiday_names)

    # --- assemble target ------------------------------------------------
    intercept_true = 0.8
    sigma_y_true = 0.02
    noise_eps = rng.normal(0.0, sigma_y_true, size=n_weeks)
    y_scaled_vec = (
        intercept_true + media_signal + control_signal + fourier_signal + holiday_signal + noise_eps
    )
    target_mean = 500_000.0
    y_actual_vec = y_scaled_vec * target_mean
    y_scaled = pd.Series(y_scaled_vec, index=dates, name="y_scaled")
    y_actual = pd.Series(y_actual_vec, index=dates, name="y_actual")

    df = (
        media_actual.add_suffix("_spend")
        .join(controls_actual)
        .assign(y_actual=y_actual, y_scaled=y_scaled)
    )

    return SyntheticTruth(
        df=df,
        media_actual=media_actual,
        media_scaled=media_scaled,
        controls_actual=controls_actual,
        controls_scaled=controls_scaled,
        holiday_distance=holiday_distance,
        fourier_features=fourier_features,
        y_actual=y_actual,
        y_scaled=y_scaled,
        target_mean=target_mean,
        holiday_dates=holiday_dates,
        channel_names=channel_names,
        control_names=control_names,
        holiday_names=holiday_names,
        beta_true=beta_true,
        gamma_true=gamma_true,
        alpha_true=alpha_true,
        sat_b_true=sat_b_true,
        sat_c_true=sat_c_true,
    )
