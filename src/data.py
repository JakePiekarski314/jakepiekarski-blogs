"""Data generation and channel mappings for the hierarchical MMM demo."""

import numpy as np
import pandas as pd

# Paid Social bucket (10 channels) – presentation labels
CHANNEL_NAME = {
    "c1": "Meta Prospecting – Broad (Conversions)",
    "c2": "Meta Prospecting – Lookalike (Purchasers)",
    "c3": "Meta TOF Video – Broad (Reach)",
    "c4": "TikTok Prospecting – Broad (Conversions)",
    "c5": "TikTok Spark Ads – Creator Whitelisting",
    "c6": "TikTok Video Views – Awareness",
    "c7": "Pinterest Prospecting – Conversions",
    "c8": "Reddit Prospecting – Community Targeting",
    "c9": "X (Twitter) Awareness – Interest Targeting",
    "c10": "LinkedIn Awareness – Always-on (Legacy)",  # weak channel
}

WEAK_CHANNEL = "c10"
WEAK_CHANNEL_LABEL = CHANNEL_NAME[WEAK_CHANNEL]


def make_synthetic_mmm_data(
    seed: int = 42,
    start_date: str = "2023-01-01",
    mu_beta: float = 0.5,
    sigma_beta: float = 0.05,
    alpha: float = 0.5,
    sigma_y: float = 0.1,
    spend_sd_strong: float = 0.6,
    spend_sd_weak: float = 0.05,
):
    """
    Generate positive-only weekly regression data with 10 channels (c1–c10)
    from start_date to today, where c10 is deliberately weakly identified.

    Conceptual data-generating process (log scale):
        log(y_t) = alpha + sum_j beta_j * log(c_{j,t}) + epsilon_t
        beta_j ~ Normal(mu_beta, sigma_beta)
        epsilon_t ~ Normal(0, sigma_y)

    After exponentiation:
        y_t > 0, c_{j,t} > 0

    Weak channel (c10):
      - much lower variance in spend
      - contributes little identifying information about its coefficient

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    start_date : str
        Start date for weekly data.
    mu_beta : float
        Population mean of channel elasticities.
    sigma_beta : float
        Population standard deviation of channel elasticities.
    alpha : float
        Intercept on the log-sales scale.
    sigma_y : float
        Noise standard deviation on the log-sales scale.
    spend_sd_strong : float
        Log-scale standard deviation of spend for strong channels.
    spend_sd_weak : float
        Log-scale standard deviation of spend for the weak channel (c10).

    Returns
    -------
    data : dict
        X (log-spend matrix), y_obs (log-outcome), coords (date, channel),
        channel_cols, df (for plotting).
    truth : dict
        alpha, beta, mu_beta, sigma_beta, weak_channel ("c10").
    """
    rng = np.random.default_rng(seed)

    # Weekly date index
    dates = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq="W")
    n = len(dates)

    n_channels = 10
    weak_idx = 9  # c10

    # True channel elasticities
    beta_true = rng.normal(mu_beta, sigma_beta, size=n_channels)

    # Spend variability (log scale)
    spend_sd = np.full(n_channels, spend_sd_strong)
    spend_sd[weak_idx] = spend_sd_weak

    # Generate positive spend (log-normal)
    log_spend = rng.normal(0.0, spend_sd, size=(n, n_channels))
    spend = np.exp(log_spend)

    # Generate sales (log-linear model)
    log_y = alpha + log_spend @ beta_true + rng.normal(0.0, sigma_y, size=n)
    y = np.exp(log_y)

    channel_cols = [f"c{i+1}" for i in range(n_channels)]
    df = pd.DataFrame(spend, columns=channel_cols)
    df.insert(0, "date", dates)
    df["y"] = y

    channel_cols_sorted = sorted(channel_cols, key=lambda s: int(s[1:]))
    coords = {
        "date": df["date"].to_numpy(),
        "channel": channel_cols_sorted,
    }

    X = np.log(df[channel_cols_sorted].to_numpy())
    y_obs = np.log(df["y"].to_numpy())

    data = {
        "X": X,
        "y_obs": y_obs,
        "coords": coords,
        "channel_cols": channel_cols_sorted,
        "df": df,
    }

    truth = {
        "alpha": alpha,
        "mu_beta": mu_beta,
        "sigma_beta": sigma_beta,
        "beta": beta_true,
        "weak_channel": "c10",
    }

    return data, truth
