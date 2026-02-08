"""Summary functions for hierarchical MMM results."""

import warnings

warnings.filterwarnings("ignore", message=".*ArviZ.*undergoing.*")

import numpy as np
import pandas as pd
import arviz as az


def extract_true_beta(truth, channel: str = "c10"):
    """Extract true beta value for a channel from truth dict."""
    channel_ids = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    if channel not in channel_ids:
        raise ValueError(f"Unknown channel: {channel}")
    idx = channel_ids.index(channel)
    return float(np.asarray(truth["beta"]).ravel()[idx])


def summarize_channel(idata, channel_label, rope=0.0):
    """
    Summarize posterior for a channel's beta: mean, 94% HDI, P(beta > rope).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples (channel coord must be human-readable after apply_labels).
    channel_label : str
        Human-readable channel label (e.g. "LinkedIn Awareness – Always-on (Legacy)").
    rope : float
        Region of practical equivalence (default 0.0).

    Returns
    -------
    dict
        mean_beta, hdi_94_low, hdi_94_high, P(beta>0) or P(beta>rope).
    """
    beta = idata.posterior["beta"].sel(channel=channel_label)
    samples = beta.stack(sample=("chain", "draw")).values
    mean = float(np.mean(samples))
    hdi_low, hdi_high = az.hdi(samples, hdi_prob=0.94).tolist()
    p_pos = float(np.mean(samples > rope))
    return {"mean_beta": mean, "hdi_94_low": hdi_low, "hdi_94_high": hdi_high, "P(beta>0)": p_pos}


def compare_models_table(idatas, model_names, channel_label):
    """
    Build a comparison table for a channel across models.

    Parameters
    ----------
    idatas : list of arviz.InferenceData
        Posterior samples from each model.
    model_names : list of str
        Model display names.
    channel_label : str
        Human-readable channel label.

    Returns
    -------
    pandas.DataFrame
        Rows: models; columns: mean_beta, hdi_94_low, hdi_94_high, P(beta>0).
    """
    rows = []
    for idata, name in zip(idatas, model_names):
        s = summarize_channel(idata, channel_label)
        s["model"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("model")


def accuracy_summary_table(
    idatas,
    model_names,
    truth,
    channel_map=None,
    channel="c10",
    hdi_prob=0.94,
):
    """
    Build a table quantifying each model's accuracy in estimating a channel's true effect.

    Reports point estimate % difference (posterior mean vs true value) and whether
    the HDI covers the true value (coverage).

    Parameters
    ----------
    idatas : list of arviz.InferenceData
        Posterior samples from each model.
    model_names : list of str
        Model display names.
    truth : dict
        True data-generating parameters (expects key 'beta').
    channel_map : dict, optional
        Mapping from channel id to display name. Defaults to CHANNEL_NAME.
    channel : str
        Channel id (e.g. "c10") to evaluate.
    hdi_prob : float
        HDI probability for coverage (default 0.94).

    Returns
    -------
    pandas.DataFrame
        Rows: models; columns: Model, Point estimate, True value, Point est. % diff, Coverage.
    """
    from .data import CHANNEL_NAME

    channel_map = channel_map or CHANNEL_NAME
    channel_label = channel_map.get(channel, channel)

    channel_ids = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    true_beta = float(np.asarray(truth["beta"]).ravel()[channel_ids.index(channel)])
    rows = []
    for idata, name in zip(idatas, model_names):
        idata_labeled = idata.copy()
        if hasattr(idata_labeled, "posterior") and "beta" in idata_labeled.posterior:
            from .plotting import apply_channel_labels_to_idata

            apply_channel_labels_to_idata(idata_labeled, channel_map)
        s = summarize_channel(idata_labeled, channel_label)
        mean_beta = s["mean_beta"]
        hdi_low, hdi_high = s["hdi_94_low"], s["hdi_94_high"]
        pct_diff = 100 * (mean_beta - true_beta) / true_beta if true_beta != 0 else np.nan
        pct_str = f"{pct_diff:.2f}%" if not np.isnan(pct_diff) else "n/a"
        coverage_str = "Yes" if hdi_low <= true_beta <= hdi_high else "No"
        rows.append(
            {
                "Model": name,
                "Point estimate": round(mean_beta, 4),
                "True value": round(true_beta, 4),
                "Point est. % diff": pct_str,
                "Coverage": coverage_str,
            }
        )
    return pd.DataFrame(rows)
