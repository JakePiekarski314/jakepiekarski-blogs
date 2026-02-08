"""Plotting functions for hierarchical MMM results."""

import warnings

warnings.filterwarnings("ignore", message=".*ArviZ.*undergoing.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import arviz as az
import xarray as xr

from .data import CHANNEL_NAME


def apply_channel_labels_to_idata(idata, channel_name_map: dict, coord_name: str = "channel"):
    """Replace idata's channel coordinate labels (e.g., c1..c10) with human-readable names."""
    if idata is None:
        return None

    def _rename_coords(ds):
        if ds is None:
            return None
        if coord_name not in ds.coords:
            return ds
        old = list(ds.coords[coord_name].values)
        new = [channel_name_map.get(str(x), str(x)) for x in old]
        return ds.assign_coords({coord_name: new})

    for grp in ["posterior", "prior", "posterior_predictive", "sample_stats", "prior_predictive"]:
        if hasattr(idata, grp):
            try:
                setattr(idata, grp, _rename_coords(getattr(idata, grp)))
            except Exception:
                pass

    return idata


def plot_weekly_data(df, weak="c10", normalize_channels=True, channel_map=None):
    """
    Plot weekly data showing y and all channels over time, with the weak channel highlighted.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'date', 'y', and channel columns (e.g., c1..c10).
    weak : str
        Name of the weak channel column to highlight (default 'c10').
    normalize_channels : bool
        If True, z-score channels so they can be compared on one axis.
        If False, plot channels on their original scale.
    channel_map : dict, optional
        Mapping from channel id to display name. Defaults to CHANNEL_NAME.
    """
    channel_map = channel_map or CHANNEL_NAME
    d = df.sort_values("date").copy()
    channel_cols = [c for c in d.columns if c.startswith("c") and c[1:].isdigit()]
    channel_cols = sorted(channel_cols, key=lambda s: int(s[1:]))

    if normalize_channels:
        X = d[channel_cols]
        Xn = (X - X.mean()) / X.std(ddof=0)
        d = d.copy()
        d[channel_cols] = Xn

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    fig.subplots_adjust(bottom=0.22, hspace=0.3)

    axes[0].plot(d["date"], d["y"])
    axes[0].set_title("Outcome Over Time")
    axes[0].set_ylabel("Outcome")

    for c in channel_cols:
        lw = 2.8 if c == weak else 1.0
        ls = "-" if c == weak else "--"
        alpha = 1.0 if c == weak else 0.55
        axes[1].plot(d["date"], d[c], linewidth=lw, linestyle=ls, alpha=alpha, label=channel_map.get(c, c))

    axes[1].set_title("Channel Spend Over Time")
    axes[1].set_ylabel("Channel Spend")
    axes[1].legend(
        ncol=2,
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )

    return fig, axes


def plot_coeff_forest_prior_posterior(
    idata, idata_prior, truth=None, channel_map=None, hdi_prob=0.94, figsize=(8, 4)
):
    """
    Forest plot for alpha + betas with prior vs posterior overlay (ArviZ).

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples.
    idata_prior : arviz.InferenceData
        Prior predictive samples.
    truth : dict, optional
        True data-generating parameters (expects key 'beta'). If provided, scatter points are added.
    channel_map : dict, optional
        Mapping from channel id to display name. Defaults to CHANNEL_NAME.
    hdi_prob : float
        HDI probability (default 0.94).
    figsize : tuple
        Figure size.

    Returns
    -------
    axes
        Matplotlib axes returned by az.plot_forest.
    """
    channel_map = channel_map or CHANNEL_NAME
    idata_with_prior = idata.copy()
    idata_with_prior.extend(idata_prior)
    apply_channel_labels_to_idata(idata_with_prior, channel_map)

    axes = az.plot_forest(
        [idata_with_prior.prior, idata_with_prior.posterior],
        model_names=["Prior", "Posterior"],
        kind="forestplot",
        var_names=["beta"],
        combined=True,
        hdi_prob=hdi_prob,
        figsize=figsize,
    )
    ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
    ax.set_title("Coefficients: prior vs posterior")
    ax.set_xlim(-0.3, 0.7)

    if truth is not None and "beta" in truth:
        beta_true = np.asarray(truth["beta"]).ravel()
        # Forest plot: ArviZ draws Prior and Posterior with offset within each row.
        # yticks are row midpoints; Prior is lower, Posterior is higher.
        # Shift to Posterior row: add half the in-row gap (0.375).
        yticks = ax.get_yticks()
        n_channels = len(beta_true)
        y_positions = []
        if len(yticks) >= n_channels:
            # One row per channel: yticks[0]=bottom=c10, yticks[-1]=top=c1
            midpoints = list(reversed(yticks[:n_channels]))
            # ArviZ uses ~0.75 gap between Prior and Posterior; Posterior is lower
            y_positions = [y - 0.375 for y in midpoints]
        else:
            for i in range(n_channels):
                idx = 2 * (n_channels - 1 - i)
                if idx + 1 < len(yticks):
                    mid = (yticks[idx] + yticks[idx + 1]) / 2
                    y_positions.append(mid - 0.375)
                else:
                    y_positions.append(yticks[min(idx, len(yticks) - 1)])
        if y_positions:
            scatter = ax.scatter(
                beta_true,
                y_positions,
                color="black",
                zorder=5,
                s=25,
                label="True value",
            )
            # Preserve Prior/Posterior legend, add True value
            existing = ax.get_legend()
            if existing is not None:
                handles = list(existing.legend_handles) + [scatter]
                labels = [t.get_text() for t in existing.get_texts()] + ["True value"]
                ax.legend(
                    handles=handles,
                    labels=labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 0.98),
                    frameon=False,
                )
            else:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.98), frameon=False)
    else:
        # Move ArviZ's legend (Prior/Posterior) outside when no truth
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.02, 0.98))
            leg.set_loc("upper left")
    fig = ax.get_figure()
    fig.subplots_adjust(right=0.82)

    return axes


def plot_coeff_forest_compare_models(
    idatas,
    model_names,
    truth=None,
    channel_map=None,
    coords=None,
    hdi_prob=0.94,
    figsize=(8, 4),
):
    """
    Forest plot comparing posteriors across multiple models (rows = models, like prior/posterior).

    Same format as plot_coeff_forest_prior_posterior: rows show model names, one posterior per row,
    truth dots aligned with posteriors. Legend shows only "Posterior" and "True value".

    Parameters
    ----------
    idatas : list of arviz.InferenceData
        Posterior samples from each model.
    model_names : list of str
        Model display names (used as row labels).
    truth : dict, optional
        True data-generating parameters (expects key 'beta').
    channel_map : dict, optional
        Mapping from channel id to display name.
    coords : dict, optional
        Coords to filter (e.g. {"channel": ["LinkedIn Awareness – Always-on (Legacy)"]}).
    hdi_prob : float
        HDI probability.
    figsize : tuple
        Figure size.

    Returns
    -------
    axes
        Matplotlib axes.
    """
    channel_map = channel_map or CHANNEL_NAME

    # Reshape: stack each model's beta for the filtered channel into one dataset with dim=model
    channel_label = None
    if coords is not None and "channel" in coords:
        filt = coords["channel"]
        channel_label = filt[0] if isinstance(filt, (list, tuple)) and len(filt) == 1 else None
    if channel_label is None:
        channel_label = CHANNEL_NAME.get("c10", "c10")

    idatas_labeled = []
    for _idata in idatas:
        idata_copy = _idata.copy()
        apply_channel_labels_to_idata(idata_copy, channel_map)
        idatas_labeled.append(idata_copy)

    betas = []
    for idata in idatas_labeled:
        b = idata.posterior["beta"].sel(channel=channel_label)
        betas.append(b)
    stacked = xr.concat(betas, dim="model")
    stacked = stacked.assign_coords(model=model_names)
    idata_combined = az.InferenceData(posterior=xr.Dataset({"beta": stacked}))

    n_models = len(model_names)
    figsize = (figsize[0], min(figsize[1], 2 + 0.8 * n_models))

    axes = az.plot_forest(
        idata_combined,
        var_names=["beta"],
        combined=True,
        hdi_prob=hdi_prob,
        kind="forestplot",
        figsize=figsize,
        legend=False,
        colors="C1",  # Same posterior color for all rows
    )
    ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
    ax.set_title("Coefficients: posteriors across models")
    ax.set_xlim(-0.3, 0.7)
    ax.set_xlabel("Effect size (beta)")

    # Alternating gray/white row backgrounds (equal-height rows from yticks)
    yticks = ax.get_yticks()
    if len(yticks) >= n_models:
        row_height = yticks[1] - yticks[0] if len(yticks) > 1 else 1.0
        boundaries = [yticks[0] - row_height / 2]
        for i in range(len(yticks) - 1):
            boundaries.append((yticks[i] + yticks[i + 1]) / 2)
        boundaries.append(yticks[-1] + row_height / 2)
        for i in range(n_models):
            if i < len(boundaries) - 1:
                # Top row (highest i) gray; alternate with white
                color = (0.93, 0.93, 0.93) if i % 2 == 1 else (1, 1, 1)
                ax.axhspan(boundaries[i], boundaries[i + 1], facecolor=color, zorder=-1)

    if truth is not None and "beta" in truth:
        beta_true = np.asarray(truth["beta"]).ravel()
        channels = list(channel_map.keys())
        channel_labels_list = [channel_map.get(c, c) for c in channels]
        try:
            idx = channel_labels_list.index(channel_label)
        except ValueError:
            idx = channels.index(channel_label) if channel_label in channels else 9
        true_val = float(beta_true[idx])

        yticks = ax.get_yticks()
        if len(yticks) >= n_models:
            y_positions = list(yticks[:n_models])
        else:
            y_positions = list(yticks[:n_models]) if len(yticks) > 0 else [float(np.mean(ax.get_ylim()))]

        scatter = ax.scatter(
            [true_val] * len(y_positions),
            y_positions,
            color="black",
            zorder=5,
            s=25,
            label="True value",
        )
        # Custom legend: Posterior + True value only
        posterior_handle = mlines.Line2D(
            [], [], color="C1", linewidth=4, solid_capstyle="round", label="Posterior"
        )
        ax.legend(
            handles=[posterior_handle, scatter],
            labels=["Posterior", "True value"],
            loc="upper left",
            bbox_to_anchor=(1.02, 0.98),
            frameon=False,
        )
        fig = ax.get_figure()
        fig.subplots_adjust(right=0.82)

    return axes
