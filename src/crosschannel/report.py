"""Render-safe reporting for the cross-channel MMM post.

Loads the committed posterior (a plain xarray Dataset of structural parameters)
and the pickled data, then recomputes contributions, ROAS and response curves
with numpy so nothing here depends on PyMC / pymc-marketing at render time.

Posterior summaries use the median with a 94% highest-density interval (HDI)
throughout, matching the rest of the site's Bayesian reporting.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from .transforms import transformed_media_draws

HDI_PROB = 0.94
L_MAX = 12

# Semantic colours (kept minimal and colour-blind friendly)
C_REFERENCE = "#4C72B0"
C_INTERACTION = "#DD8452"
C_REALISED = "#55A868"
C_ACTUAL = "#1f4e79"
C_PRED = "#c44e52"


# --------------------------------------------------------------------------- IO
def load_assets(assets_dir):
    assets_dir = Path(assets_dir)
    post = xr.open_dataset(assets_dir / "posterior.nc")
    with open(assets_dir / "data.pkl", "rb") as f:
        data = pickle.load(f)
    return post, data


# ---------------------------------------------------------------- summaries
def hdi(samples, prob: float = HDI_PROB):
    """Highest-density interval of a 1-D sample."""
    x = np.sort(np.asarray(samples).ravel())
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    k = int(np.floor(prob * n))
    if k == 0:
        return float(x[0]), float(x[-1])
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def _labels(data):
    names = data["channel_names"]
    lab = data.get("channel_labels", {})
    return [lab.get(n, n) for n in names]


# ------------------------------------------------------------- core totals
def compute_contributions(post, data=None):
    """Per-draw channel contribution totals (actual GBP) from the fitted model.

    Reads the model's own window-summed deterministics saved at fit time, so the
    reported figures match the fitted model exactly.

    Returns a dict of (S, C) arrays: reference, realised (observed) and
    interaction-increment totals.
    """
    reference_tot = post["reference_contribution_tot"].values         # (S, C)
    observed_tot = post["realised_mix_contribution_tot"].values       # (S, C)
    interaction_tot = observed_tot - reference_tot                    # draw-by-draw increment
    return {
        "reference_tot": reference_tot,
        "observed_tot": observed_tot,
        "interaction_tot": interaction_tot,
    }


# --------------------------------------------------------------------- tables
def _fmt_gbp_m(v):
    return f"£{v / 1e6:,.2f}m"


def contribution_table(post, data, contribs=None):
    """Contribution decomposition table (medians; realised = reference + increment)."""
    if contribs is None:
        contribs = compute_contributions(post, data)
    labels = _labels(data)
    ref = contribs["reference_tot"]
    obs = contribs["observed_tot"]

    # All derived quantities are formed draw by draw, then summarised, so the
    # increment reconciles exactly and the share is a genuine posterior summary.
    inc = obs - ref                                        # (S, C) interaction increment
    share_draws = 100.0 * obs / obs.sum(axis=1, keepdims=True)  # (S, C) realised-mix share

    ref_med = np.median(ref, axis=0)
    inc_med = np.median(inc, axis=0)
    obs_med = ref_med + inc_med                            # reconciles: mix = ref + increment
    share_med = np.median(share_draws, axis=0)
    obs_hdi = np.array([hdi(obs[:, k]) for k in range(obs.shape[1])])

    rows = []
    for k, name in enumerate(labels):
        rows.append({
            "Channel": name,
            "Reference contribution": _fmt_gbp_m(ref_med[k]),
            "Interaction increment": _fmt_gbp_m(inc_med[k]),
            "Realised-mix contribution": _fmt_gbp_m(obs_med[k]),
            "Realised-mix 94% HDI": f"[{obs_hdi[k, 0] / 1e6:,.2f}, {obs_hdi[k, 1] / 1e6:,.2f}]m",
            "Realised-mix share": f"{share_med[k]:.1f}%",
        })
    return pd.DataFrame(rows)


def roas_table(post, data, contribs=None):
    """Reference vs realised ROAS per channel (median [94% HDI]); raw-spend denominator."""
    if contribs is None:
        contribs = compute_contributions(post, data)
    labels = _labels(data)
    total_spend = np.asarray(data["media_actual"].sum(axis=0).values, dtype=float)  # (C,)

    ref_roas = contribs["reference_tot"] / total_spend  # (S, C)
    obs_roas = contribs["observed_tot"] / total_spend
    diff = obs_roas - ref_roas

    rows = []
    for k, name in enumerate(labels):
        rl, rh = hdi(ref_roas[:, k])
        ol, oh = hdi(obs_roas[:, k])
        rows.append({
            "Channel": name,
            "Reference ROAS": f"{np.median(ref_roas[:, k]):.2f} [{rl:.2f}, {rh:.2f}]",
            "Realised-mix ROAS": f"{np.median(obs_roas[:, k]):.2f} [{ol:.2f}, {oh:.2f}]",
            "Interaction difference": f"{np.median(diff[:, k]):+.2f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- plots
def plot_actual_vs_predicted(post, data, figsize=(11, 4.5)):
    dates = data["media_actual"].index
    y_actual = np.asarray(data["y_actual"].values, dtype=float)
    y_pred = post["y_mean"].mean("sample").values

    mape = float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100)
    r2 = float(1 - np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - y_actual.mean()) ** 2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, y_actual, label="Actual", color=C_ACTUAL, linewidth=1.6)
    ax.plot(dates, y_pred, label="Modelled (posterior mean)", color=C_PRED, linewidth=1.6, linestyle="--")
    ax.set_title("Actual vs modelled weekly sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (£)")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.99, 0.05, f"MAPE = {mape:.2f}%\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    fig.tight_layout()
    return fig, {"mape": mape, "r2": r2}


def plot_interaction_heatmap(post, data, figsize=(6.6, 5.4)):
    """Posterior-median directional interaction matrix.

    Rows = affected channel (i); columns = modifier channel (j).
    Cell = median gamma_ij (log-multiplier). Diagonal is masked (no self-interaction).
    """
    labels = _labels(data)
    g = post["gamma_eff"].values  # (S, i, j)
    g_med = np.median(g, axis=0)
    p_pos = np.mean(g > 0, axis=0)
    n = g_med.shape[0]

    masked = np.ma.array(g_med, mask=np.eye(n, dtype=bool))
    vmax = np.max(np.abs(masked)) if masked.count() else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e9e9e9")
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Modifier channel  (j — provides the effect)")
    ax.set_ylabel("Affected channel  (i — effectiveness changes)")
    ax.set_title(r"Directional interaction $\gamma_{ij}$ (point estimate)")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#888", fontsize=11)
                continue
            val = g_med[i, j]
            txt = f"{val:+.2f}\nP(>0)={p_pos[i, j]:.2f}"
            colour = "white" if abs(val) > 0.55 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", color=colour, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log-multiplier on effectiveness")
    fig.tight_layout()
    return fig


def plot_contribution_decomposition(post, data, contribs=None, figsize=(9.5, 5)):
    if contribs is None:
        contribs = compute_contributions(post, data)
    labels = _labels(data)
    ref = np.median(contribs["reference_tot"], axis=0) / 1e6
    inc = np.median(contribs["interaction_tot"], axis=0) / 1e6  # draw-by-draw increment
    obs = ref + inc                                             # reconciles with the table

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, ref, color=C_REFERENCE, label="Reference contribution")
    ax.bar(x, np.clip(inc, 0, None), bottom=ref, color=C_INTERACTION, label="Interaction increment (+)")
    neg = np.clip(inc, None, 0)
    ax.bar(x, neg, bottom=ref, color="#B5651D", alpha=0.75, label="Interaction increment (−)", hatch="//")
    for k in range(len(labels)):
        ax.text(x[k], max(ref[k], obs[k]), f" £{obs[k]:,.2f}m", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Contribution over the window (£m)")
    ax.set_title("Reference + interaction = realised-mix contribution")
    ax.legend(loc="upper right", frameon=False)
    ax.margins(y=0.14)
    fig.tight_layout()
    return fig


def plot_reference_vs_realised_roas(post, data, contribs=None, figsize=(9.5, 5)):
    if contribs is None:
        contribs = compute_contributions(post, data)
    labels = _labels(data)
    total_spend = np.asarray(data["media_actual"].sum(axis=0).values, dtype=float)
    ref_roas = contribs["reference_tot"] / total_spend
    obs_roas = contribs["observed_tot"] / total_spend

    ref_med = np.median(ref_roas, axis=0)
    obs_med = np.median(obs_roas, axis=0)
    ref_err = np.abs(np.array([hdi(ref_roas[:, k]) for k in range(len(labels))]).T - ref_med)
    obs_err = np.abs(np.array([hdi(obs_roas[:, k]) for k in range(len(labels))]).T - obs_med)

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, ref_med, w, yerr=ref_err, capsize=3, color=C_REFERENCE, label="Reference ROAS")
    ax.bar(x + w / 2, obs_med, w, yerr=obs_err, capsize=3, color=C_REALISED, label="Realised-mix ROAS")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ROAS (£ return per £ spend)")
    ax.set_title("Reference vs realised-mix ROAS (point estimate, 94% HDI)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def _steady_state_transform(spend, hist_max_k, alpha_k, b_k, c_k, l_max=L_MAX):
    """Transformed media for a channel at constant weekly `spend`, per draw.

    spend: scalar or (G,) array. alpha_k,b_k,c_k: (S,) arrays. Returns (S,) or (S,G).
    """
    x = np.asarray(spend, dtype=float) / hist_max_k
    ss = sum(alpha_k ** lag for lag in range(l_max + 1))  # (S,)
    bc = b_k * c_k  # (S,)
    if x.ndim == 0:
        return b_k * np.tanh(x * ss / bc)
    u = x[None, :] * (ss / bc)[:, None]
    return b_k[:, None] * np.tanh(u)


def plot_response_curves(post, data, focal=0, modifier=1, figsize=(9.5, 5.2)):
    """Steady-state response curve for the focal channel under three environments.

    Estimand: each curve is the focal channel's modelled weekly contribution if it
    spent a constant amount every week until its geometric adstock reaches steady
    state, i.e. transformed media m = b*tanh(x*sum_l alpha^l / (b*c)). Modifier
    channels enter only through the interaction multiplier, each likewise held at a
    constant weekly spend until steady state:
      - reference:  all modifiers off (multiplier = 1);
      - average:    every other channel at its historical-mean weekly spend;
      - strong:     the named modifier at its 90th-percentile weekly spend, others at mean.
    """
    labels = _labels(data)
    media_actual = data["media_actual"].values  # (T, C)
    hist_max = media_actual.max(axis=0)
    hist_mean = media_actual.mean(axis=0)
    hist_p90 = np.percentile(media_actual, 90, axis=0)
    target_mean = float(data["target_mean"])
    n_ch = media_actual.shape[1]

    alpha = post["adstock_alpha"].values
    b = post["saturation_b"].values
    c = post["saturation_c"].values
    beta = post["beta"].values
    gamma_eff = post["gamma_eff"].values  # (S, i, j)

    x_grid = np.linspace(0.0, float(hist_max[focal]), 160)
    m_focal = _steady_state_transform(x_grid, hist_max[focal], alpha[:, focal], b[:, focal], c[:, focal])  # (S,G)

    def curve(modifier_spends: dict):
        log_mult = np.zeros((alpha.shape[0], x_grid.size))
        for j, spend_j in modifier_spends.items():
            m_j = _steady_state_transform(spend_j, hist_max[j], alpha[:, j], b[:, j], c[:, j])  # (S,)
            log_mult = log_mult + (gamma_eff[:, focal, j] * m_j)[:, None]
        contrib = beta[:, focal][:, None] * m_focal * np.exp(log_mult) * target_mean  # (S,G)
        return np.median(contrib, axis=0), hdi_band(contrib)

    def hdi_band(arr):
        lo = np.empty(arr.shape[1]); hi = np.empty(arr.shape[1])
        for g in range(arr.shape[1]):
            lo[g], hi[g] = hdi(arr[:, g])
        return lo, hi

    others = [j for j in range(n_ch) if j != focal]
    env_none = {}
    env_avg = {j: hist_mean[j] for j in others}
    env_strong = {j: (hist_p90[modifier] if j == modifier else hist_mean[j]) for j in others}

    fig, ax = plt.subplots(figsize=figsize)
    specs = [
        (env_none, "Reference curve (interaction multiplier = 1)", "#7f7f7f", "--"),
        (env_avg, "Other channels at historical-mean weekly spend", C_REFERENCE, "-"),
        (env_strong, f"{labels[modifier]} at its 90th-percentile weekly spend", C_REALISED, "-"),
    ]
    for env, lab, col, ls in specs:
        med, (lo, hi) = curve(env)
        ax.plot(x_grid / 1e3, med, color=col, linestyle=ls, linewidth=1.9, label=lab)
        if ls != "--":
            ax.fill_between(x_grid / 1e3, lo, hi, color=col, alpha=0.15)

    ax.set_xlabel(f"{labels[focal]} steady-state weekly spend (£000s)")
    ax.set_ylabel(f"{labels[focal]} weekly contribution (£)")
    ax.set_title(f"{labels[focal]}: a family of response curves, one per media environment")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# -------------------------------------------- directional interaction helpers
def _offdiag_pairs(n):
    """Directional off-diagonal (affected i, modifier j) pairs, row-major."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


# --------------------------------------------- prior-predictive multiplier
def _mean_transformed_media(post, data):
    """Representative per-channel transformed media using posterior-mean transforms."""
    media_scaled = np.asarray(data["media_scaled"].values, dtype=float)  # (T, C)
    alpha = post["adstock_alpha"].values.mean(axis=0, keepdims=True)     # (1, C)
    b = post["saturation_b"].values.mean(axis=0, keepdims=True)
    c = post["saturation_c"].values.mean(axis=0, keepdims=True)
    m = transformed_media_draws(media_scaled, alpha, b, c, l_max=L_MAX)[0]  # (T, C)
    return m.mean(axis=0)  # (C,)


def plot_prior_multiplier(post, data, focal=0, sigma_gamma=0.40, n_draws=40000,
                          seed=0, figsize=(9.5, 4.6)):
    """Prior-predictive interaction multiplier M as more modifiers become active.

    Draws gamma ~ N(0, sigma_gamma) per directional pair and evaluates
    M = exp(sum_j gamma_ij * m_j) with each modifier held at its representative
    (historical-mean) transformed media. Shows how the prior over the *total*
    multiplier widens as the number of simultaneously active modifiers grows.
    """
    labels = _labels(data)
    n = len(labels)
    m_bar = _mean_transformed_media(post, data)  # (C,)
    modifiers = [j for j in range(n) if j != focal]

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize)
    data_by_k, positions = [], []
    for k in range(1, len(modifiers) + 1):
        active = modifiers[:k]
        gam = rng.normal(0.0, sigma_gamma, size=(n_draws, k))
        log_m = gam @ m_bar[active]
        data_by_k.append(np.exp(log_m))
        positions.append(k)

    parts = ax.violinplot(data_by_k, positions=positions, showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_INTERACTION); pc.set_alpha(0.5)
    for k, samp in zip(positions, data_by_k):
        lo, hi = np.percentile(samp, [2.5, 97.5])
        ax.plot([k, k], [lo, hi], color="black", lw=1.4)
        ax.plot(k, np.median(samp), "o", color="black", ms=4)
    ax.axhline(1.0, color="grey", ls=":", lw=0.9)
    ax.set_xlabel(f"Number of modifiers active on {labels[focal]} (each at historical-mean media)")
    ax.set_ylabel("Prior interaction multiplier  $M$")
    ax.set_title(rf"Prior-predictive $M$ under $\gamma\sim\mathcal{{N}}(0,{sigma_gamma})$ (black bars: 95%)")
    ax.set_xticks(positions)
    fig.tight_layout()
    return fig


def prior_multiplier_summary(post, data, focal=0, sigma_gamma=0.40, n_draws=40000, seed=0):
    """Companion numbers for the prior-predictive multiplier plot (95% interval of M)."""
    n = len(_labels(data))
    m_bar = _mean_transformed_media(post, data)
    modifiers = [j for j in range(n) if j != focal]
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(1, len(modifiers) + 1):
        gam = rng.normal(0.0, sigma_gamma, size=(n_draws, k))
        M = np.exp(gam @ m_bar[modifiers[:k]])
        lo, hi = np.percentile(M, [2.5, 97.5])
        rows.append({"Active modifiers": k,
                     "Prior 95% interval for M": f"[{lo:.2f}, {hi:.2f}]"})
    return pd.DataFrame(rows)


# ------------------------------------------- per-pair interaction increments
def _recompute_mix(post, data):
    """Draw-by-draw reference / realised-mix per (date, channel), plus m and params."""
    media_scaled = np.asarray(data["media_scaled"].values, dtype=float)
    target_mean = float(data["target_mean"])
    alpha = post["adstock_alpha"].values
    b = post["saturation_b"].values
    c = post["saturation_c"].values
    beta = post["beta"].values
    gamma_eff = post["gamma_eff"].values  # (S, i, j)
    m = transformed_media_draws(media_scaled, alpha, b, c, l_max=L_MAX)  # (S, T, C)
    ref = m * beta[:, None, :]                                          # (S, T, C)
    log_all = np.einsum("stj,sij->sti", m, gamma_eff)                   # (S, T, C)
    mix = ref * np.exp(log_all)
    return m, ref, mix, log_all, gamma_eff, target_mean


def interaction_pairs_table(post, data, top=6):
    """Leave-one-out realised-mix increment attributable to each directional pair.

    For modifier j acting on affected i, the increment is the realised-mix
    contribution of i minus what it would be with j alone removed from the
    multiplier, summed over the window. These do NOT sum to a channel's total
    increment, because the multiplier is non-additive across modifiers.
    """
    labels = _labels(data)
    m, ref, mix, log_all, gamma_eff, target_mean = _recompute_mix(post, data)
    n = len(labels)
    rows = []
    for i, j in _offdiag_pairs(n):
        log_wo = log_all[:, :, i] - gamma_eff[:, i, j][:, None] * m[:, :, j]
        mix_wo = ref[:, :, i] * np.exp(log_wo)
        inc = (mix[:, :, i] - mix_wo).sum(axis=1) * target_mean  # (S,)
        g = post["gamma_eff"].values[:, i, j]
        rows.append({
            "Modifier (j)": labels[j],
            "Affected (i)": labels[i],
            "γ (median [94% HDI])": f"{np.median(g):+.2f} [{hdi(g)[0]:+.2f}, {hdi(g)[1]:+.2f}]",
            "Increment (£m)": np.median(inc) / 1e6,
            "_abs": abs(np.median(inc)),
        })
    df = pd.DataFrame(rows).sort_values("_abs", ascending=False).drop(columns="_abs").head(top)
    df["Increment (£m)"] = df["Increment (£m)"].map(lambda v: f"£{v:+,.2f}m")
    return df.reset_index(drop=True)
