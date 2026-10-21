"""All report figures, drawn only from the result tables in reports/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icumort import plotting
from icumort.plotting import INK, INK_2, MUTED, SERIES, model_color

MAIN = "A+B"
GRID_LINE = "#d4d2ca"
# Takeaway titles, written after reading the results of the full run.
TITLES = {
    "data": "The parsed files match the published cohort; only the invasive pressures differ from the paper's table",
    "models": "XGBoost on summary statistics scores above every recurrent net on set C; the GRU trails by 0.011 AUROC",
    "calibration": "All learned models are calibrated on set C, but Event 2 varies up to threefold between GRU seeds",
    "challenge": "XGBoost matches the 2012 winner on Event 1 and would have placed third even with set A alone",
    "missingness": "When a variable was measured is itself predictive: counts and timing alone reach AUROC 0.79",
    "explanations": "Both model families rely on the Glasgow Coma Score first; their rankings agree (rho 0.80)",
    "dependence": "Low GCS, older age, low urine output and high BUN raise XGBoost's predicted risk most",
    "subgroups": "XGBoost and GRU discriminate in every subgroup (AUROC 0.81-0.91); risk is underestimated most at 80+",
    "training": ("The recurrent nets stop within 30 epochs; single-seed AUROC SD is at most 0.002 except for "
                 "process-only"),
}
FEATURE_LABEL = {"first": "first", "last": "last", "min": "min", "max": "max", "mean": "mean", "std": "SD",
                 "n": "count", "t_first": "hour of first", "t_last": "hour of last"}


def _suptitle(fig: plt.Figure, text: str, gap_in: float = 0.3) -> None:
    """Left-aligned takeaway title placed ``gap_in`` inches above the top of the axes."""
    fig.suptitle(text, x=0.01, y=fig.subplotpars.top + gap_in / fig.get_figheight(), ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color=INK)


def _feature_label(f: str) -> str:
    if "__" not in f:
        return f
    var, stat = f.split("__")
    return f"{var} ({FEATURE_LABEL.get(stat, stat)})"


def fig_data(avail: pd.DataFrame, cohort: pd.DataFrame, path: Path, title: str) -> Path:
    """Share of stays with at least one measurement per variable (vs the paper) and mortality per record set."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 7.2), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.35})
    a = avail.sort_values("raw_pct").reset_index(drop=True)
    y = np.arange(len(a))
    ax.hlines(y, 0, a["raw_pct"], color=SERIES[0], lw=1.2, alpha=0.35)
    ax.scatter(a["raw_pct"], y, color=SERIES[0], s=28, zorder=3, label="This project (raw files)")
    ax.scatter(a["published_pct"], y, facecolor="none", edgecolor=SERIES[1], s=60, lw=1.4, zorder=4,
               label="Silva et al. (2012), Table 1")
    ax.set_yticks(y)
    ax.set_yticklabels(a["variable"], fontsize=8.5)
    ax.set_xlim(0, 128)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Stays with at least one measurement in 48 h (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    off = a.index[a["diff_raw_vs_published"].abs() > 1]
    if len(off):
        ax.text(101.5, off.to_numpy().mean(), "paper lists\n98.4% for the\ninvasive pressures", va="center",
                fontsize=8, color=INK_2)
    ax.set_title("Measurement coverage per variable vs the published table", loc="left")
    c = cohort.set_index("set")
    bars = ax2.bar(c.index, 100 * c["mortality"], color=[SERIES[0], SERIES[0], SERIES[6]], width=0.6)
    for b, (_, r) in zip(bars, c.iterrows()):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f"{100 * r['mortality']:.1f}%\n{int(r['deaths'])} / {int(r['stays']):,}", ha="center", va="bottom",
                 fontsize=8.5, color=INK_2)
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("In-hospital mortality (%)")
    ax2.set_xlabel("Record set")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Sets A + B develop, set C tests", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def _ci_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, label: str, ref: float | None, deep: str) -> None:
    for i, (_, r) in enumerate(df.iterrows()):
        c = model_color(r["model"])
        ax.errorbar(r[metric], i, xerr=[[r[metric] - r[f"{metric}_ci_low"]], [r[f"{metric}_ci_high"] - r[metric]]],
                    fmt="o", color=c, mfc="white" if "only" in r["model"] else c, ms=7, capsize=3, lw=1.6)
        ax.text(r[f"{metric}_ci_high"], i, f"  {r[metric]:.3f}", va="center", fontsize=8.5, color=INK_2)
    if ref is not None:
        ax.axvline(ref, color=model_color("XGBoost"), lw=1, ls=":")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["model"].replace({"Blend": f"Blend (mean of XGBoost and {deep})"}), fontsize=9)
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)


def fig_models(res: pd.DataFrame, deep: str, path: Path, title: str) -> Path:
    """AUROC and AUPRC on set C with 95% bootstrap CIs, models trained on sets A + B."""
    r = res[res["regime"] == MAIN].sort_values("auroc")
    ref = r.set_index("model").loc["XGBoost"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, gridspec_kw={"wspace": 0.08})
    _ci_panel(axes[0], r, "auroc", "AUROC on set C (95% bootstrap CI)", ref["auroc"], deep)
    _ci_panel(axes[1], r, "auprc", "AUPRC on set C (95% bootstrap CI)", ref["auprc"], deep)
    axes[0].set_title("Discrimination (open markers: input ablations)", loc="left")
    axes[1].set_title(f"Precision-recall (prevalence {r['observed_rate'].iloc[0]:.1%})", loc="left")
    for ax in axes:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.08 * (hi - lo))
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_calibration(rel: pd.DataFrame, res: pd.DataFrame, seeds: pd.DataFrame, path: Path, title: str) -> Path:
    """Reliability per risk decile (the Event 2 groups), calibration error per decile, and Event 2 with seed spread."""
    r = res[res["regime"] == MAIN].set_index("model")
    models = [m for m in ("XGBoost", "GRU", "GRU-D", "LSTM", "Logistic regression", "SAPS-I") if m in r.index]
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.8),
                                       gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.1]})
    for m in ("XGBoost", "GRU", "SAPS-I"):
        g = rel[(rel["regime"] == MAIN) & (rel["model"] == m) & (rel["variant"] == "raw")]
        ax.plot(100 * g["mean_risk"], 100 * g["observed_rate"], "o-", color=model_color(m), ms=4.5, lw=1.6,
                label=f"{m} (slope {r.loc[m, 'cal_slope']:.2f})")
        ax1.plot(g["bin"], 100 * (g["observed_rate"] - g["mean_risk"]), "o-", color=model_color(m), ms=4.5, lw=1.6)
    ax.plot([0, 70], [0, 70], color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Mean predicted risk in decile (%)")
    ax.set_ylabel("Observed mortality in decile (%)")
    ax.set_title("Reliability by risk decile", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax1.axhline(0, color=MUTED, lw=1, ls=":")
    ax1.set_xticks(range(1, 11))
    ax1.set_xlabel("Risk decile (1 = lowest predicted risk)")
    ax1.set_ylabel("Observed minus predicted (percentage points)")
    ax1.set_title("Calibration error per decile", loc="left")
    y = np.arange(len(models))[::-1]
    colors = [model_color(m) for m in models]
    ax2.barh(y + 0.2, [r.loc[m, "event2"] for m in models], height=0.38, color=colors, label="risk as trained")
    ax2.barh(y - 0.2, [r.loc[m, "event2_platt"] for m in models], height=0.38, color=colors, alpha=0.4,
             label="after Platt scaling")
    s = seeds[seeds["regime"] == MAIN]
    for yi, m in zip(y, models):
        e = s.loc[s["model"] == m, "event2"]
        if len(e) > 1:
            ax2.plot([e.min(), e.max()], [yi + 0.2, yi + 0.2], color=INK, lw=1.2, marker="|", ms=8)
        ax2.text(r.loc[m, "event2"], yi + 0.36, f"{r.loc[m, 'event2']:.1f}", ha="center", va="bottom", fontsize=7.5,
                 color=INK_2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(models)
    ax2.set_xlabel("Event 2 = H / D on set C (lower is better)")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
               title="black line: range of single seeds", title_fontsize=8)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Challenge calibration score", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_challenge(ch: pd.DataFrame, path: Path, title: str) -> Path:
    """This project's Event 1 and Event 2 on set C next to the published final challenge scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"wspace": 0.62})
    ours_keep = ("XGBoost, trained on sets A+B", "XGBoost, trained on set A", "GRU, trained on sets A+B",
                 "GRU, trained on set A", "SAPS-I, trained on sets A+B")
    for ax, event in zip(axes, ("Event 1", "Event 2")):
        d = ch[(ch["event"] == event) & ~ch["entry"].str.startswith("Random")]
        d = d[d["source"].str.startswith("published") | d["entry"].isin(ours_keep)]
        d = d.sort_values("score", ascending=(event == "Event 2"))
        y = np.arange(len(d))[::-1]
        ours = d["source"].str.startswith("this").to_numpy()
        colors = [model_color(e.split(",")[0]) if o else GRID_LINE for e, o in zip(d["entry"], ours)]
        ax.barh(y, d["score"], color=colors, height=0.7)
        for yi, (_, r), o in zip(y, d.iterrows(), ours):
            lo, hi = r.get("single_seed_min", np.nan), r.get("single_seed_max", np.nan)
            spread = o and pd.notna(lo) and hi > lo
            if spread:
                ax.plot([lo, hi], [yi, yi], color=INK, lw=1.1, marker="|", ms=7)
            ax.text(max(r["score"], hi) if spread else r["score"], yi,
                    f" {r['score']:.3f}" if event == "Event 1" else f" {r['score']:.1f}", va="center", fontsize=7.8,
                    color=INK if o else INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(d["entry"], fontsize=8)
        for lab, o in zip(ax.get_yticklabels(), ours):
            lab.set_color(INK if o else INK_2)
            lab.set_fontweight("bold" if o else "normal")
        ax.grid(axis="y", visible=False)
        if event == "Event 1":
            ax.set_xlabel("Event 1 = min(sensitivity, positive predictivity), higher is better")
            ax.set_xlim(0, 0.66)
            ax.set_title("Event 1 on set C (grey: published; black line: single seeds)", loc="left")
        else:
            ax.set_xlabel("Event 2 = H / D, lower is better (black line: range of single seeds)")
            ax.set_title("Event 2 on set C (risks as trained)", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_missingness(miss: pd.DataFrame, res: pd.DataFrame, path: Path, title: str) -> Path:
    """Mortality of stays with and without a measurement, and the input ablations."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.5})
    m = miss[(miss["share_measured"] > 0.03) & (miss["share_measured"] < 0.97)].copy()
    m["gap"] = m["mortality_if_measured"] - m["mortality_if_not_measured"]
    m = m.sort_values("gap")
    y = np.arange(len(m))
    ax.hlines(y, 100 * m["mortality_if_not_measured"], 100 * m["mortality_if_measured"], color=GRID_LINE, lw=2)
    ax.scatter(100 * m["mortality_if_not_measured"], y, color=SERIES[0], s=30, zorder=3, label="not measured in 48 h")
    ax.scatter(100 * m["mortality_if_measured"], y, color=SERIES[7], s=30, zorder=3, label="measured at least once")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v} ({s:.0%})" for v, s in zip(m["variable"], m["share_measured"])], fontsize=8.5)
    ax.set_xlabel("In-hospital mortality (%), sets A + B")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Mortality by whether a variable was ever measured\n(share of stays measured in brackets)", loc="left")
    r = res[res["regime"] == MAIN].set_index("model")
    rows = [("XGBoost", "full"), ("XGBoost, values only", "values"), ("XGBoost, measurement process only", "process"),
            ("GRU", "full"), ("GRU, values only", "values"), ("GRU, measurement process only", "process")]
    labels = {"full": "values + process", "values": "values only", "process": "measurement process only"}
    y2 = np.array([5.4, 4.4, 3.4, 2, 1, 0])
    for yi, (name, kind) in zip(y2, rows):
        v = r.loc[name]
        c = model_color(name)
        ax2.errorbar(v["auroc"], yi, xerr=[[v["auroc"] - v["auroc_ci_low"]], [v["auroc_ci_high"] - v["auroc"]]],
                     fmt="o", color=c, mfc=c if kind == "full" else "white", ms=7, capsize=3)
        ax2.text(v["auroc_ci_high"], yi, f"  {v['auroc']:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([f"{n.split(',')[0]}: {labels[k]}" for n, k in rows], fontsize=8.5)
    lo = r.loc[[n for n, _ in rows], "auroc_ci_low"].min() - 0.02
    ax2.set_xlim(lo, 0.95)
    ax2.set_xlabel("AUROC on set C (95% bootstrap CI)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Input ablations (process only: measurement counts\nand timing; no values, age, sex or ICU type)",
                  loc="left")
    _suptitle(fig, title, gap_in=0.55)
    plotting.save(fig, path)
    return path


def fig_explanations(shap_var: pd.DataFrame, perm: pd.DataFrame, deep: str, path: Path, title: str) -> Path:
    """XGBoost mean |SHAP| per variable, and grouped permutation importance of XGBoost and the recurrent model."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.0), gridspec_kw={"wspace": 0.4})
    s = shap_var.head(15).iloc[::-1]
    ax.barh(s["variable"], s["mean_abs_shap"], color=model_color("XGBoost"), height=0.65)
    for i, v in enumerate(s["mean_abs_shap"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Mean |SHAP| summed over the variable's features (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("XGBoost: TreeSHAP by variable (set C)", loc="left")
    p = perm.pivot_table(index="variable", columns="model", values="auroc_drop")
    rho = p.corr(method="spearman").iloc[0, 1]
    p = p.loc[p.max(axis=1).sort_values(ascending=False).head(15).index[::-1]]
    y = np.arange(len(p))
    ax2.barh(y + 0.2, p["XGBoost"], height=0.38, color=model_color("XGBoost"), label="XGBoost")
    ax2.barh(y - 0.2, p[deep], height=0.38, color=model_color(deep), label=deep)
    ax2.set_yticks(y)
    ax2.set_yticklabels(p.index)
    ax2.set_xlabel("Drop in AUROC when the variable is shuffled across stays")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(axis="y", visible=False)
    ax2.set_title(f"Grouped permutation importance (Spearman rho {rho:.2f}, 41 variables)", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


UNITS = {"GCS": "Glasgow Coma Score (3-15)", "Age": "years", "Urine": "mL per charted output", "BUN": "mg/dL",
         "Lactate": "mmol/L", "Temp": "degrees C", "Weight": "kg", "Glucose": "mg/dL", "HR": "beats per minute",
         "Creatinine": "mg/dL"}


def fig_dependence(dep: pd.DataFrame, path: Path, title: str) -> Path:
    """TreeSHAP value against feature value for the six most important XGBoost features."""
    feats = list(dict.fromkeys(dep["feature"]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.9), gridspec_kw={"hspace": 0.62, "wspace": 0.28})
    for ax, f in zip(axes.flat, feats):
        g = dep[dep["feature"] == f]
        nan = g["value"].isna()
        ax.scatter(g.loc[~nan, "value"], g.loc[~nan, "shap"], s=5, alpha=0.25, color=model_color("XGBoost"),
                   edgecolor="none")
        ax.axhline(0, color=MUTED, lw=0.8)
        note = (f"\nmissing in {nan.mean():.0%} of stays: mean SHAP {g.loc[nan, 'shap'].mean():+.2f}"
                if nan.any() else "")
        ax.set_title(_feature_label(f) + note, loc="left", fontsize=9.5)
        var, stat = (f.split("__") + [""])[:2]
        unit = "hours since admission" if stat.startswith("t_") else "count" if stat == "n" else UNITS.get(var, "")
        ax.set_xlabel(f"standard deviation within the stay ({unit})" if stat == "std" else unit)
        ax.set_ylabel("SHAP (log-odds)")
        lo, hi = np.nanpercentile(g["value"], [0.5, 99.5])
        ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    _suptitle(fig, title, gap_in=0.65)
    plotting.save(fig, path)
    return path


def fig_subgroups(sg: pd.DataFrame, models: list[str], path: Path, title: str) -> Path:
    """AUROC per subgroup with bootstrap CIs and predicted vs observed mortality."""
    order = {"<50": 0, "50-64": 1, "65-79": 2, "80+": 3}
    groupings = list(dict.fromkeys(sg["grouping"]))
    keys = sorted(dict.fromkeys(zip(sg["grouping"], sg["group"])),
                  key=lambda k: (groupings.index(k[0]), order.get(k[1], 0), k[1]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(keys))[::-1]
    offs = np.linspace(-0.25, 0.25, len(models))
    for off, m in zip(offs, models):
        g = sg[sg["model"] == m].set_index(["grouping", "group"]).loc[keys]
        ax.errorbar(g["auroc"], y + off, xerr=[g["auroc"] - g["auroc_ci_low"], g["auroc_ci_high"] - g["auroc"]],
                    fmt="o", ms=4.5, color=model_color(m), capsize=2, lw=1.2, label=m)
        ax2.scatter(100 * g["mean_risk"], y + off, color=model_color(m), s=22, zorder=3)
    g0 = sg[sg["model"] == models[0]].set_index(["grouping", "group"]).loc[keys]
    ax2.scatter(100 * g0["observed_rate"], y, marker="|", s=260, color=INK, lw=2, zorder=4, label="observed")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k[0]}: {k[1]} (n={n:,}, {d} deaths)" for k, n, d in zip(keys, g0["n"], g0["deaths"])],
                       fontsize=8.5)
    ax.set_xlabel("AUROC on set C (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Discrimination within subgroups", loc="left")
    ax2.set_xlabel("Mortality (%): mean predicted risk (dots) vs observed (bar)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Calibration in the large", loc="left")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = models + ["observed mortality"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, bbox_to_anchor=(0.55, -0.04))
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def fig_training(hist: pd.DataFrame, seeds: pd.DataFrame, path: Path, title: str) -> Path:
    """Validation loss per epoch of every recurrent model (seed 0) and the test AUROC of each seed."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.45})
    h = hist[(hist["regime"] == MAIN) & (hist["seed"] == 0)]
    for m, g in h.groupby("model", sort=False):
        ls = ":" if "process" in m else "--" if "only" in m else "-"
        ax.plot(g["epoch"], g["val_loss"], color=model_color(m), lw=1.6, ls=ls, label=m)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (binary cross-entropy)")
    ax.legend(fontsize=8)
    ax.set_title("Training curves (seed 0; early stopping on this loss)", loc="left")
    s = seeds[seeds["regime"] == MAIN]
    order = s.groupby("model")["auroc"].mean().sort_values().index
    for i, m in enumerate(order):
        v = s.loc[s["model"] == m, "auroc"]
        ax2.scatter(v, np.full(len(v), i), color=model_color(m), s=22, alpha=0.8)
        if len(v) > 1:
            ax2.text(v.max(), i, f"  SD {v.std():.4f}", va="center", fontsize=8, color=INK_2)
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels(order, fontsize=8.5)
    lo, hi = ax2.get_xlim()
    ax2.set_xlim(lo, hi + 0.15 * (hi - lo))
    ax2.set_xlabel("AUROC on set C of each single seed")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Seed-to-seed variation", loc="left")
    _suptitle(fig, title)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figures: Path) -> list[Path]:
    """Draw every figure from the CSV tables in ``results``; returns the written paths."""
    plotting.apply_style()
    figures.mkdir(parents=True, exist_ok=True)
    deep = json.loads((results / "run_metadata.json").read_text())["best_recurrent_model"]
    res = pd.read_csv(results / "model_results.csv")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(results / f"{name}.csv")

    return [
        fig_data(read("variable_availability"), read("cohort_summary"), figures / "00_data.png", TITLES["data"]),
        fig_models(res, deep, figures / "01_model_comparison.png", TITLES["models"]),
        fig_calibration(read("calibration_deciles"), res, read("seed_results"), figures / "02_calibration.png",
                        TITLES["calibration"]),
        fig_challenge(read("challenge_comparison"), figures / "03_challenge.png", TITLES["challenge"]),
        fig_missingness(read("missingness"), res, figures / "04_missingness.png", TITLES["missingness"]),
        fig_explanations(read("shap_variables"), read("permutation_importance"), deep, figures / "05_explanations.png",
                         TITLES["explanations"]),
        fig_dependence(read("shap_dependence"), figures / "06_shap_dependence.png", TITLES["dependence"]),
        fig_subgroups(read("subgroups"), ["XGBoost", deep, "SAPS-I"], figures / "07_subgroups.png",
                      TITLES["subgroups"]),
        fig_training(read("training_history"), read("seed_results"), figures / "08_training.png", TITLES["training"]),
    ]
