"""Run every experiment end to end.

Downloads and checks the PhysioNet/CinC Challenge 2012 data, builds summary features and hourly tensors, trains the
tabular models (XGBoost with Optuna, logistic regression, SAPS-I and SOFA recalibrations) and the recurrent nets
(LSTM, GRU, GRU-D and two input ablations) under two development regimes (sets A+B, and set A alone as in the
challenge), evaluates everything on set C, and writes every table in reports/results/ and every figure in
reports/figures/. Trained weights and predictions are cached in data/cache/ under keys that include every setting,
so re-runs only recompute what changed.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test: 500 stays per set; outputs in data/cache/smoke/
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from icumort import data as D  # noqa: E402
from icumort.explain import grouped_permutation_importance, shap_by_feature, shap_by_variable  # noqa: E402
from icumort.features import (  # noqa: E402
    feature_set, feature_variable, hourly, Scaler, sequence_tensors, static_matrix, summary_features,
)
from icumort.figures import make_all  # noqa: E402
from icumort.metrics import (  # noqa: E402
    BOOT_METRICS, auprc, auroc, bootstrap_indices, bootstrap_metrics, brier, event1, event1_threshold, event2, holm,
    paired_p, platt_apply, platt_fit, reliability, summarize,
)
from icumort.models import ModelSpec, TrainConfig, build, predict_logits, to_device, train  # noqa: E402
from icumort.models import tune as tune_rnn  # noqa: E402
from icumort.tabular import (  # noqa: E402
    logreg_fit, score_model_fit, score_model_predict, tune_xgb, xgb_contributions, xgb_fit, xgb_predict,
)
from icumort.tracking import init_mlflow, log_result  # noqa: E402

torch.set_num_threads(2)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
VERSION = "v1"  # bump to invalidate every cached model and prediction

DEEP_SPECS = (
    ModelSpec("LSTM", "lstm"),
    ModelSpec("GRU", "gru"),
    ModelSpec("GRU-D", "grud"),
    ModelSpec("GRU, values only", "gru", use_process=False),
    ModelSpec("GRU, measurement process only", "gru", use_values=False, use_static=False),
)
FULL_DEEP = ("LSTM", "GRU", "GRU-D")  # candidates for "best recurrent model", chosen on validation AUROC
XGB_SETS = {"XGBoost": "full", "XGBoost, values only": "values", "XGBoost, measurement process only": "process"}
MAIN, CHALLENGE = "A+B", "A"
REGIMES = {MAIN: ("A", "B"), CHALLENGE: ("A",)}  # development sets; set C is always the test set
REFERENCE = "XGBoost"

# Final set C results of the challenge (PhysioNet challenge-2012 page, "Challenge Results"; Silva et al., 2012).
PUBLISHED_EVENT1 = [
    ("Johnson et al.", 0.5353), ("Citi and Barbieri", 0.5345), ("Vairavan et al.", 0.5009), ("Macas et al.", 0.4928),
    ("Xia et al.", 0.4923), ("Hamilton and Hamilton", 0.4872), ("Arzeno et al.", 0.4821), ("Chia et al.", 0.4564),
    ("Pantelopoulos", 0.4544), ("Bera and Nayak", 0.4513), ("SAPS-I sample entry (m-code)", 0.3125),
    ("SAPS-I sample entry (C)", 0.3097), ("Random, 86% predicted to survive", 0.1386),
]
PUBLISHED_EVENT2 = [
    ("Citi and Barbieri", 17.88), ("Kang et al.", 20.58), ("Macas et al.", 24.70), ("Ezeozue", 24.93),
    ("Johnson et al.", 29.86), ("Kayaalp", 36.38), ("Pollard and Martinez", 38.23), ("Bera and Nayak", 45.01),
    ("Bosnjak and Montilla", 48.61), ("Arzeno et al.", 51.69), ("SAPS-I sample entry (m-code)", 68.58),
    ("SAPS-I sample entry (C)", 35.21), ("Random, 86% predicted to survive", 10137.7),
]
# Unofficial set A scores of the organisers' SAPS-I sample entry, as stated on the challenge page.
SAMPLE_ENTRY_SET_A = {"event1": 0.296, "event2": 68.0}
STATIC_GROUPS = {"Age": [0, 11], "Gender": [1, 8], "Height": [2, 9], "Weight": [3, 10], "ICUType": [4, 5, 6, 7]}
SIG_PAIRS = [(m, REFERENCE) for m in ("LSTM", "GRU", "GRU-D", "Blend", "Logistic regression", "SAPS-I", "SOFA")] + [
    ("GRU-D", "GRU"), ("GRU", "LSTM"), ("GRU", "GRU, values only"), (REFERENCE, "XGBoost, values only")]
SIG_METRICS = ("auroc", "auprc", "brier")


@dataclass(frozen=True)
class Settings:
    """Splits, budgets and evaluation settings of one run."""

    split_seed: int = 0
    val_share: float = 0.2
    optuna_trials: int = 40
    optuna_trials_challenge: int = 25
    rnn_trials: int = 20
    rnn_trials_challenge: int = 12
    xgb_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    deep_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train: TrainConfig = field(default_factory=TrainConfig)  # base settings; the Optuna search overrides five of them
    n_boot: int = 1000
    perm_repeats: int = 3
    shap_check_rows: int = 200
    subsample: int = 0  # quick mode: stays kept per set


QUICK = Settings(optuna_trials=3, optuna_trials_challenge=2, rnn_trials=2, rnn_trials_challenge=2, xgb_seeds=(0,),
                 deep_seeds=(0,),
                 train=TrainConfig(max_epochs=2), n_boot=30, perm_repeats=1, shap_check_rows=50, subsample=500)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock seconds per stage; GPU stages are listed separately."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.gpu: set[str] = set()

    def run(self, name: str, fn: Callable, gpu: bool = False):
        t0 = time.perf_counter()
        out = fn()
        if gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.gpu.add(name)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out


def key_of(spec: dict) -> str:
    """Short hash of everything that changes an output."""
    return hashlib.sha1(json.dumps({**spec, "version": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached_arrays(name: str, spec: dict, fn: Callable[[], dict]) -> dict:
    """Arrays returned by ``fn``, stored as .npz under a hash of ``spec``."""
    path = CACHE / "preds" / f"{name}-{key_of(spec)}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


@dataclass
class Pred:
    """Validation and test risks of one model (ensemble mean) and of each seed."""

    val: np.ndarray
    test: np.ndarray
    val_seeds: list[np.ndarray] = field(default_factory=list)
    test_seeds: list[np.ndarray] = field(default_factory=list)


def ensemble(parts: list[dict]) -> Pred:
    return Pred(np.mean([p["val"] for p in parts], axis=0), np.mean([p["test"] for p in parts], axis=0),
                [p["val"] for p in parts], [p["test"] for p in parts])


def cohort_summary(stays: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Size, outcome and case mix of each record set."""
    rows = []
    for s, g in stays.groupby("set"):
        f = feats.loc[g.index]
        n_meas = f[[c for c in f.columns if c.endswith("__n")]].sum(axis=1)
        rows.append({"set": s, "stays": len(g), "deaths": int(g["died"].sum()), "mortality": g["died"].mean(),
                     "age_mean": g["Age"].mean(), "female_share": (g["Gender"] == 0).mean(),
                     **{f"icu_{D.ICU_TYPES[k].lower().replace(' ', '_')}": (g["ICUType"] == k).mean()
                        for k in D.ICU_TYPES},
                     "saps1_mean": g.loc[g["saps1"] >= 0, "saps1"].mean(),
                     "mechvent_share": (f["MechVent__n"] > 0).mean(), "measurements_median": n_meas.median(),
                     "stays_without_time_series": int((n_meas == 0).sum())})
    return pd.DataFrame(rows)


def missingness_table(stays: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Per variable (development sets A and B only): how often it is measured and mortality by measured or not."""
    rows = []
    y = stays["died"]
    for v in D.TIME_SERIES:
        n = feats.loc[stays.index, f"{v}__n"]
        measured = n > 0
        rows.append({"variable": v, "share_measured": measured.mean(),
                     "mortality_if_measured": y[measured].mean() if measured.any() else np.nan,
                     "mortality_if_not_measured": y[~measured].mean() if (~measured).any() else np.nan,
                     "measurements_per_stay_survivors": n[measured & (y == 0)].mean(),
                     "measurements_per_stay_deaths": n[measured & (y == 1)].mean(),
                     "auroc_of_count": auroc(n.to_numpy(), y.to_numpy())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on 500 stays per set with tiny budgets")
    args = ap.parse_args()
    S = QUICK if args.quick else Settings()
    results = CACHE / "smoke" / "results" if args.quick else ROOT / "reports" / "results"
    figures = CACHE / "smoke" / "figures" if args.quick else ROOT / "reports" / "figures"
    results.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = datetime.now(timezone.utc)
    if not args.quick:
        init_mlflow(ROOT)

    # ------------------------------------------------------------------ data and checks
    timer.run("download", lambda: D.download(RAW))
    cohort = timer.run("parse", lambda: D.load(RAW, CACHE))
    data_key = key_of({"files": D.FILES, "parser": D.PARSER_VERSION, "ranges": D.PLAUSIBLE})
    checks = D.crosscheck(cohort)
    log(f"published statistics matched: {int(checks.match.sum())}/{len(checks)}")
    checks.to_csv(results / "data_crosscheck.csv", index=False)
    D.availability_table(cohort).to_csv(results / "variable_availability.csv", index=False)
    cohort.quality.to_csv(results / "data_quality.csv", index=False)

    sample = D.read_sample_entry(RAW)
    outcomes_a = cohort.stays.loc[sample.index, "died"].to_numpy()
    e1, se, ppv = event1(sample["prediction"].to_numpy(), outcomes_a)
    e2, h, d = event2(sample["risk"].to_numpy(), outcomes_a, sample.index.to_numpy())
    scorer = pd.DataFrame([
        {"score": "Event 1", "published_set_a": SAMPLE_ENTRY_SET_A["event1"], "this_scorer": e1, "se": se, "ppv": ppv},
        {"score": "Event 2", "published_set_a": SAMPLE_ENTRY_SET_A["event2"], "this_scorer": e2, "h": h, "d": d},
        {"score": "AUROC of the sample entry's risk", "published_set_a": np.nan,
         "this_scorer": auroc(sample["risk"].to_numpy(), outcomes_a)},
    ])
    scorer.to_csv(results / "scorer_check.csv", index=False)
    log(f"scorer check on the SAPS-I sample entry, set A: Event 1 {e1:.4f} (published 0.296), Event 2 {e2:.2f} (68)")

    stays = cohort.stays
    if S.subsample:
        stays = stays.groupby("set", group_keys=False).sample(n=S.subsample, random_state=0).sort_index()
    obs = cohort.obs[cohort.obs["record_id"].isin(stays.index)]

    # ------------------------------------------------------------------ features
    def build_features() -> pd.DataFrame:
        path = CACHE / f"features-{key_of({'data': data_key, 'sub': S.subsample})}.parquet"
        if not path.exists():
            summary_features(obs, stays).to_parquet(path)
        return pd.read_parquet(path)

    feats = timer.run("features", build_features)
    hourly_all = timer.run("hourly tensors", lambda: hourly(obs, stays.index.to_numpy()))
    row_of = pd.Series(np.arange(len(stays)), index=stays.index)
    cohort_summary(stays, feats).to_csv(results / "cohort_summary.csv", index=False)
    dev_stays = stays[stays["set"].isin(REGIMES[MAIN])]
    missingness_table(dev_stays, feats).to_csv(results / "missingness.csv", index=False)

    rows = {k: [] for k in ("models", "seeds", "history", "trials", "rel", "challenge", "sig", "subgroups")}
    preds: dict[str, dict[str, Pred]] = {}
    train_info: dict[tuple, dict] = {}
    xgb_params: dict[str, dict] = {}
    rnn_cfg: dict[str, dict] = {}
    keep: dict = {}  # objects needed after the main regime (boosters, recurrent models, inputs)
    best_deep = None

    for regime, dev_sets in REGIMES.items():
        dev_ids = stays.index[stays["set"].isin(dev_sets)]
        test_ids = stays.index[stays["set"] == "C"]
        fit_ids, val_ids = train_test_split(np.asarray(dev_ids), test_size=S.val_share, random_state=S.split_seed,
                                            stratify=stays.loc[dev_ids, "died"])
        fit_ids, val_ids = np.sort(fit_ids), np.sort(val_ids)
        y_fit, y_val, y_test = (stays.loc[i, "died"].to_numpy() for i in (fit_ids, val_ids, test_ids))
        split_spec = {"data": data_key, "sub": S.subsample, "regime": regime, "seed": S.split_seed,
                      "val": S.val_share}
        P: dict[str, Pred] = {}
        log(f"regime {regime}: fit {len(fit_ids)}, validation {len(val_ids)}, test {len(test_ids)} stays "
            f"({y_fit.mean():.3f} / {y_val.mean():.3f} / {y_test.mean():.3f} mortality)")

        # -------------------------------------------------------------- severity scores and logistic regression
        for score, name in (("saps1", "SAPS-I"), ("sofa", "SOFA")):
            m, med = score_model_fit(stays.loc[fit_ids, score].to_numpy(), y_fit)
            P[name] = Pred(score_model_predict(m, med, stays.loc[val_ids, score].to_numpy()),
                           score_model_predict(m, med, stays.loc[test_ids, score].to_numpy()))
        X = feats

        def fit_logreg() -> dict:
            model, c = logreg_fit(X.loc[fit_ids], y_fit, X.loc[val_ids], y_val)
            return {"val": model.predict_proba(X.loc[val_ids])[:, 1],
                    "test": model.predict_proba(X.loc[test_ids])[:, 1], "c": np.array(c)}

        lr_out = timer.run("logistic regression", lambda: cached_arrays(f"logreg-{regime}", split_spec, fit_logreg))
        P["Logistic regression"] = Pred(lr_out["val"], lr_out["test"])
        train_info[(regime, "Logistic regression", 0)] = {"C": float(lr_out["c"])}

        # -------------------------------------------------------------- XGBoost
        trials_n = S.optuna_trials if regime == MAIN else S.optuna_trials_challenge
        tune_spec = {**split_spec, "trials": trials_n}
        tune_path = CACHE / "optuna" / f"xgb-{regime.replace('+', '')}-{key_of(tune_spec)}.json"
        if not tune_path.exists():
            best_params, trials = timer.run("optuna", lambda: tune_xgb(X.loc[fit_ids], y_fit, X.loc[val_ids], y_val,
                                                                       trials_n))
            tune_path.parent.mkdir(parents=True, exist_ok=True)
            tune_path.write_text(json.dumps({"best": best_params, "trials": trials.to_dict(orient="list")},
                                            default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        tuned = json.loads(tune_path.read_text())
        best_params = tuned["best"]
        xgb_params[regime] = best_params
        rows["trials"].append(pd.DataFrame(tuned["trials"]).assign(regime=regime, model="XGBoost"))
        log(f"{regime}: Optuna best XGBoost params {best_params}")
        sets = XGB_SETS if regime == MAIN else {"XGBoost": "full"}
        for name, fs in sets.items():
            cols = feature_set(list(X.columns), fs)
            parts = []
            for seed in S.xgb_seeds:
                spec = {**split_spec, "params": best_params, "features": fs, "seed": seed}
                model_path = CACHE / "models" / f"xgb-{slug(name)}-{regime.replace('+', '')}-{seed}-{key_of(spec)}.json"

                def fit_xgb(cols=cols, seed=seed, model_path=model_path) -> dict:
                    booster = xgb_fit(best_params, X.loc[fit_ids, cols], y_fit, X.loc[val_ids, cols], y_val, seed)
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    booster.save_model(model_path)
                    return {"val": xgb_predict(booster, X.loc[val_ids, cols]),
                            "test": xgb_predict(booster, X.loc[test_ids, cols]),
                            "rounds": np.array(booster.num_boosted_rounds())}

                out = timer.run("xgboost", lambda: cached_arrays(f"xgb-{slug(name)}-{regime}-{seed}", spec, fit_xgb))
                parts.append(out)
                if regime == MAIN and name == REFERENCE:
                    booster = xgb.Booster()
                    booster.load_model(model_path)
                    keep.setdefault("boosters", []).append(booster)
                    keep["xgb_cols"] = cols
            P[name] = ensemble(parts)
            for seed, out in zip(S.xgb_seeds, parts):
                train_info[(regime, name, seed)] = {"rounds": int(out["rounds"])}

        # -------------------------------------------------------------- recurrent models
        scaler = Scaler.fit(hourly_all.x[row_of[fit_ids].to_numpy()])
        static_all = static_matrix(feats, pd.Index(fit_ids))
        seq = sequence_tensors(hourly_all, scaler, static_all)
        inputs = {k: to_device(seq.take(row_of[i].to_numpy()), dev) for k, i in
                  (("fit", fit_ids), ("val", val_ids), ("test", test_ids))}
        n_rnn_trials = S.rnn_trials if regime == MAIN else S.rnn_trials_challenge
        rnn_tune_spec = {**split_spec, "trials": n_rnn_trials, "base": asdict(S.train)}
        rnn_tune_path = CACHE / "optuna" / f"gru-{regime.replace('+', '')}-{key_of(rnn_tune_spec)}.json"
        if not rnn_tune_path.exists():
            cfg_t, trials_t = timer.run("optuna (GRU, GPU)", lambda: tune_rnn(
                inputs["fit"], y_fit, inputs["val"], y_val, S.train, n_rnn_trials), gpu=True)
            rnn_tune_path.parent.mkdir(parents=True, exist_ok=True)
            rnn_tune_path.write_text(json.dumps({"best": asdict(cfg_t), "trials": trials_t.to_dict(orient="list")},
                                                default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        tuned_rnn = json.loads(rnn_tune_path.read_text())
        cfg = TrainConfig(**tuned_rnn["best"])
        rnn_cfg[regime] = asdict(cfg)
        rows["trials"].append(pd.DataFrame(tuned_rnn["trials"]).assign(regime=regime, model="GRU"))
        log(f"{regime}: Optuna best recurrent config {asdict(cfg)}")
        specs = DEEP_SPECS if regime == MAIN else tuple(s for s in DEEP_SPECS if s.name == best_deep)
        for spec_m in specs:
            parts = []
            for seed in S.deep_seeds:
                spec = {**split_spec, "model": asdict(spec_m), "train": asdict(cfg), "seed": seed}
                weights = (CACHE / "models"
                           / f"rnn-{slug(spec_m.name)}-{regime.replace('+', '')}-{seed}-{key_of(spec)}.pt")

                def fit_rnn(spec_m=spec_m, seed=seed, weights=weights) -> dict:
                    model, hist = train(spec_m, cfg, inputs["fit"], y_fit, inputs["val"], y_val, seed)
                    weights.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), weights)
                    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
                    return {"val": sig(predict_logits(model, inputs["val"])),
                            "test": sig(predict_logits(model, inputs["test"])),
                            **{f"hist_{c}": hist[c].to_numpy() for c in hist.columns}}

                out = timer.run(f"train {spec_m.kind}", lambda: cached_arrays(
                    f"rnn-{slug(spec_m.name)}-{regime}-{seed}", spec, fit_rnn), gpu=True)
                parts.append(out)
                hist = pd.DataFrame({c[5:]: out[c] for c in out if c.startswith("hist_")})
                rows["history"].append(hist.assign(regime=regime, model=spec_m.name, seed=seed))
                train_info[(regime, spec_m.name, seed)] = {"epochs": len(hist),
                                                           "best_epoch": int(hist["val_loss"].idxmin()) + 1,
                                                           "train_seconds": float(hist["seconds"].sum())}
                if regime == MAIN:
                    keep.setdefault("weights", {}).setdefault(spec_m.name, []).append(weights)
            P[spec_m.name] = ensemble(parts)
            log(f"{regime}: {spec_m.name} ensemble validation AUROC {auroc(P[spec_m.name].val, y_val):.4f}")
        if regime == MAIN:
            best_deep = max(FULL_DEEP, key=lambda m: auroc(P[m].val, y_val))
            keep["inputs_test"] = inputs["test"]
            log(f"best recurrent model on validation: {best_deep}")
        P["Blend"] = Pred((P[REFERENCE].val + P[best_deep].val) / 2, (P[REFERENCE].test + P[best_deep].test) / 2)

        # -------------------------------------------------------------- evaluation on set C
        draws = bootstrap_indices(len(test_ids), S.n_boot, seed=1)
        boots = {}
        for name, pr in P.items():
            thr = event1_threshold(pr.val, y_val)
            ab = platt_fit(pr.val, y_val)
            res = summarize(pr.test, y_test, test_ids, thr)
            cal = summarize(platt_apply(pr.test, ab), y_test, test_ids, thr)
            boots[name] = timer.run("bootstrap", lambda: bootstrap_metrics(pr.test, y_test, thr, draws))
            row = {"regime": regime, "model": name, "n_seeds": max(1, len(pr.test_seeds)),
                   "val_auroc": auroc(pr.val, y_val), **res}
            for m in BOOT_METRICS:
                lo, hi = np.nanpercentile(boots[name][m], [2.5, 97.5])
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = lo, hi
            row.update({f"{k}_platt": cal[k] for k in ("brier", "ece", "cal_intercept", "cal_slope", "hl_stat",
                                                         "hl_p", "event2", "log_loss")})
            row["platt_a"], row["platt_b"] = ab
            rows["models"].append(row)
            for variant, p in (("raw", pr.test), ("platt", platt_apply(pr.test, ab))):
                rows["rel"].append(reliability(p, y_test).assign(regime=regime, model=name, variant=variant))
            seeds = S.deep_seeds if pr.test_seeds and name not in XGB_SETS else S.xgb_seeds
            for seed, pv, pt in zip(seeds, pr.val_seeds, pr.test_seeds):
                r_s = summarize(pt, y_test, test_ids, event1_threshold(pv, y_val))
                rows["seeds"].append({"regime": regime, "model": name, "seed": seed, "val_auroc": auroc(pv, y_val),
                                      **{k: r_s[k] for k in ("auroc", "auprc", "brier", "ece", "event1", "event2")},
                                      **train_info.get((regime, name, seed), {})})
            if not args.quick:
                log_result(f"{regime} {name}", {"regime": regime, "model": name, "threshold": thr},
                           {k: v for k, v in res.items() if isinstance(v, float)})
        preds[regime] = P

        if regime == MAIN:
            pvals, sig_rows = [], []
            for a, b in SIG_PAIRS:
                for m in SIG_METRICS:
                    diff = boots[a][m] - boots[b][m]
                    metric_fn = {"auroc": auroc, "auprc": auprc, "brier": brier}[m]
                    point = metric_fn(P[a].test, y_test) - metric_fn(P[b].test, y_test)
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    sig_rows.append({"model": a, "versus": b, "metric": m, "diff": point, "diff_ci_low": lo,
                                     "diff_ci_high": hi, "p": paired_p(diff)})
                    pvals.append(sig_rows[-1]["p"])
            for r_, ph in zip(sig_rows, holm(pvals)):
                r_["p_holm"] = ph
            rows["sig"] = sig_rows
            keep["boots_main"] = boots
            keep["test_ids"], keep["y_test"], keep["val_thr"] = test_ids, y_test, {
                n: event1_threshold(pr.val, y_val) for n, pr in P.items()}

    # ------------------------------------------------------------------ challenge comparison
    mres = pd.DataFrame(rows["models"])
    seeds_df = pd.DataFrame(rows["seeds"])
    challenge = [{"source": "published (set C)", "entry": n, "event": event, "score": s}
                 for event, table in (("Event 1", PUBLISHED_EVENT1), ("Event 2", PUBLISHED_EVENT2)) for n, s in table]
    for regime in REGIMES:
        for name in (REFERENCE, best_deep, "Blend", "SAPS-I", "Logistic regression"):
            r = mres[(mres.regime == regime) & (mres.model == name)].iloc[0]
            label = f"{name}, trained on {'sets A+B' if regime == MAIN else 'set A'}"
            sd = seeds_df[(seeds_df["regime"] == regime) & (seeds_df["model"] == name)]
            spread = {e: (sd[e].min(), sd[e].max()) for e in ("event1", "event2")}  # NaN for single-model rows
            challenge.append({"source": "this project (set C)", "entry": label, "event": "Event 1",
                              "score": r["event1"], "ci_low": r["event1_ci_low"], "ci_high": r["event1_ci_high"],
                              "single_seed_min": spread["event1"][0], "single_seed_max": spread["event1"][1]})
            challenge.append({"source": "this project (set C)", "entry": label, "event": "Event 2",
                              "score": r["event2"], "single_seed_min": spread["event2"][0],
                              "single_seed_max": spread["event2"][1]})
            challenge.append({"source": "this project (set C)", "entry": label + ", Platt-scaled",
                              "event": "Event 2", "score": r["event2_platt"]})
    pd.DataFrame(challenge).to_csv(results / "challenge_comparison.csv", index=False)

    # ------------------------------------------------------------------ explanations (main regime, set C)
    test_ids, y_test = keep["test_ids"], keep["y_test"]
    Xt = feats.loc[test_ids, keep["xgb_cols"]]
    boosters = keep["boosters"]
    contrib = xgb_contributions(boosters[0], Xt)
    explainer_vals = shap.TreeExplainer(boosters[0]).shap_values(Xt.iloc[: S.shap_check_rows])
    shap_diff = float(np.max(np.abs(explainer_vals - contrib[: S.shap_check_rows, :-1])))
    log(f"TreeSHAP: xgboost pred_contribs vs shap.TreeExplainer max |diff| {shap_diff:.2e}")
    by_feature = shap_by_feature(contrib, list(Xt.columns))
    by_feature.to_csv(results / "shap_features.csv", index=False)
    shap_by_variable(contrib, list(Xt.columns)).to_csv(results / "shap_variables.csv", index=False)
    top = by_feature["feature"].head(6).tolist()
    pd.concat([pd.DataFrame({"feature": f, "value": Xt[f].to_numpy(), "shap": contrib[:, Xt.columns.get_loc(f)]})
               for f in top]).to_csv(results / "shap_dependence.csv", index=False)

    groups = list(D.TIME_SERIES) + ["Age", "Gender", "Height", "ICUType"]
    col_groups = {g: [c for c in Xt.columns if feature_variable(c) == g] for g in groups}

    def predict_xgb(g: str | None, perm: np.ndarray | None) -> np.ndarray:
        Xp = Xt
        if g is not None:
            Xp = Xt.copy()
            Xp[col_groups[g]] = Xt[col_groups[g]].to_numpy()[perm]
        return np.mean([xgb_predict(b, Xp) for b in boosters], axis=0)

    inputs_t = keep["inputs_test"]
    nets = []
    for w in keep["weights"][best_deep]:
        spec_m = next(s for s in DEEP_SPECS if s.name == best_deep)
        net = build(spec_m, inputs_t["x"].shape[-1], inputs_t["static"].shape[-1],
                    TrainConfig(**rnn_cfg[MAIN])).to(dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        nets.append(net)

    def predict_rnn(g: str | None, perm: np.ndarray | None) -> np.ndarray:
        inp = inputs_t
        if g is not None:
            inp = {k: v.clone() for k, v in inputs_t.items()}
            pt = torch.as_tensor(perm, device=dev)
            if g in D.TIME_SERIES:
                j = D.TIME_SERIES.index(g)
                for k in ("x", "x_ffill", "mask", "delta"):
                    inp[k][:, :, j] = inputs_t[k][pt][:, :, j]
            if g in STATIC_GROUPS:
                cols = STATIC_GROUPS[g]
                inp["static"][:, cols] = inputs_t["static"][pt][:, cols]
        return np.mean([1 / (1 + np.exp(-predict_logits(n, inp))) for n in nets], axis=0)

    perm_spec = {"data": data_key, "sub": S.subsample, "split": S.split_seed, "xgb": xgb_params[MAIN],
                 "rnn": rnn_cfg[MAIN], "deep": best_deep, "seeds": [S.xgb_seeds, S.deep_seeds],
                 "repeats": S.perm_repeats}
    perm_path = CACHE / "explain" / f"permutation-{key_of(perm_spec)}.csv"
    if not perm_path.exists():
        perm_xgb = timer.run("permutation importance", lambda: grouped_permutation_importance(
            predict_xgb, groups, y_test, S.perm_repeats))
        perm_rnn = timer.run("permutation importance (GPU)", lambda: grouped_permutation_importance(
            predict_rnn, groups, y_test, S.perm_repeats), gpu=True)
        perm_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([perm_xgb.assign(model=REFERENCE), perm_rnn.assign(model=best_deep)]).to_csv(perm_path, index=False)
    pd.read_csv(perm_path).to_csv(results / "permutation_importance.csv", index=False)

    # ------------------------------------------------------------------ subgroups (main regime, set C)
    test_stays = stays.loc[test_ids]
    age_band = pd.cut(test_stays["Age"], [0, 49.99, 64.99, 79.99, 200], labels=["<50", "50-64", "65-79", "80+"])
    group_defs = {
        "ICU type": test_stays["ICUType"].map(D.ICU_TYPES).fillna("unknown"),
        "Age": age_band.astype(str),
        "Sex": test_stays["Gender"].map({0: "Female", 1: "Male"}).fillna("unknown"),
        "Mechanical ventilation": np.where(feats.loc[test_ids, "MechVent__n"] > 0, "Ventilated", "Not ventilated"),
    }
    rng = np.random.default_rng(2)
    for gname, labels in group_defs.items():
        labels = np.asarray(labels)
        for level in sorted(set(labels)):
            idx = np.flatnonzero(labels == level)
            if level == "unknown" or len(idx) < 30:
                continue
            sub_draws = rng.integers(0, len(idx), size=(S.n_boot, len(idx)))
            for name in (REFERENCE, best_deep, "SAPS-I"):
                p, y = preds[MAIN][name].test[idx], y_test[idx]
                aucs = [auroc(p[d_], y[d_]) for d_ in sub_draws]
                pred_bin = p >= keep["val_thr"][name]
                e1_, se_, ppv_ = event1(pred_bin, y)
                rows["subgroups"].append({
                    "grouping": gname, "group": level, "model": name, "n": len(idx), "deaths": int(y.sum()),
                    "observed_rate": y.mean(), "mean_risk": p.mean(), "auroc": auroc(p, y),
                    "auroc_ci_low": np.nanpercentile(aucs, 2.5), "auroc_ci_high": np.nanpercentile(aucs, 97.5),
                    "brier": float(np.mean((p - y) ** 2)), "sensitivity": se_, "ppv": ppv_,
                    "flagged_share": pred_bin.mean()})

    # ------------------------------------------------------------------ write tables
    mres.to_csv(results / "model_results.csv", index=False)
    pd.DataFrame(rows["seeds"]).to_csv(results / "seed_results.csv", index=False)
    pd.concat(rows["history"]).to_csv(results / "training_history.csv", index=False)
    pd.concat(rows["trials"]).to_csv(results / "tuning_trials.csv", index=False)
    pd.concat(rows["rel"]).to_csv(results / "calibration_deciles.csv", index=False)
    pd.DataFrame(rows["sig"]).to_csv(results / "significance.csv", index=False)
    pd.DataFrame(rows["subgroups"]).to_csv(results / "subgroups.csv", index=False)

    meta = {
        "started_utc": started.isoformat(timespec="seconds"), "quick": args.quick,
        "settings": {k: v for k, v in asdict(S).items() if k != "train"}, "base_train_config": asdict(S.train),
        "tuned_train_config": rnn_cfg,
        "deep_specs": [asdict(s) for s in DEEP_SPECS], "best_recurrent_model": best_deep,
        "xgb_params": xgb_params,
        "shap_check_max_abs_diff": shap_diff,
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
        "gpu_stages": sorted(timer.gpu), "gpu_seconds": round(sum(timer.stages[k] for k in timer.gpu), 1),
        "total_seconds": round(sum(timer.stages.values()), 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "xgboost": xgb.__version__,
                     "optuna": optuna.__version__, "shap": shap.__version__, "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"timings (s): {meta['timings_s']}")
    for p in make_all(results, figures):
        log(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
