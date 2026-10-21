"""Tabular baselines on the summary features: XGBoost tuned with Optuna, logistic regression, and severity scores.

All hyperparameters are chosen on the validation stays only; the test set never enters a tuning decision.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from icumort.metrics import log_loss

XGB_FIXED = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "nthread": 2,
             "verbosity": 0}
MAX_ROUNDS = 3000
EARLY_STOP = 100


def xgb_fit(params: dict, X_fit: pd.DataFrame, y_fit: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
            seed: int = 0) -> xgb.Booster:
    """Train with early stopping on the validation log loss; returns the booster cut at its best iteration."""
    dfit = xgb.DMatrix(X_fit, label=y_fit)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train({**XGB_FIXED, **params, "seed": seed}, dfit, num_boost_round=MAX_ROUNDS,
                        evals=[(dval, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return booster[: booster.best_iteration + 1]


def xgb_predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predicted risk."""
    return booster.predict(xgb.DMatrix(X))


def xgb_contributions(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP contributions in log-odds, (n, n_features + 1) with the bias last."""
    return booster.predict(xgb.DMatrix(X), pred_contribs=True)


def tune_xgb(X_fit: pd.DataFrame, y_fit: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, n_trials: int,
             seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Optuna TPE search minimising the early-stopped validation log loss; returns the best params and all trials."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-2, 30, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
        }
        booster = xgb_fit(params, X_fit, y_fit, X_val, y_val, seed)
        trial.set_user_attr("rounds", booster.num_boosted_rounds())
        return log_loss(xgb_predict(booster, X_val), y_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return dict(study.best_params), trials


def logreg_fit(X_fit: pd.DataFrame, y_fit: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
               cs: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3)) -> tuple[Pipeline, float]:
    """L2 logistic regression (median imputation with missing indicators, standardisation); C by validation log loss."""
    best, best_c, best_model = np.inf, None, None
    for c in cs:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=c, max_iter=3000))
        model.fit(X_fit, y_fit)
        loss = log_loss(model.predict_proba(X_val)[:, 1], y_val)
        if loss < best:
            best, best_c, best_model = loss, c, model
    return best_model, float(best_c)


def score_model_fit(score_fit: np.ndarray, y_fit: np.ndarray) -> tuple[LogisticRegression, float]:
    """Logistic recalibration of a severity score (unknown -1 replaced by the training median)."""
    med = float(np.median(score_fit[score_fit >= 0]))
    x = np.where(score_fit >= 0, score_fit, med).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=1000).fit(x, y_fit), med


def score_model_predict(model: LogisticRegression, median: float, score: np.ndarray) -> np.ndarray:
    """Risk from a recalibrated severity score."""
    return model.predict_proba(np.where(score >= 0, score, median).reshape(-1, 1))[:, 1]
