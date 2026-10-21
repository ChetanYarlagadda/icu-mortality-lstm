"""Model explanations: TreeSHAP aggregated by clinical variable, and grouped permutation importance for any model.

Grouped permutation importance shuffles *everything* derived from one variable across stays at once (for the trees all
nine summary statistics; for the recurrent nets the values, mask and time gaps of every hour), so the importance of a
variable includes whether and when it was measured. The same definition is used for both model families, which makes
their rankings comparable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from icumort.features import feature_variable
from icumort.metrics import auroc, log_loss


def shap_by_feature(contrib: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean |SHAP| (log-odds) of every feature, sorted, with its variable."""
    imp = np.abs(contrib[:, : len(columns)]).mean(axis=0)
    df = pd.DataFrame({"feature": columns, "variable": [feature_variable(c) for c in columns], "mean_abs_shap": imp})
    return df.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def shap_by_variable(contrib: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean |sum of a variable's SHAP values| per stay: the log-odds the variable moves a prediction on average."""
    var = pd.Series([feature_variable(c) for c in columns])
    summed = pd.DataFrame(contrib[:, : len(columns)]).T.groupby(var.to_numpy()).sum().T
    out = summed.abs().mean().rename("mean_abs_shap").rename_axis("variable").reset_index()
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def grouped_permutation_importance(predict: Callable[[str | None, np.ndarray | None], np.ndarray], groups: list[str],
                                   y: np.ndarray, n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Drop in AUROC (and rise in log loss) when one variable group is shuffled across stays.

    ``predict(group, perm)`` returns risks with the inputs of ``group`` permuted by the index array ``perm``;
    ``predict(None, None)`` returns the unpermuted risks.
    """
    rng = np.random.default_rng(seed)
    base = predict(None, None)
    base_auc, base_ll = auroc(base, y), log_loss(base, y)
    rows = []
    perms = [rng.permutation(len(y)) for _ in range(n_repeats)]
    for g in groups:
        aucs, lls = [], []
        for perm in perms:
            p = predict(g, perm)
            aucs.append(auroc(p, y))
            lls.append(log_loss(p, y))
        rows.append({"variable": g, "auroc_drop": base_auc - float(np.mean(aucs)),
                     "auroc_drop_sd": float(np.std(aucs)), "log_loss_increase": float(np.mean(lls)) - base_ll})
    return pd.DataFrame(rows).sort_values("auroc_drop", ascending=False, ignore_index=True)
