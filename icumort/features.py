"""Two views of the first 48 hours: per-stay summary features for tree models, hourly tensors for recurrent nets.

Summary features: for each of the 37 variables, six *value* statistics (first, last, min, max, mean, standard
deviation) and three *measurement-process* statistics (number of measurements, hour of the first and of the last
one), plus the admission descriptors. Missing values stay NaN for the tree models.

Hourly tensors: 48 one-hour bins x 37 variables holding the mean of the measurements in the bin, a mask of which bins
were measured, and the time since the last measurement (GRU-D, Che et al., 2018).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from icumort.data import ICU_TYPES, TIME_SERIES

N_HOURS = 48
VALUE_STATS = ("first", "last", "min", "max", "mean", "std")
PROCESS_STATS = ("n", "t_first", "t_last")
ICU_COLUMNS = tuple(f"icu_{k}" for k in ICU_TYPES)
STATIC_COLUMNS = ("Age", "Gender", "Height", "Weight") + ICU_COLUMNS
# Right-skewed variables are log1p-transformed before standardisation for the recurrent nets.
LOG_VARIABLES = frozenset({"ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose", "Lactate", "PaO2",
                           "Platelets", "TroponinI", "TroponinT", "Urine", "WBC"})


def static_frame(stays: pd.DataFrame) -> pd.DataFrame:
    """Admission descriptors with the ICU type one-hot encoded (NaN where unknown)."""
    out = stays[["Age", "Gender", "Height", "Weight"]].astype(float).copy()
    for k, col in zip(ICU_TYPES, ICU_COLUMNS):
        out[col] = (stays["ICUType"] == k).astype(float).where(stays["ICUType"].notna())
    return out


def summary_features(obs: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """One row per stay: value and process statistics of every variable plus the admission descriptors.

    Columns are named ``<variable>__<statistic>``; the descriptors keep their own names.
    """
    o = obs.sort_values(["record_id", "minute"], kind="stable")
    g = o.groupby(["record_id", "param"], observed=True)
    stats = g["value"].agg(["first", "last", "min", "max", "mean", "std", "size"]).rename(columns={"size": "n"})
    stats["t_first"] = g["minute"].min() / 60.0
    stats["t_last"] = g["minute"].max() / 60.0
    wide = stats.unstack("param")
    wide.columns = [f"{var}__{stat}" for stat, var in wide.columns]
    cols = [f"{v}__{s}" for v in TIME_SERIES for s in VALUE_STATS + PROCESS_STATS]
    wide = wide.reindex(index=stays.index, columns=cols)
    for v in TIME_SERIES:
        wide[f"{v}__n"] = wide[f"{v}__n"].fillna(0)
    return pd.concat([wide, static_frame(stays)], axis=1).astype(np.float32)


def feature_variable(column: str) -> str:
    """The clinical variable a feature column is derived from (ICU one-hots map to ICUType)."""
    if column.startswith("icu_"):
        return "ICUType"
    return column.split("__")[0]


def feature_set(columns: list[str], name: str) -> list[str]:
    """Columns of a named feature set.

    ``full``: everything; ``values``: value statistics and descriptors, no counts or timing;
    ``process``: counts and timing of the measurements only (no values, no descriptors).
    """
    def stat(c: str) -> str:
        return c.split("__")[1] if "__" in c else "static"

    if name == "full":
        return list(columns)
    if name == "values":
        return [c for c in columns if stat(c) in VALUE_STATS or stat(c) == "static"]
    if name == "process":
        return [c for c in columns if stat(c) in PROCESS_STATS]
    raise ValueError(f"unknown feature set {name}")


@dataclass
class Hourly:
    """Hourly binned measurements of a set of stays (same order as ``record_id``)."""

    record_id: np.ndarray  # (n,)
    x: np.ndarray  # float32 (n, 48, 37): mean value per bin, NaN where not measured
    mask: np.ndarray  # bool (n, 48, 37)


def hourly(obs: pd.DataFrame, record_ids: np.ndarray, n_hours: int = N_HOURS) -> Hourly:
    """Bin measurements into ``n_hours`` one-hour bins (a time stamp of exactly 48:00 goes into the last bin)."""
    pos = pd.Series(np.arange(len(record_ids)), index=record_ids)
    o = obs[obs["record_id"].isin(pos.index)]
    hour = np.minimum(o["minute"].to_numpy() // 60, n_hours - 1)
    var = o["param"].cat.codes.to_numpy()
    key = pd.DataFrame({"i": pos.loc[o["record_id"]].to_numpy(), "t": hour, "v": var, "value": o["value"].to_numpy()})
    agg = key.groupby(["i", "t", "v"])["value"].mean().reset_index()
    x = np.full((len(record_ids), n_hours, len(TIME_SERIES)), np.nan, dtype=np.float32)
    x[agg["i"].to_numpy(), agg["t"].to_numpy(), agg["v"].to_numpy()] = agg["value"].to_numpy()
    return Hourly(np.asarray(record_ids), x, ~np.isnan(x))


def transform_values(x: np.ndarray) -> np.ndarray:
    """log1p for the skewed variables, identity otherwise (NaN preserved)."""
    out = x.copy()
    for j, v in enumerate(TIME_SERIES):
        if v in LOG_VARIABLES:
            out[..., j] = np.log1p(out[..., j])
    return out


@dataclass(frozen=True)
class Scaler:
    """Per-variable mean and standard deviation of the transformed measured values in the fitting stays."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        """Fit on (n, T, V) values with NaN for missing bins."""
        t = transform_values(x).reshape(-1, x.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a variable never measured in the fitting stays
            mean, std = np.nanmean(t, axis=0), np.nanstd(t, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(np.nan_to_num(mean).astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray, clip: float = 6.0) -> np.ndarray:
        """Standardised values clipped to +-clip (NaN preserved)."""
        return np.clip((transform_values(x) - self.mean) / self.std, -clip, clip).astype(np.float32)


def forward_fill(z: np.ndarray) -> np.ndarray:
    """Last observation carried forward along time; 0 (the training mean) before the first observation."""
    n, t, v = z.shape
    idx = np.where(~np.isnan(z), np.arange(t)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = np.take_along_axis(z, idx, axis=1)
    return np.nan_to_num(filled, nan=0.0).astype(np.float32)


def time_since_last(mask: np.ndarray) -> np.ndarray:
    """GRU-D time interval in hours: delta_1 = 0; delta_t = 1 + delta_{t-1} if unmeasured at t-1, else 1."""
    n, t, v = mask.shape
    delta = np.zeros((n, t, v), dtype=np.float32)
    for s in range(1, t):
        delta[:, s] = 1.0 + np.where(mask[:, s - 1], 0.0, delta[:, s - 1])
    return delta


def static_matrix(static: pd.DataFrame, fit_index: pd.Index) -> np.ndarray:
    """Standardised descriptors (statistics from ``fit_index``) with NaN set to 0 plus missing indicators."""
    cols = list(STATIC_COLUMNS)
    s = static[cols].astype(float)
    mu, sd = s.loc[fit_index].mean(), s.loc[fit_index].std().replace(0, 1.0)
    z = ((s - mu) / sd).fillna(0.0)
    missing = s[["Gender", "Height", "Weight", "Age"]].isna().astype(float).add_suffix("_missing")
    return pd.concat([z, missing], axis=1).to_numpy(np.float32)


@dataclass
class SeqTensors:
    """Model inputs for a set of stays: standardised values, LOCF values, mask, time since last measurement."""

    x: np.ndarray  # float32 (n, T, V), 0 where unmeasured
    x_ffill: np.ndarray  # float32 (n, T, V)
    mask: np.ndarray  # float32 (n, T, V)
    delta: np.ndarray  # float32 (n, T, V), hours
    static: np.ndarray  # float32 (n, S)

    def take(self, idx: np.ndarray) -> "SeqTensors":
        """Rows ``idx`` of every array."""
        return SeqTensors(self.x[idx], self.x_ffill[idx], self.mask[idx], self.delta[idx], self.static[idx])


def sequence_tensors(h: Hourly, scaler: Scaler, static: np.ndarray) -> SeqTensors:
    """Build the recurrent-net inputs from hourly bins with a scaler fitted on the training stays."""
    z = scaler(h.x)
    return SeqTensors(np.nan_to_num(z, nan=0.0), forward_fill(z), h.mask.astype(np.float32),
                      time_since_last(h.mask), static)
