"""Download, verify, parse and clean the PhysioNet/CinC Challenge 2012 data (Silva et al., 2012).

Each of the three record sets (A, B, C) holds 4,000 ICU stays as one CSV text file per stay (``Time,Parameter,Value``
with ``HH:MM`` time stamps since ICU admission), and each set has an outcomes file with the SAPS-I and SOFA scores,
length of stay, survival and in-hospital death. Files come from PhysioNet's open-data mirror on AWS (with
physionet.org as the fallback) and are checked against pinned SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URLS = (
    "https://physionet-open.s3.amazonaws.com/challenge-2012/1.0.0",  # PhysioNet open-data bucket (fast)
    "https://physionet.org/files/challenge-2012/1.0.0",
)
FILES = {
    "set-a.tar.gz": "8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920",
    "set-b.tar.gz": "b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30",
    "set-c.tar.gz": "a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1",
    "Outcomes-a.txt": "2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2",
    "Outcomes-b.txt": "1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5",
    "Outcomes-c.txt": "0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9",
    # Output of the organisers' SAPS-I sample entry (physionet2012.m) on set A, used to check the scorer.
    "Sample-SetA.txt": "dfeb2ee3aef58d4f60a5213f9836ecb4ff1c6d334ce75a5e83fe4fec4b5dd786",
}
SETS = ("a", "b", "c")
DESCRIPTORS = ("Age", "Gender", "Height", "ICUType", "Weight")
TIME_SERIES = (
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS",
    "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
    "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine",
    "WBC", "Weight",
)
PARSER_VERSION = 2  # part of the cache key of the parsed cohort
ICU_TYPES = {1: "Coronary care", 2: "Cardiac surgery recovery", 3: "Medical", 4: "Surgical"}
OUTCOME_COLUMNS = {"SAPS-I": "saps1", "SOFA": "sofa", "Length_of_stay": "los", "Survival": "survival",
                   "In-hospital_death": "died"}

# Physiologically plausible ranges (inclusive). Values outside are recording errors (zeros from disconnected
# arterial lines, temperatures in Fahrenheit, pH typed without the decimal point) and are dropped.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "Albumin": (0.5, 7), "ALP": (1, 5000), "ALT": (1, 15000), "AST": (1, 25000), "Bilirubin": (0.05, 80),
    "BUN": (0.5, 300), "Cholesterol": (20, 600), "Creatinine": (0.1, 30), "DiasABP": (10, 200), "FiO2": (0.21, 1),
    "GCS": (3, 15), "Glucose": (10, 1500), "HCO3": (2, 60), "HCT": (5, 70), "HR": (10, 300), "K": (1, 12),
    "Lactate": (0.1, 40), "Mg": (0.3, 10), "MAP": (20, 300), "MechVent": (0, 1), "Na": (90, 200),
    "NIDiasABP": (10, 200), "NIMAP": (20, 300), "NISysABP": (20, 300), "PaCO2": (5, 200), "PaO2": (10, 800),
    "pH": (6.5, 8.0), "Platelets": (1, 2000), "RespRate": (2, 100), "SaO2": (20, 100), "SysABP": (20, 300),
    "Temp": (25, 45), "TroponinI": (0.01, 100), "TroponinT": (0.005, 50), "Urine": (0, 10000), "WBC": (0.1, 300),
    "Weight": (20, 300),
}
STATIC_PLAUSIBLE: dict[str, tuple[float, float]] = {"Age": (15, 120), "Gender": (0, 1), "Height": (100, 250),
                                                    "ICUType": (1, 4), "Weight": (20, 300)}

# Published statistics of the 12,000 stays (Silva et al., 2012, section 2 and Table 1; PhysioNet challenge page).
PUBLISHED = {
    "ICU stays": 12_000,
    "stays per set": 4_000,
    "in-hospital mortality, all sets (%)": 14.2,
    "set A in-hospital mortality (%)": 13.85,  # prior used by the organisers' sample entry (physionet2012.m)
    "age, mean (years)": 64.5,
    "age, SD (years)": 17.1,
    "female (%)": 43.8,
    "male (%)": 56.1,
    "medical ICU (%)": 35.8,
    "surgical ICU (%)": 28.4,
    "cardiac surgery recovery unit (%)": 21.1,
    "coronary care unit (%)": 21.1,  # the paper's four ICU shares add up to 106.4%
    "SAPS-I, mean": 14.9,
    "SAPS-I, SD": 5.2,
    "SOFA, mean": 6.4,
    "SOFA, SD": 4.2,
    "length of stay, mean (days)": 13.4,
    "length of stay, SD (days)": 12.8,
    "time-series variables": 37,
}
# Share of the 12,000 stays with at least one measurement (Silva et al., 2012, Table 1, in %).
PUBLISHED_AVAILABILITY = {
    "DiasABP": 98.4, "MAP": 98.4, "SysABP": 98.4, "NIDiasABP": 87.3, "NIMAP": 87.2, "NISysABP": 87.6,
    "Albumin": 40.5, "ALP": 42.4, "ALT": 43.4, "AST": 43.4, "Bilirubin": 43.4, "BUN": 98.4, "Cholesterol": 7.9,
    "Creatinine": 98.4, "FiO2": 67.6, "GCS": 98.4, "Glucose": 97.5, "HCO3": 98.2, "HCT": 98.4, "HR": 98.4,
    "K": 97.9, "Lactate": 54.8, "Mg": 97.5, "MechVent": 63.1, "Na": 98.2, "PaCO2": 75.4, "PaO2": 75.4, "pH": 75.9,
    "Platelets": 98.3, "RespRate": 27.7, "SaO2": 44.7, "Temp": 98.4, "TroponinI": 4.7, "TroponinT": 21.9,
    "Urine": 97.4, "WBC": 98.2, "Weight": 67.7,
}


def sha256(path: Path) -> str:
    """Hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw: Path, timeout: float = 300.0) -> None:
    """Fetch every file in ``FILES`` into ``raw`` (skipping verified ones) and check the SHA-256 digests."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = raw / name
        if path.exists() and sha256(path) == digest:
            continue
        last_error: Exception | None = None
        for base in BASE_URLS:
            try:
                r = requests.get(f"{base}/{name}", timeout=timeout)
                r.raise_for_status()
                path.write_bytes(r.content)
                break
            except requests.RequestException as e:
                last_error = e
        else:
            raise RuntimeError(f"could not download {name}: {last_error}")
        if sha256(path) != digest:
            raise ValueError(f"SHA-256 mismatch for {name}")


def parse_record(text: str) -> tuple[int, dict[str, float], list[tuple[int, str, float]]]:
    """Parse one stay: (RecordID, descriptors at admission, time-series rows as (minute, parameter, value)).

    Descriptors carry the time stamp ``00:00``. In about 2% of files they are interleaved alphabetically with
    time-series values recorded at 00:00, so every 00:00 line naming a descriptor is taken as one; Weight is both a
    descriptor and a time series, and only its first 00:00 line is the admission weight.
    """
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith("Time"):
        raise ValueError("not a challenge record")
    descriptors: dict[str, float] = {}
    rows: list[tuple[int, str, float]] = []
    for line in lines[1:]:
        t, param, value = line.split(",")
        if t == "00:00" and param in ("RecordID",) + DESCRIPTORS and param not in descriptors:
            descriptors[param] = float(value)
            continue
        hh, mm = t.split(":")
        rows.append((int(hh) * 60 + int(mm), param, float(value)))
    record_id = int(descriptors.pop("RecordID"))
    return record_id, descriptors, rows


def read_set(raw: Path, s: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Static descriptors (one row per stay) and raw time-series observations (long format) of record set ``s``."""
    static_rows, rec, minute, param, value = [], [], [], [], []
    with tarfile.open(raw / f"set-{s}.tar.gz", "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            record_id, desc, rows = parse_record(tar.extractfile(member).read().decode("ascii"))
            static_rows.append({"record_id": record_id, **desc})
            for m, p, v in rows:
                rec.append(record_id)
                minute.append(m)
                param.append(p)
                value.append(v)
    static = pd.DataFrame(static_rows).set_index("record_id").sort_index()
    static["set"] = s.upper()
    obs = pd.DataFrame({"record_id": np.array(rec, dtype=np.int64), "minute": np.array(minute, dtype=np.int16),
                        "param": pd.Categorical(param), "value": np.array(value, dtype=np.float64)})
    return static, obs


def read_outcomes(raw: Path, s: str) -> pd.DataFrame:
    """Outcome-related descriptors of record set ``s`` with -1 (unknown) kept as recorded."""
    out = pd.read_csv(raw / f"Outcomes-{s}.txt").rename(columns={"RecordID": "record_id", **OUTCOME_COLUMNS})
    return out.set_index("record_id").sort_index()


def clean_observations(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop unknown (-1) and implausible values; return the cleaned observations and a per-variable quality table.

    FiO2 recorded as a percentage (1 < value <= 100) is divided by 100 before the range check. Rows whose parameter
    is not one of the 37 documented variables (set C has 2,403 rows with an empty name) are dropped.
    """
    obs = obs.copy()
    obs["param"] = obs["param"].astype(str).replace({"": "(empty name)"})
    fio2_pct = (obs["param"] == "FiO2") & (obs["value"] > 1) & (obs["value"] <= 100)
    obs.loc[fio2_pct, "value"] = obs.loc[fio2_pct, "value"] / 100
    lo = obs["param"].map({k: v[0] for k, v in PLAUSIBLE.items()})
    hi = obs["param"].map({k: v[1] for k, v in PLAUSIBLE.items()})
    unknown = obs["value"] == -1
    keep = ~unknown & obs["value"].between(lo, hi) & obs["param"].isin(TIME_SERIES)
    quality = (pd.DataFrame({"param": obs["param"], "unknown": unknown, "implausible": ~unknown & ~keep,
                             "fio2_percent_fixed": fio2_pct})
               .groupby("param").agg(n_raw=("unknown", "size"), n_unknown=("unknown", "sum"),
                                     n_implausible=("implausible", "sum"), n_rescaled=("fio2_percent_fixed", "sum")))
    quality["n_kept"] = quality["n_raw"] - quality["n_unknown"] - quality["n_implausible"]
    quality["share_dropped"] = 1 - quality["n_kept"] / quality["n_raw"]
    obs = obs[keep].copy()
    obs["param"] = pd.Categorical(obs["param"], categories=list(TIME_SERIES))
    return obs.reset_index(drop=True), quality.reset_index()


def clean_static(static: pd.DataFrame) -> pd.DataFrame:
    """Replace unknown (-1) and implausible admission descriptors by NaN."""
    static = static.copy()
    for col, (lo, hi) in STATIC_PLAUSIBLE.items():
        v = static[col]
        static[col] = v.where(v.between(lo, hi))
    return static


@dataclass
class Cohort:
    """All 12,000 stays: per-stay table (descriptors, set, outcomes) and cleaned long-format observations."""

    stays: pd.DataFrame  # index record_id; Age, Gender, Height, ICUType, Weight, set, saps1, sofa, los, survival, died
    obs: pd.DataFrame  # record_id, minute, param (categorical over TIME_SERIES), value
    quality: pd.DataFrame  # per-variable counts of raw, unknown, implausible and kept values
    stays_raw: pd.DataFrame  # descriptors as recorded (for the published-statistics check)
    obs_raw_availability: pd.Series  # share of stays with >= 1 raw observation per variable


def load(raw: Path, cache: Path) -> Cohort:
    """Parse, clean and join the three sets (cached as Parquet under ``cache`` keyed by the data digests)."""
    key = hashlib.sha256("".join(FILES[f"set-{s}.tar.gz"] + FILES[f"Outcomes-{s}.txt"] for s in SETS).encode()
                         + repr((PARSER_VERSION, sorted(PLAUSIBLE.items()))).encode()).hexdigest()[:12]
    d = cache / f"cohort_{key}"
    if not (d / "obs.parquet").exists():
        statics, observations = [], []
        for s in SETS:
            st, ob = read_set(raw, s)
            statics.append(st.join(read_outcomes(raw, s), how="left"))
            observations.append(ob)
        stays_raw = pd.concat(statics).sort_index()
        obs_raw = pd.concat(observations, ignore_index=True)
        obs_raw["param"] = obs_raw["param"].astype(str)
        avail = (obs_raw[obs_raw["value"] != -1].groupby("param")["record_id"].nunique() / len(stays_raw))
        obs, quality = clean_observations(obs_raw)
        stays = clean_static(stays_raw)
        d.mkdir(parents=True, exist_ok=True)
        obs.to_parquet(d / "obs.parquet")
        stays.to_parquet(d / "stays.parquet")
        stays_raw.to_parquet(d / "stays_raw.parquet")
        quality.to_parquet(d / "quality.parquet")
        avail.rename("share").to_frame().to_parquet(d / "availability_raw.parquet")
    obs = pd.read_parquet(d / "obs.parquet")
    obs["param"] = pd.Categorical(obs["param"].astype(str), categories=list(TIME_SERIES))
    return Cohort(stays=pd.read_parquet(d / "stays.parquet"), obs=obs, quality=pd.read_parquet(d / "quality.parquet"),
                  stays_raw=pd.read_parquet(d / "stays_raw.parquet"),
                  obs_raw_availability=pd.read_parquet(d / "availability_raw.parquet")["share"])


def outcome_consistency(stays: pd.DataFrame) -> pd.Series:
    """True where in-hospital death agrees with the documented rule on survival days and length of stay.

    Survival = -1 or Survival > LOS means survivor; 2 <= Survival <= LOS means in-hospital death.
    """
    died_rule = (stays["survival"] >= 2) & (stays["survival"] <= stays["los"])
    return died_rule.astype(int) == stays["died"].astype(int)


def crosscheck(cohort: Cohort) -> pd.DataFrame:
    """Compare the parsed cohort with the statistics published by the organisers.

    The paper's SOFA and length-of-stay summaries are reproduced only when the -1 ("unknown") codes are averaged in,
    so those rows are computed that way and the note gives the values without them.
    """
    s, r = cohort.stays, cohort.stays_raw
    known = lambda v: v[v >= 0]  # noqa: E731  (-1 marks unknown descriptors and scores)
    sizes = s.groupby("set").size()
    observed = {
        "ICU stays": (len(s), ""),
        "stays per set": (int(sizes.min()) if sizes.nunique() == 1 else -1, ""),
        "in-hospital mortality, all sets (%)": (100 * s["died"].mean(), ""),
        "set A in-hospital mortality (%)": (100 * s.loc[s["set"] == "A", "died"].mean(), ""),
        "age, mean (years)": (known(r["Age"]).mean(), ""),
        "age, SD (years)": (known(r["Age"]).std(), ""),
        "female (%)": (100 * (r["Gender"] == 0).mean(), f"{int((r['Gender'] == -1).sum())} stays without gender"),
        "male (%)": (100 * (r["Gender"] == 1).mean(), ""),
        "medical ICU (%)": (100 * (r["ICUType"] == 3).mean(), ""),
        "surgical ICU (%)": (100 * (r["ICUType"] == 4).mean(), ""),
        "cardiac surgery recovery unit (%)": (100 * (r["ICUType"] == 2).mean(), ""),
        "coronary care unit (%)": (100 * (r["ICUType"] == 1).mean(),
                                   "published shares sum to 106.4%; the observed four sum to 100%"),
        "SAPS-I, mean": (known(s["saps1"]).mean(), f"{int((s['saps1'] == -1).sum())} unknown (-1) excluded"),
        "SAPS-I, SD": (known(s["saps1"]).std(), ""),
        "SOFA, mean": (s["sofa"].mean(), f"-1 codes included; without them {known(s['sofa']).mean():.2f}"),
        "SOFA, SD": (s["sofa"].std(), f"-1 codes included; without them {known(s['sofa']).std():.2f}"),
        "length of stay, mean (days)": (s["los"].mean(),
                                        f"-1 codes included; without them {known(s['los']).mean():.2f}"),
        "length of stay, SD (days)": (s["los"].std(), f"-1 codes included; without them {known(s['los']).std():.2f}"),
        "time-series variables": (int(cohort.obs["param"].nunique()), ""),
    }
    rows = []
    for k, pub in PUBLISHED.items():
        obs_v, note = observed[k]
        # Published values are rounded to 1 decimal (2 for the set A prior): allow one unit in the last digit.
        tol = (0.01 if k.startswith("set A") else 0.1) if isinstance(pub, float) else 0
        rows.append({"statistic": k, "published": pub, "observed": round(float(obs_v), 2),
                     "match": abs(float(obs_v) - pub) <= tol + 1e-9, "note": note})
    ok = outcome_consistency(s)
    rows.append({"statistic": "stays whose death flag follows the documented survival/LOS rule", "published": len(s),
                 "observed": int(ok.sum()), "match": bool(ok.all()),
                 "note": f"{int((~ok).sum())} deaths have Survival = 1 day, which the documentation rules out; "
                         "the in-hospital death flag is used as recorded"})
    return pd.DataFrame(rows)


def availability_table(cohort: Cohort) -> pd.DataFrame:
    """Share of stays with at least one measurement per variable: raw, after cleaning, and as published."""
    n = len(cohort.stays)
    clean = cohort.obs.groupby("param", observed=False)["record_id"].nunique() / n
    df = pd.DataFrame({"variable": list(TIME_SERIES)})
    df["published_pct"] = df["variable"].map(PUBLISHED_AVAILABILITY)
    df["raw_pct"] = 100 * df["variable"].map(cohort.obs_raw_availability).fillna(0)
    df["clean_pct"] = 100 * df["variable"].map(clean).fillna(0)
    df["diff_raw_vs_published"] = df["raw_pct"] - df["published_pct"]
    return df


def read_sample_entry(raw: Path) -> pd.DataFrame:
    """The organisers' SAPS-I sample entry output for set A: record_id, binary prediction, risk."""
    out = pd.read_csv(raw / "Sample-SetA.txt", header=None, names=["record_id", "prediction", "risk"])
    return out.set_index("record_id")
