import numpy as np
import pandas as pd
import pytest

from icumort import data as D

RECORD = """Time,Parameter,Value
00:00,RecordID,140762
00:00,Age,76
00:00,BUN,17
00:00,Gender,1
00:00,Height,-1
00:00,ICUType,3
00:00,Weight,80
00:00,Weight,81.5
00:30,HR,88
48:00,HR,90
"""


def test_parse_record_handles_interleaved_descriptors_and_weight_twice():
    rid, desc, rows = D.parse_record(RECORD)
    assert rid == 140762
    assert desc == {"Age": 76.0, "Gender": 1.0, "Height": -1.0, "ICUType": 3.0, "Weight": 80.0}
    # BUN at 00:00 sits between the descriptors; the second 00:00 Weight is a time-series value.
    assert rows == [(0, "BUN", 17.0), (0, "Weight", 81.5), (30, "HR", 88.0), (2880, "HR", 90.0)]


def test_parse_record_rejects_other_files():
    with pytest.raises(ValueError):
        D.parse_record("RecordID,SAPS-I\n1,2\n")


def test_clean_observations_drops_unknown_implausible_and_unnamed_rows():
    obs = pd.DataFrame({
        "record_id": [1] * 7, "minute": np.arange(7, dtype=np.int16),
        "param": ["HR", "HR", "Temp", "FiO2", "FiO2", "", "pH"],
        "value": [80.0, -1.0, 0.0, 40.0, 0.5, 2.9, 735.0],
    })
    clean, quality = D.clean_observations(obs)
    got = list(zip(clean["param"].astype(str), clean["value"]))
    assert got == [("HR", 80.0), ("FiO2", 0.4), ("FiO2", 0.5)]  # FiO2 40% rescaled, pH 735 and Temp 0 dropped
    q = quality.set_index("param")
    assert q.loc["HR", "n_unknown"] == 1
    assert q.loc["(empty name)", "n_kept"] == 0
    assert q.loc["FiO2", "n_rescaled"] == 1


def test_outcome_consistency_rule():
    stays = pd.DataFrame({"los": [5, 5, 5, 5], "survival": [-1, 9, 3, 3], "died": [0, 0, 1, 0]})
    assert D.outcome_consistency(stays).tolist() == [True, True, True, False]


def test_clean_static_sets_unknown_and_implausible_to_nan():
    s = pd.DataFrame({"Age": [70.0], "Gender": [-1.0], "Height": [1.8], "ICUType": [2.0], "Weight": [-1.0]})
    out = D.clean_static(s)
    assert out["Age"].iloc[0] == 70 and out["ICUType"].iloc[0] == 2
    assert out[["Gender", "Height", "Weight"]].isna().all(axis=None)
