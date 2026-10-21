import numpy as np
import pandas as pd

from icumort import features as F
from icumort.data import TIME_SERIES


def _obs(rows):
    df = pd.DataFrame(rows, columns=["record_id", "minute", "param", "value"])
    df["param"] = pd.Categorical(df["param"], categories=list(TIME_SERIES))
    return df


def test_hourly_binning_averages_within_the_hour_and_clips_48h():
    obs = _obs([(1, 10, "HR", 80.0), (1, 50, "HR", 90.0), (1, 2880, "HR", 70.0), (2, 125, "Temp", 37.0)])
    h = F.hourly(obs, np.array([1, 2, 3]))
    hr, temp = TIME_SERIES.index("HR"), TIME_SERIES.index("Temp")
    assert h.x[0, 0, hr] == 85.0 and h.x[0, 47, hr] == 70.0
    assert h.x[1, 2, temp] == 37.0
    assert h.mask[0].sum() == 2 and h.mask[1].sum() == 1 and h.mask[2].sum() == 0  # stay 3 has no data


def test_forward_fill_and_time_since_last_follow_grud_definition():
    z = np.full((1, 6, 1), np.nan, dtype=np.float32)
    z[0, 1, 0], z[0, 4, 0] = 2.0, -1.0
    np.testing.assert_array_equal(F.forward_fill(z)[0, :, 0], [0, 2, 2, 2, -1, -1])
    delta = F.time_since_last(~np.isnan(z))
    # delta_1 = 0; then 1 + previous delta while unmeasured, reset to 1 after a measurement.
    np.testing.assert_array_equal(delta[0, :, 0], [0, 1, 1, 2, 3, 1])


def test_summary_features_statistics_and_feature_sets():
    obs = _obs([(1, 0, "HR", 80.0), (1, 60, "HR", 100.0), (1, 120, "HR", 90.0), (2, 30, "Na", 140.0)])
    stays = pd.DataFrame({"Age": [70.0, 50.0], "Gender": [1.0, 0.0], "Height": [np.nan, 170.0],
                          "Weight": [80.0, 60.0], "ICUType": [3.0, 1.0]}, index=pd.Index([1, 2], name="record_id"))
    f = F.summary_features(obs, stays)
    row = f.loc[1]
    assert (row["HR__first"], row["HR__last"], row["HR__min"], row["HR__max"], row["HR__mean"]) == (80, 90, 80, 100, 90)
    assert (row["HR__n"], row["HR__t_first"], row["HR__t_last"]) == (3, 0, 2)
    assert f.loc[2, "HR__n"] == 0 and np.isnan(f.loc[2, "HR__mean"])
    assert f.loc[1, "icu_3"] == 1 and f.loc[2, "icu_1"] == 1
    cols = list(f.columns)
    values, process = F.feature_set(cols, "values"), F.feature_set(cols, "process")
    assert not set(values) & set(process)
    assert set(values) | set(process) == set(cols)
    assert all(c.split("__")[1] in F.PROCESS_STATS for c in process)
    assert F.feature_variable("Weight") == F.feature_variable("Weight__max") == "Weight"
    assert F.feature_variable("icu_2") == "ICUType"


def test_scaler_uses_log_for_skewed_variables_and_keeps_nan():
    x = np.full((2, 3, len(TIME_SERIES)), np.nan, dtype=np.float32)
    u = TIME_SERIES.index("Urine")
    x[:, :, u] = [[0, 100, 1000], [10, 50, np.nan]]
    sc = F.Scaler.fit(x)
    z = sc(x)
    assert np.isnan(z[1, 2, u])
    assert abs(np.nanmean(z[:, :, u])) < 1e-5
    assert sc.mean[u] == np.nanmean(np.log1p(x[:, :, u]))
