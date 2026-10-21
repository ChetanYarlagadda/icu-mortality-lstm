import numpy as np
import pandas as pd
import torch

from icumort.explain import grouped_permutation_importance, shap_by_variable
from icumort.features import SeqTensors
from icumort.metrics import auroc
from icumort.models import GRUD, ModelSpec, TrainConfig, build, predict_logits, to_device, train

torch.set_num_threads(2)


def _synthetic(n: int, seed: int) -> tuple[SeqTensors, np.ndarray]:
    """Variable 0 is measured more often in deaths; variable 1's value is higher in deaths; variable 2 is noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    mask = rng.random((n, 12, 3)) < np.where(y[:, None, None] == 1, [0.6, 0.3, 0.3], [0.15, 0.3, 0.3])
    x = rng.normal(size=(n, 12, 3)).astype(np.float32)
    x[:, :, 1] += 1.5 * y[:, None]
    x = np.where(mask, x, 0).astype(np.float32)
    delta = np.ones_like(x)
    return SeqTensors(x, x.copy(), mask.astype(np.float32), delta, rng.normal(size=(n, 2)).astype(np.float32)), y


def test_grud_uses_the_measurement_when_every_value_is_observed():
    cfg = TrainConfig(hidden=8)
    torch.manual_seed(0)
    m = GRUD(ModelSpec("GRU-D", "grud"), 3, 2, cfg)
    t, _ = _synthetic(5, 0)
    inp = to_device(t, torch.device("cpu"))
    inp["mask"][:] = 1.0
    other = dict(inp, x_ffill=inp["x_ffill"] + 100.0)  # last values must not matter when everything is measured
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(**inp), m(**other))
        assert m(**inp).shape == (5,)


def test_every_architecture_learns_a_simple_signal_on_cpu():
    fit_t, y_fit = _synthetic(600, 1)
    val_t, y_val = _synthetic(300, 2)
    dev = torch.device("cpu")
    fit, val = to_device(fit_t, dev), to_device(val_t, dev)
    cfg = TrainConfig(hidden=16, max_epochs=8, patience=8, lr=3e-3, batch_size=64)
    for spec in (ModelSpec("LSTM", "lstm"), ModelSpec("GRU-D", "grud"),
                 ModelSpec("process only", "gru", use_values=False, use_static=False)):
        model, hist = train(spec, cfg, fit, y_fit, val, y_val, seed=0)
        assert len(hist) == 8 and hist["val_loss"].notna().all()
        assert auroc(predict_logits(model, val), y_val) > 0.75, spec.name


def test_build_rejects_a_model_without_inputs():
    try:
        build(ModelSpec("none", "gru", use_values=False, use_process=False), 3, 2, TrainConfig())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grouped_permutation_importance_finds_the_informative_group():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    y = (rng.random(3000) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(int)

    def predict(g, perm):
        Xp = X.copy()
        if g is not None:
            Xp[:, int(g)] = X[perm, int(g)]
        return 1 / (1 + np.exp(-2 * Xp[:, 0]))

    imp = grouped_permutation_importance(predict, ["0", "1", "2"], y, n_repeats=2)
    assert imp.iloc[0]["variable"] == "0" and imp.iloc[0]["auroc_drop"] > 0.2
    assert imp.set_index("variable").loc[["1", "2"], "auroc_drop"].abs().max() < 1e-12


def test_shap_by_variable_sums_a_variables_features_before_averaging():
    contrib = np.array([[1.0, -1.0, 0.5, 0.0], [2.0, 0.0, -0.5, 0.0]])  # last column is the bias
    out = shap_by_variable(contrib, ["HR__min", "HR__max", "Age"]).set_index("variable")["mean_abs_shap"]
    assert out["HR"] == 1.0 and out["Age"] == 0.5
    assert isinstance(out, pd.Series)
