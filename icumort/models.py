"""Recurrent mortality models in PyTorch: LSTM / GRU on values, mask and time gaps, and GRU-D (Che et al., 2018).

Every model reads the 48 hourly bins of a stay and the admission descriptors and outputs one logit for in-hospital
death. Inputs per hour and variable:

- ``x``: standardised mean value of the bin (0 where unmeasured),
- ``x_ffill``: the last measured value carried forward (0, the training mean, before the first measurement),
- ``mask``: 1 if the variable was measured in that hour,
- ``delta``: hours since the previous measurement.

The LSTM / GRU take any combination of ``x_ffill``, ``mask`` and ``delta / 48`` as the per-step input. GRU-D replaces
carry-forward imputation by a learned decay towards the training mean, x_hat = m x + (1 - m)(g x_last + (1 - g) x_mean)
with g = exp(-max(0, w * delta + b)), and decays the hidden state between steps with
g_h = exp(-max(0, W delta + b)).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from icumort.features import SeqTensors
from icumort.metrics import auroc


@dataclass(frozen=True)
class ModelSpec:
    """Architecture and inputs of one recurrent model."""

    name: str
    kind: str  # "lstm", "gru" or "grud"
    use_values: bool = True
    use_process: bool = True  # mask and time since last measurement
    use_static: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings shared by every recurrent model."""

    hidden: int = 64
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0


def _head(n_in: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))


class RNNClassifier(nn.Module):
    """LSTM or GRU over the hourly inputs; the last hidden state and the descriptors feed a small MLP head."""

    def __init__(self, spec: ModelSpec, n_var: int, n_static: int, cfg: TrainConfig) -> None:
        super().__init__()
        if spec.kind not in ("lstm", "gru"):
            raise ValueError(spec.kind)
        self.spec = spec
        n_in = n_var * (int(spec.use_values) + 2 * int(spec.use_process))
        if n_in == 0:
            raise ValueError("a model needs values or the measurement process as input")
        rnn = nn.LSTM if spec.kind == "lstm" else nn.GRU
        self.rnn = rnn(n_in, cfg.hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU()) if spec.use_static else None
        self.head = _head(cfg.hidden + (32 if spec.use_static else 0), cfg.dropout)

    def forward(self, x: torch.Tensor, x_ffill: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.spec.use_values:
            parts.append(x_ffill)
        if self.spec.use_process:
            parts += [mask, delta / 48.0]
        out, _ = self.rnn(torch.cat(parts, dim=-1))
        h = out[:, -1]
        if self.static is not None:
            h = torch.cat([h, self.static(static)], dim=-1)
        return self.head(h).squeeze(-1)


class GRUD(nn.Module):
    """GRU-D: GRU with trainable input decay towards the mean and hidden-state decay driven by the time gaps."""

    def __init__(self, spec: ModelSpec, n_var: int, n_static: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.spec = spec
        self.w_x = nn.Parameter(torch.zeros(n_var))
        self.b_x = nn.Parameter(torch.zeros(n_var))
        self.decay_h = nn.Linear(n_var, cfg.hidden)
        self.cell = nn.GRUCell(2 * n_var, cfg.hidden)
        self.register_buffer("x_mean", torch.zeros(n_var))  # inputs are standardised on the training stays
        self.static = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU()) if spec.use_static else None
        self.head = _head(cfg.hidden + (32 if spec.use_static else 0), cfg.dropout)
        self.hidden = cfg.hidden

    def forward(self, x: torch.Tensor, x_ffill: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        g_x = torch.exp(-torch.relu(self.w_x * delta + self.b_x))
        x_hat = mask * x + (1 - mask) * (g_x * x_ffill + (1 - g_x) * self.x_mean)
        g_h = torch.exp(-torch.relu(self.decay_h(delta)))
        inp = torch.cat([x_hat, mask], dim=-1)
        h = x.new_zeros(x.shape[0], self.hidden)
        for t in range(x.shape[1]):
            h = self.cell(inp[:, t], g_h[:, t] * h)
        if self.static is not None:
            h = torch.cat([h, self.static(static)], dim=-1)
        return self.head(h).squeeze(-1)


def build(spec: ModelSpec, n_var: int, n_static: int, cfg: TrainConfig) -> nn.Module:
    """Instantiate the model described by ``spec``."""
    return GRUD(spec, n_var, n_static, cfg) if spec.kind == "grud" else RNNClassifier(spec, n_var, n_static, cfg)


def to_device(t: SeqTensors, device: torch.device) -> dict[str, torch.Tensor]:
    """Move every input array to ``device`` as float32 tensors keyed like the forward() arguments."""
    return {k: torch.as_tensor(getattr(t, k), dtype=torch.float32, device=device)
            for k in ("x", "x_ffill", "mask", "delta", "static")}


def _batches(n: int, size: int, gen: torch.Generator | None, device: torch.device):
    order = torch.randperm(n, generator=gen).to(device) if gen is not None else torch.arange(n, device=device)
    for i in range(0, n, size):
        yield order[i:i + size]


@torch.no_grad()
def predict_logits(model: nn.Module, inputs: dict[str, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
    """Logits for every stay in ``inputs`` (evaluation mode)."""
    model.eval()
    n = inputs["x"].shape[0]
    out = [model(**{k: v[idx] for k, v in inputs.items()}) for idx in _batches(n, batch_size, None, inputs["x"].device)]
    return torch.cat(out).float().cpu().numpy()


def train(spec: ModelSpec, cfg: TrainConfig, fit: dict[str, torch.Tensor], y_fit: np.ndarray,
          val: dict[str, torch.Tensor], y_val: np.ndarray, seed: int) -> tuple[nn.Module, pd.DataFrame]:
    """Train with Adam and binary cross-entropy, keeping the epoch with the lowest validation loss.

    Returns the model with the best weights and the per-epoch history (train loss, validation loss and AUROC).
    """
    device = fit["x"].device
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(spec, fit["x"].shape[-1], fit["static"].shape[-1], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    yf = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, rows = np.inf, None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total = 0.0
        for idx in _batches(len(yf), cfg.batch_size, gen, device):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(**{k: v[idx] for k, v in fit.items()}), yf[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(idx)
        logits = predict_logits(model, val)
        val_loss = float(loss_fn(torch.as_tensor(logits, device=device), yv))
        rows.append({"epoch": epoch, "train_loss": total / len(yf), "val_loss": val_loss,
                     "val_auroc": auroc(logits, y_val), "seconds": time.perf_counter() - t0})
        if val_loss < best - 1e-4:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def tune(fit: dict[str, torch.Tensor], y_fit: np.ndarray, val: dict[str, torch.Tensor], y_val: np.ndarray,
         base: TrainConfig, n_trials: int, seed: int = 0) -> tuple[TrainConfig, pd.DataFrame]:
    """Optuna TPE search of the GRU's width, dropout, learning rate, weight decay and batch size.

    The objective is the best validation loss; the returned configuration is shared by every recurrent model.
    """
    spec = ModelSpec("GRU", "gru")

    def objective(trial: optuna.Trial) -> float:
        cfg = replace(base, hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
                      dropout=trial.suggest_float("dropout", 0.0, 0.5),
                      lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                      weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                      batch_size=trial.suggest_categorical("batch_size", [128, 256]))
        _, hist = train(spec, cfg, fit, y_fit, val, y_val, seed)
        trial.set_user_attr("epochs", len(hist))
        trial.set_user_attr("val_auroc", float(hist.loc[hist["val_loss"].idxmin(), "val_auroc"]))
        return float(hist["val_loss"].min())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return replace(base, **study.best_params), trials
