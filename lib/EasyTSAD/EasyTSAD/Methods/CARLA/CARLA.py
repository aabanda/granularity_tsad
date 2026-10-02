"""CARLA-style sequence-wise detector for EasyTSAD / UCR.

Self-contained adaptation of Zamanzadeh et al. (Pattern Recognition 2025):
contrastive pretext with anomaly injection + self-supervised classification
over windows. The official CARLA repo hard-codes dataset paths and lacks a
working ``data/UCR.py``; this wrapper follows the same two-stage idea on the
EasyTSAD ``TSData`` train/test split and returns a point-wise anomaly score
(higher = more anomalous) by expanding window-level normal-class uncertainty.

Not a byte-for-byte re-run of their Yahoo/MSL scripts; hyperparameters are
kept close to their univariate (Yahoo) defaults but shortened for UCR scale.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ...DataFactory import TSData
from .. import BaseMethod


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _windows(x: np.ndarray, window: int, stride: int = 1, max_windows: int | None = None) -> tuple[np.ndarray, int]:
    """Return (windows, effective_stride). Caps count on long series."""
    x = _as_1d(x)
    if len(x) < window:
        pad = np.full(window - len(x), x[-1] if len(x) else 0.0)
        x = np.concatenate([x, pad])
    eff_stride = max(1, int(stride))
    n_win = 1 + (len(x) - window) // eff_stride
    if max_windows is not None and n_win > max_windows:
        eff_stride = max(1, int(np.ceil((len(x) - window + 1) / max_windows)))
        n_win = 1 + (len(x) - window) // eff_stride
    idx = range(0, len(x) - window + 1, eff_stride)
    return np.stack([x[i : i + window] for i in idx], axis=0), eff_stride


def _inject_sub_anomaly(batch: torch.Tensor, portion: float) -> torch.Tensor:
    """Corrupt a random contiguous fraction of each window (CARLA-style)."""
    out = batch.clone()
    b, w = out.shape
    plen = max(1, int(w * float(portion)))
    for i in range(b):
        start = int(torch.randint(0, max(1, w - plen + 1), (1,)).item())
        end = start + plen
        scale = float(torch.empty(1).uniform_(0.1, 2.0).item())
        noise = torch.randn(plen, device=out.device) * 0.05
        out[i, start:end] = out[i, start:end] * scale + noise
    return out


class _Conv1dSame(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        kernel, dilation, stride = self.weight.size(2), self.dilation[0], self.stride[0]
        l_in = x.size(2)
        padding = ((l_in - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
        if padding % 2 != 0:
            x = F.pad(x, [0, 1])
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding // 2,
            self.dilation,
            self.groups,
        )


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _Conv1dSame(in_ch, out_ch, kernel_size=k, stride=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _ConvBlock(in_ch, out_ch, 8),
            _ConvBlock(out_ch, out_ch, 5),
            _ConvBlock(out_ch, out_ch, 3),
        )
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                _Conv1dSame(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        if self.proj is not None:
            x = self.proj(x)
        return y + x


class _ResNetTS(nn.Module):
    """Univariate ResNet encoder + optional classification head (CARLA-like)."""

    def __init__(self, mid_channels: int, features_dim: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ResBlock(1, mid_channels),
            _ResBlock(mid_channels, mid_channels * 2),
            _ResBlock(mid_channels * 2, mid_channels * 2),
        )
        feat_in = mid_channels * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_in, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
        )
        self.head = nn.Linear(features_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W) → (B, 1, W)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.backbone(x).mean(dim=-1)
        return self.proj(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.head(z)


def _expand_window_scores(win_scores: np.ndarray, n: int, window: int, stride: int) -> np.ndarray:
    """Max-pool window scores onto timestamps (sequence-wise → point score)."""
    out = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    for i, s in enumerate(win_scores):
        start = i * stride
        end = min(start + window, n)
        out[start:end] = np.maximum(out[start:end], s)
        counts[start:end] += 1.0
    # Leading positions never covered keep 0; covered positions already max-pooled.
    return out


class CARLA(BaseMethod):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__()
        params = params or {}
        self.window = int(params.get("window", 64))
        self.stride = int(params.get("stride", 1))
        self.max_windows = int(params.get("max_windows", 1500))
        self.mid_channels = int(params.get("mid_channels", 4))
        self.features_dim = int(params.get("features_dim", 8))
        self.n_classes = int(params.get("n_classes", 4))
        self.pretext_epochs = int(params.get("pretext_epochs", 8))
        self.classification_epochs = int(params.get("classification_epochs", 10))
        self.batch_size = int(params.get("batch_size", 32))
        self.lr = float(params.get("lr", 1e-3))
        self.seed = int(params.get("seed", 4))
        self.anomaly_portion = float(params.get("anomaly_portion", 0.3))
        self.device = self._pick_device()
        self.model: _ResNetTS | None = None
        self._mu = 0.0
        self._sigma = 1.0
        self.__anomaly_score: np.ndarray | None = None
        self.score_train = None

    @staticmethod
    def _pick_device() -> torch.device:
        try:
            if torch.cuda.is_available():
                # Probe a tiny allocation; some hosts report CUDA but fail at use.
                torch.zeros(1, device="cuda")
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")

    def _set_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

    def _normalize(self, x: np.ndarray, fit: bool) -> np.ndarray:
        x = _as_1d(x)
        if fit:
            self._mu = float(np.mean(x))
            self._sigma = float(np.std(x)) or 1.0
        return (x - self._mu) / self._sigma

    def _make_model(self) -> _ResNetTS:
        return _ResNetTS(self.mid_channels, self.features_dim, self.n_classes).to(self.device)

    def _pretext(self, train_x: np.ndarray) -> None:
        assert self.model is not None
        wins, _ = _windows(train_x, self.window, self.stride, self.max_windows)
        if len(wins) < 4:
            # Duplicate until we have a tiny batch.
            wins = np.repeat(wins, max(4 // max(len(wins), 1), 1), axis=0)
        ds = TensorDataset(torch.from_numpy(wins.astype(np.float32)))
        bs = min(self.batch_size, len(ds))
        loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.model.train()
        for _ in range(self.pretext_epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                # Positive: temporally previous neighbour (roll); negative: injected anomaly.
                pos = torch.roll(batch, shifts=1, dims=0)
                neg = _inject_sub_anomaly(batch, self.anomaly_portion)
                za, _ = self.model(batch)
                zp, _ = self.model(pos)
                zn, _ = self.model(neg)
                # Triplet margin on L2-normalized embeddings.
                za = F.normalize(za, dim=1)
                zp = F.normalize(zp, dim=1)
                zn = F.normalize(zn, dim=1)
                d_pos = (za - zp).pow(2).sum(1)
                d_neg = (za - zn).pow(2).sum(1)
                loss = F.relu(d_pos - d_neg + 0.5).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _classification(self, train_x: np.ndarray) -> int:
        """Train cluster head; return majority (normal) class id on train windows."""
        assert self.model is not None
        wins, _ = _windows(train_x, self.window, self.stride, self.max_windows)
        if len(wins) < 2:
            wins = np.repeat(wins, 2, axis=0)
        ds = TensorDataset(torch.from_numpy(wins.astype(np.float32)))
        bs = min(self.batch_size, len(ds))
        loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr * 10.0, weight_decay=1e-3)
        self.model.train()
        for _ in range(self.classification_epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                # Soft self-labelling via nearest/furthest neighbour in-batch.
                z, logits = self.model(batch)
                with torch.no_grad():
                    z_n = F.normalize(z, dim=1)
                    sim = z_n @ z_n.t()
                    sim.fill_diagonal_(-1e9)
                    nn_idx = sim.argmax(dim=1)
                    fn_idx = sim.argmin(dim=1)
                # Pull toward NN logits, push from FN; entropy regularization.
                log_p = F.log_softmax(logits, dim=1)
                p = log_p.exp()
                loss_nn = F.nll_loss(log_p, logits[nn_idx].argmax(dim=1))
                loss_fn = -F.nll_loss(log_p, logits[fn_idx].argmax(dim=1))
                ent = -(p * log_p).sum(1).mean()
                loss = loss_nn + 0.1 * loss_fn - 2.0 * ent
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model.eval()
        with torch.no_grad():
            preds = []
            for (batch,) in DataLoader(ds, batch_size=bs, shuffle=False):
                _, logits = self.model(batch.to(self.device))
                preds.append(logits.argmax(dim=1).cpu())
            pred = torch.cat(preds)
            majority = int(torch.bincount(pred, minlength=self.n_classes).argmax().item())
        return majority

    def _score_series(self, x: np.ndarray, normal_label: int) -> np.ndarray:
        assert self.model is not None
        wins, eff_stride = _windows(x, self.window, self.stride, self.max_windows)
        self.model.eval()
        scores_w = []
        with torch.no_grad():
            for i in range(0, len(wins), self.batch_size):
                batch = torch.from_numpy(wins[i : i + self.batch_size].astype(np.float32)).to(
                    self.device
                )
                _, logits = self.model(batch)
                prob = F.softmax(logits, dim=1)
                # Anomaly score = 1 - P(normal / majority class).
                s = 1.0 - prob[:, normal_label]
                scores_w.append(s.cpu().numpy())
        win_scores = np.concatenate(scores_w) if scores_w else np.zeros(1)
        return _expand_window_scores(win_scores, len(x), self.window, eff_stride)

    def train_valid_phase(self, tsTrain: TSData) -> None:
        self._set_seed()
        train = self._normalize(_as_1d(tsTrain.train), fit=True)
        valid = (
            self._normalize(_as_1d(tsTrain.valid), fit=False)
            if getattr(tsTrain, "valid", None) is not None
            else np.array([])
        )
        train_all = np.concatenate([train, valid]) if valid.size else train
        self.model = self._make_model()
        self._pretext(train_all)
        self._normal_label = self._classification(train_all)
        self.score_train = self._score_series(train, self._normal_label)

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]) -> None:
        pass

    def test_phase(self, tsData: TSData) -> None:
        if self.model is None:
            # In case train was skipped; fit on train then score test.
            self.train_valid_phase(tsData)
        test = self._normalize(_as_1d(tsData.test), fit=False)
        normal_label = getattr(self, "_normal_label", 0)
        self.__anomaly_score = self._score_series(test, normal_label)

    def anomaly_score(self) -> np.ndarray:
        assert self.__anomaly_score is not None
        return self.__anomaly_score
