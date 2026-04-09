from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from comparison_config import EncoderFitConfig, VAMPNetConfig


def build_encoder_basis(observations: np.ndarray, basis: str = "state") -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float64)
    if basis == "state":
        return obs
    if basis == "augmented":
        x = obs[:, 6]
        y = obs[:, 7]
        dist = obs[:, 0]
        pieces = [
            obs,
            (x * y)[:, None],
            (x * x)[:, None],
            (y * y)[:, None],
            np.sin(2.0 * np.pi * x)[:, None],
            np.sin(2.0 * np.pi * y)[:, None],
            np.cos(2.0 * np.pi * x)[:, None],
            np.cos(2.0 * np.pi * y)[:, None],
            (dist * x)[:, None],
            (dist * y)[:, None],
        ]
        return np.concatenate(pieces, axis=1)
    raise ValueError(f"Unknown encoder basis: {basis}")


class BaseEncoder:
    name = "base"

    def fit(self, observations: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, observation: np.ndarray) -> np.ndarray:
        return self.transform_batch(np.asarray(observation, dtype=np.float32)[None, :])[0]

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    name = "identity"

    def __init__(self, input_dim: int):
        self._output_dim = input_dim

    def fit(self, observations: np.ndarray) -> None:
        self._output_dim = int(observations.shape[1])

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class TICAEncoder(BaseEncoder):
    name = "tica"

    def __init__(self, fit_cfg: EncoderFitConfig):
        self.fit_cfg = fit_cfg
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self._output_dim = fit_cfg.n_components

    def fit(self, observations: np.ndarray) -> None:
        feats = build_encoder_basis(observations, self.fit_cfg.feature_basis)
        lag = max(1, int(self.fit_cfg.lagtime))
        if feats.shape[0] <= lag + 2:
            raise ValueError("Not enough samples to fit TICA encoder.")
        x0 = feats[:-lag]
        xt = feats[lag:]
        self.mean_ = np.mean(np.concatenate([x0, xt], axis=0), axis=0)
        x0 = x0 - self.mean_
        xt = xt - self.mean_
        cov_00 = (x0.T @ x0) / max(1, x0.shape[0] - 1)
        cov_0t = (x0.T @ xt) / max(1, x0.shape[0] - 1)
        cov_00 = 0.5 * (cov_00 + cov_00.T) + 1e-6 * np.eye(cov_00.shape[0])
        cov_sym = 0.5 * (cov_0t + cov_0t.T)
        eigvals, eigvecs = scipy.linalg.eigh(cov_sym, cov_00)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.real(eigvals[order])
        eigvecs = np.real(eigvecs[:, order])
        n_comp = min(self.fit_cfg.n_components, eigvecs.shape[1])
        self.components_ = eigvecs[:, :n_comp]
        self.eigenvalues_ = eigvals[:n_comp]
        self._output_dim = int(n_comp)

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("TICA encoder must be fitted before transform.")
        feats = build_encoder_basis(observations, self.fit_cfg.feature_basis)
        latent = (feats - self.mean_) @ self.components_
        return latent.astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class _VAMPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_sizes, output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAMPNetEncoder(BaseEncoder):
    name = "vampnet"

    def __init__(self, fit_cfg: EncoderFitConfig, vamp_cfg: VAMPNetConfig):
        self.fit_cfg = fit_cfg
        self.vamp_cfg = vamp_cfg
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.model: Optional[_VAMPNet] = None
        self._output_dim = vamp_cfg.n_components

    def fit(self, observations: np.ndarray) -> None:
        feats = build_encoder_basis(observations, self.fit_cfg.feature_basis)
        lag = max(1, int(self.vamp_cfg.lagtime))
        if feats.shape[0] <= lag + 2:
            raise ValueError("Not enough samples to fit VAMPNet encoder.")
        x0 = feats[:-lag]
        xt = feats[lag:]
        self.mean_ = np.mean(np.concatenate([x0, xt], axis=0), axis=0)
        self.std_ = np.std(np.concatenate([x0, xt], axis=0), axis=0)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        x0 = (x0 - self.mean_) / self.std_
        xt = (xt - self.mean_) / self.std_

        self.model = _VAMPNet(
            input_dim=x0.shape[1],
            hidden_sizes=list(self.vamp_cfg.hidden_sizes),
            output_dim=self.vamp_cfg.n_components,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.vamp_cfg.lr,
            weight_decay=self.vamp_cfg.weight_decay,
        )
        dataset = TensorDataset(
            torch.tensor(x0, dtype=torch.float32),
            torch.tensor(xt, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.vamp_cfg.batch_size, shuffle=True, drop_last=False)
        self.model.train()
        for _ in range(self.vamp_cfg.epochs):
            for batch_x0, batch_xt in loader:
                z0 = self.model(batch_x0)
                zt = self.model(batch_xt)
                loss = -self._vamp2_score(z0, zt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        self._output_dim = int(self.vamp_cfg.n_components)

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("VAMPNet encoder must be fitted before transform.")
        feats = build_encoder_basis(observations, self.fit_cfg.feature_basis)
        feats = (feats - self.mean_) / self.std_
        with torch.no_grad():
            latent = self.model(torch.tensor(feats, dtype=torch.float32)).cpu().numpy()
        return latent.astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _vamp2_score(self, x0: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        x0 = x0 - x0.mean(dim=0, keepdim=True)
        xt = xt - xt.mean(dim=0, keepdim=True)
        c00 = (x0.T @ x0) / max(1, x0.shape[0] - 1)
        ctt = (xt.T @ xt) / max(1, xt.shape[0] - 1)
        c0t = (x0.T @ xt) / max(1, x0.shape[0] - 1)
        eps_eye = torch.eye(c00.shape[0], dtype=x0.dtype, device=x0.device) * self.vamp_cfg.score_eps
        c00 = c00 + eps_eye
        ctt = ctt + eps_eye
        c00_inv_sqrt = self._matrix_inv_sqrt(c00)
        ctt_inv_sqrt = self._matrix_inv_sqrt(ctt)
        vamp_matrix = c00_inv_sqrt @ c0t @ ctt_inv_sqrt
        return torch.sum(vamp_matrix * vamp_matrix)

    @staticmethod
    def _matrix_inv_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=1e-6)
        inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        return inv_sqrt
