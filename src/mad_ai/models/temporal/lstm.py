from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mad_ai.core.base import BaseAnomalyModel


class _TemporalAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(inputs)
        return self.projection(encoded)


class LSTMAnomalyModel(BaseAnomalyModel):
    """LSTM autoencoder baseline with feature normalization and mini-batch training."""

    def __init__(
        self,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        hidden_size: int = 48,
        batch_size: int = 16,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.device = torch.device(device or "cpu")
        self.model: _TemporalAutoencoder | None = None
        self.input_shape: tuple[int, int] | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.training_history: list[float] = []

    def train(self, train_data, val_data=None) -> None:
        _ = val_data
        array = self._as_batch(train_data)
        normalized = self._fit_normalize(array)
        tensor = torch.from_numpy(normalized).to(self.device)
        input_size = tensor.shape[2]
        self.input_shape = tuple(array.shape[1:])
        self.model = _TemporalAutoencoder(
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.SmoothL1Loss()
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.training_history = []
        self.model.train()
        for _epoch in range(self.epochs):
            epoch_losses: list[float] = []
            for (batch_tensor,) in loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_tensor)
                loss = loss_fn(reconstructed, batch_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    def score(self, batch):
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        array = self._as_batch(batch)
        normalized = self._apply_normalize(array)
        tensor = torch.from_numpy(normalized).to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor)
            errors = torch.mean(torch.abs(reconstructed - tensor), dim=(1, 2))
        scores = errors.cpu().numpy()
        return float(scores[0]) if np.asarray(batch).ndim == 2 else scores

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_shape is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_shape": self.input_shape,
                "hidden_size": self.hidden_size,
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
            },
            path,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "kind": "temporal-autoencoder",
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "hidden_size": self.hidden_size,
                    "training_history": self.training_history,
                }
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.input_shape = tuple(payload["input_shape"])
        self.hidden_size = int(payload["hidden_size"])
        self.batch_size = int(payload.get("batch_size", self.batch_size))
        self.dropout = float(payload.get("dropout", self.dropout))
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.model = _TemporalAutoencoder(
            input_size=int(self.input_shape[1]),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _as_batch(self, data) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 2:
            return array[np.newaxis, ...]
        if array.ndim != 3:
            raise ValueError("Expected temporal data with shape (T, F) or (N, T, F).")
        return array

    def _fit_normalize(self, array: np.ndarray) -> np.ndarray:
        self.feature_mean = array.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.feature_std = array.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        return self._apply_normalize(array)

    def _apply_normalize(self, array: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Normalization statistics are not available.")
        return (array - self.feature_mean) / self.feature_std
