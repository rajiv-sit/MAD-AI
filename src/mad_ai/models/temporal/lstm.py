from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mad_ai.core.base import BaseAnomalyModel


class _TemporalAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(inputs)
        return self.decoder(encoded)


class LSTMAnomalyModel(BaseAnomalyModel):
    """Trainable LSTM-based sequence autoencoder for temporal anomaly scoring."""

    def __init__(self, epochs: int = 20, learning_rate: float = 1e-3, hidden_size: int = 32, device: str | None = None) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.device = torch.device(device or "cpu")
        self.model: _TemporalAutoencoder | None = None
        self.input_shape: tuple[int, int] | None = None

    def train(self, train_data, val_data=None) -> None:
        _ = val_data
        array = self._as_batch(train_data)
        tensor = torch.from_numpy(array).to(self.device)
        input_size = tensor.shape[2]
        self.input_shape = tuple(array.shape[1:])
        self.model = _TemporalAutoencoder(input_size=input_size, hidden_size=self.hidden_size).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model(tensor)
            loss = loss_fn(reconstructed, tensor)
            loss.backward()
            optimizer.step()

    def score(self, batch):
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        array = self._as_batch(batch)
        tensor = torch.from_numpy(array).to(self.device)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor)
            errors = torch.mean(torch.abs(reconstructed - tensor), dim=(1, 2))
        scores = errors.cpu().numpy()
        return float(scores[0]) if np.asarray(batch).ndim == 2 else scores

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_shape is None:
            raise RuntimeError("Model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_shape": self.input_shape,
                "hidden_size": self.hidden_size,
            },
            path,
        )
        path.with_suffix(".json").write_text(json.dumps({"kind": "temporal-autoencoder"}), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.input_shape = tuple(payload["input_shape"])
        self.hidden_size = int(payload["hidden_size"])
        self.model = _TemporalAutoencoder(input_size=int(self.input_shape[1]), hidden_size=self.hidden_size).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _as_batch(self, data) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 2:
            return array[np.newaxis, ...]
        if array.ndim != 3:
            raise ValueError("Expected temporal data with shape (T, F) or (N, T, F).")
        return array
