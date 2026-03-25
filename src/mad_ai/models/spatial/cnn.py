from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mad_ai.core.base import BaseAnomalyModel


class _SpatialAutoencoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CNNAnomalyModel(BaseAnomalyModel):
    """Trainable convolutional autoencoder for spatial anomaly scoring."""

    def __init__(self, epochs: int = 20, learning_rate: float = 1e-3, device: str | None = None) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device or "cpu")
        self.model: _SpatialAutoencoder | None = None
        self.input_shape: tuple[int, int, int] | None = None

    def train(self, train_data, val_data=None) -> None:
        _ = val_data
        array = self._as_batch(train_data)
        tensor = self._to_tensor(array)
        channels = tensor.shape[1]
        self.input_shape = tuple(array.shape[1:])
        self.model = _SpatialAutoencoder(channels).to(self.device)

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
        tensor = self._to_tensor(array)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor)
            errors = torch.mean(torch.abs(reconstructed - tensor), dim=(1, 2, 3))
        scores = errors.cpu().numpy()
        return float(scores[0]) if np.asarray(batch).ndim == 3 else scores

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_shape is None:
            raise RuntimeError("Model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_shape": self.input_shape,
            },
            path,
        )
        path.with_suffix(".json").write_text(json.dumps({"kind": "spatial-autoencoder"}), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.input_shape = tuple(payload["input_shape"])
        channels = int(self.input_shape[2])
        self.model = _SpatialAutoencoder(channels).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _as_batch(self, data) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 3:
            return array[np.newaxis, ...]
        if array.ndim != 4:
            raise ValueError("Expected spatial data with shape (H, W, C) or (N, H, W, C).")
        return array

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
