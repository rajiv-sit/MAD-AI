from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mad_ai.core.base import BaseAnomalyModel


class _SpatialAutoencoder(nn.Module):
    def __init__(self, channels: int, latent_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        return self.decoder(encoded)


class CNNAnomalyModel(BaseAnomalyModel):
    """Convolutional autoencoder baseline with normalization and mini-batch training."""

    def __init__(
        self,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        latent_channels: int = 32,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_channels = latent_channels
        self.dropout = dropout
        self.device = torch.device(device or "cpu")
        self.model: _SpatialAutoencoder | None = None
        self.input_shape: tuple[int, int, int] | None = None
        self.channel_mean: np.ndarray | None = None
        self.channel_std: np.ndarray | None = None
        self.training_history: list[float] = []

    def train(self, train_data, val_data=None) -> None:
        _ = val_data
        array = self._as_batch(train_data)
        normalized = self._fit_normalize(array)
        tensor = self._to_tensor(normalized)
        channels = tensor.shape[1]
        self.input_shape = tuple(array.shape[1:])
        self.model = _SpatialAutoencoder(
            channels=channels,
            latent_channels=self.latent_channels,
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
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    def score(self, batch):
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        array = self._as_batch(batch)
        normalized = self._apply_normalize(array)
        tensor = self._to_tensor(normalized)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor)
            errors = torch.mean(torch.abs(reconstructed - tensor), dim=(1, 2, 3))
        scores = errors.cpu().numpy()
        return float(scores[0]) if np.asarray(batch).ndim == 3 else scores

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_shape is None or self.channel_mean is None or self.channel_std is None:
            raise RuntimeError("Model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_shape": self.input_shape,
                "channel_mean": self.channel_mean.tolist(),
                "channel_std": self.channel_std.tolist(),
                "latent_channels": self.latent_channels,
                "dropout": self.dropout,
            },
            path,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "kind": "spatial-autoencoder",
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "training_history": self.training_history,
                }
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self.input_shape = tuple(payload["input_shape"])
        self.channel_mean = np.asarray(payload["channel_mean"], dtype=np.float32)
        self.channel_std = np.asarray(payload["channel_std"], dtype=np.float32)
        self.latent_channels = int(payload.get("latent_channels", self.latent_channels))
        self.dropout = float(payload.get("dropout", self.dropout))
        channels = int(self.input_shape[2])
        self.model = _SpatialAutoencoder(
            channels=channels,
            latent_channels=self.latent_channels,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _as_batch(self, data) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 3:
            return array[np.newaxis, ...]
        if array.ndim != 4:
            raise ValueError("Expected spatial data with shape (H, W, C) or (N, H, W, C).")
        return array

    def _fit_normalize(self, array: np.ndarray) -> np.ndarray:
        self.channel_mean = array.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        self.channel_std = array.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        self.channel_std = np.where(self.channel_std < 1e-6, 1.0, self.channel_std)
        return self._apply_normalize(array)

    def _apply_normalize(self, array: np.ndarray) -> np.ndarray:
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError("Normalization statistics are not available.")
        return (array - self.channel_mean) / self.channel_std

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
