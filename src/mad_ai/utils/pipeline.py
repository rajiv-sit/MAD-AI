from __future__ import annotations

from pathlib import Path

import pandas as pd

from mad_ai.datasets import SpatialGridBuilder, TemporalSequenceBuilder
from mad_ai.features import ResidualFeatureBuilder, TemporalFeatureBuilder
from mad_ai.utils.persistence import ArtifactStore
from mad_ai.wmm import WMMMagneticModel


def build_processed_artifacts(
    raw: pd.DataFrame,
    store: ArtifactStore,
    wmm_model: WMMMagneticModel,
    spatial_name: str = "spatial_grid",
    sequence_name: str = "temporal_sequences",
    enriched_name: str = "enriched_samples",
    temporal_name: str = "temporal_samples",
    metadata_name: str = "dataset_metadata",
) -> dict[str, Path]:
    enriched = ResidualFeatureBuilder(wmm_model).transform(raw)
    temporal = TemporalFeatureBuilder().transform(enriched)
    grid = SpatialGridBuilder().build(temporal)
    sequences = TemporalSequenceBuilder().build(temporal)

    return {
        "enriched_samples": store.save_dataframe(enriched_name, enriched),
        "temporal_samples": store.save_dataframe(temporal_name, temporal),
        "spatial_grid": store.save_array(spatial_name, grid),
        "temporal_sequences": store.save_array(sequence_name, sequences),
        "dataset_metadata": store.save_json(
            metadata_name,
            {
                "enriched_rows": int(len(enriched)),
                "temporal_rows": int(len(temporal)),
                "spatial_grid_shape": list(grid.shape),
                "temporal_sequences_shape": list(sequences.shape),
                "wmm_cache_path": str(wmm_model.cache_path) if wmm_model.cache_path is not None else None,
            },
        ),
    }
