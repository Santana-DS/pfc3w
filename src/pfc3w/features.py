"""Preparação rastreável de instâncias e features do 3W Dataset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pfc3w.dataset import InstanceIdentity

FEATURE_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "skewness",
    "kurtosis",
    "median",
    "iqr",
    "diff1_std",
    "diff2_std",
    "max_zscore",
)

FEATURE_METADATA_COLUMNS = (
    "instance_id",
    "legacy_instance_id",
    "well_id",
    "nominal_start",
    "relative_path",
    "fault_class",
    "source_type",
    "window_start",
    "window_label",
)


def expected_feature_columns(sensors: Sequence[str]) -> list[str]:
    """Retorna as onze features esperadas para cada sensor, em ordem estável."""
    return [f"{sensor}_{statistic}" for sensor in sensors for statistic in FEATURE_STATISTICS]


def clean_real_instance(
    frame: pd.DataFrame,
    identity: InstanceIdentity,
    *,
    sensors: Sequence[str],
    ffill_limit: int,
    critical_sensor: str,
    max_missing_ratio: float,
) -> pd.DataFrame | None:
    """Replica a limpeza original em uma instância real com identidade única."""
    if identity.source_type != "REAL" or identity.well_id is None:
        raise ValueError("A função aceita somente instâncias reais com well_id")
    if ffill_limit < 0:
        raise ValueError("ffill_limit não pode ser negativo")
    if not 0 <= max_missing_ratio <= 1:
        raise ValueError("max_missing_ratio precisa estar entre zero e um")
    if "class" not in frame.columns:
        raise KeyError("A instância não contém a coluna operacional 'class'")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("As amostras da instância não estão em ordem temporal")

    cleaned = frame.copy()
    for sensor in sensors:
        if sensor not in cleaned.columns:
            cleaned[sensor] = np.nan

    numeric_columns = list(cleaned.select_dtypes(include=[np.number]).columns)
    fill_columns = [column for column in numeric_columns if column not in {"class", "state"}]
    if fill_columns and ffill_limit:
        cleaned[fill_columns] = cleaned[fill_columns].ffill(limit=ffill_limit)

    if critical_sensor not in cleaned.columns:
        return None
    if float(cleaned[critical_sensor].isna().mean()) > max_missing_ratio:
        return None

    cleaned["instance_id"] = identity.instance_id
    cleaned["fault_class"] = identity.class_id
    cleaned["source_type"] = identity.source_type
    return cleaned


def attach_feature_identity(
    features: pd.DataFrame,
    identity: InstanceIdentity,
    *,
    relative_path: str,
    sensors: Sequence[str],
) -> pd.DataFrame:
    """Acrescenta identidade física e força um contrato estável de colunas."""
    if features.empty:
        return pd.DataFrame(columns=[*FEATURE_METADATA_COLUMNS, *expected_feature_columns(sensors)])

    enriched = features.copy()
    enriched["instance_id"] = identity.instance_id
    enriched["legacy_instance_id"] = identity.legacy_instance_id
    enriched["well_id"] = identity.well_id
    enriched["nominal_start"] = identity.nominal_start
    enriched["relative_path"] = relative_path
    enriched["fault_class"] = identity.class_id
    enriched["source_type"] = identity.source_type

    feature_columns = expected_feature_columns(sensors)
    return enriched.reindex(columns=[*FEATURE_METADATA_COLUMNS, *feature_columns])
