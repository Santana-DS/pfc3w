import numpy as np
import pandas as pd
import pytest

from pfc3w.dataset import InstanceIdentity
from pfc3w.features import (
    FEATURE_METADATA_COLUMNS,
    attach_feature_identity,
    clean_real_instance,
    expected_feature_columns,
)


def real_identity() -> InstanceIdentity:
    return InstanceIdentity(
        instance_id="1/WELL-00001_20200101000000",
        legacy_instance_id="WELL-00001_20200101000000",
        class_id=1,
        source_type="REAL",
        well_id="WELL-00001",
        source_index=None,
        nominal_start="20200101000000",
    )


def test_clean_real_instance_forward_fills_only_up_to_limit() -> None:
    frame = pd.DataFrame(
        {
            "P-TPT": [1.0, np.nan, np.nan, 4.0],
            "P-PDG": [2.0, np.nan, np.nan, 5.0],
            "class": pd.Series([0, 0, 1, 1], dtype="Int16"),
            "state": pd.Series([0, 0, 1, 1], dtype="Int16"),
        },
        index=pd.date_range("2020-01-01", periods=4, freq="s"),
    )

    cleaned = clean_real_instance(
        frame,
        real_identity(),
        sensors=["P-TPT", "P-PDG"],
        ffill_limit=1,
        critical_sensor="P-TPT",
        max_missing_ratio=0.5,
    )

    assert cleaned is not None
    assert cleaned["P-PDG"].tolist()[:2] == [2.0, 2.0]
    assert np.isnan(cleaned["P-PDG"].iloc[2])
    assert cleaned["instance_id"].nunique() == 1
    assert cleaned["source_type"].unique().tolist() == ["REAL"]


def test_clean_real_instance_rejects_missing_critical_sensor() -> None:
    frame = pd.DataFrame(
        {"class": pd.Series([0, 1], dtype="Int16")},
        index=pd.date_range("2020-01-01", periods=2, freq="s"),
    )
    assert (
        clean_real_instance(
            frame,
            real_identity(),
            sensors=["P-PDG"],
            ffill_limit=60,
            critical_sensor="P-TPT",
            max_missing_ratio=0.5,
        )
        is None
    )


def test_clean_real_instance_requires_temporal_order() -> None:
    frame = pd.DataFrame(
        {"P-TPT": [1.0, 2.0], "class": [0, 1]},
        index=pd.to_datetime(["2020-01-02", "2020-01-01"]),
    )
    with pytest.raises(ValueError, match="ordem temporal"):
        clean_real_instance(
            frame,
            real_identity(),
            sensors=["P-TPT"],
            ffill_limit=60,
            critical_sensor="P-TPT",
            max_missing_ratio=0.5,
        )


def test_attach_feature_identity_has_stable_schema() -> None:
    raw_features = pd.DataFrame(
        {
            "instance_id": ["legacy"],
            "fault_class": [1],
            "source_type": ["WELL"],
            "window_start": [0],
            "window_label": [0],
            "P-TPT_mean": [1.5],
        }
    )
    enriched = attach_feature_identity(
        raw_features,
        real_identity(),
        relative_path="1/WELL-00001_20200101000000.parquet",
        sensors=["P-TPT", "QGL"],
    )

    assert enriched.columns.tolist() == [
        *FEATURE_METADATA_COLUMNS,
        *expected_feature_columns(["P-TPT", "QGL"]),
    ]
    assert enriched.loc[0, "well_id"] == "WELL-00001"
    assert enriched.loc[0, "instance_id"] == "1/WELL-00001_20200101000000"
    assert np.isnan(enriched.loc[0, "QGL_mean"])
