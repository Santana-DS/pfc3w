from pathlib import Path

import pytest

from pfc3w.dataset import parse_instance_identity


def test_parse_real_instance(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    path = dataset / "8" / "WELL-00042_20240102030405.parquet"

    identity = parse_instance_identity(path, dataset)

    assert identity.instance_id == "8/WELL-00042_20240102030405"
    assert identity.legacy_instance_id == "WELL-00042_20240102030405"
    assert identity.source_type == "REAL"
    assert identity.well_id == "WELL-00042"
    assert identity.nominal_start == "20240102030405"


@pytest.mark.parametrize("source", ["SIMULATED", "DRAWN"])
def test_artificial_instance_has_no_physical_well(tmp_path: Path, source: str) -> None:
    dataset = tmp_path / "dataset"
    path = dataset / "1" / f"{source}_00001.parquet"

    identity = parse_instance_identity(path, dataset)

    assert identity.instance_id == f"1/{source}_00001"
    assert identity.source_type == source
    assert identity.well_id is None
    assert identity.source_index == 1


def test_class_prefix_prevents_cross_class_collision(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    first = parse_instance_identity(dataset / "1" / "SIMULATED_00001.parquet", dataset)
    second = parse_instance_identity(dataset / "2" / "SIMULATED_00001.parquet", dataset)

    assert first.legacy_instance_id == second.legacy_instance_id
    assert first.instance_id != second.instance_id


def test_rejects_unknown_filename(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    with pytest.raises(ValueError, match="não reconhecido"):
        parse_instance_identity(dataset / "1" / "unknown.parquet", dataset)
