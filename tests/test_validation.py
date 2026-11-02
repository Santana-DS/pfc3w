import pytest

from pfc3w.validation import assert_disjoint_groups


def test_accepts_disjoint_wells() -> None:
    assert_disjoint_groups(["WELL-00001", "WELL-00002"], ["WELL-00003"])


def test_rejects_well_leakage() -> None:
    with pytest.raises(ValueError, match="WELL-00002"):
        assert_disjoint_groups(
            ["WELL-00001", "WELL-00002"],
            ["WELL-00002", "WELL-00003"],
        )
