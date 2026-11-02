import math
import re

import pytest

from pfc3w.identity import (
    canonical_specification,
    manifest_scientific_specification,
    scientific_run_id,
    slug_component,
)


def example_specification() -> dict[str, object]:
    return {
        "dataset_fingerprint": "data-sha256",
        "split_fingerprint": "split-sha256",
        "source_fingerprint": "source-sha256",
        "seed": 42,
        "preprocessing": {"window_size": 300, "step_size": 150},
        "hyperparameters": {"max_depth": 4, "n_estimators": 500},
    }


def test_identifier_is_order_independent_and_contains_no_timestamp() -> None:
    specification = example_specification()
    reordered = dict(reversed(list(specification.items())))

    first = scientific_run_id("E3", "XGBoost", "Causal OOF", specification)
    second = scientific_run_id("E3", "XGBoost", "Causal OOF", reordered)

    assert first == second
    assert re.fullmatch(r"e3-xgboost-causal-oof__[0-9a-f]{10}", first)
    assert not re.search(r"20\d{6}T\d{6}Z", first)


def test_identifier_changes_when_scientific_specification_changes() -> None:
    original = example_specification()
    changed = {**original, "seed": 7}

    assert scientific_run_id("E3", "rf", "causal", original) != scientific_run_id(
        "E3", "rf", "causal", changed
    )


def test_canonical_specification_rejects_nan() -> None:
    with pytest.raises(ValueError, match="JSON canônico"):
        canonical_specification({"metric": math.nan})


@pytest.mark.parametrize("length", [0, 7, 65])
def test_rejects_unsafe_digest_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="entre 8 e 64"):
        scientific_run_id("E3", "rf", "causal", example_specification(), digest_length=length)


def test_slug_rejects_empty_component() -> None:
    with pytest.raises(ValueError, match="não pode ser vazio"):
        slug_component(" --- ")


def test_manifest_specification_excludes_results_and_temporal_metadata() -> None:
    manifest = {
        "run_id": "legacy_e3_rf_causal_oof",
        "started_at_utc": "2026-10-10T17:13:46+00:00",
        "finished_at_utc": "2026-10-10T17:15:24+00:00",
        "project_commit": "project-sha",
        "baseline_commit": "baseline-sha",
        "source_features_sha256": "features-sha",
        "experiment_id": "E3",
        "model": "rf",
        "metrics": {"f1_macro": 0.18},
        "outputs": {"metrics": "metrics.json"},
    }

    specification = manifest_scientific_specification(manifest)

    assert specification == {
        "experiment_id": "E3",
        "model": "rf",
        "project_commit": "project-sha",
        "baseline_commit": "baseline-sha",
        "source_features_sha256": "features-sha",
    }


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"features_sha256": "features-sha"}, "project_commit"),
        ({"project_commit": "project-sha"}, "impressão digital"),
    ],
)
def test_manifest_specification_requires_code_and_data(
    manifest: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        manifest_scientific_specification(manifest)
