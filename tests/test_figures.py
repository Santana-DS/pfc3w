from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pfc3w.figures import build_f001_shap_global, load_shap_feature_importance


def _make_shap_run(run_dir: Path, *, shares: tuple[float, ...] = (0.6, 0.4)) -> None:
    run_dir.mkdir(parents=True)
    sensors = [f"S{index}" for index in range(1, len(shares) + 1)]
    pd.DataFrame(
        {
            "feature": [f"{sensor}_mean" for sensor in sensors],
            "sensor": sensors,
            "statistic": ["mean"] * len(shares),
            "predicted_output_share": shares,
        }
    ).to_csv(run_dir / "global_feature_importance.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "shap_run",
                "status": "completed",
                "experiment_id": "E3",
                "model": "random_forest",
                "fold": 0,
                "project_commit": "abc",
                "explanation_sample": {
                    "n_windows": 10,
                    "selection": "stratified",
                    "maximum_additivity_residual": 1e-12,
                },
            }
        ),
        encoding="utf-8",
    )


def test_builds_vector_and_raster_f001_with_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _make_shap_run(run_dir)
    outputs = build_f001_shap_global(
        run_dir=run_dir,
        output_dir=tmp_path / "figures",
        manuscript_figures_dir=tmp_path / "manuscript",
        top_n=2,
        scientific_run_id="e3-rf-shap-fold0__testdigest",
    )

    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    assert outputs["png"].read_bytes().startswith(b"\x89PNG")
    provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    assert provenance["figure_id"] == "F001"
    assert provenance["n_explanation_windows"] == 10
    assert provenance["run_id"] == "e3-rf-shap-fold0__testdigest"
    assert set(provenance["outputs"]) == {outputs["pdf"].name, outputs["png"].name}


def test_f001_is_byte_reproducible(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _make_shap_run(run_dir)
    first = build_f001_shap_global(
        run_dir=run_dir,
        output_dir=tmp_path / "first",
        top_n=2,
    )
    second = build_f001_shap_global(
        run_dir=run_dir,
        output_dir=tmp_path / "second",
        top_n=2,
    )

    for artifact in ("pdf", "png", "provenance"):
        assert first[artifact].read_bytes() == second[artifact].read_bytes()


def test_rejects_feature_shares_that_do_not_sum_to_one(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _make_shap_run(run_dir, shares=(0.6, 0.3))

    with pytest.raises(ValueError, match="não somam um"):
        load_shap_feature_importance(run_dir)
