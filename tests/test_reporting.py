from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pfc3w.reporting import (
    build_e0_e3_tables,
    build_support_and_shap_tables,
    collect_experiment_matrix,
    collect_metric_rows,
    collect_r0_reproduction,
    sha256_file,
)

METRICS = {
    "f1_macro_all_labels": 0.4,
    "f1_macro_physically_identifiable": 0.42,
    "f1_weighted": 0.8,
    "balanced_accuracy": 0.39,
    "accuracy": 0.82,
}


def _make_run(
    results_dir: Path,
    *,
    run_id: str,
    experiment_id: str,
    model: str,
    f1_macro: float,
    n_windows: int = 100,
) -> None:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    metrics = {**METRICS, "f1_macro_all_labels": f1_macro}
    (run_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment_id": experiment_id,
                "model": model,
                "project_commit": "abc123",
                "dataset": {
                    "n_windows": n_windows,
                    "n_wells": 2,
                    "all_labels": [0, 1],
                },
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    half = n_windows // 2
    (run_dir / "per_well_metrics.csv").write_text(
        "well_id,n_windows,n_labels,f1_macro_present_labels,accuracy\n"
        f"WELL-1,{half},2,{f1_macro},0.8\n"
        f"WELL-2,{n_windows - half},2,{f1_macro + 0.1},0.9\n",
        encoding="utf-8",
    )


def _config(results_dir: Path, config_path: Path) -> None:
    runs = []
    values = {
        "rf": {"E0": 0.40, "E1": 0.10, "E2": 0.46, "E3": 0.18},
        "xgboost": {"E0": 0.42, "E1": 0.12, "E2": 0.50, "E3": 0.20},
    }
    for model, experiments in values.items():
        for experiment_id, value in experiments.items():
            run_id = f"run_{experiment_id}_{model}"
            _make_run(
                results_dir,
                run_id=run_id,
                experiment_id=experiment_id,
                model=model,
                f1_macro=value,
            )
            runs.append(
                {
                    "experiment_id": experiment_id,
                    "model": model,
                    "display_model": "RF" if model == "rf" else "XGBoost",
                    "run_id": run_id,
                    "scientific_run_id": f"{experiment_id.lower()}-{model}-oof__test",
                }
            )
    config_path.write_text(
        yaml.safe_dump(
            {
                "table_set": "test",
                "runs": runs,
                "contrasts": [
                    {
                        "contrast_id": "E1-E0",
                        "minuend": "E1",
                        "subtrahend": "E0",
                        "factor_latex": "episódio para poço",
                    },
                    {
                        "contrast_id": "E2-E0",
                        "minuend": "E2",
                        "subtrahend": "E0",
                        "factor_latex": "retrospectivo para causal",
                    },
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_builds_deterministic_tables_and_signed_contrasts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "reports" / "tables"
    manuscript_dir = tmp_path / "manuscript" / "tables"
    _config(results_dir, config_path)

    outputs = build_e0_e3_tables(
        config_path=config_path,
        results_dir=results_dir,
        output_dir=output_dir,
        manuscript_tables_dir=manuscript_dir,
    )

    metrics_tex = outputs["metrics_tex"].read_text(encoding="utf-8")
    contrasts_tex = outputs["contrasts_tex"].read_text(encoding="utf-8")
    per_well_tex = outputs["per_well_tex"].read_text(encoding="utf-8")
    metrics_csv = outputs["metrics_csv"].read_text(encoding="utf-8")
    provenance = outputs["provenance"].read_text(encoding="utf-8")
    assert "RF & E0 & 0,4000" in metrics_tex
    assert "XGBoost & E3 & 0,2000" in metrics_tex
    assert "E1 $-$ E0" in contrasts_tex
    assert "$-0,3000$ & $-0,3000$" in contrasts_tex
    assert "$+0,0600$ & $+0,0800$" in contrasts_tex
    assert "RF & E0 & 0,4500 & 0,4500 & 0,8500 & 0,8500" in per_well_tex
    assert "e0-rf-oof__test" in metrics_csv
    assert "run_E0_rf" not in metrics_csv
    assert "e3-xgboost-oof__test" in provenance
    assert "run_E3_xgboost" not in provenance
    assert (manuscript_dir / outputs["metrics_tex"].name).read_bytes() == outputs[
        "metrics_tex"
    ].read_bytes()

    first = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    build_e0_e3_tables(
        config_path=config_path,
        results_dir=results_dir,
        output_dir=output_dir,
        manuscript_tables_dir=manuscript_dir,
    )
    second = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert second == first


def test_rejects_population_mismatch(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    _config(results_dir, config_path)
    manifest_path = results_dir / "run_E3_xgboost" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["n_windows"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="População incompatível"):
        collect_metric_rows(results_dir, config)


def test_rejects_per_well_population_mismatch(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    _config(results_dir, config_path)
    per_well_path = results_dir / "run_E3_xgboost" / "per_well_metrics.csv"
    content = per_well_path.read_text(encoding="utf-8").replace("WELL-1,50", "WELL-X,50")
    per_well_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="População por poço incompatível"):
        build_e0_e3_tables(
            config_path=config_path,
            results_dir=results_dir,
            output_dir=tmp_path / "output",
        )


def test_builds_state_support_and_shap_tables(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    support_run_id = "support_run"
    shap_run_id = "shap_run"
    support_dir = results_dir / support_run_id
    shap_dir = results_dir / shap_run_id
    support_dir.mkdir(parents=True)
    shap_dir.mkdir(parents=True)

    features = pd.DataFrame(
        {
            "window_label": [0, 0, 1, 1],
            "instance_id": ["i1", "i1", "i2", "i3"],
            "well_id": ["w1", "w1", "w2", "w3"],
        }
    )
    features_path = support_dir / "features_causal_real.parquet"
    features.to_parquet(features_path, index=False)
    (support_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": support_run_id,
                "status": "completed",
                "project_commit": "abc",
                "n_windows": 4,
                "n_processed_instances": 3,
                "n_wells": 3,
                "windows_by_operational_label": {"0": 2, "1": 2},
                "outputs": {"features_sha256": sha256_file(features_path)},
            }
        ),
        encoding="utf-8",
    )
    (shap_dir / "global_sensor_importance.csv").write_text(
        "sensor,predicted_output_share\nS1,0.6\nS2,0.4\n", encoding="utf-8"
    )
    (shap_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": shap_run_id,
                "status": "completed",
                "experiment_id": "E3",
                "model": "random_forest",
                "project_commit": "abc",
                "explanation_sample": {
                    "n_windows": 2,
                    "maximum_additivity_residual": 1e-12,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "table_set": "test",
                "runs": [{"run_id": "unused"}],
                "contrasts": [{"contrast_id": "unused"}],
                "state_support": {"run_id": support_run_id},
                "shap_sensor_importance": {
                    "run_id": shap_run_id,
                    "share_column": "predicted_output_share",
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = build_support_and_shap_tables(
        config_path=config_path,
        results_dir=results_dir,
        output_dir=tmp_path / "output",
        manuscript_tables_dir=tmp_path / "manuscript",
    )

    assert "0 & 2 & 1 & 1 & 1 & 2 & 2 & 2" in outputs["support_tex"].read_text(
        encoding="utf-8"
    )
    assert r"\texttt{S1} & 60,00\%" in outputs["shap_tex"].read_text(encoding="utf-8")


def test_collects_experiment_matrix_and_r0_reproduction(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "config" / "experiments"
    experiment_dir.mkdir(parents=True)
    definitions = [
        (
            "R0",
            ["REAL", "SIMULATED", "DRAWN"],
            "legacy_instance_id",
            "historical_published_metrics",
        ),
        ("E0", ["REAL"], "instance_id", "retrospective_baseline_revision"),
        ("E1", ["REAL"], "well_id", "retrospective_baseline_revision"),
        ("E2", ["REAL"], "instance_id", "causal"),
        ("E3", ["REAL"], "well_id", "causal"),
    ]
    paths = []
    for experiment_id, sources, group, variant in definitions:
        path = experiment_dir / f"{experiment_id.lower()}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment_id": experiment_id,
                    "data": {"sources": sources},
                    "split": {"group_column": group},
                    "preprocessing": {"variant": variant},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path.relative_to(tmp_path).as_posix())

    results_dir = tmp_path / "results"
    r0_specs = []
    for model, reproduced in (("rf", 0.41), ("xgboost", 0.42)):
        run_id = f"r0_{model}"
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True)
        comparison = {
            "f1_macro": {
                "published": 0.40,
                "reproduced": reproduced,
                "absolute_difference": abs(reproduced - 0.40),
            },
            "f1_weighted": {
                "published": 0.80,
                "reproduced": 0.81,
                "absolute_difference": 0.01,
            },
            "accuracy": {
                "published": 0.82,
                "reproduced": 0.83,
                "absolute_difference": 0.01,
            },
        }
        (run_dir / "metrics.json").write_text(
            json.dumps({"published_comparison": comparison}), encoding="utf-8"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "experiment_id": "R0",
                    "model": model,
                    "imputation": "global",
                    "dataset": {"n_windows": 100, "n_features": 2, "n_classes": 2},
                    "published_comparison": comparison,
                    "published_metrics_sha256": "published-hash",
                }
            ),
            encoding="utf-8",
        )
        r0_specs.append(
            {
                "model": model,
                "display_model": "RF" if model == "rf" else "XGBoost",
                "run_id": run_id,
            }
        )
    config = {"experiment_matrix_configs": paths, "r0_reproduction": r0_specs}

    matrix_rows, _ = collect_experiment_matrix(tmp_path, config)
    r0_rows, _ = collect_r0_reproduction(results_dir, config)

    assert matrix_rows[0]["population"] == "todas as origens"
    assert matrix_rows[-1] == {
        "experiment_id": "E3",
        "population": "real",
        "fold_group": "poço",
        "preprocessing": "causal",
    }
    assert len(r0_rows) == 6
    assert r0_rows[3]["display_model"] == "XGBoost"
