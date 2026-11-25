"""Executa uma prova de conceito SHAP em um fold causal por poço (E3)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.causal import impute_from_training_fold  # noqa: E402
from pfc3w.dataset import sha256_file  # noqa: E402
from pfc3w.features import (  # noqa: E402
    FEATURE_METADATA_COLUMNS,
    FEATURE_STATISTICS,
)
from pfc3w.paths import load_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_run", type=Path)
    parser.add_argument("splits_run", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=6)
    return parser.parse_args()


def git_head(repository: Path) -> str | None:
    process = subprocess.run(
        ["git", f"--git-dir={repository / '.git'}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def split_feature_name(feature: str) -> tuple[str, str]:
    for statistic in sorted(FEATURE_STATISTICS, key=len, reverse=True):
        suffix = f"_{statistic}"
        if feature.endswith(suffix):
            return feature[: -len(suffix)], statistic
    raise ValueError(f"Feature sem estatística reconhecida: {feature}")


def stratified_positions(
    labels: np.ndarray,
    *,
    maximum_per_class: int,
    random_seed: int,
) -> np.ndarray:
    if maximum_per_class <= 0:
        raise ValueError("max-per-class precisa ser positivo")
    random = np.random.default_rng(random_seed)
    positions: list[int] = []
    for label in np.sort(np.unique(labels)):
        candidates = np.flatnonzero(labels == label)
        size = min(maximum_per_class, len(candidates))
        positions.extend(random.choice(candidates, size=size, replace=False).tolist())
    return np.asarray(sorted(positions), dtype=int)


def highest_confidence_diverse_examples(
    candidates: np.ndarray,
    confidence: np.ndarray,
    instance_ids: pd.Series,
    *,
    maximum: int,
) -> list[int]:
    """Prioriza confiança sem repetir a mesma instância entre exemplos locais."""
    ranked = candidates[np.argsort(confidence[candidates])[::-1]]
    selected: list[int] = []
    seen_instances: set[str] = set()
    for position in ranked:
        instance_id = str(instance_ids.iloc[position])
        if instance_id not in seen_instances:
            selected.append(int(position))
            seen_instances.add(instance_id)
        if len(selected) == maximum:
            return selected
    for position in ranked:
        if int(position) not in selected:
            selected.append(int(position))
        if len(selected) == maximum:
            break
    return selected


def main() -> None:
    args = parse_args()
    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    features_run = args.features_run.resolve()
    splits_run = args.splits_run.resolve()
    features_path = features_run / "features_causal_real.parquet"
    assignment_path = splits_run / "splits" / "e1_assignments.csv"
    published_path = baseline_repo / "results" / "metrics" / "rf_window_class_metrics.json"
    for required in (features_path, assignment_path, published_path):
        if not required.exists():
            raise FileNotFoundError(required)

    with published_path.open(encoding="utf-8") as stream:
        parameters = json.load(stream)["best_params"]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_E3_rf_shap_pilot")
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "purpose": "PFC1_SHAP_proof_of_concept_on_E3",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "experiment_id": "E3",
        "model": "random_forest",
        "fold": args.fold,
        "random_seed": 42,
        "maximum_explanation_samples_per_true_class": args.max_per_class,
        "source_features_sha256": sha256_file(features_path),
        "assignment_sha256": sha256_file(assignment_path),
        "hyperparameters": parameters,
        "environment": {
            "python": sys.version,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "shap": importlib.metadata.version("shap"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        print("[1/5] Carregando E3 e o fold por poço...", flush=True)
        frame = pd.read_parquet(features_path)
        feature_columns = [
            column for column in frame.columns if column not in FEATURE_METADATA_COLUMNS
        ]
        assignments = pd.read_csv(assignment_path)
        fold_map = assignments.set_index("well_id")["fold"]
        fold_ids = frame["well_id"].map(fold_map)
        if fold_ids.isna().any() or frame["well_id"].nunique() != len(assignments):
            raise ValueError("Atribuições de poço incompatíveis com a tabela causal")
        available_folds = sorted(fold_ids.unique().astype(int))
        if args.fold not in available_folds:
            raise ValueError(f"Fold {args.fold} ausente; disponíveis: {available_folds}")

        labels = frame["window_label"].to_numpy(dtype=int)
        test_mask = fold_ids.to_numpy(dtype=int) == args.fold
        train_mask = ~test_mask
        raw_features = frame[feature_columns].to_numpy(dtype=float)

        print("[2/5] Imputando com estatísticas exclusivas do treino...", flush=True)
        features = impute_from_training_fold(raw_features, train_mask)
        estimator = RandomForestClassifier(
            **parameters,
            class_weight="balanced",
            random_state=42,
            n_jobs=args.n_jobs,
        )
        print("[3/5] Treinando o Random Forest do fold...", flush=True)
        estimator.fit(features[train_mask], labels[train_mask])
        test_features = features[test_mask]
        test_labels = labels[test_mask]
        predictions = estimator.predict(test_features).astype(int)
        probabilities = estimator.predict_proba(test_features)

        sample_positions = stratified_positions(
            test_labels,
            maximum_per_class=args.max_per_class,
            random_seed=42,
        )
        sample_features = test_features[sample_positions]
        sample_labels = test_labels[sample_positions]
        sample_predictions = predictions[sample_positions]
        sample_probabilities = probabilities[sample_positions]

        print(
            f"[4/5] Calculando SHAP para {len(sample_positions)} janelas estratificadas...",
            flush=True,
        )
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(sample_features, check_additivity=False)
        shap_values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        if shap_values.shape != (
            len(sample_positions),
            len(feature_columns),
            len(estimator.classes_),
        ):
            raise ValueError(f"Formato SHAP inesperado: {shap_values.shape}")

        predicted_class_indices = np.searchsorted(estimator.classes_, sample_predictions)
        predicted_output_shap = shap_values[
            np.arange(len(sample_positions)), :, predicted_class_indices
        ]
        reconstructed_probabilities = base_values + shap_values.sum(axis=1)
        maximum_additivity_residual = float(
            np.abs(reconstructed_probabilities - sample_probabilities).max()
        )

        feature_rows = []
        mean_abs_all_outputs = np.abs(shap_values).mean(axis=(0, 2))
        mean_abs_predicted_output = np.abs(predicted_output_shap).mean(axis=0)
        for index, feature in enumerate(feature_columns):
            sensor, statistic = split_feature_name(feature)
            feature_rows.append(
                {
                    "feature": feature,
                    "sensor": sensor,
                    "statistic": statistic,
                    "mean_abs_shap_all_outputs": float(mean_abs_all_outputs[index]),
                    "mean_abs_shap_predicted_output": float(
                        mean_abs_predicted_output[index]
                    ),
                }
            )
        feature_importance = pd.DataFrame(feature_rows).sort_values(
            "mean_abs_shap_predicted_output", ascending=False
        )
        total_predicted_importance = feature_importance[
            "mean_abs_shap_predicted_output"
        ].sum()
        feature_importance["predicted_output_share"] = (
            feature_importance["mean_abs_shap_predicted_output"]
            / total_predicted_importance
        )
        feature_importance_path = run_dir / "global_feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)

        sensor_importance = (
            feature_importance.groupby("sensor", as_index=False)[
                ["mean_abs_shap_all_outputs", "mean_abs_shap_predicted_output"]
            ]
            .sum()
            .sort_values("mean_abs_shap_predicted_output", ascending=False)
        )
        sensor_importance["predicted_output_share"] = (
            sensor_importance["mean_abs_shap_predicted_output"]
            / total_predicted_importance
        )
        sensor_importance_path = run_dir / "global_sensor_importance.csv"
        sensor_importance.to_csv(sensor_importance_path, index=False)

        statistic_importance = (
            feature_importance.groupby("statistic", as_index=False)[
                ["mean_abs_shap_all_outputs", "mean_abs_shap_predicted_output"]
            ]
            .sum()
            .sort_values("mean_abs_shap_predicted_output", ascending=False)
        )
        statistic_importance["predicted_output_share"] = (
            statistic_importance["mean_abs_shap_predicted_output"]
            / total_predicted_importance
        )
        statistic_importance_path = run_dir / "global_statistic_importance.csv"
        statistic_importance.to_csv(statistic_importance_path, index=False)

        test_global_positions = np.flatnonzero(test_mask)
        sample_global_positions = test_global_positions[sample_positions]
        sample_metadata = frame.iloc[sample_global_positions][
            ["instance_id", "well_id", "window_start", "window_label"]
        ].reset_index(drop=True)
        explanation_frame = sample_metadata.assign(
            row_index=sample_global_positions,
            y_pred=sample_predictions,
            prediction_confidence=sample_probabilities.max(axis=1),
            correct=sample_predictions == sample_labels,
        )
        value_frame = pd.DataFrame(
            sample_features,
            columns=[f"value__{feature}" for feature in feature_columns],
        )
        shap_frame = pd.DataFrame(
            predicted_output_shap,
            columns=[f"shap__{feature}" for feature in feature_columns],
        )
        explanation_frame = pd.concat(
            [explanation_frame, value_frame, shap_frame], axis=1
        )
        explanations_path = run_dir / "sample_explanations.parquet"
        explanation_frame.to_parquet(explanations_path, index=False)

        confidence = sample_probabilities.max(axis=1)
        selected_local: list[int] = []
        for correctness in (False, True):
            candidates = np.flatnonzero((sample_predictions == sample_labels) == correctness)
            selected_local.extend(
                highest_confidence_diverse_examples(
                    candidates,
                    confidence,
                    sample_metadata["instance_id"],
                    maximum=5,
                )
            )
        local_rows = []
        for sample_index in selected_local:
            ranked_features = np.argsort(np.abs(predicted_output_shap[sample_index]))[::-1][:10]
            for rank, feature_index in enumerate(ranked_features, start=1):
                feature = feature_columns[feature_index]
                sensor, statistic = split_feature_name(feature)
                local_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "row_index": int(sample_global_positions[sample_index]),
                        "instance_id": sample_metadata.loc[sample_index, "instance_id"],
                        "well_id": sample_metadata.loc[sample_index, "well_id"],
                        "window_start": int(sample_metadata.loc[sample_index, "window_start"]),
                        "y_true": int(sample_labels[sample_index]),
                        "y_pred": int(sample_predictions[sample_index]),
                        "correct": bool(
                            sample_predictions[sample_index] == sample_labels[sample_index]
                        ),
                        "prediction_confidence": float(confidence[sample_index]),
                        "rank": rank,
                        "feature": feature,
                        "sensor": sensor,
                        "statistic": statistic,
                        "feature_value": float(sample_features[sample_index, feature_index]),
                        "shap_for_predicted_class": float(
                            predicted_output_shap[sample_index, feature_index]
                        ),
                    }
                )
        local_path = run_dir / "local_top_contributions.csv"
        pd.DataFrame(local_rows).to_csv(local_path, index=False)

        print("[5/5] Gerando figura e manifesto...", flush=True)
        figure, axes = plt.subplots(1, 2, figsize=(13, 6))
        top_features = feature_importance.head(15).sort_values(
            "mean_abs_shap_predicted_output"
        )
        axes[0].barh(
            top_features["feature"],
            top_features["mean_abs_shap_predicted_output"],
            color="#276FBF",
        )
        axes[0].set_title("15 features mais influentes")
        axes[0].set_xlabel("média de |SHAP| para a classe prevista")
        top_sensors = sensor_importance.sort_values("mean_abs_shap_predicted_output")
        axes[1].barh(
            top_sensors["sensor"],
            top_sensors["predicted_output_share"],
            color="#D95D39",
        )
        axes[1].set_title("Importância agregada por sensor")
        axes[1].set_xlabel("fração da importância SHAP")
        figure.suptitle(f"E3 — Random Forest — SHAP — fold {args.fold}")
        figure.tight_layout()
        figure_path = run_dir / "global_importance.png"
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        all_labels = np.sort(frame["window_label"].unique().astype(int))
        test_metrics = {
            "f1_macro_all_labels": float(
                f1_score(
                    test_labels,
                    predictions,
                    labels=all_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "accuracy": float(accuracy_score(test_labels, predictions)),
        }
        sample_counts = pd.Series(sample_labels).value_counts().sort_index()
        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "dataset": {
                    "n_train_windows": int(train_mask.sum()),
                    "n_test_windows": int(test_mask.sum()),
                    "n_train_wells": int(frame.loc[train_mask, "well_id"].nunique()),
                    "n_test_wells": int(frame.loc[test_mask, "well_id"].nunique()),
                    "train_labels": np.unique(labels[train_mask]).tolist(),
                    "test_labels": np.unique(test_labels).tolist(),
                },
                "fold_metrics": test_metrics,
                "explanation_sample": {
                    "n_windows": len(sample_positions),
                    "counts_by_true_label": {
                        str(label): int(count) for label, count in sample_counts.items()
                    },
                    "maximum_additivity_residual": maximum_additivity_residual,
                    "selection": "up_to_maximum_per_class_uniform_without_replacement",
                },
                "local_example_selection": (
                    "five_highest_confidence_errors_and_correct_predictions; "
                    "distinct_instances_prioritized"
                ),
                "outputs": {
                    "global_feature_importance": feature_importance_path.name,
                    "global_sensor_importance": sensor_importance_path.name,
                    "global_statistic_importance": statistic_importance_path.name,
                    "sample_explanations": explanations_path.name,
                    "local_top_contributions": local_path.name,
                    "global_importance_figure": figure_path.name,
                },
                "interpretation_guardrail": (
                    "SHAP attributes model output; it does not establish physical causality."
                ),
            }
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
