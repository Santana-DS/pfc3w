"""Executa avaliação OOF de E0–E3 com folds congelados."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.causal import impute_from_training_fold  # noqa: E402
from pfc3w.dataset import sha256_file  # noqa: E402
from pfc3w.features import FEATURE_METADATA_COLUMNS  # noqa: E402
from pfc3w.paths import load_paths  # noqa: E402

BASELINE_METRICS = {
    "rf": "rf_window_class_metrics.json",
    "xgboost": "xgboost_window_class_metrics.json",
}
GROUP_COLUMNS = {
    "E0": "instance_id",
    "E1": "well_id",
    "E2": "instance_id",
    "E3": "well_id",
}
ASSIGNMENT_EXPERIMENT = {"E0": "E0", "E1": "E1", "E2": "E0", "E3": "E1"}
CAUSAL_EXPERIMENTS = {"E2", "E3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(GROUP_COLUMNS))
    parser.add_argument("model", choices=sorted(BASELINE_METRICS))
    parser.add_argument("features_run", type=Path)
    parser.add_argument(
        "--splits-run",
        type=Path,
        help=(
            "Run que contém os folds congelados; obrigatório para E2/E3 e "
            "omitido em E0/E1 para usar features_run."
        ),
    )
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


def impute_per_instance(
    raw_features: np.ndarray,
    instance_ids: np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    frame = pd.DataFrame(raw_features, columns=feature_columns)
    frame["_instance_id"] = instance_ids
    medians = frame.groupby("_instance_id", observed=True)[feature_columns].transform("median")
    return frame[feature_columns].fillna(medians).fillna(0.0).to_numpy()


def make_estimator(
    model: str,
    parameters: dict[str, object],
    *,
    n_jobs: int,
    n_classes: int,
):
    if model == "rf":
        return RandomForestClassifier(
            **parameters,
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs,
        )
    return XGBClassifier(
        **parameters,
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0,
    )


def metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    all_labels: np.ndarray,
    identifiable_labels: np.ndarray,
) -> dict[str, float]:
    return {
        "f1_macro_all_labels": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=all_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_macro_physically_identifiable": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=identifiable_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def main() -> None:
    args = parse_args()
    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    source_run = args.features_run.resolve()
    causal = args.experiment in CAUSAL_EXPERIMENTS
    if causal and args.splits_run is None:
        raise ValueError("--splits-run é obrigatório para E2/E3")
    split_run = (args.splits_run or source_run).resolve()
    features_filename = (
        "features_causal_real.parquet" if causal else "features_original_real.parquet"
    )
    features_path = source_run / features_filename
    split_dir = split_run / "splits"
    split_manifest_path = split_dir / "split_manifest.json"
    group_column = GROUP_COLUMNS[args.experiment]
    assignment_experiment = ASSIGNMENT_EXPERIMENT[args.experiment]
    assignment_path = split_dir / f"{assignment_experiment.lower()}_assignments.csv"
    if not features_path.exists() or not assignment_path.exists():
        raise FileNotFoundError("Features ou folds congelados não encontrados")

    with split_manifest_path.open(encoding="utf-8") as stream:
        split_manifest = json.load(stream)
    published_path = baseline_repo / "results" / "metrics" / BASELINE_METRICS[args.model]
    with published_path.open(encoding="utf-8") as stream:
        published = json.load(stream)
    parameters = published["best_params"]

    variant_name = "causal" if causal else "retrospective"
    run_id = datetime.now(UTC).strftime(
        f"%Y%m%dT%H%M%SZ_{args.experiment}_{args.model}_{variant_name}_oof"
    )
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"

    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "experiment_id": args.experiment,
        "model": args.model,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "source_run_id": source_run.name,
        "source_features_sha256": sha256_file(features_path),
        "split_source_run_id": split_manifest["source_run_id"],
        "split_source_features_sha256": split_manifest["source_features_sha256"],
        "group_column": group_column,
        "assignment_experiment": assignment_experiment,
        "assignment_sha256": sha256_file(assignment_path),
        "preprocessing": {
            "variant": "causal" if causal else "retrospective_baseline_revision",
            "imputation": (
                "training_fold_median_then_zero"
                if causal
                else "full_instance_median_then_zero"
            ),
        },
        "hyperparameters": {
            "source": "R0 published best parameters; no new search",
            "values": parameters,
        },
        "environment": {
            "python": sys.version,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    try:
        print("[1/4] Carregando features e folds...", flush=True)
        frame = pd.read_parquet(features_path)
        feature_columns = [
            column for column in frame.columns if column not in FEATURE_METADATA_COLUMNS
        ]
        assignments = pd.read_csv(assignment_path)
        fold_map = assignments.set_index(group_column)["fold"]
        fold_ids = frame[group_column].map(fold_map)
        if fold_ids.isna().any():
            raise ValueError("Há janelas sem fold")
        if frame[group_column].nunique() != len(assignments):
            raise ValueError("Número de grupos diverge das atribuições")

        labels = frame["window_label"].to_numpy(dtype=int)
        all_labels = np.sort(frame["window_label"].unique().astype(int))
        well_support = frame.groupby("window_label")["well_id"].nunique()
        identifiable_labels = np.sort(well_support[well_support >= 2].index.to_numpy(dtype=int))

        raw_features = frame[feature_columns].to_numpy()
        if causal:
            print("[2/4] Imputação será ajustada dentro de cada fold...", flush=True)
            features = None
        else:
            print("[2/4] Imputando pela trajetória completa de cada instância...", flush=True)
            features = impute_per_instance(
                raw_features,
                frame["instance_id"].to_numpy(),
                feature_columns,
            )

        predictions = np.full(len(frame), -1, dtype=int)
        train_label_seen = np.zeros(len(frame), dtype=bool)
        fold_results: list[dict[str, object]] = []
        print("[3/4] Treinando os cinco folds...", flush=True)
        for fold in sorted(fold_ids.unique()):
            test_mask = fold_ids.to_numpy() == fold
            train_mask = ~test_mask
            y_train = labels[train_mask]
            y_test = labels[test_mask]
            fold_features = (
                impute_from_training_fold(raw_features, train_mask)
                if causal
                else features
            )
            if fold_features is None:
                raise RuntimeError("Matriz de features não foi preparada")
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            unseen_test = np.setdiff1d(test_classes, train_classes)
            train_label_seen[test_mask] = np.isin(y_test, train_classes)

            if args.model == "xgboost":
                encoder = LabelEncoder().fit(y_train)
                y_train_model = encoder.transform(y_train)
                estimator = make_estimator(
                    args.model,
                    parameters,
                    n_jobs=args.n_jobs,
                    n_classes=len(encoder.classes_),
                )
                sample_weight = compute_sample_weight("balanced", y_train)
                estimator.fit(
                    fold_features[train_mask],
                    y_train_model,
                    sample_weight=sample_weight,
                )
                fold_predictions = encoder.inverse_transform(
                    estimator.predict(fold_features[test_mask]).astype(int)
                )
            else:
                estimator = make_estimator(
                    args.model,
                    parameters,
                    n_jobs=args.n_jobs,
                    n_classes=len(train_classes),
                )
                estimator.fit(fold_features[train_mask], y_train)
                fold_predictions = estimator.predict(fold_features[test_mask]).astype(int)

            predictions[test_mask] = fold_predictions
            fold_metric = metric_summary(
                y_test,
                fold_predictions,
                all_labels=all_labels,
                identifiable_labels=identifiable_labels,
            )
            fold_results.append(
                {
                    "fold": int(fold),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "train_labels": train_classes.tolist(),
                    "test_labels": test_classes.tolist(),
                    "unseen_test_labels": unseen_test.tolist(),
                    **fold_metric,
                }
            )
            print(f"  fold {fold}: {fold_metric}", flush=True)

        if np.any(predictions < 0):
            raise RuntimeError("Há janelas sem previsão OOF")

        print("[4/4] Consolidando métricas...", flush=True)
        metrics = metric_summary(
            labels,
            predictions,
            all_labels=all_labels,
            identifiable_labels=identifiable_labels,
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=all_labels,
            zero_division=0,
        )
        per_class = {
            str(label): {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
                "well_support": int(well_support.loc[label]),
                "physically_identifiable": bool(label in identifiable_labels),
            }
            for index, label in enumerate(all_labels)
        }

        predictions_path = run_dir / "oof_predictions.parquet"
        pd.DataFrame(
            {
                "row_index": np.arange(len(frame)),
                "instance_id": frame["instance_id"],
                "well_id": frame["well_id"],
                "fold": fold_ids.to_numpy(dtype=int),
                "y_true": labels,
                "y_pred": predictions,
                "train_label_seen": train_label_seen,
            }
        ).to_parquet(predictions_path, index=False)

        per_well_rows = []
        for well_id, index in frame.groupby("well_id").groups.items():
            positions = np.asarray(list(index), dtype=int)
            present_labels = np.unique(labels[positions])
            per_well_rows.append(
                {
                    "well_id": well_id,
                    "n_windows": int(len(positions)),
                    "n_labels": int(len(present_labels)),
                    "f1_macro_present_labels": float(
                        f1_score(
                            labels[positions],
                            predictions[positions],
                            labels=present_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "accuracy": float(accuracy_score(labels[positions], predictions[positions])),
                }
            )
        per_well_path = run_dir / "per_well_metrics.csv"
        pd.DataFrame(per_well_rows).sort_values("well_id").to_csv(per_well_path, index=False)

        metrics_payload = {
            "metrics": metrics,
            "all_labels": all_labels.tolist(),
            "physically_identifiable_labels": identifiable_labels.tolist(),
            "folds": fold_results,
            "per_class": per_class,
        }
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)

        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "dataset": {
                    "n_windows": int(len(frame)),
                    "n_features": len(feature_columns),
                    "n_instances": int(frame["instance_id"].nunique()),
                    "n_wells": int(frame["well_id"].nunique()),
                    "all_labels": all_labels.tolist(),
                    "physically_identifiable_labels": identifiable_labels.tolist(),
                },
                "metrics": metrics,
                "integrity": {
                    "groups_crossing_folds": int(
                        pd.DataFrame({group_column: frame[group_column], "fold": fold_ids})
                        .groupby(group_column)["fold"]
                        .nunique()
                        .gt(1)
                        .sum()
                    ),
                    "windows_with_unseen_training_label": int((~train_label_seen).sum()),
                },
                "outputs": {
                    "metrics": metrics_path.name,
                    "oof_predictions": predictions_path.name,
                    "per_well_metrics": per_well_path.name,
                },
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
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
