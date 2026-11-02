"""Reproduz métricas OOF do R0 com os hiperparâmetros publicados no baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.paths import load_paths  # noqa: E402

META_COLUMNS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
BASELINE_METRIC_FILES = {
    "rf": "rf_window_class_metrics.json",
    "xgboost": "xgboost_window_class_metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(BASELINE_METRIC_FILES))
    parser.add_argument("--preprocessing-run", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--imputation",
        choices=["global", "per-instance"],
        required=True,
        help=(
            "'global' reproduz o código que gerou as métricas publicadas; "
            "'per-instance' reproduz a revisão posterior dos scripts."
        ),
    )
    return parser.parse_args()


def git_head(repository: Path) -> str | None:
    process = subprocess.run(
        ["git", f"--git-dir={repository / '.git'}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "joblib"]
    return {name: importlib.metadata.version(name) for name in names}


def impute_per_legacy_instance(
    features: np.ndarray, groups: np.ndarray, feature_columns: list[str]
) -> np.ndarray:
    """Replica a imputação dos classificadores de estado do baseline."""
    frame = pd.DataFrame(features, columns=feature_columns)
    frame["_gid"] = groups
    instance_medians = frame.groupby("_gid")[feature_columns].transform("median")
    frame[feature_columns] = frame[feature_columns].fillna(instance_medians).fillna(0.0)
    return frame[feature_columns].to_numpy()


def make_estimator(model: str, parameters: dict[str, object], n_jobs: int, n_classes: int):
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


def rounded_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def main() -> None:
    args = parse_args()
    preprocessing_run = args.preprocessing_run.resolve()
    features_path = preprocessing_run / "features_window_class.parquet"
    audit_path = preprocessing_run / "artifact_audit.json"
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    published_path = baseline_repo / "results" / "metrics" / BASELINE_METRIC_FILES[args.model]
    with published_path.open(encoding="utf-8") as stream:
        published = json.load(stream)
    parameters = published["best_params"]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_R0_{args.model}_{args.imputation}_published_params_oof"
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"

    feature_hash = None
    if audit_path.exists():
        with audit_path.open(encoding="utf-8") as stream:
            feature_hash = json.load(stream)["features"]["sha256"]
    if feature_hash is None:
        feature_hash = sha256_file(features_path)

    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "experiment_id": "R0",
        "model": args.model,
        "mode": "oof_evaluation_with_published_best_parameters",
        "imputation": args.imputation,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "preprocessing_run_id": preprocessing_run.name,
        "features_sha256": feature_hash,
        "published_metrics_file": str(published_path.relative_to(baseline_repo)),
        "published_metrics_sha256": sha256_file(published_path),
        "published_metrics_code_commit": "853b1cde6115dbbb8bb6a6e85ec2a1354afc50f4",
        "parameters": parameters,
        "validation": {
            "splitter": "GroupKFold",
            "n_splits": 5,
            "group": "legacy instance_id",
            "random_state": 42,
            "n_jobs_per_estimator": args.n_jobs,
            "fold_execution": "sequential",
        },
        "environment": {"python": sys.version, "packages": package_versions()},
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    try:
        print("[1/4] Carregando features...")
        frame = pd.read_parquet(features_path)
        feature_columns = [column for column in frame.columns if column not in META_COLUMNS]
        features = frame[feature_columns].to_numpy()
        labels_original = frame["window_label"].to_numpy()
        groups = frame["instance_id"].to_numpy()

        if args.imputation == "global":
            print("[2/4] Replicando imputação global anterior à validação...")
            features = SimpleImputer(strategy="median").fit_transform(features)
        else:
            print("[2/4] Replicando imputação pela mediana da instância completa...")
            features = impute_per_legacy_instance(features, groups, feature_columns)

        encoder = LabelEncoder()
        if args.model == "xgboost":
            labels_model = encoder.fit_transform(labels_original)
            sample_weights = compute_sample_weight("balanced", labels_original)
        else:
            labels_model = labels_original
            sample_weights = None

        classes = np.unique(labels_original)
        predictions_model = np.empty_like(labels_model)
        fold_ids = np.empty(len(frame), dtype=np.int8)
        folds: list[dict[str, object]] = []
        splitter = GroupKFold(n_splits=5)

        print("[3/4] Gerando predições out-of-fold...")
        for fold, (train_index, test_index) in enumerate(
            splitter.split(features, labels_model, groups), start=1
        ):
            estimator = make_estimator(args.model, parameters, args.n_jobs, len(classes))
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights[train_index]
            estimator.fit(features[train_index], labels_model[train_index], **fit_kwargs)
            fold_prediction = estimator.predict(features[test_index])
            predictions_model[test_index] = fold_prediction
            fold_ids[test_index] = fold

            if args.model == "xgboost":
                fold_true = encoder.inverse_transform(labels_model[test_index])
                fold_pred = encoder.inverse_transform(fold_prediction)
            else:
                fold_true = labels_model[test_index]
                fold_pred = fold_prediction
            fold_metrics = rounded_metrics(fold_true, fold_pred)
            fold_entry = {
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "n_train_groups": int(np.unique(groups[train_index]).size),
                "n_test_groups": int(np.unique(groups[test_index]).size),
                **fold_metrics,
            }
            folds.append(fold_entry)
            print(f"  fold {fold}/5: {fold_metrics}")

        if args.model == "xgboost":
            predictions_original = encoder.inverse_transform(predictions_model)
        else:
            predictions_original = predictions_model

        fold_assignment = pd.DataFrame({"group": groups, "fold": fold_ids})
        groups_crossing_folds = int(
            (fold_assignment.groupby("group", observed=True)["fold"].nunique() > 1).sum()
        )
        if groups_crossing_folds:
            raise RuntimeError(f"{groups_crossing_folds} grupos atravessaram folds.")

        print("[4/4] Salvando métricas e predições...")
        metrics = rounded_metrics(labels_original, predictions_original)
        report = classification_report(
            labels_original,
            predictions_original,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        expected = published["metrics_concat_folds"]
        comparison = {
            key: {
                "published": float(expected[key]),
                "reproduced": float(metrics[key]),
                "absolute_difference": round(abs(float(metrics[key]) - float(expected[key])), 4),
            }
            for key in metrics
        }

        prediction_path = run_dir / "oof_predictions.parquet"
        pd.DataFrame(
            {
                "row_index": np.arange(len(frame)),
                "instance_id": groups,
                "fold": fold_ids,
                "y_true": labels_original,
                "y_pred": predictions_original,
            }
        ).to_parquet(prediction_path, index=False)
        joblib.dump(estimator, run_dir / "last_fold_estimator.joblib")

        result = {
            "metrics": metrics,
            "published_comparison": comparison,
            "folds": folds,
            "classification_report": report,
        }
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)

        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "dataset": {
                    "n_windows": int(len(frame)),
                    "n_features": int(len(feature_columns)),
                    "n_legacy_instances": int(np.unique(groups).size),
                    "n_classes": int(len(classes)),
                },
                "metrics": metrics,
                "published_comparison": comparison,
                "integrity": {
                    "all_rows_predicted_once": bool(np.all(fold_ids > 0)),
                    "groups_crossing_folds": groups_crossing_folds,
                    "fold_sizes": {
                        str(key): int(value)
                        for key, value in pd.Series(fold_ids).value_counts().sort_index().items()
                    },
                },
                "outputs": {
                    "metrics": metrics_path.name,
                    "oof_predictions": prediction_path.name,
                    "last_fold_estimator": "last_fold_estimator.joblib",
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
