"""Executa um classificador R0 sem modificar o clone canônico do baseline."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.paths import load_paths  # noqa: E402

MODEL_SCRIPTS = {
    "rf": "train_rf_window_class.py",
    "xgboost": "train_xgboost_window_class.py",
}
METRIC_FILES = {
    "rf": "rf_window_class_metrics.json",
    "xgboost": "xgboost_window_class_metrics.json",
}


class Tee:
    """Replica texto no terminal e em um arquivo de log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODEL_SCRIPTS))
    parser.add_argument(
        "--preprocessing-run",
        type=Path,
        required=True,
        help="Diretório R0 que contém features_window_class.parquet.",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-iter-search", type=int, default=None)
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


def main() -> None:
    args = parse_args()
    preprocessing_run = args.preprocessing_run.resolve()
    features_path = preprocessing_run / "features_window_class.parquet"
    artifact_audit_path = preprocessing_run / "artifact_audit.json"
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    baseline_script = baseline_repo / "scripts" / MODEL_SCRIPTS[args.model]

    sys.path.insert(0, str(baseline_repo))
    baseline_config = importlib.import_module("config")
    n_jobs = args.n_jobs if args.n_jobs is not None else baseline_config.N_JOBS
    n_iter_search = (
        args.n_iter_search if args.n_iter_search is not None else baseline_config.N_ITER_SEARCH
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_R0_{args.model}_full"
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    models_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    for directory in (models_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    baseline_config.FEATURES_BY_FILTER = {
        **baseline_config.FEATURES_BY_FILTER,
        "gaussian": features_path,
    }
    baseline_config.MODELS_DIR = models_dir
    baseline_config.METRICS_DIR = metrics_dir
    baseline_config.FIGURES_DIR = figures_dir
    baseline_config.N_JOBS = n_jobs
    baseline_config.N_ITER_SEARCH = n_iter_search

    feature_hash = None
    if artifact_audit_path.exists():
        with artifact_audit_path.open(encoding="utf-8") as stream:
            feature_hash = json.load(stream)["features"]["sha256"]
    if feature_hash is None:
        feature_hash = sha256_file(features_path)

    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "experiment_id": "R0",
        "model": args.model,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "baseline_script": str(baseline_script.relative_to(baseline_repo)),
        "baseline_script_sha256": sha256_file(baseline_script),
        "preprocessing_run_id": preprocessing_run.name,
        "features_sha256": feature_hash,
        "parameters": {
            "filter": "gaussian",
            "n_splits_cv": baseline_config.N_SPLITS_CV,
            "n_iter_search": n_iter_search,
            "n_jobs": n_jobs,
            "random_state": baseline_config.RANDOM_STATE,
        },
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    original_argv = sys.argv[:]
    os.environ.setdefault("MPLBACKEND", "Agg")
    log_path = run_dir / "training.log"
    try:
        sys.argv = [str(baseline_script), "--filter", "gaussian"]
        with log_path.open("w", encoding="utf-8") as log_stream:
            stdout = Tee(sys.stdout, log_stream)
            stderr = Tee(sys.stderr, log_stream)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runpy.run_path(str(baseline_script), run_name="__main__")

        metric_path = metrics_dir / METRIC_FILES[args.model]
        with metric_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "metrics_file": str(metric_path.relative_to(run_dir)),
                "metrics": metrics.get("metrics_concat_folds"),
                "best_parameters": metrics.get("best_params"),
                "best_cv_score": metrics.get("cv", {}).get("best_score_cv"),
            }
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        sys.argv = original_argv
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
