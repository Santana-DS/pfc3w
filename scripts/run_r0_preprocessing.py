"""Executa o pré-processamento completo R0 no código canônico do baseline."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.paths import load_paths  # noqa: E402


def git_head(repository: Path) -> str | None:
    git_dir = repository / ".git"
    process = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--max-instances-per-class",
        type=int,
        default=None,
        help="Limite somente para ensaios; omitido executa o R0 completo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    dataset_root = paths["dataset_root"].resolve()

    sys.path.insert(0, str(baseline_repo))
    baseline_config = importlib.import_module("config")
    feature_engineering = importlib.import_module("src.feature_engineering")
    baseline_config.RAW_DATA_DIR = dataset_root
    feature_engineering.RAW_DATA_DIR = dataset_root

    suffix = "full" if args.max_instances_per_class is None else "sample"
    run_id = datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ_R0_preprocessing_{suffix}")
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cleaned_path = run_dir / "cleaned.parquet"
    features_path = run_dir / "features_window_class.parquet"
    manifest_path = run_dir / "run_manifest.json"

    inventory_manifest_path = PROJECT_ROOT / "results" / "inventory" / "data_manifest.json"
    with inventory_manifest_path.open(encoding="utf-8") as stream:
        inventory_manifest = json.load(stream)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "experiment_id": "R0",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "dataset_version": inventory_manifest["dataset_version"],
        "dataset_fingerprint_sha256": inventory_manifest["dataset_fingerprint_sha256"],
        "parameters": {
            "max_instances_per_class": args.max_instances_per_class,
            "batch_size": args.batch_size,
            "sensors": list(baseline_config.KEY_SENSORS),
            "window_size": baseline_config.WINDOW_SIZE,
            "step_size": baseline_config.STEP_SIZE,
            "normalize": True,
            "label_strategy": "window",
            "smooth_filter": "gaussian",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    try:
        feature_engineering.run_pipeline_chunked(
            max_instances_per_class=args.max_instances_per_class,
            batch_size=args.batch_size,
            sensors=list(baseline_config.KEY_SENSORS),
            window_size=baseline_config.WINDOW_SIZE,
            step_size=baseline_config.STEP_SIZE,
            normalize=True,
            label_strategy="window",
            smooth_filter="gaussian",
            cleaned_path=cleaned_path,
            features_path=features_path,
            verbose=True,
        )

        features = pd.read_parquet(
            features_path,
            columns=["instance_id", "fault_class", "source_type", "window_label"],
        )
        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "outputs": {
                    "cleaned_parquet": cleaned_path.name,
                    "features_parquet": features_path.name,
                    "cleaned_size_bytes": cleaned_path.stat().st_size,
                    "features_size_bytes": features_path.stat().st_size,
                },
                "n_windows": int(len(features)),
                "n_legacy_instance_ids": int(features["instance_id"].nunique()),
                "classes": sorted(int(value) for value in features["window_label"].unique()),
                "source_counts": {
                    str(key): int(value)
                    for key, value in features["source_type"].value_counts().items()
                },
            }
        )
    except Exception as exc:
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

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
