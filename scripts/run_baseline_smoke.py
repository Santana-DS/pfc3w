"""Executa uma amostra mínima do pré-processamento canônico do baseline."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.paths import load_paths  # noqa: E402


def main() -> None:
    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    baseline_repo = paths["baseline_repo"].resolve()
    dataset_root = paths["dataset_root"].resolve()

    sys.path.insert(0, str(baseline_repo))
    baseline_config = importlib.import_module("config")
    feature_engineering = importlib.import_module("src.feature_engineering")

    # O caminho é a única adaptação: algoritmos e parâmetros permanecem no
    # commit canônico do baseline.
    baseline_config.RAW_DATA_DIR = dataset_root
    feature_engineering.RAW_DATA_DIR = dataset_root

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_baseline_smoke")
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cleaned_path = run_dir / "cleaned.parquet"
    features_path = run_dir / "features_window_class.parquet"

    feature_engineering.run_pipeline_chunked(
        max_instances_per_class=1,
        batch_size=1,
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

    cleaned = pd.read_parquet(
        cleaned_path,
        columns=["instance_id", "fault_class", "source_type", "class"],
    )
    features = pd.read_parquet(features_path)
    manifest = {
        "run_id": run_id,
        "purpose": "compatibility_smoke_test",
        "baseline_repository": str(baseline_repo),
        "dataset_root": str(dataset_root),
        "instances_per_class_limit": 1,
        "window_size": baseline_config.WINDOW_SIZE,
        "step_size": baseline_config.STEP_SIZE,
        "normalization": "full_instance_zscore",
        "filter": "gaussian",
        "n_clean_rows": int(len(cleaned)),
        "n_instances": int(cleaned["instance_id"].nunique()),
        "n_windows": int(len(features)),
        "n_feature_columns": int(
            len(
                set(features.columns)
                - {"instance_id", "fault_class", "source_type", "window_start", "window_label"}
            )
        ),
        "classes": sorted(int(value) for value in cleaned["fault_class"].unique()),
    }
    with (run_dir / "smoke_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
