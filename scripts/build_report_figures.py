"""Gera as figuras rastreáveis selecionadas para o relatório."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRIVATE_REPORT_CONFIG = PROJECT_ROOT / "config" / "reporting" / "e0_e3_tables.yaml"
PUBLIC_REPORT_CONFIG = PROJECT_ROOT / "config" / "reporting" / "public" / "results.yaml"
DEFAULT_REPORT_CONFIG = (
    PRIVATE_REPORT_CONFIG if PRIVATE_REPORT_CONFIG.is_file() else PUBLIC_REPORT_CONFIG
)

from pfc3w.figures import (  # noqa: E402
    build_f001_shap_global,
    build_f002_e0_e3_metrics,
    build_f003_state_support,
    build_f004_per_well_distribution,
    build_f005_e3_confusion_matrices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REPORT_CONFIG,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    parser.add_argument(
        "--manuscript-figures-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "reports" / "pfc1" / "latex" / "manuscript" / "figures"
        ),
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    shap_specification = config["shap_sensor_importance"]
    runs = config["runs"]
    public_run_id = lambda specification: specification.get(  # noqa: E731
        "scientific_run_id", specification["run_id"]
    )
    outputs = build_f001_shap_global(
        run_dir=(PROJECT_ROOT / "results" / shap_specification["run_id"]).resolve(),
        output_dir=args.output_dir.resolve(),
        manuscript_figures_dir=args.manuscript_figures_dir.resolve(),
        top_n=args.top_n,
        scientific_run_id=public_run_id(shap_specification),
    )
    for kind, path in outputs.items():
        print(f"F001_{kind}: {path}")
    builders = [
        build_f002_e0_e3_metrics(
            metrics_csv=PROJECT_ROOT / "reports" / "tables" / "tab_t004_e0_e3_metrics.csv",
            output_dir=args.output_dir.resolve(),
            manuscript_figures_dir=args.manuscript_figures_dir.resolve(),
        ),
        build_f003_state_support(
            support_csv=PROJECT_ROOT / "reports" / "tables" / "tab_t003_state_support.csv",
            output_dir=args.output_dir.resolve(),
        ),
        build_f004_per_well_distribution(
            run_specifications=[
                (
                    specification["experiment_id"],
                    specification["model"],
                    specification["run_id"],
                    public_run_id(specification),
                )
                for specification in runs
                if specification["experiment_id"] in {"E1", "E3"}
            ],
            results_dir=PROJECT_ROOT / "results",
            output_dir=args.output_dir.resolve(),
        ),
        build_f005_e3_confusion_matrices(
            run_specifications=[
                (
                    "Random Forest"
                    if specification["model"] == "rf"
                    else specification["display_model"],
                    specification["run_id"],
                    public_run_id(specification),
                )
                for specification in runs
                if specification["experiment_id"] == "E3"
            ],
            results_dir=PROJECT_ROOT / "results",
            output_dir=args.output_dir.resolve(),
        ),
    ]
    for figure_index, figure_outputs in enumerate(builders, start=2):
        for kind, path in figure_outputs.items():
            print(f"F{figure_index:03d}_{kind}: {path}")


if __name__ == "__main__":
    main()
