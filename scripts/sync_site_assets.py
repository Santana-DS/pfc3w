"""Sincroniza no projeto Quarto somente os artefatos leves selecionados."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ASSETS = PROJECT_ROOT / "site" / "assets"

ASSETS = {
    "figures": (
        "fig_f001_shap_global_e3.png",
        "fig_f002_e0_e3_f1_macro.png",
        "fig_f004_per_well_f1_distribution.png",
        "fig_f005_e3_confusion_matrices.png",
    ),
    "tables": (
        "tab_t004_e0_e3_metrics.csv",
        "tab_t005_e0_e3_contrasts.csv",
        "tab_t006_per_well_summary.csv",
        "tab_t007_shap_sensor_importance.csv",
    ),
}


def main() -> None:
    for kind, names in ASSETS.items():
        source_dir = PROJECT_ROOT / "reports" / kind
        destination_dir = SITE_ASSETS / kind
        destination_dir.mkdir(parents=True, exist_ok=True)
        expected = set(names)
        for existing in destination_dir.iterdir():
            if existing.is_file() and existing.name not in expected:
                existing.unlink()
        for name in names:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Artefato selecionado ausente: {source}")
            shutil.copyfile(source, destination_dir / name)


if __name__ == "__main__":
    main()
