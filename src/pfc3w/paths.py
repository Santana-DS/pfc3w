"""Resolução explícita de caminhos locais, sem caminhos absolutos no código."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_paths(config_path: str | Path) -> dict[str, Path]:
    """Carrega caminhos locais e os normaliza como objetos ``Path``."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return {key: Path(value).expanduser() for key, value in values.items()}
