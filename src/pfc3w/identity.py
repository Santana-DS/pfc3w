"""Identificadores estáveis para especificações experimentais."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

IDENTITY_SCHEMA_VERSION = 1
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")
_MANIFEST_SPECIFICATION_FIELDS = (
    "experiment_id",
    "model",
    "mode",
    "imputation",
    "purpose",
    "project_commit",
    "baseline_commit",
    "dataset_version",
    "dataset_fingerprint_sha256",
    "features_sha256",
    "source_features_sha256",
    "split_source_features_sha256",
    "group_column",
    "assignment_experiment",
    "assignment_sha256",
    "preprocessing",
    "parameters",
    "hyperparameters",
    "validation",
    "published_metrics_sha256",
    "published_metrics_code_commit",
    "random_seed",
    "fold",
    "maximum_explanation_samples_per_true_class",
)
_DATA_FINGERPRINT_FIELDS = (
    "dataset_fingerprint_sha256",
    "features_sha256",
    "source_features_sha256",
)


def slug_component(value: str) -> str:
    """Converte um componente legível em um slug estável."""
    slug = _SLUG_SEPARATOR.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Componente do identificador não pode ser vazio")
    return slug


def canonical_specification(specification: Mapping[str, Any]) -> bytes:
    """Serializa uma especificação JSON sem depender da ordem das chaves."""
    try:
        serialized = json.dumps(
            specification,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Especificação precisa ser JSON canônico e sem NaN") from error
    return serialized.encode("utf-8")


def scientific_run_id(
    experiment_id: str,
    model: str,
    variant: str,
    specification: Mapping[str, Any],
    *,
    digest_length: int = 10,
) -> str:
    """Gera ``<experimento>-<modelo>-<variante>__<digest>``.

    O identificador representa uma especificação científica. Repetir a mesma
    especificação produz o mesmo ID; horário e duração pertencem ao manifesto
    da execução, e não ao nome.
    """
    if not 8 <= digest_length <= 64:
        raise ValueError("digest_length precisa estar entre 8 e 64")

    components = {
        "experiment_id": slug_component(experiment_id),
        "model": slug_component(model),
        "variant": slug_component(variant),
    }
    payload = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        **components,
        "specification": specification,
    }
    digest = hashlib.sha256(canonical_specification(payload)).hexdigest()[:digest_length]
    readable = "-".join(components.values())
    return f"{readable}__{digest}"


def manifest_scientific_specification(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extrai somente entradas científicas de um manifesto legado.

    Métricas, horários, duração, caminhos e nomes de outputs são excluídos para
    que o ID represente a especificação que produziu o resultado.
    """
    if not manifest.get("project_commit"):
        raise ValueError("Manifesto sem project_commit")
    if not any(manifest.get(field) for field in _DATA_FINGERPRINT_FIELDS):
        raise ValueError("Manifesto sem impressão digital dos dados/features")

    return {
        field: manifest[field]
        for field in _MANIFEST_SPECIFICATION_FIELDS
        if field in manifest and manifest[field] is not None
    }
