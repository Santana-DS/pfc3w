"""Invariantes do protocolo experimental."""

from __future__ import annotations

from collections.abc import Iterable


def assert_disjoint_groups(
    train_groups: Iterable[str],
    test_groups: Iterable[str],
    *,
    group_name: str = "well_id",
) -> None:
    """Falha quando a mesma unidade experimental aparece em treino e teste."""
    train = set(train_groups)
    test = set(test_groups)
    overlap = train & test
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"Vazamento em {group_name}: {len(overlap)} grupo(s) compartilhado(s): {examples}"
        )
