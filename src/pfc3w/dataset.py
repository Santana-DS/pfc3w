"""Identidade e inventário estrutural do 3W Dataset."""

from __future__ import annotations

import configparser
import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REAL_INSTANCE_PATTERN = re.compile(
    r"^(?P<well_id>WELL-\d{5})_(?P<timestamp>\d{14})$",
    flags=re.IGNORECASE,
)
ARTIFICIAL_INSTANCE_PATTERN = re.compile(
    r"^(?P<source_type>SIMULATED|DRAWN)_(?P<source_index>\d{5})$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class InstanceIdentity:
    """Identificadores derivados exclusivamente da estrutura oficial do 3W."""

    instance_id: str
    legacy_instance_id: str
    class_id: int
    source_type: str
    well_id: str | None
    source_index: int | None
    nominal_start: str | None


@dataclass(frozen=True)
class InventoryRecord:
    """Metadados de uma instância sem carregar sua série completa."""

    relative_path: str
    instance_id: str
    legacy_instance_id: str
    class_id: int
    source_type: str
    well_id: str | None
    source_index: int | None
    nominal_start: str | None
    size_bytes: int
    num_rows: int
    num_columns: int
    schema_names: tuple[str, ...]
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converte o registro em tipos serializáveis."""
        values = asdict(self)
        values["schema_names"] = list(self.schema_names)
        return values


def iter_parquet_files(dataset_root: str | Path) -> Iterable[Path]:
    """Percorre deterministicamente os Parquet das classes 0 a 9."""
    root = Path(dataset_root)
    for class_id in range(10):
        yield from sorted((root / str(class_id)).glob("*.parquet"))


def parse_instance_identity(path: str | Path, dataset_root: str | Path) -> InstanceIdentity:
    """Extrai identidade da instância sem confundir arquivo com poço físico.

    ``instance_id`` inclui a classe porque nomes como ``SIMULATED_00001`` se
    repetem entre diretórios. ``legacy_instance_id`` preserva o identificador
    usado no baseline de Gabriel. Apenas arquivos ``WELL-*`` recebem
    ``well_id``; dados simulados ou desenhados não representam, por si só, um
    poço físico identificável.
    """
    root = Path(dataset_root).resolve()
    file_path = Path(path).resolve()
    try:
        relative_path = file_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Arquivo fora do dataset: {file_path}") from exc

    if len(relative_path.parts) != 2 or relative_path.suffix.lower() != ".parquet":
        raise ValueError(f"Estrutura de instância inesperada: {relative_path}")

    try:
        class_id = int(relative_path.parent.name)
    except ValueError as exc:
        raise ValueError(f"Diretório de classe inválido: {relative_path.parent.name}") from exc
    if class_id not in range(10):
        raise ValueError(f"Classe fora do intervalo 0..9: {class_id}")

    stem = relative_path.stem.upper()
    real_match = REAL_INSTANCE_PATTERN.fullmatch(stem)
    if real_match:
        return InstanceIdentity(
            instance_id=f"{class_id}/{stem}",
            legacy_instance_id=stem,
            class_id=class_id,
            source_type="REAL",
            well_id=real_match.group("well_id").upper(),
            source_index=None,
            nominal_start=real_match.group("timestamp"),
        )

    artificial_match = ARTIFICIAL_INSTANCE_PATTERN.fullmatch(stem)
    if artificial_match:
        return InstanceIdentity(
            instance_id=f"{class_id}/{stem}",
            legacy_instance_id=stem,
            class_id=class_id,
            source_type=artificial_match.group("source_type").upper(),
            well_id=None,
            source_index=int(artificial_match.group("source_index")),
            nominal_start=None,
        )

    raise ValueError(f"Nome de instância 3W não reconhecido: {relative_path.name}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 por streaming para limitar o uso de memória."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_instance(
    path: str | Path,
    dataset_root: str | Path,
    *,
    include_hash: bool = False,
) -> InventoryRecord:
    """Lê metadados Parquet e, opcionalmente, o hash do arquivo."""
    root = Path(dataset_root).resolve()
    file_path = Path(path).resolve()
    identity = parse_instance_identity(file_path, root)
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    return InventoryRecord(
        relative_path=file_path.relative_to(root).as_posix(),
        **asdict(identity),
        size_bytes=file_path.stat().st_size,
        num_rows=metadata.num_rows,
        num_columns=metadata.num_columns,
        schema_names=tuple(parquet_file.schema_arrow.names),
        sha256=sha256_file(file_path) if include_hash else None,
    )


def read_dataset_version(dataset_root: str | Path) -> str:
    """Lê a versão declarada pelo arquivo oficial ``dataset.ini``."""
    config_path = Path(dataset_root) / "dataset.ini"
    parser = configparser.ConfigParser()
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(config_path)
    return parser["VERSION"]["DATASET"]


def dataset_fingerprint(records: Iterable[InventoryRecord]) -> str:
    """Agrega caminhos e hashes em uma impressão digital determinística."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        if record.sha256 is None:
            raise ValueError("Todos os registros precisam de SHA-256 para gerar o fingerprint")
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
