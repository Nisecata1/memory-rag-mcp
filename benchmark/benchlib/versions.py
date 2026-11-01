"""Discover experiment packages without importing their adapters or models."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "memory-retrieval-v1"


@dataclass(frozen=True)
class Version:
    id: str
    directory: Path
    label: str
    aliases: tuple[str, ...]
    result_key: str
    default: bool
    adapter: Path
    config: Path
    source_files: tuple[str, ...]


def _local_file(directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Version file paths must be non-empty strings")
    candidate = (directory / value).resolve()
    if not candidate.is_relative_to(directory.resolve()) or not candidate.is_file():
        raise ValueError(f"Missing or out-of-package version file: {value}")
    return candidate


def discover_versions(root: Path = ROOT) -> dict[str, Version]:
    versions: dict[str, Version] = {}
    identifiers: dict[str, str] = {}
    for directory in sorted((root / "versions").iterdir()):
        if not directory.is_dir() or directory.name.startswith(("_", ".")):
            continue
        path = directory / "version.json"
        if not path.is_file():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, dict) or item.get("schema_version") != 1:
            raise ValueError(f"Invalid version manifest: {path}")
        if type(item.get("enabled", True)) is not bool:
            raise ValueError(f"enabled must be boolean: {path}")
        if not item.get("enabled", True):
            continue
        version_id = item.get("id")
        if version_id != directory.name or not re.fullmatch(r"[a-z][a-z0-9_]*", str(version_id)):
            raise ValueError(f"Version id must match its snake_case directory: {path}")
        if item.get("protocol") != PROTOCOL:
            raise ValueError(f"Unsupported evaluation protocol: {path}")
        aliases = item.get("aliases", [])
        result_key = item.get("result_key", version_id)
        files = item.get("source_files", [])
        if (not isinstance(aliases, list) or not all(isinstance(a, str) and re.fullmatch(r"[a-z][a-z0-9_]*", a) for a in aliases)
                or not isinstance(result_key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", result_key)
                or type(item.get("default", False)) is not bool
                or not isinstance(files, list) or not all(isinstance(f, str) and f and not Path(f).is_absolute() and ".." not in Path(f).parts for f in files)):
            raise ValueError(f"Invalid aliases/result_key/default/source_files: {path}")
        for name in [version_id, *aliases]:
            if name in identifiers:
                raise ValueError(f"Duplicate version id/alias {name}: {identifiers[name]} and {version_id}")
            identifiers[name] = version_id
        versions[version_id] = Version(
            version_id, directory.resolve(), str(item.get("label", version_id)), tuple(aliases),
            result_key, item.get("default", False), _local_file(directory, item.get("adapter")),
            _local_file(directory, item.get("config")), tuple(files),
        )
    if not versions:
        raise ValueError("No enabled experiment packages found under benchmark/versions/")
    return versions


def select_versions(selection: str | None = None, *, all_versions: bool = False,
                    root: Path = ROOT) -> list[Version]:
    available = discover_versions(root)
    if all_versions:
        return list(available.values())
    if selection is None:
        selected = [v for v in available.values() if v.default]
        if not selected:
            raise ValueError("Select --versions or mark a version as default")
        return selected
    names = [name.strip() for name in selection.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one version must be selected")
    lookup = {name: v for v in available.values() for name in [v.id, *v.aliases]}
    result = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"Unknown version {name!r}; run 'python run_benchmark.py list'")
        if lookup[name] not in result:
            result.append(lookup[name])
    return result


def create_retriever(version: Version, records: list[dict], **kwargs):
    name = f"_rag_benchmark_{version.id}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, version.adapter, submodule_search_locations=[str(version.directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load adapter: {version.adapter}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, "create_retriever", None)
        if not callable(factory):
            raise ValueError(f"Adapter must export create_retriever: {version.adapter}")
        return factory(records, **kwargs)
    except BaseException:
        sys.modules.pop(name, None)
        raise


def version_main(directory: Path) -> None:
    """Entry point used by each package's run.py; no package-specific scoring."""
    from benchlib.runner import main
    main(pinned_version=directory.resolve().name)
