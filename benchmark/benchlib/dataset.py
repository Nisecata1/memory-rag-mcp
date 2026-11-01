"""Download and normalize the pinned, public LoCoMo text-only retrieval demo.

This module uses only the standard library.  It never imports the memory server
or opens its databases.  QA annotations are kept outside the retrieval corpus.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.request import Request, urlopen


LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_DATA_GIT_BLOB = "d95b872480b413d935821fdc3c84f8a8f5f29e73"
LOCOMO_DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_DATA_BYTES = 2_805_274
LOCOMO_REPOSITORY = "https://github.com/snap-research/locomo"
LOCOMO_DATA_URL = (
    f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}"
    "/data/locomo10.json"
)
LOCOMO_LICENSE_URL = (
    f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}"
    "/LICENSE.txt"
)
SCHEMA_VERSION = "memory-retrieval-dataset-v1"
RAW_SOURCE_DIRECTORY = Path("data") / "raw" / "locomo"
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
IMAGE_FIELDS = {"img_url", "image_url", "image_urls", "image", "images", "img", "blip_caption", "caption"}
_SESSION_RE = re.compile(r"session_(\d+)\Z")
_EVIDENCE_RE = re.compile(r"D\d+:\d+\Z")


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options = {"ensure_ascii": False, "sort_keys": True, "allow_nan": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(data)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_dataset_manifest(path: Path, manifest: dict[str, Any]) -> None:
    # Cache relocation is not a new evaluation dataset. Keep the exact existing
    # manifest bytes when only its historical raw-source location differs: run
    # compatibility also hashes this file, not only corpus/queries.
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict):
            previous_content = {key: value for key, value in previous.items() if key != "raw_source_manifest"}
            new_content = {key: value for key, value in manifest.items() if key != "raw_source_manifest"}
            if previous_content == new_content:
                return
    _write_bytes(path, _json_bytes(manifest, pretty=True))


def _download_once(url: str, path: Path, *, maximum_bytes: int) -> tuple[bytes, str]:
    if path.exists():
        return path.read_bytes(), "reused_local_file"
    request = Request(url, headers={"User-Agent": "memory-rag-benchmark-demo/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(f"Source exceeds the permitted download size: {url}")
    _write_bytes(path, data)
    return data, datetime.now(timezone.utc).isoformat()


def _source(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = root / RAW_SOURCE_DIRECTORY / LOCOMO_COMMIT
    data_path = raw_dir / "locomo10.json"
    license_path = raw_dir / "LICENSE.txt"
    data_bytes, downloaded_at = _download_once(LOCOMO_DATA_URL, data_path, maximum_bytes=8 * 1024 * 1024)
    git_blob = hashlib.sha1(b"blob " + str(len(data_bytes)).encode("ascii") + b"\0" + data_bytes).hexdigest()
    if len(data_bytes) != LOCOMO_DATA_BYTES or git_blob != LOCOMO_DATA_GIT_BLOB or _sha256(data_bytes) != LOCOMO_DATA_SHA256:
        raise ValueError("LoCoMo source does not match the pinned official Git blob; existing files were not replaced.")
    license_bytes, license_downloaded_at = _download_once(LOCOMO_LICENSE_URL, license_path, maximum_bytes=128 * 1024)
    if b"Attribution-NonCommercial 4.0 International" not in license_bytes:
        raise ValueError("The downloaded LoCoMo license is not the expected CC BY-NC 4.0 text.")
    manifest_path = raw_dir / "source_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    source = {
        "dataset": "LoCoMo locomo10",
        "repository": LOCOMO_REPOSITORY,
        "commit": LOCOMO_COMMIT,
        "url": LOCOMO_DATA_URL,
        "file": "locomo10.json",
        "size_bytes": len(data_bytes),
        "sha256": _sha256(data_bytes),
        "git_blob_sha1": git_blob,
        "downloaded_at": previous.get("downloaded_at", downloaded_at),
        "license": "CC-BY-NC-4.0",
        "license_url": LOCOMO_LICENSE_URL,
        "license_sha256": _sha256(license_bytes),
        "license_downloaded_at": previous.get("license_downloaded_at", license_downloaded_at),
    }
    _write_bytes(manifest_path, _json_bytes(source, pretty=True))
    samples = json.loads(data_bytes)
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("Expected the pinned LoCoMo release containing ten conversations.")
    return samples, source


def _scenario_splits(samples: list[dict[str, Any]]) -> dict[str, str]:
    ids = [str(sample["sample_id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source scenario IDs.")
    ordered = sorted(ids, key=lambda value: (_sha256(value.encode("utf-8")), value))
    development_count = max(1, math.floor(len(ordered) * 0.6))
    return {scenario_id: "dev" if index < development_count else "test" for index, scenario_id in enumerate(ordered)}


def _records_for_sample(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, tuple[str, dict[str, Any]]]]:
    scenario_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    session_keys = sorted(
        (key for key in conversation if _SESSION_RE.fullmatch(key)),
        key=lambda key: int(_SESSION_RE.fullmatch(key).group(1)),
    )
    records: list[dict[str, Any]] = []
    evidence_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for session_key in session_keys:
        session_number = int(_SESSION_RE.fullmatch(session_key).group(1))
        safe_scenario = re.sub(r"[^A-Za-z0-9_-]", "-", scenario_id)
        record_id = f"eventMemId-bench-locomo-{safe_scenario}-s{session_number:03d}"
        session_date = str(conversation.get(f"{session_key}_date_time", "timestamp unavailable"))
        messages = []
        for turn in conversation[session_key]:
            message_id = str(turn["dia_id"])
            if message_id in evidence_map:
                raise ValueError(f"Duplicate dia_id inside scenario {scenario_id}: {message_id}")
            speaker = str(turn.get("speaker", "unknown speaker"))
            original_text = str(turn.get("text", ""))
            content = f"{speaker}: {original_text}"
            caption = turn.get("blip_caption")
            if caption:
                if not isinstance(caption, str):
                    caption = json.dumps(caption, ensure_ascii=False, sort_keys=True)
                content += f"\n[Dataset-provided image caption; image not loaded]: {caption}"
            messages.append({"message_id": message_id, "role": "user", "content": content, "speaker": speaker})
            evidence_map[message_id] = (record_id, turn)
        records.append({
            "scenario_id": scenario_id,
            "record_id": record_id,
            "title": f"LoCoMo {scenario_id} {session_key} | {session_date}",
            "messages": messages,
        })
    return records, evidence_map


def _queries_for_sample(
    sample: dict[str, Any],
    evidence_map: dict[str, tuple[str, dict[str, Any]]],
    split: str,
    categories: set[int],
) -> tuple[list[dict[str, Any]], Counter]:
    scenario_id = str(sample["sample_id"])
    queries: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for source_index, qa in enumerate(sample.get("qa", [])):
        stats["source_queries"] += 1
        category = qa.get("category")
        if category not in categories:
            stats[f"category_{category}_excluded"] += 1
            continue
        query = qa.get("question")
        if not isinstance(query, str) or not query.strip():
            stats["empty_question_excluded"] += 1
            continue
        evidence = qa.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            stats["no_evidence_excluded"] += 1
            continue
        if any(not isinstance(item, str) or not _EVIDENCE_RE.fullmatch(item.strip()) for item in evidence):
            stats["unparseable_evidence_excluded"] += 1
            continue
        evidence_ids = list(dict.fromkeys(item.strip() for item in evidence))
        if any(message_id not in evidence_map for message_id in evidence_ids):
            stats["unmapped_evidence_excluded"] += 1
            continue
        if any(any(turn.get(key) for key in IMAGE_FIELDS) for _, turn in (evidence_map[item] for item in evidence_ids)):
            stats["image_evidence_excluded"] += 1
            continue
        duplicate_key = (" ".join(query.casefold().split()), tuple(sorted(evidence_ids)))
        if duplicate_key in seen:
            stats["duplicate_query_excluded"] += 1
            continue
        seen.add(duplicate_key)
        evidence_groups = []
        for group_index, message_id in enumerate(evidence_ids, start=1):
            record_id, _ = evidence_map[message_id]
            evidence_groups.append({
                "group_id": f"evidence-{group_index:03d}",
                "alternatives": [{"record_id": record_id, "message_id": message_id}],
            })
        answer = qa.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False, sort_keys=True)
        queries.append({
            "query_id": f"locomo-{scenario_id}-qa-{source_index:04d}",
            "scenario_id": scenario_id,
            "split": split,
            "category": CATEGORY_NAMES[category],
            "query": query.strip(),
            "answerable": True,
            "relevant_record_ids": sorted({group["alternatives"][0]["record_id"] for group in evidence_groups}),
            "evidence_groups": evidence_groups,
            "reference_answer": answer,
        })
        stats["eligible_queries"] += 1
    return queries, stats


def validate_dataset(records: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, int]:
    """Validate IDs, scenario isolation and complete evidence references.

    Labels have no semantic completeness guarantee: this checks the exact
    published annotations, rather than asserting they are minimal sufficient
    evidence or an exhaustive set of every possible correct source.
    """
    if not records or not queries:
        raise ValueError("A retrieval dataset must contain records and queries.")
    by_id: dict[str, dict[str, Any]] = {}
    message_ids: dict[str, set[str]] = {}
    scenarios = set()
    for record in records:
        record_id = record["record_id"]
        if record_id in by_id:
            raise ValueError(f"Duplicate record_id: {record_id}")
        by_id[record_id] = record
        scenarios.add(record["scenario_id"])
        ids: set[str] = set()
        if not record["messages"]:
            raise ValueError(f"Empty session: {record_id}")
        for message in record["messages"]:
            if message["message_id"] in ids:
                raise ValueError(f"Duplicate message in {record_id}")
            ids.add(message["message_id"])
            if message["role"] != "user" or not isinstance(message["content"], str):
                raise ValueError("LoCoMo speaker mapping contract was violated.")
        message_ids[record_id] = ids
    query_ids = set()
    scenario_split: dict[str, str] = {}
    evidence_count = 0
    for query in queries:
        if query["query_id"] in query_ids:
            raise ValueError(f"Duplicate query_id: {query['query_id']}")
        query_ids.add(query["query_id"])
        scenario_id = query["scenario_id"]
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown query scenario: {scenario_id}")
        split = query["split"]
        if split not in {"dev", "test"} or scenario_split.setdefault(scenario_id, split) != split:
            raise ValueError("One conversation must not appear in both dev and test queries.")
        if query["answerable"] is not True or not query["evidence_groups"]:
            raise ValueError("This profile only supports answerable, evidence-annotated queries.")
        group_ids = set()
        evidence_records = set()
        for group in query["evidence_groups"]:
            if group["group_id"] in group_ids or not group["alternatives"]:
                raise ValueError("Evidence groups require unique IDs and nonempty alternatives.")
            group_ids.add(group["group_id"])
            for alternative in group["alternatives"]:
                record_id = alternative["record_id"]
                if record_id not in by_id or by_id[record_id]["scenario_id"] != scenario_id:
                    raise ValueError(f"Evidence crosses scenarios or references a missing record: {record_id}")
                if alternative["message_id"] not in message_ids[record_id]:
                    raise ValueError(f"Evidence message is missing: {alternative}")
                evidence_records.add(record_id)
            evidence_count += 1
        if set(query["relevant_record_ids"]) != evidence_records:
            raise ValueError("relevant_record_ids must agree with the evidence annotations.")
    return {
        "scenarios": len(scenarios),
        "records": len(records),
        "messages": sum(len(record["messages"]) for record in records),
        "queries": len(queries),
        "evidence_groups": evidence_count,
    }


def prepare_locomo(root: Path, profile: str = "demo", max_queries: int = 24) -> Path:
    """Prepare a reproducible, derived text-only retrieval dataset.

    ``root`` is the benchmark workspace, not the memory-server repository.
    ``demo`` uses one full dev conversation and caps single-hop queries.
    ``full`` uses all conversations and eligible categories 1/2/4; its query
    count is not capped by ``max_queries``.
    """
    root = Path(root).resolve()
    if profile not in {"demo", "full"}:
        raise ValueError("profile must be 'demo' or 'full'.")
    if type(max_queries) is not int or max_queries <= 0:
        raise ValueError("max_queries must be a positive integer.")
    samples, source = _source(root)
    scenario_splits = _scenario_splits(samples)
    samples = sorted(samples, key=lambda sample: str(sample["sample_id"]))
    if profile == "demo":
        samples = [next(sample for sample in samples if scenario_splits[str(sample["sample_id"])] == "dev")]
    categories = {4} if profile == "demo" else {1, 2, 4}
    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    filtering: Counter = Counter()
    for sample in samples:
        sample_records, evidence_map = _records_for_sample(sample)
        sample_queries, sample_stats = _queries_for_sample(
            sample, evidence_map, scenario_splits[str(sample["sample_id"])], categories,
        )
        records.extend(sample_records)
        queries.extend(sample_queries)
        filtering.update(sample_stats)
    if profile == "demo":
        filtering["query_limit_excluded"] = max(0, len(queries) - max_queries)
        queries = queries[:max_queries]
    filtering["selected_queries"] = len(queries)
    counts = validate_dataset(records, queries)
    corpus_bytes = b"".join(_json_bytes(record) for record in records)
    queries_bytes = b"".join(_json_bytes(query) for query in queries)
    fingerprint_inputs = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source["commit"],
        "source_sha256": source["sha256"],
        "profile": profile,
        "corpus_sha256": _sha256(corpus_bytes),
        "queries_sha256": _sha256(queries_bytes),
    }
    manifest = {
        **fingerprint_inputs,
        "dataset_id": f"locomo-{profile}-v1",
        "dataset_fingerprint": _sha256(_json_bytes(fingerprint_inputs)),
        "description": "Derived text-only retrieval task; not official LoCoMo QA or end-to-end memory benchmark scores.",
        "source": {key: source[key] for key in ("repository", "commit", "url", "sha256", "size_bytes", "license", "license_url")},
        "raw_source_manifest": str(RAW_SOURCE_DIRECTORY / LOCOMO_COMMIT / "source_manifest.json"),
        "split_strategy": "Sort scenario IDs by SHA256(sample_id); first floor(60%) are dev, remainder test. No query-level random split.",
        "scenario_splits": scenario_splits,
        "included_scenarios": sorted({record["scenario_id"] for record in records}),
        "query_categories": [CATEGORY_NAMES[category] for category in sorted(categories)],
        "max_queries": max_queries if profile == "demo" else None,
        "counts": counts,
        "query_counts_by_split": dict(sorted(Counter(query["split"] for query in queries).items())),
        "query_counts_by_category": dict(sorted(Counter(query["category"] for query in queries).items())),
        "filtering": dict(sorted(filtering.items())),
        "speaker_mapping": "Every source speaker maps to role=user, with original speaker retained as metadata and a content prefix. These are two conversational peers, not user/assistant turns.",
        "text_only_policy": "No image downloads. Dataset-provided BLIP captions remain in the corpus with a caption marker; questions whose evidence messages have nonempty image fields are excluded as whole questions.",
        "evidence_semantics": "Each deduplicated published evidence dialogue ID becomes a required group with one alternative. All-annotated-evidence coverage is not a claim of minimal sufficient evidence or exhaustive relevance judgments.",
        "retrieval_visibility": "Only original conversation text, speaker, session timestamp and dataset-provided image captions. QA labels, reference answers, observations, event summaries and session summaries are never included in corpus.jsonl.",
        "scenario_isolation": "For each query, retrieve only records with the same scenario_id; pooling conversations would change the evaluation protocol.",
        "limitations": [
            "English synthetic conversations edited by humans; not representative of the user's Chinese personal records.",
            "The demo is a development-only pipeline smoke test and contains no held-out test scenario.",
            "Categories open_domain and adversarial are excluded; no no-answer detection or abstention claims are supported.",
            "Evidence annotations can be incomplete, redundant or semantically imperfect; exact-label metrics are diagnostic.",
        ],
    }
    output_dir = root / "data" / "processed" / f"locomo-{profile}-v1"
    _write_bytes(output_dir / "corpus.jsonl", corpus_bytes)
    _write_bytes(output_dir / "queries.jsonl", queries_bytes)
    _write_dataset_manifest(output_dir / "manifest.json", manifest)
    return output_dir


def load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(path)
    if path.is_file():
        path = path.parent
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    corpus_bytes = (path / "corpus.jsonl").read_bytes()
    queries_bytes = (path / "queries.jsonl").read_bytes()
    for field, data in (("corpus_sha256", corpus_bytes), ("queries_sha256", queries_bytes)):
        if manifest.get(field) != _sha256(data):
            raise ValueError(f"Dataset checksum does not match manifest: {field}")
    records = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()]
    queries = [json.loads(line) for line in queries_bytes.decode("utf-8").splitlines() if line.strip()]
    counts = validate_dataset(records, queries)
    if counts != manifest["counts"]:
        raise ValueError("Dataset counts do not match manifest.")
    return manifest, records, queries
