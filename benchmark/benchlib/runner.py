"""Prepare, validate, run and compare a reproducible memory retrieval benchmark."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
PROJECT_SOURCE_FILES = (
    "server.py",
    "server_utils.py",
    "embedding_utils.py",
    "memory_domain.py",
    "memory_store.py",
    "memory_access.py",
    "config/settings.yaml",
    "requirements.txt",
)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_run_config(args: argparse.Namespace, argv: list[str], *, config_path: Path | None = None,
                     skip_fields: set[str] | None = None) -> argparse.Namespace:
    if config_path is None and not getattr(args, "config", None):
        return args
    path = Path(config_path or args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "dataset", "retrievers", "versions", "split", "scenario", "query_limit", "top_k", "window_radius", "message_top_k", "project_dir", "adapter_options"}
    if not isinstance(config, dict) or config.get("schema_version") != 1 or set(config) - allowed:
        raise ValueError("Run config must have schema_version=1 and only documented run options.")
    if "versions" in config and "retrievers" in config:
        raise ValueError("Use versions or legacy retrievers in a config, not both")
    configured = set(getattr(args, "_configured_fields", set()))
    for name, value in config.items():
        if name == "schema_version" or name in (skip_fields or set()):
            continue
        if name in {"versions", "retrievers"} and any(
            item.split("=", 1)[0] in {"--versions", "--version", "--retrievers", "--all-versions"} for item in argv
        ):
            continue
        flag = "--" + name.replace("_", "-")
        if any(item == flag or item.startswith(flag + "=") or (name == "versions" and item.split("=", 1)[0] == "--version") for item in argv):
            continue
        if name in {"query_limit", "top_k", "window_radius", "message_top_k"}:
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        elif name in {"retrievers", "versions"}:
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                value = ",".join(value)
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be a string or list of strings")
        elif name == "adapter_options":
            if not isinstance(value, dict):
                raise ValueError("adapter_options must be a JSON object")
        elif name in {"dataset", "project_dir"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty path")
            value = str((ROOT / value).resolve())
        elif not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        setattr(args, name, value)
        configured.add(name)
    if args.split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test or all")
    if config_path is None:
        args.config = str(path)
        args._configured_fields = configured
    return args


def source_snapshot(project_dir: Path, destination: Path, *, files: tuple[str, ...] = PROJECT_SOURCE_FILES) -> dict:
    # Check the whole set before writing a partial, unusable service snapshot.
    for relative in files:
        source = project_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing project source: {source}")
    hashes = {}
    for relative in files:
        source = project_dir / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = hash_file(target)
        if hash_file(source) != hashes[relative]:
            raise RuntimeError("Project source changed during snapshot; retry after freezing its files.")
    commit = subprocess.run(["git", "-C", str(project_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {"project_dir": str(project_dir), "git_commit": commit.stdout.strip() if commit.returncode == 0 else None, "files_sha256": hashes, "fingerprint": fingerprint(hashes), "note": "The file hashes and saved snapshot identify the actual working tree, including uncommitted edits."}


def validate_prediction(prediction: dict, records: list[dict], top_k: int) -> None:
    if prediction.get("runtime_status", "ok") != "ok":
        return
    expected = {(r["record_id"], m["message_id"]): m["content"].strip() for r in records for m in r["messages"]}
    known_records = {r["record_id"] for r in records}
    ranked = prediction.get("ranked_record_ids")
    if not isinstance(ranked, list) or len(ranked) > top_k or len(set(ranked)) != len(ranked) or set(ranked) - known_records:
        raise ValueError("Prediction has invalid, duplicate or out-of-scope ranked records.")
    returned = prediction.get("returned_messages")
    if not isinstance(returned, list):
        raise ValueError("Prediction must contain returned_messages.")
    seen = set()
    for message in returned:
        identity = (message.get("record_id"), message.get("message_id"))
        if identity in seen or identity not in expected or identity[0] not in ranked:
            raise ValueError("Prediction has duplicate, unknown or unranked returned evidence.")
        if message.get("content", "").strip() != expected[identity]:
            raise ValueError("Returned message ID does not match the actual original message content.")
        seen.add(identity)


def execute(args: argparse.Namespace) -> Path:
    from benchlib.dataset import load_dataset
    from benchlib.metrics import aggregate, score_query
    from benchlib.reporting import report, write_json, write_jsonl
    from benchlib.contracts import normalize_records
    from benchlib.versions import create_retriever, discover_versions

    version = discover_versions()[args.version_id]

    dataset_dir = Path(args.dataset).resolve()
    dataset_manifest, records, queries = load_dataset(dataset_dir)
    selected = [q for q in queries if args.split == "all" or q["split"] == args.split]
    if args.scenario:
        selected = [q for q in selected if q["scenario_id"] == args.scenario]
    if args.query_limit:
        selected = selected[:args.query_limit]
    if not selected:
        raise ValueError("No queries selected. Check split/scenario/query-limit.")
    names = [version.result_key]
    if args.top_k < 1 or args.top_k > 20 or args.window_radius < 0 or args.message_top_k < 1:
        raise ValueError("Require 1<=top-k<=20, window-radius>=0, message-top-k>=1")
    if args.query_limit < 0:
        raise ValueError("query-limit must be >=0")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = ROOT / "runs" / version.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    args._run_dir = run_dir
    dataset_hashes = {name: hash_file(dataset_dir / name) for name in ("manifest.json", "corpus.jsonl", "queries.jsonl")}
    for name, expected_hash in dataset_hashes.items():
        snapshot_path = run_dir / "dataset_snapshot" / name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_dir / name, snapshot_path)
        if hash_file(snapshot_path) != expected_hash:
            raise RuntimeError("Dataset changed while taking the run snapshot; retry with fixed input files.")
    # The snapshot, rather than previously read mutable files, is the actual input.
    dataset_manifest, records, queries = load_dataset(run_dir / "dataset_snapshot")
    selected = [q for q in queries if args.split == "all" or q["split"] == args.split]
    if args.scenario:
        selected = [q for q in selected if q["scenario_id"] == args.scenario]
    if args.query_limit:
        selected = selected[:args.query_limit]
    if not selected:
        raise ValueError("The frozen dataset contains no queries matching this selection.")
    if version.source_files:
        source = source_snapshot(Path(args.project_dir).resolve(), run_dir / "source_snapshot", files=version.source_files)
    else:
        source = {"project_dir": None, "git_commit": None, "files_sha256": {}, "fingerprint": fingerprint({}), "note": "This version does not execute project source."}
    manifest = {"schema_version": 1, "run_id": run_id, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "dataset": {"name": dataset_manifest.get("name", dataset_manifest.get("dataset_id", dataset_dir.name)), "directory": str(dataset_dir), "files_sha256": dataset_hashes, "source_manifest": dataset_manifest}, "dataset_fingerprint": fingerprint(dataset_hashes), "query_set_fingerprint": fingerprint(selected), "selection": {"split": args.split, "query_count": len(selected), "query_ids": [q["query_id"] for q in selected]}, "evaluation": {"top_k": args.top_k, "window_radius": args.window_radius, "message_top_k": args.message_top_k, "warmup_queries_per_scenario": 1, "latency_percentile_method": "nearest-rank", "evidence_metric": "coverage of published annotated evidence groups, not official QA accuracy"}, "source": source, "benchmark_source": {str(p.relative_to(ROOT)): hash_file(p) for p in sorted([ROOT / "run_benchmark.py", *list((ROOT / "benchlib").glob("*.py"))])}, "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "cpu": platform.processor(), "logical_cpu_count": os.cpu_count()}, "retrievers": {}}
    source_paths = [ROOT / "run_benchmark.py", *sorted((ROOT / "benchlib").glob("*.py"))]
    source_paths += sorted(p for p in version.directory.rglob("*") if p.is_file()
                           and not any(part in {"__pycache__", ".venv", "runs", ".git"} for part in p.relative_to(version.directory).parts)
                           and p.suffix in {".py", ".json", ".md", ".toml", ".yaml", ".yml", ".txt"})
    manifest["benchmark_source"] = {p.relative_to(ROOT).as_posix(): hash_file(p) for p in source_paths}
    version_hashes = {p.relative_to(version.directory).as_posix(): hash_file(p) for p in source_paths if p.is_relative_to(version.directory)}
    manifest["experiment_version"] = {"id": version.id, "label": version.label, "protocol": "memory-retrieval-v1", "files_sha256": version_hashes, "fingerprint": fingerprint(version_hashes), "adapter_options": getattr(args, "adapter_options", {})}
    write_json(run_dir / "run_manifest.json", manifest)
    manifest["metric_fingerprint"] = hash_file(ROOT / "benchlib/metrics.py")
    for relative, expected_hash in manifest["benchmark_source"].items():
        snapshot_path = run_dir / "benchmark_source_snapshot" / relative
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, snapshot_path)
        if hash_file(snapshot_path) != expected_hash:
            raise RuntimeError("Benchmark code changed during snapshot; retry after freezing its files.")
    if getattr(args, "config", None):
        config_path = Path(args.config)
        manifest["configuration_file"] = {"path": str(config_path), "sha256": hash_file(config_path)}
        shutil.copyfile(config_path, run_dir / "configuration.json")
    scenario_records = defaultdict(list)
    scenario_queries = defaultdict(list)
    for record in records:
        scenario_records[record["scenario_id"]].append(record)
    for query in selected:
        scenario_queries[query["scenario_id"]].append(query)
    all_predictions, all_scores = [], []
    try:
        for name in names:
            manifest["retrievers"][name] = {"scenarios": {}, "initialization_ms": 0}
            for scenario_id, scenario_selected in scenario_queries.items():
                print(f"[{name}] preparing {scenario_id}: {len(scenario_records[scenario_id])} records, {len(scenario_selected)} queries", flush=True)
                start = time.perf_counter()
                retriever = None
                try:
                    safe_id = hashlib.sha256(scenario_id.encode()).hexdigest()[:16]
                    retriever = create_retriever(
                        version, normalize_records(scenario_records[scenario_id]),
                        project_dir=Path(args.project_dir).resolve(), runtime_dir=run_dir / "runtime" / safe_id,
                        window_radius=args.window_radius, message_top_k=args.message_top_k,
                        options=copy.deepcopy(getattr(args, "adapter_options", {})),
                    )
                    setup_ms = (time.perf_counter() - start) * 1000
                    meta = {"initialization_ms": setup_ms, "build_stats": getattr(retriever, "build_stats", {})}
                    manifest["retrievers"][name]["initialization_ms"] += setup_ms
                    manifest["retrievers"][name]["scenarios"][scenario_id] = meta
                    warmup = retriever.search(scenario_selected[0]["query"], top_k=args.top_k)
                    if warmup.get("runtime_status", "ok") != "ok":
                        raise RuntimeError(f"Warmup failed: {warmup.get('error', warmup)}")
                    for index, query in enumerate(scenario_selected, 1):
                        began = time.perf_counter()
                        try:
                            prediction = retriever.search(query["query"], top_k=args.top_k)
                            validate_prediction(prediction, scenario_records[scenario_id], args.top_k)
                        except Exception as exc:
                            prediction = {"runtime_status": "error", "error": f"{type(exc).__name__}: {exc}", "ranked_record_ids": [], "scores": [], "returned_messages": [], "latency_ms": (time.perf_counter() - began) * 1000}
                        prediction.update({"retriever": name, "query_id": query["query_id"], "scenario_id": scenario_id, "query": query["query"]})
                        all_predictions.append(prediction)
                        score = score_query(query, prediction, k=args.top_k)
                        score.update({"retriever": name, "query_id": query["query_id"], "scenario_id": scenario_id, "category": query["category"]})
                        all_scores.append(score)
                        print(f"[{name}] {index}/{len(scenario_selected)} {query['query_id']} evidence={score.get('evidence_coverage')} ms={prediction.get('latency_ms')}", flush=True)
                except Exception as exc:
                    # Initialization errors are explicit failed queries, never omitted successes.
                    manifest["retrievers"][name]["scenarios"].setdefault(scenario_id, {})["setup_error"] = f"{type(exc).__name__}: {exc}"
                    already = {(row["retriever"], row["query_id"]) for row in all_predictions}
                    for query in scenario_selected:
                        if (name, query["query_id"]) in already:
                            continue
                        prediction = {"retriever": name, "query_id": query["query_id"], "scenario_id": scenario_id, "query": query["query"], "runtime_status": "error", "error": f"setup: {type(exc).__name__}: {exc}", "ranked_record_ids": [], "scores": [], "returned_messages": [], "latency_ms": None}
                        all_predictions.append(prediction)
                        score = score_query(query, prediction, k=args.top_k)
                        score.update({"retriever": name, "query_id": query["query_id"], "scenario_id": scenario_id, "category": query["category"]})
                        all_scores.append(score)
                    print(f"[{name}] SETUP ERROR: {exc}", file=sys.stderr, flush=True)
                finally:
                    if retriever is not None and hasattr(retriever, "close"):
                        retriever.close()
                write_jsonl(run_dir / "predictions.jsonl", all_predictions)
                write_jsonl(run_dir / "scores.jsonl", all_scores)
                write_json(run_dir / "run_manifest.json", manifest)
        summaries = {name: aggregate([row for row in all_scores if row["retriever"] == name]) for name in names}
        manifest["status"] = "completed_with_errors" if any(row.get("error") for row in all_scores) else "completed"
        source_unchanged = all((Path(args.project_dir).resolve() / relative).is_file() and hash_file(Path(args.project_dir).resolve() / relative) == expected for relative, expected in source["files_sha256"].items())
        benchmark_unchanged = all((ROOT / relative).is_file() and hash_file(ROOT / relative) == expected for relative, expected in manifest["benchmark_source"].items())
        manifest["source_unchanged_during_run"] = source_unchanged
        manifest["benchmark_unchanged_during_run"] = benchmark_unchanged
        if not source_unchanged or not benchmark_unchanged:
            manifest["status"] = "invalid_source_changed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "summary.json", summaries)
        write_json(run_dir / "run_manifest.json", manifest)
        report(run_dir, manifest, summaries, all_scores, all_predictions)
        if manifest["status"] == "completed":
            (run_dir.parent / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
            (ROOT / "runs" / "latest.txt").write_text(run_dir.relative_to(ROOT / "runs").as_posix() + "\n", encoding="utf-8")
        print(f"Report: {run_dir / 'report.md'}", flush=True)
        if manifest["status"] != "completed":
            raise RuntimeError(f"Run status is {manifest['status']}; inspect {run_dir / 'report.md'}")
        return run_dir
    except BaseException:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted_or_failed"
            write_json(run_dir / "run_manifest.json", manifest)
            write_jsonl(run_dir / "predictions.jsonl", all_predictions)
            write_jsonl(run_dir / "scores.jsonl", all_scores)
        raise


def run_selected(args: argparse.Namespace, argv: list[str], *, pinned_version: str | None = None) -> list[Path]:
    from benchlib.reporting import write_json
    from benchlib.versions import select_versions

    selection = getattr(args, "versions", None) or getattr(args, "retrievers", None)
    if pinned_version:
        selected = select_versions(pinned_version)
        if selection or getattr(args, "all_versions", False):
            raise ValueError("A version's run.py runs only its own version; use the root dispatcher to select others")
    else:
        selected = select_versions(selection, all_versions=args.all_versions)
    suite_dir = None
    suite = {"schema_version": 1, "status": "running", "versions": {}}
    if len(selected) > 1:
        suite_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        suite_dir = ROOT / "runs" / "suites" / suite_id
        write_json(suite_dir / "suite_manifest.json", suite)
    results, failures = [], []
    for version in selected:
        current = copy.deepcopy(args)
        current.version_id = version.id
        current = apply_run_config(current, argv, config_path=version.config,
                                   skip_fields=set(getattr(args, "_configured_fields", set())) | {"versions", "retrievers"})
        print(f"[version={version.id}] {version.label}", flush=True)
        try:
            result = execute(current)
            results.append(result)
            suite["versions"][version.id] = {"status": "completed", "run_directory": result.relative_to(ROOT).as_posix()}
        except Exception as exc:
            failures.append(version.id)
            entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if getattr(current, "_run_dir", None):
                entry["run_directory"] = current._run_dir.relative_to(ROOT).as_posix()
            suite["versions"][version.id] = entry
            print(f"[version={version.id}] FAILED: {exc}", file=sys.stderr, flush=True)
        finally:
            if suite_dir:
                write_json(suite_dir / "suite_manifest.json", suite)
    if suite_dir:
        suite["status"] = "completed_with_errors" if failures else "completed"
        write_json(suite_dir / "suite_manifest.json", suite)
        lines = ["# 多版本运行索引", "", "各版本独立建库、评分并保存报告；本页只组织运行结果。", "", "| 版本 | 状态 | 报告 |", "| --- | --- | --- |"]
        for name, item in suite["versions"].items():
            target = ROOT / item["run_directory"] / "report.md" if item.get("run_directory") else None
            link = f"[report.md]({os.path.relpath(target, suite_dir).replace(chr(92), '/')})" if target and target.is_file() else "运行失败，见 suite_manifest.json"
            lines.append(f"| {name} | {item['status']} | {link} |")
        (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (ROOT / "runs" / "latest-suite.txt").write_text(suite_dir.relative_to(ROOT / "runs").as_posix() + "\n", encoding="utf-8")
        print(f"Suite: {suite_dir / 'report.md'}", flush=True)
    if failures:
        raise RuntimeError(f"Failed versions: {', '.join(failures)}; inspect their independent run reports")
    return results


def main(argv: list[str] | None = None, *, pinned_version: str | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if pinned_version and (not argv or argv[0].startswith("-")):
        argv.insert(0, "run")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Discover enabled version packages without importing models.")
    prepare = commands.add_parser("prepare", help="Download pinned public data and normalize it.")
    prepare.add_argument("--profile", choices=["demo", "full"], default="demo")
    prepare.add_argument("--max-queries", type=int, default=24)
    validate = commands.add_parser("validate", help="Check normalized data and all evidence references.")
    validate.add_argument("--dataset", default=str(ROOT / "data/processed/locomo-demo-v1"))
    for name in ("run", "demo"):
        run = commands.add_parser(name, help="Run baselines; demo prepares public data first.")
        run.add_argument("--dataset", default=str(ROOT / "data/processed/locomo-demo-v1"))
        run.add_argument("--config", help="JSON run config; explicit command-line flags take precedence.")
        selection = run.add_mutually_exclusive_group()
        selection.add_argument("--versions", "--version", help="Comma-separated version ids or aliases.")
        selection.add_argument("--all-versions", action="store_true", help="Run every enabled version sequentially.")
        selection.add_argument("--retrievers", help="Compatibility alias for --versions (bm25,project still work).")
        run.add_argument("--split", choices=["dev", "test", "all"], default="dev")
        run.add_argument("--scenario")
        run.add_argument("--query-limit", type=int, default=0)
        run.add_argument("--top-k", type=int, default=5)
        run.add_argument("--window-radius", type=int, default=10)
        run.add_argument("--message-top-k", type=int, default=3)
        run.add_argument("--project-dir", default=str(ROOT.parent))
        run.set_defaults(adapter_options={})
    compare = commands.add_parser("compare", help="Compare two runs after checking input compatibility.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", default=str(ROOT / "runs/comparison.md"))
    compare.add_argument("--baseline-retriever", help="Select a scheme in a historical multi-scheme run.")
    compare.add_argument("--candidate-retriever", help="Select the scheme to compare against it.")
    args = parser.parse_args(argv)
    if args.command == "list":
        from benchlib.versions import discover_versions
        for version in discover_versions().values():
            print(f"{version.id}\t{version.label}\taliases={','.join(version.aliases) or '-'}")
    elif args.command == "prepare":
        from benchlib.dataset import prepare_locomo
        print(prepare_locomo(ROOT, profile=args.profile, max_queries=args.max_queries))
    elif args.command == "validate":
        from benchlib.dataset import load_dataset, validate_dataset
        _, records, queries = load_dataset(Path(args.dataset))
        print(json.dumps(validate_dataset(records, queries), ensure_ascii=False, indent=2))
    elif args.command == "compare":
        from benchlib.reporting import compare_runs
        compare_runs(Path(args.baseline).resolve(), Path(args.candidate).resolve(), Path(args.output).resolve(),
                     baseline_retriever=args.baseline_retriever, candidate_retriever=args.candidate_retriever)
        print(Path(args.output).resolve())
    else:
        args = apply_run_config(args, argv)
        if args.command == "demo" and Path(args.dataset).resolve() == ROOT / "data/processed/locomo-demo-v1":
            from benchlib.dataset import prepare_locomo
            args.dataset = str(prepare_locomo(ROOT, profile="demo", max_queries=24))
        run_selected(args, argv, pinned_version=pinned_version)


if __name__ == "__main__":
    main()
