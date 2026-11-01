"""Write reports from measured results; no model-generated interpretation."""
from __future__ import annotations

import html
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def number(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report(run_dir: Path, manifest: dict, summaries: dict, scored: list[dict], predictions: list[dict]) -> None:
    k = manifest["evaluation"]["top_k"]
    columns = ["方案", "题数", f"Hit@{k}", f"Recall@{k}", f"MRR@{k}", "标注证据覆盖", "全部标注证据覆盖", "返回字符均值", "查询 p95 ms", "错误数"]
    table = []
    for name, item in summaries.items():
        table.append([name, str(item["query_count"]), number(item["hit_at_k"]), number(item["recall_at_k"]), number(item["mrr_at_k"]), number(item["evidence_coverage"]), number(item["evidence_complete"]), number(item["returned_characters_mean"], 0), number(item["latency_p95_ms"], 1), str(item["error_count"])])
    lines = ["# 记忆检索 benchmark 实测报告", "", f"运行：`{manifest['run_id']}`；状态：`{manifest['status']}`。", "", f"数据集：`{manifest['dataset']['name']}`；评测范围：`{manifest['selection']['split']}`；实际题数：{manifest['selection']['query_count']}。", "", "> 本报告是 LoCoMo 文本证据检索派生任务，不是官方 LoCoMo 问答成绩。Demo 仅用于开发和诊断。", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    if manifest.get("experiment_version"):
        version = manifest["experiment_version"]
        lines[2:2] = ["", f"实验版本：`{version['id']}`；实现指纹：`{version['fingerprint']}`。"]
    if manifest["status"] == "invalid_source_changed":
        lines[1:1] = ["", "> 本次运行期间源码或benchmark逻辑发生变化，成绩来源不稳定；保留结果用于排查，不能作为有效版本对照。", ""]
    lines.extend("| " + " | ".join(row) + " |" for row in table)
    lines += ["", "## 怎样解释这些数字", "", "- 主记录指标按 session 记忆计算；证据按 (record_id, message_id) 联合定位。", "- ‘全部标注证据覆盖’衡量是否取回发布数据给出的全部证据，不保证这些证据是最小必要集合，也不代表最终回答正确率。", "- 返回字符数统计实际返回消息正文；没有将字符数当作 token 数。", "- 查询计时包含当前适配器的检索和详情展开；初始化、模型加载和建索引另列。每方案每场景预热一次，不计入成绩。", "- p50/p95 为本次查询样本的 nearest-rank 分位数，单次小样本不构成生产性能承诺。BM25 和项目适配器的运行开销边界及锚点策略不同，见 baseline 设计稿。", "- 服务异常记为错误；有答案题的质量分记 0，不从总体均值中删掉。", "- 数据集、实际查询集合、源码和配置指纹见 run_manifest.json；所有逐题结果均保留。", ""]
    for name, meta in manifest["retrievers"].items():
        lines += [f"## {name} 初始化与实现信息", "", "```json", json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    lines += ["## 按题型拆分", ""]
    for name, summary in summaries.items():
        lines += [f"### {name}", "", "| 题型 | 题数 | 主记录命中 | 标注证据覆盖 | 错误数 |", "| --- | --- | --- | --- | --- |"]
        for category, item in summary.get("by_category", {}).items():
            lines.append(f"| {category} | {item['query_count']} | {number(item['hit_at_k'])} | {number(item['evidence_coverage'])} | {item['error_count']} |")
        lines.append("")
    lines += ["## 优先查看的失败案例", ""]
    failures = [row for row in scored if row.get("error") or (row.get("evidence_complete") is not None and row["evidence_complete"] < 1)]
    lookup = {(row["retriever"], row["query_id"]): row for row in predictions}
    if not failures:
        lines += ["本次没有发现按当前发布标注计的失败；仍需扩大题目覆盖，不能据此推断系统普遍可靠。", ""]
    for row in failures[:10]:
        prediction = lookup.get((row["retriever"], row["query_id"]), {})
        if row.get("error"):
            reason = "运行错误"
        elif not row.get("hit_at_k"):
            reason = "未找到任何已标注相关主记录"
        elif row.get("recall_at_k", 0) < 1:
            reason = "部分已标注主记录未找回，可能同时存在上下文损失"
        else:
            reason = "相关主记录已找到，但返回正文未覆盖全部已标注证据"
        lines += [f"- **{row['retriever']} / {row['query_id']}**：{reason}；证据覆盖={number(row.get('evidence_coverage'))}。", f"  查询：{prediction.get('query', '')}"]
    lines += ["", "## 可复现文件", "", "- run_manifest.json：协议、输入/源码哈希与运行环境。", "- predictions.jsonl：每题实际排名、锚点、返回正文与耗时。", "- scores.jsonl：每题按照固定标注计算的分数。", "- summary.json：机器可读汇总；report.html：浏览器查看。", "- runtime/：本次独立的项目数据库和索引，不与正式记忆库混用。", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in table)
    details = "".join(f"<li><strong>{html.escape(r['retriever'])} · {html.escape(r['query_id'])}</strong> — evidence coverage {number(r.get('evidence_coverage'))}</li>" for r in failures[:10]) or "<li>本次没有按当前标注判定的失败。</li>"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>记忆检索评测</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#172b38;background:#f4f3ef;margin:0;padding:40px 5vw}}main{{max-width:1200px;margin:auto}}h1{{font-size:32px;line-height:1.2}}.note{{border-left:4px solid #327d70;padding:12px 20px;background:white}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}}th{{color:#306758}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}pre{{background:white;padding:20px;white-space:pre-wrap}}a{{color:#216b5e}}</style><main><p>MEMORY RETRIEVAL · DEVELOPMENT REPORT</p><h1>先看证据是否找回，再看代价</h1><p><code>{html.escape(manifest['run_id'])}</code></p><p class="note">LoCoMo 文本检索派生任务 · {manifest['selection']['query_count']} 道 {html.escape(manifest['selection']['split'])} 题。不是官方问答成绩，也不是最终回答准确率。</p><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div><h2>需要检查的案例</h2><ul>{details}</ul><h2>如何复核</h2><p>查看 <a href="report.md">完整说明</a>、<a href="run_manifest.json">运行协议</a>、<a href="predictions.jsonl">逐题检索输出</a> 与 <a href="scores.jsonl">逐题分数</a>。耗时为本次查询样本的描述统计，初始化和建索引另列。</p><p>“全部标注证据覆盖”只反映发布证据的交付情况。小样本成绩不证明普遍能力；不同锚点策略、返回文本量和运行边界需一起解释。</p></main></html>"""
    (run_dir / "report.html").write_text(page, encoding="utf-8")


def compare_runs(baseline_dir: Path, candidate_dir: Path, output: Path, *,
                 baseline_retriever: str | None = None, candidate_retriever: str | None = None) -> dict:
    baseline = json.loads((baseline_dir / "run_manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if any(item.get("status") not in {"completed", "completed_with_errors"} for item in (baseline, candidate)):
        raise ValueError("Only completed or completed_with_errors runs can be compared; running, interrupted and invalid runs are excluded.")
    for key in ("dataset_fingerprint", "query_set_fingerprint"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare: {key} differs. Use the same dataset and exact queries.")
    if not baseline.get("metric_fingerprint") or baseline.get("metric_fingerprint") != candidate.get("metric_fingerprint"):
        raise ValueError("Cannot compare: scoring implementation fingerprint is missing or differs. Re-score both runs with one metric version.")
    if baseline["evaluation"]["top_k"] != candidate["evaluation"]["top_k"]:
        raise ValueError("Cannot compare retrieval metrics at different top_k values.")
    old = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    new = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    if baseline_retriever is not None or candidate_retriever is not None:
        a = baseline_retriever or (next(iter(old)) if len(old) == 1 else None)
        b = candidate_retriever or (next(iter(new)) if len(new) == 1 else None)
        if a not in old or b not in new:
            raise ValueError("Select existing --baseline-retriever and --candidate-retriever names")
        pairs = [(a, b)]
    elif set(old) & set(new):
        pairs = [(name, name) for name in sorted(set(old) & set(new))]
    elif len(old) == len(new) == 1:
        pairs = [(next(iter(old)), next(iter(new)))]
    else:
        raise ValueError("Ambiguous schemes; select --baseline-retriever and --candidate-retriever")
    for before, after in pairs:
        if old[before].get("query_count") != baseline["selection"]["query_count"] or new[after].get("query_count") != candidate["selection"]["query_count"]:
            raise ValueError("Cannot compare: scored query count does not match the selected query set.")
    delta = {}
    keys = ["hit_at_k", "recall_at_k", "mrr_at_k", "evidence_coverage", "evidence_complete", "returned_characters_mean", "latency_p95_ms", "error_count", "error_rate"]
    lines = ["# 两次固定题集运行的比较", "", f"基线：`{baseline['run_id']}`（{baseline['status']}）；候选：`{candidate['run_id']}`（{candidate['status']}）。", "", "差值为候选减基线。质量指标通常越高越好；字符、延迟与错误率通常越低越好。该表是描述比较，不是统计显著性检验。", "", "运行配置、源码或环境的差异需要同时解释，不能单凭总分归因于某一功能。错误请求仍纳入质量分母，错误数单列。", "", "| 方案 | 指标 | 基线 | 候选 | 差值 |", "| --- | --- | --- | --- | --- |"]
    for before, after in pairs:
        name = before if before == after else f"{before} -> {after}"
        delta[name] = {}
        for key in keys:
            a, b = old[before].get(key), new[after].get(key)
            d = None if a is None or b is None else b - a
            delta[name][key] = d
            lines.append(f"| {name} | {key} | {number(a)} | {number(b)} | {number(d)} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output.with_suffix(".json"), {"baseline": baseline["run_id"], "candidate": candidate["run_id"], "scheme_pairs": pairs, "version_before": baseline.get("experiment_version"), "version_after": candidate.get("experiment_version"), "delta": delta, "evaluation_before": baseline["evaluation"], "evaluation_after": candidate["evaluation"]})
    return delta
