# 实验版本隔离与新增方法

本次整理将“被评测的实现”与“评分规则”分开管理。版本目录可独立启动，公共代码统一校验与打分，外层入口按需调度。增加版本通常只新增目录，不修改外层 runner。

## 三种版本各管什么

| 版本 | 记录位置 | 什么时候变化 |
| --- | --- | --- |
| 实验实现 | `versions/<id>/`，每次保存全部实现文件指纹 | 换检索方法、数据库、切片或接入协议 |
| 数据集与评测协议 | 数据 manifest、实际题集指纹、`metrics.py` 指纹 | 增删题目、改标注、改指标定义 |
| 一次运行 | `runs/<id>/<run_id>/` | 每执行一次，包括同一版本反复运行 |

同一 `project_v1` 连接不同项目代码，仍会得到不同的项目源码指纹。目录名隔离实验代码，不能代替被测源码的版本管理。保留历史成绩时，不修改已有 run 内的快照或 manifest。

## 目录的职责

每个版本包含：

- `version.json`：ID、别名、协议、适配器入口、需追踪的项目源码。
- `config.json`：这个版本的默认评测参数和 `adapter_options`。
- `run.py`：独立入口，支持从任意工作目录调用。
- `adapter.py`：接收语料，返回检索器；可以直接实现检索，也可连接这个目录自己的 worker。
- `worker.py`：按需提供，负责数据库、索引、业务接口等实现细节。
- `README.md`：实验假设、控制条件、依赖、与前版的差异。

公共的 `benchlib` 负责数据准备、ID/正文校验、评分、报告、指纹、运行隔离和 JSON-lines 传输。每个版本不要复制一套评分实现；改评测协议需要单独审查并重新评分两边。

## 新增一个版本

1. 复制 `versions/_template/` 到例如 `versions/raw_fts_v1/`。模板目录不会出现在自动发现或全部运行中。
2. 修改 `version.json` 的 `id` 与目录名一致，填写实验说明；别名不能与其他版本重复。
3. 实现 `adapter.py`，必要时添加自己的 `worker.py`。`config.json` 的 `adapter_options` 可放后端专用参数，因此公共 runner 不必增加专用分支。
4. 使用项目源码的版本，列出 `source_files`，包含实际影响行为的模块、配置与依赖文件。更换数据库后应更新这份清单；纯独立实现可以留空。
5. 实现完成后将 `enabled` 改为 `true`。独立运行、检查报告，再加入需要的套件配置。

```powershell
# 在 benchmark 目录执行；目标目录不存在时使用。
Copy-Item -LiteralPath .\versions\_template -Destination .\versions\raw_fts_v1 -Recurse

# 完成上面的配置与适配器后：
python .\run_benchmark.py list
python .\versions\raw_fts_v1\run.py
python .\run_benchmark.py run --versions bm25_v1,raw_fts_v1
```

`raw_fts_v1` 只是新增方法的示例名称，本次没有实现或验证截图中的系统或论文结论。其检索粒度、返回预算和评测任务需要另外设计。

## 适配器约定

公共调度调用：

```python
create_retriever(
    records,
    project_dir=project_dir,
    runtime_dir=fresh_runtime_dir,
    window_radius=10,
    message_top_k=3,
    options={},
)
```

`records` 只保留原始语料字段，不携带 QA 答案、证据标记或生成式总结。检索器提供：

```python
retriever.search(query_text, top_k=5)  # 输入不含标注
retriever.close()
retriever.build_stats               # 可选：建库/模型/实现信息
```

`search` 至少返回：

```python
{
    "ranked_record_ids": ["record-1"],
    "returned_messages": [
        {"record_id": "record-1", "message_id": "message-1", "content": "原始消息正文"}
    ],
    "latency_ms": 12.3,
    "runtime_status": "ok",
}
```

还可返回 `scores`、`selected_message_ids` 等诊断信息。评分器只承认实际交付的原始正文，不承认适配器自己计算的正确率。

worker 可使用共享 `JsonlWorkerClient`，但初始化、数据写入、搜索接口与数据库隔离都由当前版本自己的 worker 实现。SQLite 的 audit hook 留在 `project_v1/worker.py`，其他数据库版本应有自己的隔离方案。

## 配置优先级与依赖

显式 CLI 参数 > 显式 `--config` 套件配置 > 当前版本 `config.json` > 公共默认值。JSON 中 `dataset` 与 `project_dir` 都相对 benchmark 根目录解析，和命令执行位置无关。

各版本可维护自己的依赖说明或 `requirements.txt`，并用各自 Python 环境启动 `run.py`。根入口同一批次使用同一 Python；依赖冲突时分别运行后比较报告。

## 复现旧实现与比较

项目适配器默认连接仓库当前工作树。要复现历史实现，准备该提交的独立 checkout、相同依赖与模型，然后传入其路径：

```powershell
python .\versions\project_v1\run.py --project-dir C:\path\to\historical-checkout
```

每次运行同时保存版本目录、共享 benchmark 源码和被测项目源文件快照，并在结束时检查这些文件是否变化。文件名相同、目录名相同都不意味着被测代码相同，以 manifest 与指纹为准。

新版本独立报告可以直接比较。双方只有一个方案时，即使方案名称不同也会自动配对；历史多方案报告有歧义时明确指定：

```powershell
python .\run_benchmark.py compare `
  --baseline .\runs\20261029T070259_918552Z `
  --candidate .\runs\raw_fts_v1\RUN_ID `
  --baseline-retriever project `
  --candidate-retriever raw_fts_v1 `
  --output .\runs\comparison.md
```

比较仍要求同一数据、同一实际题集、同一评分实现与相同 k，不能通过改版本名绕过。窗口、模型、设备和依赖的差异另结合 manifest 解释。dev 用于调整，test 用于稳定版本验证。
