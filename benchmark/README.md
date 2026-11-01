# memory-rag-mcp 的 benchmark 工作区

这里实现一个可以反复运行的**记忆检索评估流程**：固定公开语料和标注，分别运行 BM25 与当前项目的真实 BGE-M3 检索，保存逐题证据、成绩和报告。

实现已按实验版本拆分：`versions/bm25_v1/`、`versions/project_v1/` 各有自己的配置、适配器与 `run.py`，项目版还有自己的 `worker.py`。公共数据与评分保持一致；根入口用于发现版本、选择运行、比较报告。

第一版测的是“能否找回记录和原始证据”，不是最终 AI 回答准确率，也不是官方 LoCoMo 问答成绩。

## 先读什么

1. [基线设计稿](docs/baseline-design.md)：任务、假设、对照、固定条件、指标与结论边界。这就是本项目的 baseline 设计参考。
2. [数据说明](docs/dataset.md)：官方来源、固定提交、许可证、如何从原始对话构造检索语料与证据。
3. [动手学习顺序](docs/learning-path.md)：先看一题，再读评分，最后运行和解释对照实验。
4. [第一轮实测逐题讲解](docs/first-run-walkthrough.md)：已经跑出的成绩、失败如何定位，以及哪些结论暂时不能下。
5. [版本隔离与新增方法](docs/versioning.md)：新增版本、worker 接入、配置优先级和历史复现。

首个固定双基线记录在 `baselines/locomo-demo-v1.json`，指向历史目录 `runs/20261029T070259_918552Z/`。该目录及原成绩保留。新运行按版本隔离，`runs/latest.txt` 记录最近成功运行相对 `runs/` 的路径，不重新定义固定 baseline。

## 目录约定

```text
benchmark/
  run_benchmark.py                 # 可选调度：list / prepare / validate / run / demo / compare
  run_demo.ps1                     # 当前 Windows 环境的一键演练入口
  configs/demo.json                # 指定多版本的运行套件
  versions/
    bm25_v1/                      # version.json / config.json / adapter.py / run.py / README
    project_v1/                   # 同上，另含自己的 worker.py
    _template/                    # 新版本模板；默认不参与发现或运行
  benchlib/
    dataset.py                    # 下载、版本固定、映射、验证
    runner.py                     # 公共运行、校验与快照流程
    versions.py                   # 发现版本并加载其适配器
    contracts.py                  # 只把原始语料字段传给适配器
    worker_client.py              # 与存储无关的 JSON-lines 客户端
    retrievers.py                 # 旧导入路径兼容层
    project_worker.py             # 旧 worker 入口兼容层
    metrics.py                    # 确定性评分
    reporting.py                  # JSON、Markdown、HTML 报告与运行比较
  tests/                          # 评分与流程边界验证
  docs/                           # 方案、数据与学习说明
  data/processed/locomo-demo-v1/   # demo: manifest / corpus / queries
  data/processed/locomo-full-v1/   # 扩展: 固定的 dev/test 划分
  data/raw/locomo/<commit>/       # 公开原始数据、许可证和来源 manifest
  runs/<version_id>/<run_id>/     # 各版本独立的运行记录
    source_snapshot/              # 当次项目源码与配置副本
    benchmark_source_snapshot/    # 当次评测逻辑副本
    dataset_snapshot/             # 当次实际使用的标准化输入
    runtime/                      # 本次项目适配器独立 DB 与索引
    run_manifest.json
    predictions.jsonl
    scores.jsonl
    summary.json
    report.md / report.html
  runs/suites/<suite_id>/         # 多版本运行索引，链接各自报告
  runs/20261029T.../              # 已有历史运行，原样保留
```

公开数据、标准化数据、每次实验的数据库是三种不同产物。**不把公开问题、参考答案或证据标记存进检索库。** 正式记忆服务和项目源码无需改动。

原始数据与标准化数据均由脚本准备，`data/raw/` 与 `data/processed/` 不进入 Git。旧私有测试库副本已移到仓库外的 `AI-memory/memory-rag-mcp-backups/`，不再放在 benchmark 中。

## 环境

数据准备、BM25、评分和报告只使用 Python 标准库，Python 3.10+ 即可。

真实项目适配器复用 `../requirements.txt` 中的依赖和当前本地模型。当前机器已经有可用解释器：

```powershell
$BenchPython = 'C:\Users\ndir\miniconda3\envs\project-beta\python.exe'
```

换机器时将它替换为安装好项目依赖的 Python 路径，并按项目 README 准备模型。无需购买 API 或调用在线 LLM 裁判。

## 第一次手动走通

在 `benchmark` 目录运行：

```powershell
# 1. 下载固定版本公开数据并标准化；已下载且哈希正确则复用。
& $BenchPython .\run_benchmark.py prepare --profile demo

# 2. 检查ID、证据引用、场景和划分。
& $BenchPython .\run_benchmark.py validate

# 3. 跑同一套开发题的两个基线。
& $BenchPython .\run_benchmark.py run --config .\configs\demo.json
```

每个版本也可以独立启动，支持从任意工作目录通过完整路径调用：

```powershell
& $BenchPython .\versions\bm25_v1\run.py
& $BenchPython .\versions\project_v1\run.py
```

根入口按需选择，不必每轮运行所有版本：

```powershell
python .\run_benchmark.py list
python .\run_benchmark.py run --versions bm25_v1
& $BenchPython .\run_benchmark.py run --versions bm25_v1,project_v1
& $BenchPython .\run_benchmark.py run --all-versions
```

`--all-versions` 依次运行所有启用版本，包括将来新增的版本；模板和禁用版本除外。未指定选择时，只运行 `version.json` 中 `default: true` 的版本。旧的 `--retrievers bm25,project` 仍作为别名兼容。

或者：

```powershell
.\run_demo.ps1
```

真实 BGE-M3 在 CPU 上需要加载模型并编码全部演练语料，首次通常比只跑 BM25 慢很多。初始化/建库和查询延迟在报告里分开记录；具体耗时以实测为准。

只检查轻量流程：

```powershell
python .\run_benchmark.py demo --retrievers bm25
python -m unittest discover -s tests -v
```

## 看哪些结果

终端打印各版本自己的 `report.md`。`runs/<version_id>/latest.txt` 保存该版本最近成功的 run_id，`runs/latest.txt` 保存最近成功运行的相对路径；多版本批次索引位于 `runs/suites/`，由 `runs/latest-suite.txt` 指向。失败运行仍保存排查记录，但不替换成功运行指针。

先看报告的证据覆盖、返回字符和失败案例，再到 `predictions.jsonl` 查看实际排名与正文。`summary.json` 方便后续脚本读取。

24 题 demo 是一段完整公开对话中的可映射、无图像证据单跳题，按固定原始顺序取前24题；它用于演练，不是随机代表性抽样。分数变化一题约占4.17个百分点，不能据此宣传微小提升或中文个人记忆效果。

## 项目改动后怎样复测

影响检索、索引或上下文展开的改动，先跑同一开发集：

```powershell
& $BenchPython .\run_benchmark.py run --config .\configs\demo.json

# 用实际 run_id 替换下面两个目录。
& $BenchPython .\run_benchmark.py compare `
  --baseline .\runs\project_v1\BASELINE_RUN_ID `
  --candidate .\runs\project_v1\CANDIDATE_RUN_ID `
  --output .\runs\comparison.md
```

比较器要求数据、实际查询集合、评分实现和 top-k 相同。它生成差值表，不自动推断改动原因或统计显著性。窗口/锚点/模型等条件变化必须结合两份 run_manifest 解释；基线原始成绩保留，不在每轮重新定义基线。每次还保留标准化数据与benchmark代码快照，后续重新prepare也不会覆盖历史运行的输入证据。

跨不同实验版本同样使用 `compare`。单方案报告可自动配对；历史多方案报告有歧义时，使用 `--baseline-retriever` 和 `--candidate-retriever` 指定。批次 `runs/suites/` 只是运行索引，比较时应传入具体版本的运行目录。

`project_v1` 默认读取项目当前工作树，并记录实际源码哈希；目录名不会冻结项目提交。复现历史版本需准备对应 checkout，并通过 `--project-dir` 指定。新增数据库或检索方法的步骤见 [版本隔离说明](docs/versioning.md)。

扩展到更多公开数据：

```powershell
& $BenchPython .\run_benchmark.py prepare --profile full
& $BenchPython .\run_benchmark.py run --dataset .\data\processed\locomo-full-v1 --split dev

# 方案稳定后才跑这个版本的 test；不要每天看它来调参数。
& $BenchPython .\run_benchmark.py run --dataset .\data\processed\locomo-full-v1 --split test
```

full 按完整对话场景分组，dev 与 test 对话不重叠。demo 属于 dev 的练习子集。以后还应增加中文项目记忆查询和无答案题；当前公开子任务没有无答案质量成绩。

## 公开来源

- [LoCoMo 官方数据与任务](https://github.com/snap-research/locomo)：第一版数据，固定提交与 SHA256 见数据说明。原数据使用 **CC BY-NC 4.0**，保留署名并遵守非商业使用条件。
- [LongMemEval 官方检索实验](https://github.com/xiaowu0162/LongMemEval#memory-retrieval)：参考会话/消息两级检索、BM25 与向量对照的组织方式；当前未把其数据混入本题集。
- [信息检索评估基础](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)与[BM25 原理](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)：指标和简单词面基线的理论参考。
