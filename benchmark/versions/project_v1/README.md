# Project v1：BGE-M3 / SQLite 接入方案

用途：B1 当前项目检索方案。`adapter.py` 连接本目录的 `worker.py`；worker 配置隔离 SQLite/FAISS，调用真实 `search_records` 和携带锚点的 `get_detail_records_by_ids`。

```powershell
# 在 benchmark 目录；使用安装好项目依赖的 Python。
python .\versions\project_v1\run.py
python .\versions\project_v1\run.py --query-limit 2
python .\versions\project_v1\run.py --project-dir C:\path\to\another-checkout
```

`v1` 标识这套实验接入与检索方案，不表示项目源码永久停在某个提交。默认评当前工作树，每个 run 记录实际 Git 提交、未提交文件的哈希和源码快照。历史实验需连接对应 checkout 并匹配依赖和模型。

数据库、索引和 worker.log 位于本次 `runs/project_v1/<run_id>/runtime/<scenario>/`。正式记忆库和已有私有 data-test 副本不参与运行。

未来换数据库或写入协议时，新增例如 `project_v2/`，维护自己的 adapter/worker 和 `source_files`，继续输出相同的记录排名和实际消息正文。无需把新后端逻辑加到公共评分器或根入口。
