# 新实验模板（默认禁用）

将本目录复制为一个新的 snake_case 版本目录，修改 `version.json` 的 ID、说明、别名和源码清单，完成 `adapter.py` 后再设置 `enabled: true`。

入口 `run.py` 可直接复用。实现可以是 BM25/FTS 等进程内检索，也可以连接本目录新增的 `worker.py`。公共 JSON-lines 客户端位于 `benchlib/worker_client.py`；公共数据校验、评分与报告继续复用。

模板本身不会被 `list` 或 `--all-versions` 运行。详细步骤与输出契约见 [版本隔离说明](../../docs/versioning.md)。
