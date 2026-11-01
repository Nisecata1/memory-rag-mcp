# BM25 v1

用途：B0 词面检索对照。保留首轮 baseline 的检索逻辑：标题与完整消息分词、Okapi BM25、主记录 max 聚合、按名次和间距选消息锚点、前后窗口展开。

```powershell
# 在 benchmark 目录；已有数据时直接运行。
python .\versions\bm25_v1\run.py
# 首次准备固定版本的 demo 数据：
python .\versions\bm25_v1\run.py demo
```

只需 Python 标准库，不导入项目、不加载模型、不要求项目数据库模块存在。`source_files` 留空；本目录实现、配置和公共评测源码仍保留快照。

原始质量结果由 `baselines/locomo-demo-v1.json` 固定引用。本次整理没有改动分词、打分、锚点或评分规则。后续改这些行为时应新增版本目录，配置实验参数与代码指纹以各次 run 为准。
