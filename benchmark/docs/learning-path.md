# 动手理解这套 benchmark

## 第一步：手算一题

打开 `data/processed/locomo-demo-v1/queries.jsonl`，取第一题。记录 query、relevant_record_ids 和 evidence_groups。到 corpus.jsonl 找到对应的原始消息，确认“标签为什么这样写”。

再看一次运行的 predictions.jsonl，手算：

- 正确会话是否在前5条？第一条正确会话排第几？
- 发布证据消息是否真的出现在 returned_messages 中？
- 返回了多少条不一定需要的消息？

对照 scores.jsonl。先理解一题，比先读完所有代码更容易建立评价标准。

## 第二步：读评分函数

看 `benchlib/metrics.py` 的 `score_query` 和 `aggregate`，以及 tests/test_metrics.py。

重点观察三个边界：同名消息在不同record不能算同一条；失败请求不能被剔除；没有相关证据的题不能除以零。然后运行：

```powershell
python -m unittest discover -s tests -v
```

## 第三步：读基线设计稿，再跑一遍

先读 docs/baseline-design.md，确认固定条件，然后按README的四步命令prepare、validate、run、查看report。

用自己的话解释：BM25为什么是一个有意义的简单对照？当前向量流程相比它改变了哪些环节？为什么不能把所有分差都归因于BGE？

## 第四步：追踪一条真实检索链路

按这个顺序读实现：

`run_benchmark.py execute → ProjectRetriever → project_worker._initialize/_search → 原项目 search_records → get_detail_records_by_ids`

观察每次run的独立runtime目录。源码没有复制QA答案进索引，也没有复用正式私人记忆。语料、标准答案和系统输出是独立产物。

## 第五步：解释一次失败

从报告选一题失败，定位到：

1. 主记录候选里是否已经有证据所属会话？
2. 锚点是否落在需要的消息附近？
3. 展开窗口是否覆盖目标消息？
4. 如果取回了等价信息但没有标注ID，是否属于标注局限？

写下一个可以检验的改进假设，再设计对照。先保留原始基线，不为让某个方案看起来更好而修改题目或评分。

## 第六步：形成持续验证习惯

影响检索的代码改动后，先跑回归测试和固定开发集，保留新run及compare报告。方案稳定后再跑完整test。数据、标注或协议变化时升级版本，重新建立基线。

这份demo还没有验证中文项目档案、知识更新后的正确性、无答案过滤、并发和大规模性能。它们应逐项成为有明确数据、条件和指标的后续实验。
