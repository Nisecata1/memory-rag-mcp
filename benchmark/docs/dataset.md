# 公共数据与标准化约定

第一版使用 [Snap Research 官方 LoCoMo](https://github.com/snap-research/locomo)，固定到提交 `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。官方原文件 `data/locomo10.json` 为 2,805,274 bytes，包含 10 段长对话、1,986 道 QA。数据与代码按仓库的 **CC BY-NC 4.0** 条款提供；保留作者、来源与许可证，适用于非商业学习与研究。本工作区不把它变成新的开放商用数据集。

- [固定版本原数据](https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json)
- [固定版本许可证](https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt)
- [官方数据结构说明](https://github.com/snap-research/locomo#data)

选择它作为第一步，是因为原文件较小，且 `qa.evidence` 能映射到原始对话的 `dia_id`，可以不调用回答模型，先测试真实文本检索。它是由模型生成、人工编辑的英语长对话；目前的结果不能代表中文个人记忆检索效果。

## 原始文件与已有测试数据的关系

下载目录是：

```text
data/raw/locomo/<固定提交>/
  locomo10.json
  LICENSE.txt
  source_manifest.json
```

准备脚本仅在 `data/raw/locomo` 保存公开文件，不读取私有记忆数据库。`source_manifest.json` 保存固定提交、原始下载 URL、实际 SHA256、文件大小、下载时间和许可证指纹。原文件还按固定版本的 Git blob SHA1 校验，已有文件不符时直接报错。

旧私有测试库副本已移到仓库外的 `AI-memory/memory-rag-mcp-backups/`。现有题集 manifest 中的 `raw_source_manifest` 记录其生成时的位置，运行评测不会读取这个路径。仅移动原始文件的位置时，重复准备会保留既有 manifest 的字节和指纹；新生成题集记录 `data/raw/locomo`。语料、问题、标注或其他协议内容变化时，manifest 仍会更新，比较器照常检查一致性。

## 两种 profile

`prepare_locomo(root, profile="demo", max_queries=24)` 的 `root` 是 `benchmark` 工作区路径。函数返回标准化数据目录。

| profile | 语料 | 查询 | 用途 |
| --- | --- | --- | --- |
| demo | 固定选取一段开发集对话，保留全部 session、消息和干扰内容 | 最多 24 道有完整可映射证据的单跳题；只来自 dev | 跑通下载、标准化、索引、检索和评分流程 |
| full | 全部 10 段对话，按 scenario 隔离检索 | 有效的类别 1/2/4，即多跳、时间与单跳题；不受 `max_queries` 限制 | 扩展的派生检索实验 |

对全部 `sample_id` 计算 SHA256，并按哈希排序。前 6 个 scenario 分到 dev，后 4 个分到 test。demo 在 dev 中按 `sample_id` 排序选择第一个 scenario，按原始 QA 顺序选前若干有效单跳题。这一选择不使用运行分数。一个对话的改写题和证据不会通过随机按题划分而进入两个 split。

demo 没有封存测试题，成绩应称为开发集演示成绩。full 的 test 应保持封存，开发时只运行 dev 查询。`manifest.json` 同时保留完整 split 映射和实际纳入的 scenario，避免把 demo 的映射误认为它实际运行了 10 个 scenario。

## 标准化语料

每个原始 session 形成一个主记录：

```json
{
  "scenario_id": "conv-26",
  "record_id": "eventMemId-bench-locomo-conv-26-s001",
  "title": "LoCoMo conv-26 session_1 | 原始 session 时间",
  "messages": [
    {
      "message_id": "D1:1",
      "role": "user",
      "speaker": "原说话者",
      "content": "原说话者: 原始文本"
    }
  ]
}
```

LoCoMo 的两个说话者是对话双方，不能伪称一个是 AI 助手。为满足当前记忆存储的角色字段，两者均映射为 `role=user`；原名同时留在 `speaker` 和文本前缀。消息 ID 保留 `dia_id`，只在所属记录／scenario 内定位；相关性标注采用 `record_id + message_id`，不可只用跨对话重复的 `D1:1`。

时间写在 session 标题，所有检索方案看到相同信息。保留原有图片的官方 BLIP caption 文本，并明确添加“image not loaded”的标记；不下载图片，不声称测试视觉理解。语料不会写入 `qa`、参考答案、`event_summary`、`observation` 或 `session_summary`。

## 查询和证据

每个查询保存 `query_id`、`scenario_id`、`split`、`category`、`query`、`answerable=true`、`relevant_record_ids`、`evidence_groups` 和 `reference_answer`。参考答案留在标注文件，不能进入检索文本。

每个去重后的原始 evidence 消息 ID 形成一个证据组，当前每组只有一个备选来源。因此这里的证据覆盖是 **对已公布标注的覆盖率**。尤其多跳题，“全部已标注证据”不一定恰好等于“完成任务的最小充分证据”，标注也不保证穷尽所有等价正确来源。

过滤采用整题跳过，不会偷偷移除坏证据后改变正确答案定义：

- demo 只保留类别 4；full 只保留类别 1/2/4。
- 类别 3 的开放域问题可能需要外部知识；类别 5 为对抗／无答案问题。本版都排除。**类别 5 也可能带非空 evidence，不能据此当成正例。**
- 空问题、空证据、无法按 `D数字:数字` 解析的证据、任何一个 evidence ID 无法映射的题整题排除。
- 任何证据消息带非空图片字段或图片 caption 的题保守整题排除；其原始消息仍留在完整语料，保留干扰。
- 相同 scenario 中，规范化问题文本与证据集合相同的重复 QA 只保留第一条。
- demo 超过上限的有效题暂不纳入；不是把它们当测试题。

`manifest.json` 记录各类过滤数量、入选题目数、类别分布和 split 分布。没有伪造无答案标签，所以这一版不能评价拒答能力。

## 文件与确定性

```text
data/processed/locomo-demo-v1/
  manifest.json
  corpus.jsonl
  queries.jsonl
data/processed/locomo-full-v1/
  manifest.json
  corpus.jsonl
  queries.jsonl
```

JSON 与 JSONL 全部使用 UTF-8、固定键排序和稳定顺序。manifest 保存语料及查询文件的 SHA256，并以版本、来源指纹、profile 和这两个文件指纹生成 `dataset_fingerprint`。下载时间不参与数据集指纹，因此相同准备命令重复执行会得到相同数据指纹。`load_dataset(path)` 会检查文件指纹、数量、稳定 ID、split 隔离和全部证据引用。

2026-10-29 已实际完成下载、标准化与引用完整性校验，并确认重复准备 demo 的指纹一致：

| profile | scenario | session 主记录 | 消息 | 查询 | 证据组 |
| --- | --- | --- | --- | --- | --- |
| demo | `conv-26` | 19 | 419 | 24 道 dev 单跳题 | 24 |
| full | 10 个，dev/test 为 6/4 | 272 | 5,882 | 792 道，其中 dev 483、test 309 | 952 |

完整源文件 SHA256 为 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`，demo 数据指纹为 `c1c2535360e64e8dc3df01096981813dc9f2309b2541187e1ac141020fd5cb6d`。full 题目从 1,986 条中排除了开放域 96 条、对抗题 446 条、图片证据题 641 条、证据格式无法解析 3 条、证据 ID 不存在 3 条、重复题 5 条。full 的标准化只是数据结构检查，不代表已经运行或查看测试集检索成绩。

每道题只允许搜索同一 `scenario_id` 的记录。把十个人物对话混成同一个候选库会改变问题难度，也会脱离本版协议。原始对话中的所有 sessions 都是候选，不得用 evidence 先筛出正确 session 再进行检索。

## 如何命名成绩

本任务应写成“LoCoMo 文本检索派生任务”“LoCoMo demo 开发集检索成绩”等。它既不是官方回答 F1，也不是模型记忆系统的完整端到端成绩。报告时同时给出 profile、scenario 数量、查询数量、过滤规则、检索粒度、top-k、输出文本量与数据指纹。

后续可以接入 [LongMemEval cleaned 官方数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)。它提供会话级 `answer_session_ids` 与消息级 `has_answer`，更贴近 user/assistant 历史记忆。完整 S cleaned 文件约 277 MB；oracle 文件约 15.4 MB，但只包含正确证据会话，不能把 oracle 的容易检索结果当成带完整干扰语料的正式检索成绩。
