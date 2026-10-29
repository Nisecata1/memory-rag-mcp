# 检索与详情接口契约

本文是 `search` 与 `get_details_by_ids` 的完整行为说明；[README](../README.md) 保留入口和速查，其他写接口仍在 README 中说明。以下 JSON 使用虚构数据，表示工具的业务请求或返回，不包含 MCP 协议外层。

## 先区分三种读取方式

锚点是调用者选中的消息，用于定位前后文。以下只针对 `chat_event`，且默认 `results.returnMessageNum=10`。

| 详情请求 | 返回内容 | `raw_dialogue.context` |
| --- | --- | --- |
| 只传 `ids`，未传消息定位 | 每条会话末尾最多 21 条消息；不足 21 条则全部返回 | `mode=tail_preview`，`reason=no_anchors` |
| 传 `ids` 和分组的 `matched_message_ids` | 每个有效锚点及其前后各 10 条，窗口合并、重叠去重 | `mode=matched_context`，`reason=matched_messages` |
| 传 `full_dialogue=true` | 完整原始对话，忽略锚点，不限制消息数量 | 不添加 `context` 或省略项 |

**只传主记忆 ID，不会自动读取上一次 search 的命中消息，也不会默认读取长会话全文。** 预览不是检索证据；漏传定位可能让调用者错过实际命中的内容。

- `ids` 填主记忆 ID，例如 `eventMemId-*`、`factMemId-*`，不是 `msg-*`。
- `matched_message_ids` 中的消息 ID 只在所属会话内定位；当前代码不保证它在全库唯一，所以必须按主记忆 ID 分组。
- 详情不重新检索、不重新评分，也不保存“上一次 search”的会话状态。调用者可以根据搜索评分只选择部分锚点。
- 限制按每条会话计算，不是整批详情共享额度。消息按数组中的单条记录计数，不是一问一答。

## `search`：获取候选和代表消息

### 输入与配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `query` | 必填 | 查询文本 |
| `top_k` | `5` | 主记忆数量上限，不是消息数量；实际范围为 1 到 `search.max_top_k`，当前配置上限为 20 |
| `message_top_k` | `null` | 每条聊天记忆的代表消息上限；省略或 `null` 使用配置，配置缺失默认 3 |

显式 `message_top_k` 必须为正整数，拒绝布尔值、小数、字符串、零和负数。`message_top_k=3`、`top_k=5` 表示最多 5 条主记忆，每条聊天记忆最多 3 个代表消息，不是全库选 3 条消息。

```yaml
search:
  max_top_k: 20
  min_similarity: null
  message_score_gap: 0.2
  message_top_k: 3

results:
  returnMessageNum: 10
```

- `message_score_gap` 为非负有限数，控制同一会话中消息候选与最高消息分的差距；缺失默认 0.2。
- `returnMessageNum` 为非负整数，既是详情前后扩展的半径，也是代表消息的排除距离；缺失默认 10，设为 0 时不扩展邻近消息。
- `min_similarity` 为 `null` 时关闭主记忆最低分过滤，或设为 0 到 1 的数值；先折叠主记忆并取最高分，再过滤，最后截取 `top_k`。
- 配置在启动时读取，修改后需重启相关 MCP 进程。以上结果控制参数不改变索引签名，不需要重算向量或迁移数据。
- 旧的 `results.max_total_messages` 已取消，残留键被忽略；详情不再有固定 40 条上限。

### 消息选择规则

1. 同一消息的多个向量切片取最高分，排除非有限消息分数。
2. 每条会话独立计算最高消息分，保留分差不超过 `message_score_gap` 的候选；浮点边界沿用现有容差，不先四舍五入。消息分不与标题等字段分数混比。
3. 按消息分降序，同分按原消息位置、再按消息 ID 排序。
4. 依次选择候选，只有与所有已选锚点的位置距离都大于 `returnMessageNum` 才保留，最多选 `message_top_k` 个；不足不凑数。

只有已选锚点会排除候选，被跳过的候选不会继续连锁排除。半径 10 时，位置 20 和 30 冲突，20 和 31 可以同时入选；它们展开后的正文窗口允许重叠。

主记忆 `score` 是主记录及其子记录的最高相似度，不是整段对话的平均分。消息选择不改变主记忆排序、`top_k`、主记忆 `score` 或 `matched_fields`。

### 返回字段

| 字段 | 含义 |
| --- | --- |
| `query` | 实际查询文本 |
| `top_k` | 归一化后的主记忆上限 |
| `message_top_k` | 实际生效的每会话代表消息上限，正常和不可用返回均包含 |
| `results` | 轻量候选列表，不包含消息正文 |
| `results[].score` | 主记忆相似度 |
| `results[].matched_fields` | 有字段级命中时返回，按贡献度排序，最多 3 个字段名 |
| `results[].matched_messages` | 仅聊天记忆返回，包含代表消息 ID 和原始相似度，按评分降序 |
| `results[].matched_message_ids` | 仅聊天记忆返回，与 `matched_messages` 的 ID、顺序和数量严格对应 |

轻量业务字段由 `results.search_fields` 配置；当前包括 `id`、`memory_kind`、`created_at`、`updated_at`、`title`、`short_summary`、`overview_summary`、`project_id`、`tags`、`reference_doc_path`。聊天没有可用消息候选时两个消息列表均为空；非聊天结果不添加这两个字段。

索引正在后台重建或上次重建失败时，返回 `results=[]`、`search_available=false`，并附 `rebuild_state`（`running` 或 `failed`）、`index_refresh_reason`、`message`。运行中缺失计数非零时另有 `missing_entry_count`；失败时另有 `last_error`。调用者必须将其视为检索不可用，而不是“没有相关记忆”。正常返回不包含这些状态字段。

## `get_details_by_ids`：按指定定位读取

### 输入与基础返回

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ids` | 必填 | 主记忆 ID 数组，单条也传数组 |
| `matched_message_ids` | `null` | `{主记忆ID: [消息ID, ...]}`；键必须属于本次 `ids` |
| `full_dialogue` | `false` | 为 `true` 时返回原始全文，忽略锚点和消息数量限制 |

不能将 `msg-*` 当作 `ids` 中的主记忆 ID；接口不会反查消息所属会话，找不到同名主记录时只会将该 ID 列入 `missing_ids`。内部 `fmem-*` 索引记录会被明确拒绝读取。

接口按主 ID 从 SQLite 物理详情表读取，不使用向量索引或统一公共视图。未知旧 ID 前缀回查 `memory_registry`。部分主 ID 不存在时仍返回其他记录，缺失项列入 `missing_ids`。

外层固定为 `ids`、`records`、`missing_ids`。主 ID 去重保留首次出现，`records` 按请求顺序返回。每条记录包含 `id`、`memory_kind`、`created_at`、`updated_at` 和对应类型的业务字段：

- `chat_event`：`title`、`short_summary`、`raw_dialogue`、标签和来源字段，不返回旧的 `detailed_summary`。
- `project_record`：项目背景、分析、步骤、验证、`detailed_summary` 和 `project_id` 等；不进行消息裁剪。
- `fact`：标题、摘要、`detailed_summary`、标签和来源字段；不进行消息裁剪。
- `project_registry`：标题、`overview_summary`、标签和来源字段，保留 `project_id=id` 的别名。

`tags_json`、`source_paths_json` 转为 `tags`、`source_paths`；内部 `retrieval_fields_json` 不对外返回。

### 定位失效、窗口和只读标记

| 定位情况（`full_dialogue=false`） | 行为 |
| --- | --- |
| 未传定位、`null`、空映射或该记录的空列表 | 末尾预览，`reason=no_anchors` |
| 至少一个有效消息 ID | 使用所有有效锚点，失效项列入 `unresolved_message_ids` |
| 传入的消息 ID 全部失效 | 末尾预览，`reason=anchors_not_found`，显式列出失效项 |

窗口按当前 SQLite 正文的消息位置定位，去重锚点并保留输入顺序。每个锚点完整展开前后各 `returnMessageNum` 条，对窗口取并集，最终按原对话顺序输出。默认 3 个锚点、半径 10，单条会话最多展开 63 条真实消息；首尾边界、重叠会减少数量。

直接传入更多锚点不受 `message_top_k` 约束，也不会截断锚点或正文到 40 条。无锚点的预览数量为 `min(消息总数, 2 * returnMessageNum + 1)`。

消息的 ID、角色、原文和可选时间戳、元数据保留。所有被跳过的连续区间（含开头和结尾）插入以下独立省略项；重叠或相邻窗口之间没有省略项：

```json
{"type": "omission", "omitted_count": 128, "content": "[... 省略 128 条消息 ...]"}
```

省略项不是用户或助手消息，不计入 `returned_messages`。默认视图的 `raw_dialogue.context` 字段如下：

| 字段 | 含义 |
| --- | --- |
| `mode` | `matched_context` 或 `tail_preview` |
| `reason` | `matched_messages`、`no_anchors` 或 `anchors_not_found` |
| `total_messages` | 原始真实消息总数 |
| `returned_messages` | 本次真实消息数，不含省略项 |
| `truncated` | 是否省略了真实消息；不表示单条正文被截短 |
| `anchor_message_ids` | 实际使用的有效锚点 |
| `unresolved_message_ids` | 无法在当前正文定位的消息 ID |
| `omitted_anchor_count` | 兼容字段，固定为 0，不再因数量额度丢弃锚点 |

**默认详情始终是只读视图，即使 `truncated=false` 也不能直接回写正文。** `save_chatEvent`、`update` 会拒绝含只读 `context` 或省略项的对话。需要查重、合并、替换正文时，先显式读取全文并校验没有这些标记；自动归档的历史合并和恢复也遵循此约定。

旧的非结构化对话不能自动定位消息，默认读取会报错，需使用 `full_dialogue=true`。所有视图都不会改变 SQLite 保存的完整正文。

## 调用示例：评分选取后读取上下文

第一步，调用 `search`：

```json
{"query": "某项目的故障排查过程", "top_k": 5, "message_top_k": 3}
```

返回中的聊天候选片段（省略其他轻量字段）：

```json
{
  "id": "eventMemId-example",
  "memory_kind": "chat_event",
  "score": 0.95,
  "matched_message_ids": ["msg-A", "msg-B"],
  "matched_messages": [
    {"message_id": "msg-A", "score": 0.90},
    {"message_id": "msg-B", "score": 0.83}
  ]
}
```

第二步，调用者根据消息评分选择 A，并将候选 `id` 与消息 ID 一起交给详情接口。记录分可能来自标题等字段，未必等于最高消息分：

```json
{
  "ids": ["eventMemId-example"],
  "matched_message_ids": {"eventMemId-example": ["msg-A"]},
  "full_dialogue": false
}
```

这返回 A 及前后文，不返回 B 的独立窗口（若 B 位于 A 的前后文中，仍会出现）。以下两个请求的含义则不同：

```json
{"ids": ["eventMemId-example"]}
```

这是末尾预览，不会记住第一步的查询或自动选 A。

```json
{"ids": ["eventMemId-example"], "full_dialogue": true}
```

这是显式全文读取，不能用来代替日常检索后的局部读取。

## 调用者检查与文档维护

- 检索不可用时不要把空 `results` 解释为无证据；读取详情后检查 `missing_ids`、`context.mode` 和失效消息 ID。
- 需要命中证据却收到 `tail_preview` 时，检查是否漏传、错传定位；不要将末尾预览冒充命中内容。
- 相似度不是事实正确的概率，历史命中不代表当前事实；时间戳仅在原消息提供时存在，不要臆造时间或将旧观点当成最新状态。
- 这些规则只控制消息数量，不截断单条长消息，也不限制整批字符或 token 数；批量会话、更多锚点或全文仍可能产生很大的响应。
- 接口行为变更时同时更新实现、`tests/test_dialogue_context.py`、本文、README 速查以及 `server.py` 中的工具和参数描述；不要手工覆盖客户端生成的工具缓存。
- Markdown 文档可直接阅读；工具描述或配置更新需要客户端重连并重启对应 MCP 进程，确认工具列表已刷新。

在临时数据上验证本契约的针对性测试：

```powershell
python -m unittest discover -s tests -p test_dialogue_context.py -v
```

覆盖消息评分、分差和距离边界、每会话独立限额、预览、有效与失效定位、窗口并集、省略计数、全文开关、回写保护、配置兼容及真实 stdio 请求；不向正式记忆库写入测试数据。
