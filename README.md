# 项目说明

当前项目名是 `memory-rag-mcp`。这个目录里放的是一个独立的 Python MCP server，用来保存高信息密度的项目经验、会话事件和事实记忆，生成按最近更新时间整理的时间线，并支持向量相似度检索。

## 运行方式

- Conda 环境名称：`project-beta`
- 服务入口：`AI-memory/memory-rag-mcp/server.py`
- 项目配置：`AI-memory/memory-rag-mcp/config/settings.yaml`
- 传输方式：`stdio`

## 项目特点

**内部记忆类型**  
"field_record" 内部类型（仅为了提升召回，不开放给外部接口使用）  
**目的**：为了解决**粒度不一致**的问题（比如单条记忆太大quary太小）导致检索漏关键信息  
**字段描述**：在每一条记忆详细入库之后，也顺便把该条记忆的所有入库字段单独拆出来，作为一条 memory_kind: field_record 类型的记忆，再入一次库

## 项目配置文件

- 当前不再通过环境变量覆盖模型路径和设备。
- 运行参数统一放在：`AI-memory/memory-rag-mcp/config/settings.yaml`
- 这份 YAML 里已经按类别收好了：
  - `server`
  - `paths`
  - `embedding`
  - `memory`
  - `summary`
  - `search`
  - `retrieval`
  - `results`
- 代码启动时会直接读取这份 YAML，并把它收口成当前运行常量。

## 项目 Requirements

如果你的目标是“后面把这个项目传到 GitHub，clone 下来后直接安装依赖并在本地继续使用”，最少需要看清两类东西：代码仓库里会带什么，以及哪些私有资产需要你自己接回本地。

### clone 仓库后即可获得的内容

- `memory-rag-mcp/` 项目代码
- `config/settings.yaml`
- `requirements.txt`
- `README.md`

其中：

- `config/settings.yaml` 的位置固定就是项目内的 `config/settings.yaml`
- clone 完成后，直接在项目根目录执行 `pip install -r requirements.txt` 即可安装当前项目的 Python 直接依赖

### 不会随仓库自动带上的私有资产

- 本地 embedding 模型目录
  - 实际位置由 `config/settings.yaml` 里的 `embedding.model_path` 决定
  - 如果想保留回退能力，再额外准备一个可切换的回退模型目录
- 私有记忆数据目录
  - `AI-memory/memory-rag-mcp-data/`
  - 如果要延续旧记忆，里面至少应包含：
    - `memory.json`
    - `memory_embeddings.npy`
    - `memory.faiss`
    - `memory_meta.json`
- 人类视图文件
  - `AI-memory/memory-rag-mcp-data/RAG记忆库时间线.md`
  - 这个文件不是主数据源，但建议跟数据目录一起接回，避免换机后丢失人类可读视图
- Codex 本地注册配置
  - 新机器上的 `C:\Users\<你的用户名>\.codex\config.toml` 里需要注册 `memory-rag-mcp`

### 两种使用模式

- 空库启动模式
  - 只接代码和本地模型，不接旧数据
  - 这种情况下也可以直接开始 `save`
  - 首次保存后，服务会自动生成主数据、时间线、embedding 缓存、FAISS 和 meta
- 旧记忆延续模式
  - 在上面的基础上，再把私有模型和私有记忆数据接回本地
  - 这样 `search` 才能直接命中历史 AI 记忆

部署时最容易漏掉的不是代码，而是下面三样：

- 本地 embedding 模型目录
- `memory-rag-mcp-data/` 里的私有记忆数据文件
- Codex 的 MCP 注册块

## 记忆文件位置

- `AI-memory/memory-rag-mcp-data/memory.json`
- `AI-memory/memory-rag-mcp-data/memory_embeddings.npy`
- `AI-memory/memory-rag-mcp-data/memory.faiss`
- `AI-memory/memory-rag-mcp-data/memory_meta.json`
- `AI-memory/memory-rag-mcp-data/RAG记忆库时间线.md`

其中：

- `memory-rag-mcp-data/` 目录承载项目记忆系统自己的数据和索引文件，避免把技术文件散落在 `AI-memory` 根目录。
- `memory.json` 是唯一主数据源，保存完整字段。
- `memory_embeddings.npy` 是已计算向量的本地缓存矩阵，用来减少重复 embedding。
- `memory.faiss` 是由本地向量缓存派生出来的搜索结构。
- `memory_meta.json` 只保存索引元数据、当前检索字段签名以及 `FAISS` 行号到记录 `id` 的映射。
- `RAG记忆库时间线.md` 放在 `memory-rag-mcp-data/` 目录中，作为按最近更新时间查看的人类可读视图。

# 工具接口描述

## 调用约定

- 这些接口面向已经整理好的结构化输入，不负责替调用方阅读原始 Markdown、日志或网页。
- 调用 `save` / `update` 前，应先把内容整理成结构化字段；如果用户指定了现成文档，应先阅读文档，再提取并补充结构化信息。
- `detailed_summary` 是正式入库字段，不是临时中间值；写进去的内容会直接进入主数据。
- 当前不接受外部传业务时间字段；`created_at` 和 `updated_at` 都由服务端自动维护。
- `memory.json` 是唯一事实源；向量索引、时间线和内部 `field_record` 都是基于主数据生成的衍生产物。

## 接口总览

- `save`
  - 作用：保存一条高信息密度的项目经验、会话事件或事实记忆。
  - 副作用：更新主数据 JSON、重写时间线 Markdown，并优先复用本地 embedding 缓存来刷新向量索引。
- `update`
  - 作用：按 `id` 更新一条已有记忆。
  - 返回方式：补丁式修改主数据字段；如果规范化后内容未变化，返回 `updated=false` 且不重建。
- `search`
  - 作用：按向量相似度检索最相关的历史记忆候选。
  - 返回方式：先查 `FAISS`，再回源主数据 JSON，只返回轻量候选字段和相似度分数；如果命中的是内部字段级子记忆，会先折叠回源到主记忆。
- `get_details_by_ids`
  - 作用：按一批记录 `ids` 读取完整详情。
  - 返回方式：直接回源主数据 JSON，统一返回 `records` 列表和 `missing_ids`；仅在显式调试模式下附带内部调试字段。
- `delete_by_ids`
  - 作用：按一批记录 `ids` 从主数据中硬删除记忆。
  - 返回方式：删除命中的记录、同步更新时间线，并基于本地 embedding 缓存重建向量索引后返回最小删除结果。

## `save` 接口

### 功能说明

- 用于把 AI 已经整理好的结构化记忆写入长期库。
- `save` 不做“读文档并理解”的工作；那部分是 AI 在调用前完成的。
- 如果本次写入和历史记录完全一致，`save` 会去重，并返回 `deduped=true`。
- 每次 `save` 成功后，都会同步刷新时间线和向量索引；当前实现会优先只对新记录做增量 embedding。
- 如果缓存缺失或当前库状态无法安全复用旧向量，服务端会先写入主数据，再把全量 embedding 放到后台执行，并返回一个 `rebuilding` 提示；这段时间 `search` 不可用。
- 推荐使用流程：
  1. 如果本次要存的是非事件记忆，先 `search` 看看是否已有与当前主题相近的记忆。
  2. 如果已存在相近记忆，优先判断是否应该对已有记忆执行 `update`。
  3. 如果不存在合适的已有记忆，再调用 `save` 新建一条记忆。
  4. 每次 `save` 或 `update` 之后，再判断这条记忆里是否真的包含长期可复用、跨会话稳定、对长期陪伴有帮助的事实；只有确实存在这类事实时，才考虑继续按 `fact` 流程处理：先 `search` 相近 `fact`，若已有语义相近 `fact` 则优先 `update`，若没有相近 `fact` 则再 `save` 新 `fact`。如果内容主要是项目问题的排查、调试和解决过程，而 `project_record` 已经完整记录，则通常不需要再额外提取 `fact`。
- 上面的流程是推荐使用策略，不是当前服务端自动执行逻辑。

### 输入字段

- `memory_kind`
  - 必填。
  - 记忆类型。
  - 允许值：
    - `project_record`：项目经验类记录，重点记录已经解决的问题和可复用思路。要写清问题是什么、为什么这样判断、怎么解决、验证结果是什么；不要记流水账，默认应记录已经解决的问题与思路，而不是半成品过程日志。
    - `chat_event`：会话事件记忆，适合保存会话中的片段性、情境性、带时间顺序的事件痕迹。一般没有特殊说明，都作为一般事件类型存储。
    - `fact`：稳定事实记忆，以个人长期事实为主，更适合保存个人信息、长期偏好、稳定约束、长期状态、最近目标、已确认关系共识这类跨会话可复用事实。不建议把 `project_record` 里已经完整记录的项目 debug 过程、排错流水和解决链路再重复存成 `fact`；只有当项目内容被提炼成真正稳定、可跨会话复用的规则时，才考虑转成 `fact`。
- `title`
  - 必填。
  - 记录标题，用一句话概括本次记忆。
- `short_summary`
  - 必填。
  - 面向嵌入检索的简短摘要，建议控制在一两句话内。
  - 这个字段由 AI 主动总结，放在 `retrieval_fields` 的第二位，优先服务于 512 token 预算下的检索命中率。
- `detailed_summary`
  - 必填。
  - AI 整理后的最终详细总结，也是正式入库正文。
  - 这个字段默认不参与 embedding 拼接，主要用于详情阅读和长期归档。
  - 如果用户指定了参考文档，这里的信息不能比参考文档更少。
- `problem_background`
  - `project_record` 必填，其他类型可选。
  - 用于保存问题以及背景、故障现象，以及这条已解决问题为什么值得记录；不要写成流水账。
- `analysis`
  - `project_record` 必填，其他类型可选。
  - 用于保存原因判断、方案分析、取舍依据、排查结论，以及为什么这样判断、为什么这么做；要留下后续可复用的思路。
- `action_steps`
  - `project_record` 必填，其他类型可选。
  - 用于保存真正解决问题时执行的关键步骤、命令、操作顺序或具体处理动作，不要堆无用流水过程。
- `validation_result`
  - `project_record` 必填，其他类型可选。
  - 用于保存测试、验证、复现是否消失、验收结果等验证结论；默认应记录已经解决并验证过的问题。
- `tags`
  - 可选传入，但最终存储必有。
  - 推荐由 AI 先生成标签数组。
  - 如果缺失，服务端会做基础兜底补全。
- `source_paths`
  - 可选。
  - 普通来源材料路径或链接列表，例如日志、网页、截图、工单链接。
- `reference_doc_path`
  - 可选。
  - 参考文档路径。
  - 只在“用户指定一份现成文档让 AI 归档”时使用，不与 `source_paths` 混用。

### 返回字段

- `id`
  - 本条记录的稳定唯一标识。
- `updated_at`
  - 最近一次入库或更新的时间。
- `created_at`
  - 第一次入库时间。
- `tags`
  - 最终落库标签数组。
- `deduped`
  - 是否命中了去重。
  - `true` 表示这次没有新增记录，而是复用了已有记录。
- `total_entries`
  - 当前主库总记录数。
- `timeline_summary`
  - 最近几条时间线摘要，便于调用方快速确认入库结果。
- `index_stats`
  - 本次索引重建统计信息。
  - 当前包含：
    - `count`：索引中的记录条数。
    - `dim`：向量维度。
- `index_refresh_state`
  - 仅在这次写入已经接受、但索引仍在后台重建时返回。
  - 当前固定值为 `rebuilding`。
- `index_refresh_mode`
  - 仅在后台重建时返回。
  - 当前固定值为 `background_full_rebuild`。
- `search_available`
  - 仅在后台重建时返回。
  - `false` 表示主数据已写入，但 `search` 需要等后台重建完成后才能继续使用。
- `message`
  - 仅在后台重建时返回。
  - 用于提示调用方：主数据已经写入，但索引仍在后台补齐。
- `fact_extraction_reminder`
  - 仅在本次真实写入的是非 `fact` 记忆时返回。
  - 作用：提示调用方是否值得继续提取长期可复用 `fact`。
  - 这里只返回结构化建议，不代表服务端已经自动完成 `fact` 提取。
  - 当前固定包含：
    - `decision_by_caller`：恒为 `true`，表示是否提取由调用方判断。
    - `message`：简短提醒文案。
    - `recommended_steps`：固定的后续推荐步骤。

### `save` 的核心字段

- `memory_kind`
  - 必填。
  - 只能是 `project_record`、`chat_event` 或 `fact`。
- `title`
  - 两类记忆都必填。
- `short_summary`
  - 最终存储必有。
  - 两类记忆都必填。
  - 用来保存面向嵌入检索的简短摘要，建议控制在一两句话内。
  - 这个字段需要 AI 主动总结，不再由服务端为新记录自动补造。
- `detailed_summary`
  - 两类记忆都必填。
  - 这是 AI 整理后的最终详细总结，也是正式入库字段。
  - 默认不会进入 embedding 检索文本，主要保留给详情读取和长期归档。
- `problem_background`、`analysis`、`action_steps`、`validation_result`
  - `project_record` 必填。
  - `chat_event` 和 `fact` 可为空；如果传了，也会正常存储并参与检索拼接。
  - `problem_background` 用来保存“问题以及背景”，而不只是一个简短问题名。
  - `analysis` 用来保存原因判断、方案分析、排查结论，以及为什么这么做。
  - `action_steps` 用来保存实际执行的步骤、命令、操作顺序或具体处理动作。
  - `validation_result` 用来保存测试、验证、复现是否消失、验收结果等验证结论。
- `updated_at`
  - 由服务端自动维护。
  - 表示该记录最近一次入库或更新的时间。
- `created_at`
  - 由服务端自动维护。
  - 表示该记录第一次入库的时间，写入后不再变化。
- `tags`
  - 最终存储必有。
  - 推荐由 AI 先生成；缺失时服务端会做基础兜底补全。
- `source_paths`
  - 普通来源材料列表，比如日志、网页、截图来源。
- `reference_doc_path`
  - 参考文档路径。
  - 只在“用户指定现成文档让 AI 入库”时填写，不与 `source_paths` 混用。

### 个人笔记

save：先规范化新记录，读老主数据进内存，然后拼接成新主数据列表，然后按照脚本规则进行排序，此时内存中有了最新的主数据  
然后取meta里的id列表，并和缓存矩阵构成一个 dict【id->向量】:  
-- 从 cached_embeddings + cached_row_ids 生成 cached_vector_by_id

然后向量化新记录并把新向量补进字典:  
-- cached_vector_by_id[appended_entry_id] = appended_embedding

然后脚本根据这个 dict 以及新主数据的 id 顺序（current_entry_ids ），去重拼并落盘新faiss、meta和npy：  
memory_embeddings.npy <- rebuilt_embeddings  
memory.faiss <- 基于 rebuilt_embeddings 重建  
memory_meta.json <- sorted_entries 的 id 顺序

## `update` 接口

### 功能说明

- 用于按记录 `id` 对一条已有记忆做补丁式更新。
- `update` 不是整条记录替换；`changes` 里传什么字段，就只更新什么字段。
- 如果规范化后的结果和当前记录完全一致，接口会返回 `updated=false`，不重建索引，也不刷新 `updated_at`。
- 如果更新后的内容与另一条已有记录完全一致，接口会直接报冲突，不会自动合并。
- 当前实现会优先只重算被更新主记忆本体及其 `field_record` 家族的向量。
- 如果缓存缺失或当前库状态无法安全复用旧向量，服务端会先写入主数据，再把全量 embedding 放到后台执行，并返回一个 `rebuilding` 提示；这段时间 `search` 不可用。
- `field_record` 属于内部索引资产，不能通过 `update` 直接修改。

### 输入字段

- `id`
  - 必填。
  - 要更新的目标记录 id。
- `changes`
  - 必填。
  - 一个补丁对象，只传本次要修改的业务字段。
  - 允许字段包括：
    - `memory_kind`
    - `title`
    - `short_summary`
    - `detailed_summary`
    - `problem_background`
    - `analysis`
    - `action_steps`
    - `validation_result`
    - `tags`
    - `source_paths`
    - `reference_doc_path`
  - 不允许直接修改：
    - `id`
    - `fingerprint`
    - `created_at`
    - `updated_at`
    - `retrieval_fields`
  - 清空语义：
    - 可空文本字段用 `null` 清空
    - `source_paths` 用 `[]` 清空
    - `tags` 传 `[]` 时仍沿用当前服务端兜底补标签规则，最终结果不保证为空
    - 空字符串按真实值处理，不自动当清空

### 返回字段

- `id`
  - 本条记录的稳定唯一标识。
- `updated_at`
  - 最近一次真实内容变化时间。
- `created_at`
  - 第一次入库时间。
- `tags`
  - 更新后的最终标签数组。
- `updated`
  - 是否真的发生了更新。
  - `true` 表示已更新并重建。
  - `false` 表示这是一次 no-op。
- `total_entries`
  - 当前主库总记录数。
- `timeline_summary`
  - 最近几条时间线摘要。
- `index_stats`
  - 当前索引统计信息。
- `index_refresh_state`
  - 仅在这次更新已经接受、但索引仍在后台重建时返回。
  - 当前固定值为 `rebuilding`。
- `index_refresh_mode`
  - 仅在后台重建时返回。
  - 当前固定值为 `background_full_rebuild`。
- `search_available`
  - 仅在后台重建时返回。
  - `false` 表示主数据已更新，但 `search` 需要等后台重建完成后才能继续使用。
- `message`
  - 仅在后台重建时返回。
  - 用于提示调用方：主数据已经更新，但索引仍在后台补齐。
- `fact_extraction_reminder`
  - 仅在本次真实更新后，且更新后的记忆类型不是 `fact` 时返回。
  - 作用：提示调用方是否值得继续提取长期可复用 `fact`。
  - 这里只返回结构化建议，不代表服务端已经自动完成 `fact` 提取。
  - 当前固定包含：
    - `decision_by_caller`：恒为 `true`，表示是否提取由调用方判断。
    - `message`：简短提醒文案。
    - `recommended_steps`：固定的后续推荐步骤。

## `search` 接口

### 功能说明

- 用于按语义相似度检索历史项目经验、会话事件和事实记忆。
- `search` 不直接从时间线 Markdown 检索，而是先查 `FAISS`，再用 `meta` 中的 `row_to_id` 回源主数据 JSON。
- `search` 只负责返回轻量候选结果，不承担完整详情读取。
- 如果命中的是内部 `field_record`，服务端会先折叠回源到它的源头主记忆，再把主记忆返回给调用方。
- 如果需要完整字段，应继续调用 `get_details_by_ids`。
- 如果后台全量重建仍在运行，`search` 会直接拒绝并返回状态对象，不会继续返回旧索引结果。
- 如果上一次后台重建失败，`search` 也会返回状态对象，并附带最近一次重建错误文本。

### 输入字段

- `query`
  - 必填。
  - 查询文本，可以是问题、关键词、故障现象、主题、组件名、报错信息等。
- `top_k`
  - 可选。
  - 期望返回的结果条数。
  - 默认 `5`，最大 `20`。

### 返回字段

- `query`
  - 本次实际执行检索的查询文本。
- `top_k`
  - 归一化后的返回条数上限。
- `results`
  - 检索命中的记录列表。
  - 每条结果包含轻量候选字段和一个 `score`。
  - 当前轻量字段包括：
    - `id`
    - `score`
    - `memory_kind`
    - `updated_at`
    - `created_at`
    - `title`
    - `short_summary`
    - `tags`
    - `reference_doc_path`
    - `matched_fields`
      - 可选。
      - 只有在该主记忆是因为内部字段级子记忆命中而被召回时才返回。
      - 值是字段名数组，按贡献度排序，最多 3 个。
- `search_available`
  - 仅在索引正在后台重建或上一次重建失败时返回。
  - `false` 表示当前不能安全使用 `search`。
- `rebuild_state`
  - 仅在 `search_available=false` 时返回。
  - 当前可能值包括：
    - `running`
    - `failed`
- `message`
  - 仅在 `search_available=false` 时返回。
  - 用于说明当前为什么不能继续 `search`。
- `last_error`
  - 仅在 `rebuild_state=failed` 时返回。
  - 保存最近一次后台重建失败时的错误文本。

## `get_details_by_ids` 接口

### 功能说明

- 用于按记录 `ids` 批量读取项目经验、会话事件或事实记忆的完整详情。
- 这个接口不做向量搜索，只负责按主键回源主数据。
- 即使只读取一条，也应传单元素数组。
- 如果部分 id 不存在，接口不会整体报错，而是通过 `missing_ids` 显式返回缺失项。
- `include_debug=true` 时，才会附加内部调试字段。
- `field_record` 属于内部索引资产，不能通过这个接口直接读取。

### 输入字段

- `ids`
  - 必填。
  - 要读取的记录 id 数组。
  - 即使只查一条，也要传 `ids=["pmem-xxxx"]`。
  - 通常来自 `search` 结果中的 `id`。
- `include_debug`
  - 可选。
  - 默认 `false`。
  - 为 `true` 时，会额外返回 `fingerprint` 和 `retrieval_fields`。

### 返回字段

- `ids`
  - 清洗、去重后的请求 id 列表。
- `records`
  - 已成功命中的详情记录列表。
  - 顺序与输入 `ids` 保持一致。
  - 每条记录默认返回完整业务字段：
    - `id`
    - `updated_at`
    - `created_at`
    - `memory_kind`
    - `title`
    - `short_summary`
    - `problem_background`
    - `analysis`
    - `action_steps`
    - `validation_result`
    - `detailed_summary`
    - `tags`
    - `source_paths`
    - `reference_doc_path`
  - `include_debug=true` 时，每条记录额外返回：
    - `fingerprint`
    - `retrieval_fields`
- `missing_ids`
  - 本次请求中未找到的 id 列表。

## `delete_by_ids` 接口

### 功能说明

- 用于按记录 `ids` 从主数据中硬删除记忆。
- 删除成功后，会同步刷新主数据、时间线和向量索引；当前默认基于本地 embedding 缓存重建 FAISS，不重复调用 embedding 模型。
- 如果缓存缺失或当前库状态无法安全复用旧向量，服务端会先写入主数据，再把全量 embedding 放到后台执行，并返回一个 `rebuilding` 提示；这段时间 `search` 不可用。
- 如果部分 id 不存在，接口不会整体报错，而是通过 `missing_ids` 返回缺失项。
- 这是一个简单删除接口，不做软删除、归档或恢复站。
- 推荐调用顺序是：先 `search`，再 `get_details_by_ids` 确认内容，最后再调用 `delete_by_ids`。
- `field_record` 属于内部索引资产，不能通过这个接口直接删除。

### 输入字段

- `ids`
  - 必填。
  - 要删除的记录 id 数组。
  - 即使只删一条，也要传 `ids=["pmem-xxxx"]`。
  - 通常来自 `search` 结果中的 `id`。

### 返回字段

- `ids`
  - 清洗、去重后的请求 id 列表。
- `deleted_ids`
  - 本次实际删除成功的 id 列表。
- `missing_ids`
  - 本次请求中未找到的 id 列表。
- `deleted_count`
  - 本次实际删除成功的记录数量。
- `index_refresh_state`
  - 仅在这次删除已经接受、但索引仍在后台重建时返回。
  - 当前固定值为 `rebuilding`。
- `index_refresh_mode`
  - 仅在后台重建时返回。
  - 当前固定值为 `background_full_rebuild`。
- `search_available`
  - 仅在后台重建时返回。
  - `false` 表示主数据已删除，但 `search` 需要等后台重建完成后才能继续使用。
- `message`
  - 仅在后台重建时返回。
  - 用于提示调用方：主数据已经删除，但索引仍在后台补齐。

## 调试字段说明

- `fingerprint`
  - 这是内部去重和一致性校验字段。
  - 它的主要作用是判断“这条记录和历史记录是否完全相同”，不适合作为默认业务返回字段。
- `retrieval_fields`
  - 这是内部检索审计字段，不是 `save` 的输入参数。
  - 它的作用是说明“这条记录在服务端生成检索文本时，实际用了哪些字段参与 embedding 拼接”，主要用于排查 RAG 命中效果，而不是面向普通使用者展示。
- 当前默认 embedding 字段不包含 `detailed_summary`，而是优先使用 `title`、`short_summary`、`problem_background`、`analysis`、`action_steps`、`validation_result`、`tags`、`reference_doc_path`。
- 因此这两个字段默认隐藏，只在 `get_details_by_ids(include_debug=true)` 时返回。

## 当前实现特点

- `search` 会先命中 `FAISS`，再回源到 `memory.json` 返回轻量候选字段。
- 向量索引里同时包含主记忆和内部 `field_record`；`search` 命中 `field_record` 时会折叠回源到主记忆，并用 `matched_fields` 解释命中来源。
- `get_details_by_ids` 是完整详情的唯一读取入口。
- `delete_by_ids` 会在主数据删除成功后同步更新时间线，并基于本地 embedding 缓存重建向量索引，不会重复调用 embedding 模型。
- 服务端内部会自动生成检索文本，不再接收外部传入的 `memory_text`。
- `detailed_summary` 会保留在主数据里，但默认不进入检索文本，以避免 CPU 全量重建时被长正文拖慢。
- 当前索引链路已经分成四层：主数据 JSON、embedding 缓存矩阵、FAISS 索引、meta 映射；优化重点是减少重复 embedding，而不是完全不重建 FAISS。
- `fact` 走同库新类型，不单独建第二套 store/index；当前版本不做自动提纯或自动分片。
- 旧主数据里的 `chat_fragment` 会在读写时迁移为 `chat_event`，旧的 `knowledge_record` 会迁移为 `fact`。
- `field_record` 现在按 one-hot 形态存储：`source_field_name` 是什么，就只让那个字段非空；向量化时也只取该字段原值，不再混入额外摘要壳。
- `field_record` 会跟随主记忆自动生成、同步和删除，但不会进入时间线，也不会计入对外展示的 `total_entries`。
- 现有数据会在读写时自动按新结构兼容，不需要手工迁移旧版 `memory_text` 字段。
- 向量加载和归一化逻辑在当前目录下的 `embedding_utils.py` 中，不再依赖其他项目文件。

# 一键部署

## 模型准备与配置

部署前需要先准备本地 embedding 模型。GitHub 仓库只包含代码，不附带本地模型文件。

- `embedding.model_path` 的参数意义是：**本地 embedding 模型目录**。
- clone 仓库后，需要先自行下载模型，再把这个参数改成你自己机器上的实际模型目录。
- 改完后要重启 MCP 或重新加载 Codex 配置，并触发一次索引重建或执行一次 `save`，让新的模型配置真正生效。

### 模型参数速查表

| 模型名      | 下载链接                                   | 模型类型                        | 最大序列长度 | 向量维度 | 适用场景                                             | 当前定位 |
| ----------- | ------------------------------------------ | ------------------------------- | ------------ | -------- | ---------------------------------------------------- | -------- |
| `bge-m3`    | `https://huggingface.co/BAAI/bge-m3`       | 多语言 embedding 模型           | `8192`       | `1024`   | 更适合长总结、长问题背景、长操作步骤的检索           | 默认模型 |
| `m3e-large` | `https://huggingface.co/moka-ai/m3e-large` | Sentence Transformers / BERT 系 | `512`        | `1024`   | 适合较短中文语义检索；当前主要问题是长检索文本会截断 | 回退模型 |

参数口径说明：

- `m3e-large` 的参数来自其 Hugging Face 模型配置文件。
- `bge-m3` 的参数来自其 Hugging Face 模型配置文件与官方 README。

### 使用 `bge-m3`

1. 从 `https://huggingface.co/BAAI/bge-m3` 下载模型到你的本地目录。
2. 修改 `config/settings.yaml` 里的 `embedding.model_path`，把它指向你自己的 `bge-m3` 本地模型目录。
3. 重启 MCP 或重新加载 Codex 配置。
4. 触发一次索引重建或执行一次 `save`，确认 `memory.faiss` 和 `memory_meta.json` 已按新模型重建。

补充说明：

- `bge-m3` 的主要收益是把上下文长度从 `512` 提升到更长级别，更适合当前这类高信息密度的长归档文本检索。
- 当前项目虽然有 GPU 硬件，但这次切换不包含 CUDA 环境整改，先按现有环境完成模型替换和重建。

### 切换到 `m3e-large`（回退）

1. 从 `https://huggingface.co/moka-ai/m3e-large` 下载模型到你的本地目录。
2. 修改 `config/settings.yaml` 里的 `embedding.model_path`，把它指向你自己的 `m3e-large` 本地模型目录。
3. 重启 MCP 或重新加载 Codex 配置。
4. 触发一次索引重建或执行一次 `save`，确认 `memory.faiss` 和 `memory_meta.json` 已按回退模型重建。

补充说明：

- `m3e-large` 当前保留作为回退模型。
- 它的主要限制是 `512 token`，对较长问题背景、分析、处理步骤这类复杂项目记录，截断风险更高。

默认使用方式：

1. clone 仓库：把 `memory-rag-mcp/` 放到你的本地工作目录
2. 安装依赖：在项目根目录执行（如果有虚拟环境先激活环境）：

   ```bash
   pip install -r requirements.txt
   ```

3. 确认配置文件仍位于项目内固定位置：
   - `config/settings.yaml`
4. 如果你本地的项目目录、数据目录或时间线路径与默认布局不同，修改 `settings.yaml` 里的 `paths.*`
5. 准备本地 embedding 模型，并按上面的“模型准备与配置”完成下载和参数修改
6. 如果只是想把项目跑起来，不接旧数据也可以，后续直接开始 `save` 即可
7. 如果想继续使用旧 AI 记忆，再把私有数据目录 `memory-rag-mcp-data/` 整体接回本地；时间线文件现在也包含在这个目录里
8. 在 `~/.codex/config.toml` 里注册 `memory-rag-mcp`(可以让ai完成)
9. 做一次最小校验：
   - `python -m py_compile server.py`
   - 再执行一次 `search` 冒烟测试，确认服务可用

要点说明：

- `requirements.txt` 只能解决 Python 依赖安装问题
- 本地 embedding 模型和私有记忆数据不会随 GitHub 仓库自动带上
- `settings.yaml` 的位置不需要改，真正可能需要按本机调整的是它里面的 `paths.*`
