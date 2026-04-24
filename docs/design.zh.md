# pg_retrieval_engine v0.2 设计文档（研发接手版）

## 1. 文档目的

本设计文档面向后续接手 `pg_retrieval_engine` 的研发同学，目标是：

- 说明当前插件的核心架构与执行模型。
- 说明本期新增能力：
  - P0 可观测性
  - P1 混合检索
  - P1 自动调参
  - P1 批量查询优化
- 给出可直接落地的开发、测试、发布接手流程。

## 2. 范围与边界

### 2.1 本期范围

- 采用函数式索引对象 API（不实现 PostgreSQL Index AM）。
- 输入主通道：`vector` / `vector[]`。
- 索引类型：`hnsw` / `ivfflat` / `ivfpq`。
- 设备路径：CPU + 可选 GPU。

### 2.2 非目标

- 不做 shared-memory 全局索引注册（当前是 backend-local）。
- 不做 WAL 级别一致性复制。
- 不做 SQL 谓词解析型混合检索（当前采用 ID allow-list 方式）。

## 3. 版本与对比基线

- PostgreSQL：18.3
- pgvector：0.8.2（性能对比基线）
- FAISS：1.14.1
- 插件版本：0.2.0

## 4. 总体架构

### 4.1 组件

- 入口文件：`src/faiss_in_pg/faiss_engine.cpp`
- 元信息定义：`src/faiss_in_pg/faiss_engine.hpp`
- SQL 定义：`sql/pg_retrieval_engine--0.2.0.sql`
- 注册表：backend 本地 `HTAB`（键为 index name）
- 运行态对象：
  - `cpu_index`（必有）
  - `gpu_index`（可选，GPU 构建时）

### 4.2 生命周期

1. `create`：创建索引对象并注册到 hash。
2. `train`：训练 IVF/IVFPQ。
3. `add`：按显式 ID 批量写入。
4. `search`：单查 / 批查 / 混合检索。
5. `save/load`：索引落盘与恢复。
6. `drop/reset`：释放单索引或当前 backend 全部索引。

## 5. API 总览（含新增）

| 函数 | 用途 | 返回 |
|---|---|---|
| `pg_retrieval_engine_index_create` | 创建索引 | `void` |
| `pg_retrieval_engine_index_train` | 训练索引 | `void` |
| `pg_retrieval_engine_index_add` | 批量写入 | `bigint` |
| `pg_retrieval_engine_index_search` | 单查询 | `table(id, distance)` |
| `pg_retrieval_engine_index_search_batch` | 批查询（优化路径） | `table(query_no, id, distance)` |
| `pg_retrieval_engine_index_search_filtered` | 混合检索（单查，ID 过滤） | `table(id, distance)` |
| `pg_retrieval_engine_index_search_batch_filtered` | 混合检索（批查，ID 过滤） | `table(query_no, id, distance)` |
| `pg_retrieval_engine_index_autotune` | 自动调参 | `jsonb` |
| `pg_retrieval_engine_metrics_reset` | 重置运行时指标 | `void` |
| `pg_retrieval_engine_index_save/load` | 持久化与恢复 | `void` |
| `pg_retrieval_engine_index_stats` | 元信息+运行指标 | `jsonb` |
| `pg_retrieval_engine_index_drop/reset` | 清理资源 | `void` |

## 6. Feature #4：可观测性（P0）

### 6.1 设计目标

- 线上可快速回答：
  - 查询调用量是多少？
  - 单查/批查/混合检索各自耗时如何？
  - 最近一次候选集大小和 batch 大小是多少？
  - 自动调参是否被执行过？

### 6.2 指标模型

每个索引维护 runtime counters：

- 调用量：`train_calls`, `add_calls`, `search_single_calls`, `search_batch_calls`, `search_filtered_calls`
- 数据量：`add_vectors_total`, `search_query_total`, `search_result_total`
- 耗时：`search_single_ms_total`, `search_batch_ms_total`, `search_filtered_ms_total`
- 运维：`save_calls`, `load_calls`, `autotune_calls`, `error_calls`
- 最近参数：`last_candidate_k`, `last_batch_size`, `preferred_batch_size`, `last_autotune_mode`

### 6.3 暴露方式

- `pg_retrieval_engine_index_stats(name)` 新增 `runtime` 字段。
- `pg_retrieval_engine_metrics_reset(name default null)` 可重置单索引或全部索引的 runtime 计数。

### 6.4 写入点

- `train/add/search*/save/load/autotune` 成功路径更新计数。
- FAISS 异常路径统一增加 `error_calls`。

## 7. Feature #5：混合检索能力（P1）

### 7.1 设计目标

支持工业场景中的“先业务过滤，再 ANN 召回”，例如：

- 多租户隔离（`tenant_id`）
- 品类过滤（`category`）
- 安全级别过滤（`policy_level`）

### 7.2 API 设计

- `pg_retrieval_engine_index_search_filtered`
- `pg_retrieval_engine_index_search_batch_filtered`

两者都接收 `filter_ids bigint[]`（allow-list）。

### 7.3 执行策略

1. 使用 FAISS ANN 搜索候选集合（`candidate_k`）。
2. 按 ID allow-list 做过滤。
3. 取每个 query 的前 `k`。

### 7.4 参数语义

`search_params` 新增：

- `candidate_k`：候选深度（过滤场景默认放大）
- `batch_size`：批量查询分块大小
- 原有：`ef_search`, `nprobe`

### 7.5 限制说明

- 当前混合检索是“ID 预过滤”，不直接解析 SQL 谓词。
- 如需“过滤后精排”，建议在业务 SQL 层做二次 rerank。

## 8. Feature #6：自动调参（P1）

### 8.1 API

`pg_retrieval_engine_index_autotune(name, mode, options)`

- `mode`: `latency` / `balanced` / `recall`
- `options`:
  - `target_recall`（默认 0.95）
  - `min_batch_size`（默认 32）
  - `max_batch_size`（默认 4096）

### 8.2 调参策略（启发式）

- HNSW：基于 `sqrt(ntotal)` 推导 `ef_search`，按 mode 调整倍数。
- IVF：基于 `sqrt(nlist)` 推导 `nprobe`，按 mode 调整倍数。
- Batch：根据设备（CPU/GPU）、维度、mode 推导 `preferred_batch_size`。

### 8.3 生效方式

- 更新 `entry` 默认参数。
- 同步写回 FAISS 运行态对象（`IndexHNSW` / `IndexIVF`）。
- 返回 `jsonb`，含 old/new 对比，便于变更审计。

## 9. Feature #8：批量查询优化（P1）

### 9.1 问题

原始批查会一次性分配 `num_queries * k` 缓冲区；大批量时内存高、cache 友好性差。

### 9.2 优化点

- 分块执行：`batch_size` 控制每次送入 FAISS 的 query 数。
- 有过滤场景支持更深候选：`candidate_k`。
- 查询参数应用逻辑统一，避免重复设置/恢复代码。

### 9.3 复杂度收益

- 内存峰值从 `O(num_queries * candidate_k)` 降为 `O(batch_size * candidate_k)`。
- 更适合在线服务高并发批请求。

## 10. 持久化与恢复

- 主文件：FAISS 二进制索引（`path`）
- 侧车元数据：`path.meta`
- 保存时以 CPU 可写格式为基准，保证可恢复。
- 加载时根据 meta 恢复 metric/index_type/参数。

## 11. 错误模型

- 参数错误：`ERRCODE_INVALID_PARAMETER_VALUE`
- 对象不存在：`ERRCODE_UNDEFINED_OBJECT`
- 状态错误（未训练等）：`ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE`
- 外部 FAISS 异常：`ERRCODE_EXTERNAL_ROUTINE_EXCEPTION`

## 12. 测试策略

### 12.1 回归测试（已覆盖）

`test/sql/pg_retrieval_engine_test.sql` 覆盖：

- create/train/add/search/search_batch
- filtered search / filtered batch
- autotune + runtime stats 验证
- metrics_reset
- save/load/drop/reset

### 12.2 TAP（现有）

- `010_recall.pl`：召回验证
- `020_perf_cpu_vs_pgvector.pl`：CPU 性能对比
- `030_perf_gpu_vs_pgvector.pl`：GPU 性能对比（条件执行）

## 13. 研发接手清单

### 13.1 新增索引类型

1. 在 `PgRetrievalEngineIndexType` 增加枚举。
2. 修改 `parse_index_type/index_type_name/build_index`。
3. 更新 SQL/API 文档与回归测试。

### 13.2 新增可观测指标

1. 在 `PgRetrievalEngineIndexEntry` 增加字段。
2. 在对应成功/异常路径更新计数。
3. 在 `pg_retrieval_engine_index_stats` 暴露字段。
4. 补回归断言。

### 13.3 发版检查

1. `make && make installcheck`
2. TAP（至少 recall）
3. README/API/Design 中英文同步
4. SQL 脚本和升级脚本同步

## 14. 后续建议

- 引入 shared-memory 级全局观测视图（跨 backend）。
- 增加基于真实召回反馈的闭环 autotune。
- 扩展为 SQL 谓词驱动的混合检索（SPI + 安全白名单）。
