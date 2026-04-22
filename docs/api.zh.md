# pg_retrieval_engine v0.2 API 参考（完整参数版）

## 1. 全局说明

- 所有索引对象都存在于当前 PostgreSQL backend 进程内（非全局共享）。
- 输入类型依赖 `pgvector`：`vector` / `vector[]`。
- `metric='cosine'` 使用“归一化 + inner product”；返回值转换为 cosine distance（`1 - ip`）。
- 当 `k > ntotal` 时，实际返回 `min(k, ntotal)`。

## 2. 函数清单

| 函数 | 返回 | 主要用途 |
|---|---|---|
| `pg_retrieval_engine_index_create` | `void` | 创建索引对象 |
| `pg_retrieval_engine_index_train` | `void` | 训练 IVF/IVFPQ |
| `pg_retrieval_engine_index_add` | `bigint` | 批量写入向量 |
| `pg_retrieval_engine_index_search` | `table(id, distance)` | 单向量检索 |
| `pg_retrieval_engine_index_search_batch` | `table(query_no, id, distance)` | 批量检索（优化路径） |
| `pg_retrieval_engine_index_search_filtered` | `table(id, distance)` | 混合检索（单查，ID 过滤） |
| `pg_retrieval_engine_index_search_batch_filtered` | `table(query_no, id, distance)` | 混合检索（批查，ID 过滤） |
| `pg_retrieval_engine_index_autotune` | `jsonb` | 自动调参 |
| `pg_retrieval_engine_metrics_reset` | `void` | 重置 runtime 指标 |
| `pg_retrieval_engine_index_save` | `void` | 保存索引 |
| `pg_retrieval_engine_index_load` | `void` | 加载索引 |
| `pg_retrieval_engine_index_stats` | `jsonb` | 查看元信息+运行指标 |
| `pg_retrieval_engine_index_drop` | `void` | 删除索引 |
| `pg_retrieval_engine_reset` | `void` | 清空当前 backend 全部索引 |

## 3. 接口详情

### 3.1 `pg_retrieval_engine_index_create`

```sql
pg_retrieval_engine_index_create(
  name text,
  dim int,
  metric text,
  index_type text,
  options jsonb default '{}'::jsonb,
  device text default 'cpu'
) returns void
```

参数：

- `name`: 索引名，当前 backend 内唯一，最长 63 字符。
- `dim`: 维度，范围 `1..65535`。
- `metric`: `l2` / `ip` / `inner_product` / `cosine`。
- `index_type`: `hnsw` / `ivfflat` / `ivf_flat` / `ivfpq` / `ivf_pq`。
- `options`: 索引参数。
- `device`: `cpu`（默认）/ `gpu`。

`options` 字段：

- `m`（HNSW，默认 32）
- `ef_construction`（HNSW，默认 200）
- `ef_search`（HNSW，默认 64）
- `nlist`（IVF，默认 4096）
- `nprobe`（IVF，默认 32）
- `pq_m`（IVFPQ，默认 64）
- `pq_bits`（IVFPQ，默认 8）
- `gpu_device`（GPU 卡号，默认 0）

### 3.2 `pg_retrieval_engine_index_train`

```sql
pg_retrieval_engine_index_train(name text, training_vectors vector[]) returns void
```

- `training_vectors` 需为一维、非空、无 NULL、维度匹配。

### 3.3 `pg_retrieval_engine_index_add`

```sql
pg_retrieval_engine_index_add(name text, ids bigint[], vectors vector[]) returns bigint
```

- `ids` 与 `vectors` 数量必须相同。
- 返回写入条数。

### 3.4 `pg_retrieval_engine_index_search`

```sql
pg_retrieval_engine_index_search(
  name text,
  query vector,
  k int,
  search_params jsonb default '{}'::jsonb
) returns table(id bigint, distance real)
```

`search_params`：

- `ef_search`（HNSW 查询宽度）
- `nprobe`（IVF 探测桶数）
- `candidate_k`（候选集深度，默认 `k`）

### 3.5 `pg_retrieval_engine_index_search_batch`

```sql
pg_retrieval_engine_index_search_batch(
  name text,
  queries vector[],
  k int,
  search_params jsonb default '{}'::jsonb
) returns table(query_no int, id bigint, distance real)
```

`search_params`：

- `ef_search`
- `nprobe`
- `candidate_k`
- `batch_size`（批处理分块大小，默认使用索引的 `preferred_batch_size`）

### 3.6 `pg_retrieval_engine_index_search_filtered`

```sql
pg_retrieval_engine_index_search_filtered(
  name text,
  query vector,
  k int,
  filter_ids bigint[],
  search_params jsonb default '{}'::jsonb
) returns table(id bigint, distance real)
```

作用：ANN 检索后按 `filter_ids` allow-list 过滤，实现混合检索。

### 3.7 `pg_retrieval_engine_index_search_batch_filtered`

```sql
pg_retrieval_engine_index_search_batch_filtered(
  name text,
  queries vector[],
  k int,
  filter_ids bigint[],
  search_params jsonb default '{}'::jsonb
) returns table(query_no int, id bigint, distance real)
```

作用：批量混合检索。每个 query 在过滤后返回 top-k。

### 3.8 `pg_retrieval_engine_index_autotune`

```sql
pg_retrieval_engine_index_autotune(
  name text,
  mode text default 'balanced',
  options jsonb default '{}'::jsonb
) returns jsonb
```

参数：

- `mode`: `latency` / `balanced` / `recall`
- `options`:
  - `target_recall`（默认 0.95）
  - `min_batch_size`（默认 32）
  - `max_batch_size`（默认 4096）

返回：包含 `hnsw_ef_search` / `ivf_nprobe` / `preferred_batch_size` 的 old/new 对比。

### 3.9 `pg_retrieval_engine_metrics_reset`

```sql
pg_retrieval_engine_metrics_reset(name text default null) returns void
```

- `name is null`：重置当前 backend 全部索引 runtime 指标。
- `name not null`：只重置目标索引 runtime 指标。

### 3.10 `pg_retrieval_engine_index_save`

```sql
pg_retrieval_engine_index_save(name text, path text) returns void
```

- 主索引写到 `path`
- 元数据写到 `path.meta`

### 3.11 `pg_retrieval_engine_index_load`

```sql
pg_retrieval_engine_index_load(name text, path text, device text default 'cpu') returns void
```

- 将磁盘索引加载为新索引名。

### 3.12 `pg_retrieval_engine_index_stats`

```sql
pg_retrieval_engine_index_stats(name text) returns jsonb
```

包含三类信息：

- 元信息：`name/version/dim/metric/index_type/device`
- 参数：`hnsw/ivf/ivfpq`
- 运行指标：`runtime.*`（调用量、耗时、候选深度、batch 参数、autotune 状态等）

### 3.13 `pg_retrieval_engine_index_drop`

```sql
pg_retrieval_engine_index_drop(name text) returns void
```

### 3.14 `pg_retrieval_engine_reset`

```sql
pg_retrieval_engine_reset() returns void
```

## 4. 常见错误

- 参数类型或范围非法：`ERRCODE_INVALID_PARAMETER_VALUE`
- 索引不存在：`ERRCODE_UNDEFINED_OBJECT`
- 状态非法（例如未训练就写入 IVF）：`ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE`
- FAISS 运行异常：`ERRCODE_EXTERNAL_ROUTINE_EXCEPTION`

## 5. 混合检索示例（工业常见模式）

```sql
WITH allow_list AS (
  SELECT array_agg(id ORDER BY id) AS ids
  FROM product_embedding
  WHERE tenant_id = 42
    AND category = 'electronics'
    AND is_active = true
)
SELECT *
FROM pg_retrieval_engine_index_search_filtered(
  'prod_idx',
  '[0.1,0.2,0.3,0.4]'::vector,
  20,
  (SELECT ids FROM allow_list),
  '{"candidate_k":200,"ef_search":128}'::jsonb
);
```
