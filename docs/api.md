# pg_retrieval_engine v0.2 API Reference (Detailed)

## 1. Global Notes

- Index objects are backend-local (not globally shared across sessions).
- Input types are provided by `pgvector`: `vector` / `vector[]`.
- `metric='cosine'` is implemented as normalized inner product; returned distance is `1 - ip`.
- Effective return count is always `min(k, ntotal)`.

## 2. Function List

| Function | Return | Purpose |
|---|---|---|
| `pg_retrieval_engine_index_create` | `void` | Create index object |
| `pg_retrieval_engine_index_train` | `void` | Train IVF/IVFPQ |
| `pg_retrieval_engine_index_add` | `bigint` | Bulk insert vectors |
| `pg_retrieval_engine_index_search` | `table(id, distance)` | Single-query ANN |
| `pg_retrieval_engine_index_search_batch` | `table(query_no, id, distance)` | Batch ANN (optimized path) |
| `pg_retrieval_engine_index_search_filtered` | `table(id, distance)` | Hybrid retrieval (single query, ID filter) |
| `pg_retrieval_engine_index_search_batch_filtered` | `table(query_no, id, distance)` | Hybrid retrieval (batch query, ID filter) |
| `pg_retrieval_engine_index_autotune` | `jsonb` | Auto tune defaults |
| `pg_retrieval_engine_metrics_reset` | `void` | Reset runtime counters |
| `pg_retrieval_engine_index_save` | `void` | Persist index |
| `pg_retrieval_engine_index_load` | `void` | Load index |
| `pg_retrieval_engine_index_stats` | `jsonb` | Metadata + runtime metrics |
| `pg_retrieval_engine_index_drop` | `void` | Drop index |
| `pg_retrieval_engine_reset` | `void` | Drop all indexes in current backend |

## 3. API Details

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

Arguments:

- `name`: backend-local unique name, max 63 chars.
- `dim`: vector dimension, range `1..65535`.
- `metric`: `l2` / `ip` / `inner_product` / `cosine`.
- `index_type`: `hnsw` / `ivfflat` / `ivf_flat` / `ivfpq` / `ivf_pq`.
- `options`: index build options.
- `device`: `cpu` (default) / `gpu`.

Supported `options`:

- `m` (HNSW, default 32)
- `ef_construction` (HNSW, default 200)
- `ef_search` (HNSW, default 64)
- `nlist` (IVF, default 4096)
- `nprobe` (IVF, default 32)
- `pq_m` (IVFPQ, default 64)
- `pq_bits` (IVFPQ, default 8)
- `gpu_device` (GPU id, default 0)

### 3.2 `pg_retrieval_engine_index_train`

```sql
pg_retrieval_engine_index_train(name text, training_vectors vector[]) returns void
```

- `training_vectors` must be one-dimensional, non-empty, no NULLs, and dimension-matched.

### 3.3 `pg_retrieval_engine_index_add`

```sql
pg_retrieval_engine_index_add(name text, ids bigint[], vectors vector[]) returns bigint
```

- `ids` and `vectors` must have identical length.
- Returns number of vectors inserted.

### 3.4 `pg_retrieval_engine_index_search`

```sql
pg_retrieval_engine_index_search(
  name text,
  query vector,
  k int,
  search_params jsonb default '{}'::jsonb
) returns table(id bigint, distance real)
```

`search_params`:

- `ef_search` (HNSW query breadth)
- `nprobe` (IVF probes)
- `candidate_k` (candidate depth, default `k`)

### 3.5 `pg_retrieval_engine_index_search_batch`

```sql
pg_retrieval_engine_index_search_batch(
  name text,
  queries vector[],
  k int,
  search_params jsonb default '{}'::jsonb
) returns table(query_no int, id bigint, distance real)
```

`search_params`:

- `ef_search`
- `nprobe`
- `candidate_k`
- `batch_size` (chunk size, default = index `preferred_batch_size`)

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

Performs ANN retrieval and keeps only IDs from `filter_ids`.

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

Batch hybrid retrieval with per-query top-k after filtering.

### 3.8 `pg_retrieval_engine_index_autotune`

```sql
pg_retrieval_engine_index_autotune(
  name text,
  mode text default 'balanced',
  options jsonb default '{}'::jsonb
) returns jsonb
```

Arguments:

- `mode`: `latency` / `balanced` / `recall`
- `options`:
  - `target_recall` (default `0.95`)
  - `min_batch_size` (default `32`)
  - `max_batch_size` (default `4096`)

Returns JSON old/new diffs for `hnsw_ef_search`, `ivf_nprobe`, and `preferred_batch_size`.

### 3.9 `pg_retrieval_engine_metrics_reset`

```sql
pg_retrieval_engine_metrics_reset(name text default null) returns void
```

- `name is null`: reset runtime counters for all indexes in current backend.
- `name is not null`: reset only the target index.

### 3.10 `pg_retrieval_engine_index_save`

```sql
pg_retrieval_engine_index_save(name text, path text) returns void
```

- Main index file: `path`
- Sidecar metadata: `path.meta`

### 3.11 `pg_retrieval_engine_index_load`

```sql
pg_retrieval_engine_index_load(name text, path text, device text default 'cpu') returns void
```

Loads a persisted index under a new runtime name.

### 3.12 `pg_retrieval_engine_index_stats`

```sql
pg_retrieval_engine_index_stats(name text) returns jsonb
```

Returns:

- Metadata: `name/version/dim/metric/index_type/device`
- Config snapshots: `hnsw/ivf/ivfpq`
- Runtime metrics: `runtime.*` (call counts, timing totals/averages, latest candidate/batch knobs, autotune state)

### 3.13 `pg_retrieval_engine_index_drop`

```sql
pg_retrieval_engine_index_drop(name text) returns void
```

### 3.14 `pg_retrieval_engine_reset`

```sql
pg_retrieval_engine_reset() returns void
```

## 4. Common Errors

- invalid arguments/ranges: `ERRCODE_INVALID_PARAMETER_VALUE`
- missing index: `ERRCODE_UNDEFINED_OBJECT`
- invalid runtime state (for example IVF untrained): `ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE`
- FAISS runtime failures: `ERRCODE_EXTERNAL_ROUTINE_EXCEPTION`

## 5. Hybrid Retrieval Example (Production Pattern)

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
