# pg_retrieval_engine v0.2 Design (Handover Edition)

## 1. Purpose

This document is for engineers taking over `pg_retrieval_engine`. It explains:

- Core plugin architecture and execution model.
- Newly implemented capabilities:
  - P0 observability
  - P1 hybrid retrieval
  - P1 auto tuning
  - P1 batch query optimization
- A practical handover and release checklist.

## 2. Scope and Non-goals

### 2.1 In scope

- Function-based index object API (no PostgreSQL Index AM implementation).
- Primary input path: `vector` / `vector[]`.
- Index families: `hnsw`, `ivfflat`, `ivfpq`.
- CPU path + optional GPU path.

### 2.2 Out of scope

- Shared-memory global index registry (current state is backend-local).
- WAL-level replication semantics.
- SQL predicate parsing in the extension for hybrid retrieval.

## 3. Version Matrix

- PostgreSQL: 18.3
- pgvector: 0.8.2 (performance baseline)
- FAISS: 1.14.1
- pg_retrieval_engine extension: 0.2.0

## 4. Architecture

### 4.1 Components

- Entry implementation: `src/faiss_in_pg/faiss_engine.cpp`
- Shared type definitions: `src/faiss_in_pg/faiss_engine.hpp`
- SQL API definitions: `sql/pg_retrieval_engine--0.2.0.sql`
- Registry: backend-local `HTAB` keyed by index name
- Runtime index handles:
  - `cpu_index` (always present)
  - `gpu_index` (optional)

### 4.2 Lifecycle

1. `create`: build and register FAISS index.
2. `train`: train IVF/IVFPQ indexes.
3. `add`: add vectors with explicit IDs.
4. `search`: single, batch, or hybrid retrieval.
5. `save/load`: persistence and restore.
6. `drop/reset`: resource cleanup.

## 5. API Surface (including new functions)

| Function | Purpose | Return |
|---|---|---|
| `pg_retrieval_engine_index_create` | Create index | `void` |
| `pg_retrieval_engine_index_train` | Train index | `void` |
| `pg_retrieval_engine_index_add` | Bulk insert | `bigint` |
| `pg_retrieval_engine_index_search` | Single-query ANN | `table(id, distance)` |
| `pg_retrieval_engine_index_search_batch` | Batch ANN (optimized path) | `table(query_no, id, distance)` |
| `pg_retrieval_engine_index_search_filtered` | Hybrid single-query (ANN + ID filter) | `table(id, distance)` |
| `pg_retrieval_engine_index_search_batch_filtered` | Hybrid batch-query (ANN + ID filter) | `table(query_no, id, distance)` |
| `pg_retrieval_engine_index_autotune` | Auto-tune defaults | `jsonb` |
| `pg_retrieval_engine_metrics_reset` | Reset runtime counters | `void` |
| `pg_retrieval_engine_index_save/load` | Persist and restore | `void` |
| `pg_retrieval_engine_index_stats` | Metadata + runtime metrics | `jsonb` |
| `pg_retrieval_engine_index_drop/reset` | Cleanup | `void` |

## 6. Feature #4: Observability (P0)

### 6.1 Goal

Expose enough runtime data to answer:

- What is the call volume by search path?
- How expensive are single vs batch vs filtered queries?
- Which candidate depth and batch size were used recently?
- Was auto-tuning applied?

### 6.2 Metrics model

Per-index runtime counters include:

- Calls: `train_calls`, `add_calls`, `search_single_calls`, `search_batch_calls`, `search_filtered_calls`
- Volume: `add_vectors_total`, `search_query_total`, `search_result_total`
- Time totals: `search_single_ms_total`, `search_batch_ms_total`, `search_filtered_ms_total`
- Ops: `save_calls`, `load_calls`, `autotune_calls`, `error_calls`
- Last known tuning/runtime: `last_candidate_k`, `last_batch_size`, `preferred_batch_size`, `last_autotune_mode`

### 6.3 Exposure

- `pg_retrieval_engine_index_stats(name)` now includes a `runtime` object.
- `pg_retrieval_engine_metrics_reset(name default null)` resets runtime counters for one index or all indexes.

## 7. Feature #5: Hybrid Retrieval (P1)

### 7.1 Goal

Support common production patterns: business filter + ANN retrieval, such as:

- tenant isolation
- product category constraints
- policy/security tier constraints

### 7.2 API

- `pg_retrieval_engine_index_search_filtered`
- `pg_retrieval_engine_index_search_batch_filtered`

Both accept `filter_ids bigint[]` as an allow-list.

### 7.3 Execution

1. Run FAISS ANN to get `candidate_k` results.
2. Keep only IDs in the allow-list.
3. Return top `k` per query.

### 7.4 Search params

`search_params` supports:

- `candidate_k`: candidate depth before filter
- `batch_size`: chunk size for batch execution
- existing params: `ef_search`, `nprobe`

### 7.5 Current limitations

- Hybrid mode is currently ID-allow-list based.
- SQL predicate parsing/execution is expected to remain in business SQL layer.

## 8. Feature #6: Auto Tuning (P1)

### 8.1 API

`pg_retrieval_engine_index_autotune(name, mode, options)`

- `mode`: `latency` / `balanced` / `recall`
- `options`:
  - `target_recall` (default `0.95`)
  - `min_batch_size` (default `32`)
  - `max_batch_size` (default `4096`)

### 8.2 Heuristics

- HNSW: derive `ef_search` from `sqrt(ntotal)` with mode multipliers.
- IVF: derive `nprobe` from `sqrt(nlist)` with mode multipliers.
- Batch: derive `preferred_batch_size` from device, dimension, and mode.

### 8.3 Application

- Updates both entry defaults and active FAISS runtime knobs.
- Returns JSON old/new diff for auditability.

## 9. Feature #8: Batch Query Optimization (P1)

### 9.1 Problem

A monolithic batch search allocates memory proportional to `num_queries * k`, which scales poorly.

### 9.2 Optimization

- Chunked execution controlled by `batch_size`.
- Candidate widening (`candidate_k`) for filtered/hybrid paths.
- Unified search-parameter apply/restore path.

### 9.3 Result

- Peak memory drops from `O(num_queries * candidate_k)` to `O(batch_size * candidate_k)`.
- Better cache behavior and more stable throughput under large batch workloads.

## 10. Persistence and Recovery

- Primary file: FAISS binary index (`path`)
- Sidecar metadata: `path.meta`
- Persist in CPU-writable format to ensure recoverability.
- Load path reconstructs runtime config from metadata.

## 11. Error Model

- Invalid arguments: `ERRCODE_INVALID_PARAMETER_VALUE`
- Missing object: `ERRCODE_UNDEFINED_OBJECT`
- Invalid state (for example untrained index): `ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE`
- FAISS internal failures: `ERRCODE_EXTERNAL_ROUTINE_EXCEPTION`

## 12. Testing

### 12.1 Regression coverage

`test/sql/pg_retrieval_engine_test.sql` covers:

- lifecycle: create/train/add/search/save/load/drop/reset
- filtered search and filtered batch search
- autotune + runtime metrics checks
- metrics reset

### 12.2 TAP coverage

- `010_recall.pl`: recall validation
- `020_perf_cpu_vs_pgvector.pl`: CPU benchmark vs pgvector
- `030_perf_gpu_vs_pgvector.pl`: GPU benchmark vs pgvector (conditional)

## 13. Handover Checklist

### 13.1 Adding a new index type

1. Update enum in `PgRetrievalEngineIndexType`.
2. Extend `parse_index_type`, `index_type_name`, and `build_index`.
3. Update SQL/docs/regression tests.

### 13.2 Adding a new metric/counter

1. Add fields to `PgRetrievalEngineIndexEntry`.
2. Update success/error write points.
3. Expose in `pg_retrieval_engine_index_stats`.
4. Add regression assertions.

### 13.3 Release checklist

1. `make && make installcheck`
2. TAP recall at minimum
3. Keep README/API/design docs aligned in English + Chinese
4. Keep both fresh install SQL and upgrade SQL aligned

## 14. Follow-up Recommendations

- Add shared-memory global observability views across backends.
- Add closed-loop auto-tuning based on real recall/latency feedback.
- Add SQL-predicate-driven hybrid retrieval (SPI with allowlisted templates).
