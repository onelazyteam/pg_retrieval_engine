-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pg_retrieval_engine" to load this file. \quit

CREATE FUNCTION pg_retrieval_engine_index_create(
    name text,
    dim integer,
    metric text,
    index_type text,
    options jsonb DEFAULT '{}'::jsonb,
    device text DEFAULT 'cpu'
) RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_create'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_train(
    name text,
    training_vectors vector[]
) RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_train'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_add(
    name text,
    ids bigint[],
    vectors vector[]
) RETURNS bigint
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_add'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_search(
    name text,
    query vector,
    k integer,
    search_params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_search'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_search_batch(
    name text,
    queries vector[],
    k integer,
    search_params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_search_batch'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_search_filtered(
    name text,
    query vector,
    k integer,
    filter_ids bigint[],
    search_params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_search_filtered'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_search_batch_filtered(
    name text,
    queries vector[],
    k integer,
    filter_ids bigint[],
    search_params jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_search_batch_filtered'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_save(
    name text,
    path text
) RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_save'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_load(
    name text,
    path text,
    device text DEFAULT 'cpu'
) RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_load'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_autotune(
    name text,
    mode text DEFAULT 'balanced',
    options jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_autotune'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_metrics_reset(name text DEFAULT NULL)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_metrics_reset'
LANGUAGE C VOLATILE;

CREATE FUNCTION pg_retrieval_engine_index_stats(name text)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_stats'
LANGUAGE C STABLE STRICT;

CREATE FUNCTION pg_retrieval_engine_index_drop(name text)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_index_drop'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION pg_retrieval_engine_reset()
RETURNS void
AS 'MODULE_PATHNAME', 'pg_retrieval_engine_reset'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION pg_retrieval_engine_index_create(text, integer, text, text, jsonb, text)
IS 'Create a FAISS index. index_type: hnsw|ivfflat|ivfpq, metric: l2|ip|cosine, device: cpu|gpu.';

COMMENT ON FUNCTION pg_retrieval_engine_index_train(text, vector[])
IS 'Train IVF indexes using vector[] input.';

COMMENT ON FUNCTION pg_retrieval_engine_index_add(text, bigint[], vector[])
IS 'Bulk add vectors with explicit IDs.';

COMMENT ON FUNCTION pg_retrieval_engine_index_search(text, vector, integer, jsonb)
IS 'Search nearest neighbors and return (id, distance).';

COMMENT ON FUNCTION pg_retrieval_engine_index_search_batch(text, vector[], integer, jsonb)
IS 'Batch nearest-neighbor search and return (query_no, id, distance).';

COMMENT ON FUNCTION pg_retrieval_engine_index_search_filtered(text, vector, integer, bigint[], jsonb)
IS 'Hybrid retrieval: ANN search + ID prefilter list, return (id, distance).';

COMMENT ON FUNCTION pg_retrieval_engine_index_search_batch_filtered(text, vector[], integer, bigint[], jsonb)
IS 'Hybrid retrieval batch path: ANN search + ID prefilter list.';

COMMENT ON FUNCTION pg_retrieval_engine_index_save(text, text)
IS 'Persist index to disk. Metadata is stored at <path>.meta.';

COMMENT ON FUNCTION pg_retrieval_engine_index_load(text, text, text)
IS 'Load persisted index from disk.';

COMMENT ON FUNCTION pg_retrieval_engine_index_autotune(text, text, jsonb)
IS 'Auto tune search defaults (ef_search/nprobe/batch_size) for latency|balanced|recall targets.';

COMMENT ON FUNCTION pg_retrieval_engine_metrics_reset(text)
IS 'Reset runtime observability counters for one index or all indexes when name is NULL.';

COMMENT ON FUNCTION pg_retrieval_engine_index_stats(text)
IS 'Return index metadata and runtime statistics as jsonb.';

COMMENT ON FUNCTION pg_retrieval_engine_index_drop(text)
IS 'Drop one in-memory index.';

COMMENT ON FUNCTION pg_retrieval_engine_reset()
IS 'Drop all in-memory indexes in current backend process.';
