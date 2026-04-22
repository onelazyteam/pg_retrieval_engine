-- Upgrade script from 0.1.0 to 0.2.0

DROP FUNCTION IF EXISTS pg_retrieval_engine_version();
DROP FUNCTION IF EXISTS pg_retrieval_engine_init();
DROP FUNCTION IF EXISTS pg_retrieval_engine_create_index(text, integer, text, text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_drop_index(text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_list_indexes();
DROP FUNCTION IF EXISTS pg_retrieval_engine_index_stats(text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_get_index_info(text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_add_vectors(text, real[], bigint[], integer);
DROP FUNCTION IF EXISTS pg_retrieval_engine_add_vector_array(text, real[], bigint);
DROP FUNCTION IF EXISTS pg_retrieval_engine_search(text, real[], integer);
DROP FUNCTION IF EXISTS pg_retrieval_engine_search_with_ids(text, real[], integer);
DROP FUNCTION IF EXISTS pg_retrieval_engine_search_both(text, real[], integer);
DROP FUNCTION IF EXISTS pg_retrieval_engine_search_array(text, real[], integer);
DROP FUNCTION IF EXISTS pg_retrieval_engine_train(text, real[]);
DROP FUNCTION IF EXISTS pg_retrieval_engine_save_index(text, text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_load_index(text, text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_set_data_directory(text);
DROP FUNCTION IF EXISTS pg_retrieval_engine_reset();

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
