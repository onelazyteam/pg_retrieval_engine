CREATE EXTENSION vector;
CREATE EXTENSION pg_retrieval_engine;

SELECT pg_retrieval_engine_reset() IS NOT NULL AS reset_ok;

SELECT pg_retrieval_engine_index_create(
    'idx_h',
    4,
    'l2',
    'hnsw',
    '{"m":16,"ef_construction":128,"ef_search":128}'::jsonb,
    'cpu'
) IS NOT NULL AS create_hnsw_ok;

SELECT pg_retrieval_engine_index_add(
    'idx_h',
    ARRAY[1,2,3,4]::bigint[],
    ARRAY[
        '[1,0,0,0]'::vector,
        '[2,0,0,0]'::vector,
        '[3,0,0,0]'::vector,
        '[4,0,0,0]'::vector
    ]::vector[]
) AS added_hnsw;

SELECT (pg_retrieval_engine_index_stats('idx_h')->>'num_vectors')::int AS hnsw_num_vectors;

SELECT id
FROM pg_retrieval_engine_index_search('idx_h', '[1,0,0,0]'::vector, 2, '{}'::jsonb)
ORDER BY distance, id;

SELECT id
FROM pg_retrieval_engine_index_search_filtered(
    'idx_h',
    '[1,0,0,0]'::vector,
    2,
    ARRAY[2,4]::bigint[],
    '{"candidate_k":4}'::jsonb
)
ORDER BY distance, id;

SELECT count(*) AS batch_rows
FROM pg_retrieval_engine_index_search_batch(
    'idx_h',
    ARRAY['[1,0,0,0]'::vector, '[4,0,0,0]'::vector]::vector[],
    2,
    '{}'::jsonb
);

SELECT count(*) AS filtered_batch_rows
FROM pg_retrieval_engine_index_search_batch_filtered(
    'idx_h',
    ARRAY['[1,0,0,0]'::vector, '[4,0,0,0]'::vector]::vector[],
    2,
    ARRAY[1,4]::bigint[],
    '{"candidate_k":4,"batch_size":1}'::jsonb
);

SELECT (pg_retrieval_engine_index_autotune('idx_h', 'balanced', '{"target_recall":0.97}'::jsonb)
        -> 'preferred_batch_size' ->> 'new')::int > 0 AS autotune_ok;

SELECT (pg_retrieval_engine_index_stats('idx_h')->'runtime'->>'search_filtered_calls')::int >= 2 AS runtime_filtered_ok;

SELECT pg_retrieval_engine_metrics_reset('idx_h') IS NOT NULL AS metrics_reset_ok;
SELECT (pg_retrieval_engine_index_stats('idx_h')->'runtime'->>'search_query_total')::int AS search_query_total_after_reset;

SELECT pg_retrieval_engine_index_save('idx_h', '/tmp/pg_retrieval_engine_regress.idx') IS NOT NULL AS save_ok;
SELECT pg_retrieval_engine_index_drop('idx_h') IS NOT NULL AS drop_hnsw_ok;
SELECT pg_retrieval_engine_index_load('idx_h', '/tmp/pg_retrieval_engine_regress.idx', 'cpu') IS NOT NULL AS load_ok;

SELECT id
FROM pg_retrieval_engine_index_search('idx_h', '[1,0,0,0]'::vector, 1, '{}'::jsonb);

SELECT pg_retrieval_engine_index_create(
    'idx_ivf',
    4,
    'cosine',
    'ivfflat',
    '{"nlist":2,"nprobe":2}'::jsonb,
    'cpu'
) IS NOT NULL AS create_ivf_ok;

SELECT pg_retrieval_engine_index_train(
    'idx_ivf',
    ARRAY[
        '[1,0,0,0]'::vector,
        '[0,1,0,0]'::vector,
        '[0,0,1,0]'::vector,
        '[0,0,0,1]'::vector
    ]::vector[]
) IS NOT NULL AS train_ivf_ok;

SELECT pg_retrieval_engine_index_add(
    'idx_ivf',
    ARRAY[11,12,13,14]::bigint[],
    ARRAY[
        '[1,0,0,0]'::vector,
        '[0,1,0,0]'::vector,
        '[0,0,1,0]'::vector,
        '[0,0,0,1]'::vector
    ]::vector[]
) AS added_ivf;

SELECT id
FROM pg_retrieval_engine_index_search('idx_ivf', '[1,0,0,0]'::vector, 1, '{"nprobe":2}'::jsonb);

SELECT pg_retrieval_engine_index_drop('idx_h') IS NOT NULL AS drop_loaded_ok;
SELECT pg_retrieval_engine_index_drop('idx_ivf') IS NOT NULL AS drop_ivf_ok;
SELECT pg_retrieval_engine_reset() IS NOT NULL AS final_reset_ok;
