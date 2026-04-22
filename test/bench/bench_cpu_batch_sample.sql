\set ON_ERROR_STOP on
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_retrieval_engine;

DROP TABLE IF EXISTS items;
CREATE TABLE items (id bigint PRIMARY KEY, embedding vector(128));
INSERT INTO items
SELECT i,
       ARRAY(
         SELECT (sin(i * 0.001 + j * 0.01) + 1.0)::real
         FROM generate_series(1,128) AS g(j)
       )::vector(128)
FROM generate_series(1,20000) AS t(i);

DROP TABLE IF EXISTS bench_queries;
CREATE TABLE bench_queries AS
SELECT row_number() OVER () AS qno, embedding
FROM (SELECT embedding FROM items WHERE id % 667 = 0 ORDER BY id LIMIT 30) s;

ANALYZE items;
ANALYZE bench_queries;

DROP INDEX IF EXISTS items_hnsw;
DROP INDEX IF EXISTS items_ivf;
CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_l2_ops);
ANALYZE items;

SELECT pg_retrieval_engine_reset();
SELECT pg_retrieval_engine_index_create('faiss_hnsw', 128, 'l2', 'hnsw', '{"m":32,"ef_construction":200,"ef_search":64}'::jsonb, 'cpu');
SELECT pg_retrieval_engine_index_add(
  'faiss_hnsw',
  (SELECT array_agg(id ORDER BY id) FROM items),
  (SELECT array_agg(embedding ORDER BY id) FROM items)
);

DROP TABLE IF EXISTS bench_results;
CREATE TEMP TABLE bench_results (
  impl text,
  scenario text,
  total_ms double precision,
  avg_ms double precision,
  recall_at_10 double precision
);

SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;
CREATE TEMP TABLE expected_hnsw AS
SELECT q.qno, e.id
FROM bench_queries q
CROSS JOIN LATERAL (
  SELECT id FROM items ORDER BY embedding <-> q.embedding LIMIT 10
) e;
RESET enable_indexscan;
RESET enable_bitmapscan;
RESET enable_seqscan;

DO $$
DECLARE
  q RECORD;
  total_ms double precision := 0;
  q_count int := 0;
  t0 timestamptz;
  recall_val double precision;
BEGIN
  CREATE TEMP TABLE pgv_hnsw_raw (qno int, id bigint);

  FOR q IN SELECT qno, embedding FROM bench_queries ORDER BY qno LOOP
    q_count := q_count + 1;
    PERFORM set_config('enable_indexscan', 'on', true);
    PERFORM set_config('enable_bitmapscan', 'on', true);
    PERFORM set_config('enable_seqscan', 'off', true);
    PERFORM set_config('hnsw.ef_search', '512', true);

    t0 := clock_timestamp();
    INSERT INTO pgv_hnsw_raw
    SELECT q.qno, id
    FROM (
      SELECT id FROM items ORDER BY embedding <-> q.embedding LIMIT 10
    ) s;
    total_ms := total_ms + extract(epoch FROM clock_timestamp() - t0) * 1000;
  END LOOP;

  SELECT avg(coalesce(h.hit_count, 0) / 10.0) INTO recall_val
  FROM bench_queries b
  LEFT JOIN (
    SELECT e.qno, count(*) AS hit_count
    FROM expected_hnsw e
    JOIN pgv_hnsw_raw p USING (qno, id)
    GROUP BY e.qno
  ) h ON h.qno = b.qno;

  INSERT INTO bench_results VALUES
    ('pgvector', 'hnsw', total_ms, total_ms / q_count, recall_val);
END
$$;

DO $$
DECLARE
  t0 timestamptz;
  total_ms double precision;
  q_count int;
  recall_val double precision;
BEGIN
  SELECT count(*) INTO q_count FROM bench_queries;

  t0 := clock_timestamp();
  CREATE TEMP TABLE pff_hnsw_raw AS
  SELECT query_no AS qno, id
  FROM pg_retrieval_engine_index_search_batch(
    'faiss_hnsw',
    (SELECT array_agg(embedding ORDER BY qno)::vector[] FROM bench_queries),
    10,
    '{"ef_search":128}'::jsonb
  );
  total_ms := extract(epoch FROM clock_timestamp() - t0) * 1000;

  SELECT avg(coalesce(h.hit_count, 0) / 10.0) INTO recall_val
  FROM bench_queries b
  LEFT JOIN (
    SELECT e.qno, count(*) AS hit_count
    FROM expected_hnsw e
    JOIN pff_hnsw_raw p USING (qno, id)
    GROUP BY e.qno
  ) h ON h.qno = b.qno;

  INSERT INTO bench_results VALUES
    ('pg_retrieval_engine_batch', 'hnsw', total_ms, total_ms / q_count, recall_val);
END
$$;

DROP INDEX items_hnsw;
CREATE INDEX items_ivf ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 256);
ANALYZE items;

SELECT pg_retrieval_engine_index_create('faiss_ivf', 128, 'l2', 'ivfflat', '{"nlist":256,"nprobe":16}'::jsonb, 'cpu');
SELECT pg_retrieval_engine_index_train('faiss_ivf', (SELECT array_agg(embedding ORDER BY id) FROM items));
SELECT pg_retrieval_engine_index_add(
  'faiss_ivf',
  (SELECT array_agg(id ORDER BY id) FROM items),
  (SELECT array_agg(embedding ORDER BY id) FROM items)
);

SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;
CREATE TEMP TABLE expected_ivf AS
SELECT q.qno, e.id
FROM bench_queries q
CROSS JOIN LATERAL (
  SELECT id FROM items ORDER BY embedding <-> q.embedding LIMIT 10
) e;
RESET enable_indexscan;
RESET enable_bitmapscan;
RESET enable_seqscan;

DO $$
DECLARE
  q RECORD;
  total_ms double precision := 0;
  q_count int := 0;
  t0 timestamptz;
  recall_val double precision;
BEGIN
  CREATE TEMP TABLE pgv_ivf_raw (qno int, id bigint);

  FOR q IN SELECT qno, embedding FROM bench_queries ORDER BY qno LOOP
    q_count := q_count + 1;
    PERFORM set_config('enable_indexscan', 'on', true);
    PERFORM set_config('enable_bitmapscan', 'on', true);
    PERFORM set_config('enable_seqscan', 'off', true);
    PERFORM set_config('ivfflat.probes', '16', true);

    t0 := clock_timestamp();
    INSERT INTO pgv_ivf_raw
    SELECT q.qno, id
    FROM (
      SELECT id FROM items ORDER BY embedding <-> q.embedding LIMIT 10
    ) s;
    total_ms := total_ms + extract(epoch FROM clock_timestamp() - t0) * 1000;
  END LOOP;

  SELECT avg(coalesce(h.hit_count, 0) / 10.0) INTO recall_val
  FROM bench_queries b
  LEFT JOIN (
    SELECT e.qno, count(*) AS hit_count
    FROM expected_ivf e
    JOIN pgv_ivf_raw p USING (qno, id)
    GROUP BY e.qno
  ) h ON h.qno = b.qno;

  INSERT INTO bench_results VALUES
    ('pgvector', 'ivfflat', total_ms, total_ms / q_count, recall_val);
END
$$;

DO $$
DECLARE
  t0 timestamptz;
  total_ms double precision;
  q_count int;
  recall_val double precision;
BEGIN
  SELECT count(*) INTO q_count FROM bench_queries;

  t0 := clock_timestamp();
  CREATE TEMP TABLE pff_ivf_raw AS
  SELECT query_no AS qno, id
  FROM pg_retrieval_engine_index_search_batch(
    'faiss_ivf',
    (SELECT array_agg(embedding ORDER BY qno)::vector[] FROM bench_queries),
    10,
    '{"nprobe":16}'::jsonb
  );
  total_ms := extract(epoch FROM clock_timestamp() - t0) * 1000;

  SELECT avg(coalesce(h.hit_count, 0) / 10.0) INTO recall_val
  FROM bench_queries b
  LEFT JOIN (
    SELECT e.qno, count(*) AS hit_count
    FROM expected_ivf e
    JOIN pff_ivf_raw p USING (qno, id)
    GROUP BY e.qno
  ) h ON h.qno = b.qno;

  INSERT INTO bench_results VALUES
    ('pg_retrieval_engine_batch', 'ivfflat', total_ms, total_ms / q_count, recall_val);
END
$$;

SELECT impl, scenario,
       round(total_ms::numeric, 2) AS total_ms,
       round(avg_ms::numeric, 2) AS avg_ms,
       round(recall_at_10::numeric, 4) AS recall_at_10
FROM bench_results
ORDER BY scenario, impl;

WITH p AS (
  SELECT scenario, avg_ms, recall_at_10 FROM bench_results WHERE impl = 'pgvector'
), f AS (
  SELECT scenario, avg_ms, recall_at_10 FROM bench_results WHERE impl = 'pg_retrieval_engine_batch'
)
SELECT p.scenario,
       round((p.avg_ms / f.avg_ms)::numeric, 2) AS speedup_x,
       round(p.recall_at_10::numeric, 4) AS pgvector_recall,
       round(f.recall_at_10::numeric, 4) AS pg_retrieval_engine_recall
FROM p JOIN f USING (scenario)
ORDER BY p.scenario;
