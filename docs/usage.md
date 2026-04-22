# pg_retrieval_engine v0.2 Usage Guide

## 1. Prerequisites and Build

### 1.1 Build/install pgvector

```bash
cd contrib/pgvector
make
make install
```

### 1.2 Build/install FAISS CPU (v1.14.1)

```bash
brew install cmake libomp

git clone --branch v1.14.1 https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/faiss-install \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib
cmake --build build -j
cmake --install build
```

### 1.3 Build/install pg_retrieval_engine (CPU)

```bash
cd contrib/pg_retrieval_engine
make \
  PG_CPPFLAGS="-I$HOME/faiss-install/include -I/usr/local/opt/libomp/include -std=c++17" \
  SHLIB_LINK="-L$HOME/faiss-install/lib -lfaiss -L/usr/local/opt/libomp/lib -lomp -framework Accelerate -lc++ -lc++abi -bundle_loader $(pg_config --bindir)/postgres"
make install
```

### 1.4 GPU build (optional)

```bash
cd contrib/pg_retrieval_engine
make USE_FAISS_GPU=1 FAISS_GPU_LIBS="-lfaiss -lcudart -lcublas"
make install
```

## 2. Enable extension

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_retrieval_engine;
```

## 3. Quick Start

### 3.1 Create and insert

```sql
SELECT pg_retrieval_engine_index_create(
  'docs_hnsw', 768, 'cosine', 'hnsw',
  '{"m":32,"ef_construction":200,"ef_search":64}'::jsonb,
  'cpu'
);

SELECT pg_retrieval_engine_index_add(
  'docs_hnsw',
  ARRAY[1,2,3]::bigint[],
  ARRAY[
    '[0.1,0.2,0.3]'::vector,
    '[0.3,0.2,0.1]'::vector,
    '[0.0,0.5,0.5]'::vector
  ]::vector[]
);
```

### 3.2 Single-query search

```sql
SELECT *
FROM pg_retrieval_engine_index_search(
  'docs_hnsw',
  '[0.1,0.2,0.3]'::vector,
  10,
  '{"ef_search":128}'::jsonb
);
```

### 3.3 Batch search (optimized path)

```sql
SELECT *
FROM pg_retrieval_engine_index_search_batch(
  'docs_hnsw',
  ARRAY['[0.1,0.2,0.3]'::vector, '[0.0,0.5,0.5]'::vector]::vector[],
  5,
  '{"batch_size":256}'::jsonb
);
```

## 4. New Capabilities

### 4.1 Observability

```sql
SELECT pg_retrieval_engine_index_stats('docs_hnsw');
SELECT pg_retrieval_engine_metrics_reset('docs_hnsw');
```

### 4.2 Hybrid retrieval (ANN + business filtering)

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
  'docs_hnsw',
  '[0.1,0.2,0.3]'::vector,
  20,
  (SELECT ids FROM allow_list),
  '{"candidate_k":200,"ef_search":128}'::jsonb
);
```

### 4.3 Auto tuning

```sql
SELECT pg_retrieval_engine_index_autotune(
  'docs_hnsw',
  'balanced',
  '{"target_recall":0.97,"min_batch_size":64,"max_batch_size":2048}'::jsonb
);
```

### 4.4 Batch hybrid retrieval

```sql
SELECT *
FROM pg_retrieval_engine_index_search_batch_filtered(
  'docs_hnsw',
  ARRAY['[0.1,0.2,0.3]'::vector, '[0.0,0.5,0.5]'::vector]::vector[],
  10,
  ARRAY[1,2,3,4,5]::bigint[],
  '{"candidate_k":100,"batch_size":128}'::jsonb
);
```

## 5. Persistence

```sql
SELECT pg_retrieval_engine_index_save('docs_hnsw', '/tmp/docs_hnsw.faiss');
SELECT pg_retrieval_engine_index_drop('docs_hnsw');
SELECT pg_retrieval_engine_index_load('docs_hnsw', '/tmp/docs_hnsw.faiss', 'cpu');
```

## 6. Testing

```bash
make installcheck
prove -I ./test/perl test/t/010_recall.pl
```

Heavy benchmark:

```bash
pg_retrieval_engine_RUN_PERF=1 \
pg_retrieval_engine_PERF_ROWS=1000000 \
pg_retrieval_engine_PERF_DIM=768 \
pg_retrieval_engine_PERF_QUERIES=100 \
prove -I ./test/perl test/t/020_perf_cpu_vs_pgvector.pl
```

## 7. Read Next

- API details: `docs/api.md`
- design details: `docs/design.md`
- Chinese docs: `README.zh.md` / `docs/api.zh.md` / `docs/design.zh.md`
