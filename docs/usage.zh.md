# pg_retrieval_engine v0.2 使用文档

## 1. 前置依赖与安装

### 1.1 安装 pgvector

```bash
cd contrib/pgvector
make
make install
```

### 1.2 安装 FAISS CPU（v1.14.1）

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

### 1.3 构建 pg_retrieval_engine（CPU）

```bash
cd contrib/pg_retrieval_engine
make \
  PG_CPPFLAGS="-I$HOME/faiss-install/include -I/usr/local/opt/libomp/include -std=c++17" \
  SHLIB_LINK="-L$HOME/faiss-install/lib -lfaiss -L/usr/local/opt/libomp/lib -lomp -framework Accelerate -lc++ -lc++abi -bundle_loader $(pg_config --bindir)/postgres"
make install
```

### 1.4 GPU 构建（可选）

```bash
cd contrib/pg_retrieval_engine
make USE_FAISS_GPU=1 FAISS_GPU_LIBS="-lfaiss -lcudart -lcublas"
make install
```

## 2. 启用扩展

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_retrieval_engine;
```

## 3. 快速开始

### 3.1 创建与写入

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

### 3.2 单查询

```sql
SELECT *
FROM pg_retrieval_engine_index_search(
  'docs_hnsw',
  '[0.1,0.2,0.3]'::vector,
  10,
  '{"ef_search":128}'::jsonb
);
```

### 3.3 批查询（优化路径）

```sql
SELECT *
FROM pg_retrieval_engine_index_search_batch(
  'docs_hnsw',
  ARRAY['[0.1,0.2,0.3]'::vector, '[0.0,0.5,0.5]'::vector]::vector[],
  5,
  '{"batch_size":256}'::jsonb
);
```

## 4. 新能力使用示例

### 4.1 可观测性

```sql
SELECT pg_retrieval_engine_index_stats('docs_hnsw');
SELECT pg_retrieval_engine_metrics_reset('docs_hnsw');
```

### 4.2 混合检索（ANN + 业务过滤）

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

### 4.3 自动调参

```sql
SELECT pg_retrieval_engine_index_autotune(
  'docs_hnsw',
  'balanced',
  '{"target_recall":0.97,"min_batch_size":64,"max_batch_size":2048}'::jsonb
);
```

### 4.4 批量混合检索

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

## 5. 持久化

```sql
SELECT pg_retrieval_engine_index_save('docs_hnsw', '/tmp/docs_hnsw.faiss');
SELECT pg_retrieval_engine_index_drop('docs_hnsw');
SELECT pg_retrieval_engine_index_load('docs_hnsw', '/tmp/docs_hnsw.faiss', 'cpu');
```

## 6. 测试

```bash
make installcheck
prove -I ./test/perl test/t/010_recall.pl
```

重性能测试：

```bash
pg_retrieval_engine_RUN_PERF=1 \
pg_retrieval_engine_PERF_ROWS=1000000 \
pg_retrieval_engine_PERF_DIM=768 \
pg_retrieval_engine_PERF_QUERIES=100 \
prove -I ./test/perl test/t/020_perf_cpu_vs_pgvector.pl
```

## 7. 推荐阅读

- API 细节：`docs/api.zh.md`
- 设计文档：`docs/design.zh.md`
- 英文文档：`README.md` / `docs/api.md` / `docs/design.md`
