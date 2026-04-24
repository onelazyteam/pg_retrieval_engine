# pg_retrieval_engine

FAISS-based vector search extension for PostgreSQL (v0.2).

中文文档请见 [README.zh.md](README.zh.md).

## Highlights

- Function-oriented API built around `vector` / `vector[]`
- Index types: `hnsw`, `ivfflat`, `ivfpq`
- Metrics: `l2`, `ip`, `cosine`
- CPU path + optional GPU path (`USE_FAISS_GPU=1`)
- P0 observability: built-in runtime counters and timings
- P1 hybrid retrieval: `ANN + ID allow-list` for single and batch search
- P1 auto tuning: `latency/balanced/recall` modes
- P1 batch query optimization: chunked execution (`batch_size`) + candidate depth (`candidate_k`)
- Versioned extension SQL (`pg_retrieval_engine--0.2.0.sql`)
- Regression + TAP tests (correctness, recall, performance)

## Repository Layout (Current Phase)

```text
pg-retrieval-engine/
├── README.md
├── README.zh.md
├── docs/
│   ├── api*.md / design*.md / usage*.md
│   ├── architecture.md            # new scaffold
│   ├── benchmark-methodology.md   # new scaffold
│   ├── evaluation-protocol.md     # new scaffold
│   ├── observability.md           # new scaffold
│   └── tradeoffs.md               # new scaffold
├── src/
│   ├── faiss_in_pg/               # implemented
│   │   ├── faiss_engine.cpp
│   │   └── faiss_engine.hpp
│   ├── disk_graph/                # scaffold only
│   ├── fts_rerank/                # scaffold only
│   └── rrf_sql/                   # scaffold only
├── sql/
│   ├── pg_retrieval_engine--*.sql # active extension SQL
│   ├── schema.sql                 # scaffold
│   ├── indexes.sql                # scaffold
│   ├── hybrid_search.sql          # scaffold
│   └── eval_queries.sql           # scaffold
├── bench/
│   ├── run_bench.py               # scaffold
│   ├── run_ablation.py            # scaffold
│   ├── configs/
│   └── results/
├── evals/
│   ├── queries.jsonl              # sample scaffold
│   ├── qrels.tsv                  # sample scaffold
│   ├── metrics.py                 # scaffold
│   └── run_eval.py                # scaffold
├── demo/
│   ├── cli_demo.py                # scaffold
│   └── screenshots/
└── .github/workflows/
```

## Baseline Versions

- PostgreSQL: 18.3
- pgvector: 0.8.2 (`EXTVERSION = 0.8.2` in `contrib/pgvector/Makefile`)
- FAISS: 1.14.1 (CPU build from source)

## Performance Comparison (pg_retrieval_engine vs pgvector)

### Acceptance Targets

Benchmarks are evaluated under the same recall constraint (`Recall@10 >= 95%`), same dataset, and same query set.

| Scenario | Baseline (pgvector) | pg_retrieval_engine Target | Speedup Target |
|---|---:|---:|---:|
| CPU ANN (HNSW / IVFFlat) | 1.0x | >= 5.0x | >= 5x |
| GPU ANN (FAISS GPU path) | 1.0x | >= 10.0x | >= 10x |

### Measured CPU Results On This Machine (2026-04-14, Batch Query Path)

Method: same dataset, same query set, same recall constraint (`Recall@10 >= 95%`), baseline = pgvector 0.8.2.  
Scale: `20,000 x 128`, `29` queries, `k=10`.  
pg_retrieval_engine uses `pg_retrieval_engine_index_search_batch`; pgvector runs per-query loops.

| Scenario | pgvector avg_ms | pg_retrieval_engine(batch) avg_ms | Speedup | pgvector Recall@10 | pg_retrieval_engine Recall@10 |
|---|---:|---:|---:|---:|---:|
| HNSW | 1.13 | 0.10 | 11.32x | 0.9552 | 1.0000 |
| IVFFlat | 0.76 | 0.07 | 10.31x | 1.0000 | 1.0000 |

Notes:
- These are one-run CPU numbers collected on this machine and checked into README for traceability.
- Parameters in this run: pgvector `hnsw.ef_search=512`, `ivfflat.probes=16`; pg_retrieval_engine `ef_search=128`, `nprobe=16`.
- Reproduction SQL: `contrib/pg_retrieval_engine/test/bench/bench_cpu_batch_sample.sql`.

### TAP Heavy Benchmarks (Acceptance)

| Script | Focus | Default Heavy Scale |
|---|---|---|
| `test/t/020_perf_cpu_vs_pgvector.pl` | CPU vs pgvector (`hnsw`, `ivfflat`) | `1M x 768`, `100` queries |
| `test/t/030_perf_gpu_vs_pgvector.pl` | GPU vs pgvector (`hnsw`) | `1M x 768`, `100` queries |

## Install FAISS (CPU)

```bash
# Dependencies (macOS)
brew install cmake libomp

# Build/install FAISS 1.14.1 (CPU)
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

For Linux, remove the macOS OpenMP-specific flags and keep `FAISS_ENABLE_GPU=OFF`.

## Quick Build

```bash
cd contrib/pg_retrieval_engine
make
make install
```

If FAISS is installed under `$HOME/faiss-install` (not a default system path), use:

```bash
cd contrib/pg_retrieval_engine
make \
  PG_CPPFLAGS="-I$HOME/faiss-install/include -I/usr/local/opt/libomp/include -std=c++17" \
  SHLIB_LINK="-L$HOME/faiss-install/lib -lfaiss -L/usr/local/opt/libomp/lib -lomp -framework Accelerate -lc++ -lc++abi -bundle_loader $(pg_config --bindir)/postgres"
make install
```

GPU build example:

```bash
cd contrib/pg_retrieval_engine
make USE_FAISS_GPU=1 FAISS_GPU_LIBS="-lfaiss -lcudart -lcublas"
make install
```

## Test & Benchmark

```bash
# Regression
make installcheck

# Recall TAP
prove -I ./test/perl test/t/010_recall.pl

# CPU perf TAP (heavy)
pg_retrieval_engine_RUN_PERF=1 \
pg_retrieval_engine_PERF_ROWS=1000000 \
pg_retrieval_engine_PERF_DIM=768 \
pg_retrieval_engine_PERF_QUERIES=100 \
prove -I ./test/perl test/t/020_perf_cpu_vs_pgvector.pl

# GPU perf TAP (heavy)
pg_retrieval_engine_RUN_PERF_GPU=1 \
pg_retrieval_engine_PERF_GPU_ROWS=1000000 \
pg_retrieval_engine_PERF_GPU_DIM=768 \
pg_retrieval_engine_PERF_GPU_QUERIES=100 \
prove -I ./test/perl test/t/030_perf_gpu_vs_pgvector.pl
```

## API

- `pg_retrieval_engine_index_create(name, dim, metric, index_type, options, device)`
- `pg_retrieval_engine_index_train(name, training_vectors)`
- `pg_retrieval_engine_index_add(name, ids, vectors)`
- `pg_retrieval_engine_index_search(name, query, k, search_params)`
- `pg_retrieval_engine_index_search_batch(name, queries, k, search_params)`
- `pg_retrieval_engine_index_search_filtered(name, query, k, filter_ids, search_params)`
- `pg_retrieval_engine_index_search_batch_filtered(name, queries, k, filter_ids, search_params)`
- `pg_retrieval_engine_index_autotune(name, mode, options)`
- `pg_retrieval_engine_metrics_reset(name default null)`
- `pg_retrieval_engine_index_save(name, path)` / `pg_retrieval_engine_index_load(name, path, device)`
- `pg_retrieval_engine_index_stats(name)`
- `pg_retrieval_engine_index_drop(name)` / `pg_retrieval_engine_reset()`

See the API reference for full parameter semantics and error behavior.

## C++ Style

- `src/faiss_in_pg/faiss_engine.cpp` and `src/faiss_in_pg/faiss_engine.hpp` are formatted to Google C++ style.
- Style config: `contrib/pg_retrieval_engine/.clang-format`.
- Format locally: `make -C contrib/pg_retrieval_engine format`
- Check formatting: `make -C contrib/pg_retrieval_engine format-check`

## GitHub Project Website

- Site source directory: `contrib/pg_retrieval_engine/site`
- Includes language toggle (Chinese/English), project purpose, usage, performance, and doc links.
- GitHub Pages workflow: `.github/workflows/pages.yml`

Enable it:
1. Push to `main` or `master`.
2. In repository Settings -> Pages, set Source to **GitHub Actions**.
3. Run `pg_retrieval_engine-pages` workflow and use the generated Pages URL (`https://onelazyteam.github.io/pg_retrieval_engine/`).
4. Set this URL in repository **About -> Website** to get one-click navigation like `pg_llm`.

## Docs

- API Reference: [docs/api.md](docs/api.md)
- Design: [docs/design.md](docs/design.md)
- Usage: [docs/usage.md](docs/usage.md)
