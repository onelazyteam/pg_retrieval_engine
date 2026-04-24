# pg_retrieval_engine

基于 FAISS 的 PostgreSQL 向量检索扩展（v0.2）。

English version: [README.md](README.md)

## 主要特性

- 基于 `vector` / `vector[]` 的函数式 API
- 索引类型：`hnsw`、`ivfflat`、`ivfpq`
- 距离度量：`l2`、`ip`、`cosine`
- 支持 CPU 路径 + 可选 GPU 路径（`USE_FAISS_GPU=1`）
- P0 可观测性：内置 runtime 计数与耗时统计
- P1 混合检索：`ANN + ID allow-list`（单查/批查）
- P1 自动调参：`latency/balanced/recall` 模式
- P1 批量查询优化：分块执行（`batch_size`）+ 候选深度（`candidate_k`）
- 版本化扩展脚本（`pg_retrieval_engine--0.2.0.sql`）
- 回归测试 + TAP（正确性、召回、性能）

## 目录结构（当前阶段）

```text
pg-retrieval-engine/
├── README.md
├── README.zh.md
├── docs/
│   ├── api*.md / design*.md / usage*.md
│   ├── architecture.md                 # 新增：总体架构草案
│   ├── benchmark-methodology.md        # 新增：基准方法草案
│   ├── evaluation-protocol.md          # 新增：评测协议草案
│   ├── observability.md                # 新增：可观测性草案
│   └── tradeoffs.md                    # 新增：权衡分析草案
├── src/
│   ├── faiss_in_pg/                    # 已实现
│   │   ├── faiss_engine.cpp
│   │   └── faiss_engine.hpp
│   ├── disk_graph/                     # 目录占位，暂未实现
│   ├── fts_rerank/                     # 目录占位，暂未实现
│   └── rrf_sql/                        # 目录占位，暂未实现
├── sql/
│   ├── pg_retrieval_engine--*.sql      # 扩展安装脚本（已使用）
│   ├── schema.sql                      # 占位
│   ├── indexes.sql                     # 占位
│   ├── hybrid_search.sql               # 占位
│   └── eval_queries.sql                # 占位
├── bench/
│   ├── run_bench.py                    # 占位
│   ├── run_ablation.py                 # 占位
│   ├── configs/
│   └── results/
├── evals/
│   ├── queries.jsonl                   # 占位样例
│   ├── qrels.tsv                       # 占位样例
│   ├── metrics.py                      # 占位
│   └── run_eval.py                     # 占位
├── demo/
│   ├── cli_demo.py                     # 占位
│   └── screenshots/
└── .github/workflows/
```

## 对比基线版本

- PostgreSQL：18.3
- pgvector：0.8.2（`contrib/pgvector/Makefile` 中 `EXTVERSION = 0.8.2`）
- FAISS：1.14.1（CPU，源码安装）

## 性能对比（pg_retrieval_engine vs pgvector）

### 验收目标数据

对比条件：同召回约束（`Recall@10 >= 95%`）、同数据集、同查询集。

| 场景 | 基线（pgvector） | pg_retrieval_engine 目标 | 提升目标 |
|---|---:|---:|---:|
| CPU ANN（HNSW / IVFFlat） | 1.0x | >= 5.0x | >= 5x |
| GPU ANN（FAISS GPU 路径） | 1.0x | >= 10.0x | >= 10x |

### 本机 CPU 实测（2026-04-14，批量查询路径）

测试口径：同数据集、同查询集、同召回约束（`Recall@10 >= 95%`），基线为 pgvector 0.8.2。  
数据规模：`20,000 x 128`，`29` queries，`k=10`。  
pg_retrieval_engine 使用 `pg_retrieval_engine_index_search_batch`，pgvector 使用逐条查询循环。

| 场景 | pgvector avg_ms | pg_retrieval_engine(batch) avg_ms | Speedup | pgvector Recall@10 | pg_retrieval_engine Recall@10 |
|---|---:|---:|---:|---:|---:|
| HNSW | 1.13 | 0.10 | 11.32x | 0.9552 | 1.0000 |
| IVFFlat | 0.76 | 0.07 | 10.31x | 1.0000 | 1.0000 |

说明：
- 以上是你机器上的一次 CPU 实测数据，用于 README 固化对比结果。
- 该实测使用参数：pgvector `hnsw.ef_search=512`、`ivfflat.probes=16`；pg_retrieval_engine `ef_search=128`、`nprobe=16`。
- 复现脚本：`contrib/pg_retrieval_engine/test/bench/bench_cpu_batch_sample.sql`。

### TAP 重基准脚本（验收用）

| 脚本 | 对比内容 | 默认重载规模 |
|---|---|---|
| `test/t/020_perf_cpu_vs_pgvector.pl` | CPU 对比 pgvector（`hnsw`、`ivfflat`） | `1M x 768`，`100` queries |
| `test/t/030_perf_gpu_vs_pgvector.pl` | GPU 对比 pgvector（`hnsw`） | `1M x 768`，`100` queries |

## 安装 FAISS（CPU）

```bash
# 依赖（macOS）
brew install cmake libomp

# 编译安装 FAISS 1.14.1（CPU）
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

如果使用 Linux，可去掉上面的 OpenMP macOS 专用参数，保留 `FAISS_ENABLE_GPU=OFF` 即可。

## 快速构建

```bash
cd contrib/pg_retrieval_engine
make
make install
```

如果 FAISS 安装在 `$HOME/faiss-install`（非系统默认路径），可使用：

```bash
cd contrib/pg_retrieval_engine
make \
  PG_CPPFLAGS="-I$HOME/faiss-install/include -I/usr/local/opt/libomp/include -std=c++17" \
  SHLIB_LINK="-L$HOME/faiss-install/lib -lfaiss -L/usr/local/opt/libomp/lib -lomp -framework Accelerate -lc++ -lc++abi -bundle_loader $(pg_config --bindir)/postgres"
make install
```

GPU 构建示例：

```bash
cd contrib/pg_retrieval_engine
make USE_FAISS_GPU=1 FAISS_GPU_LIBS="-lfaiss -lcudart -lcublas"
make install
```

## 测试与基准

```bash
# 回归测试
make installcheck

# Recall TAP
prove -I ./test/perl test/t/010_recall.pl

# CPU 性能 TAP（重负载）
pg_retrieval_engine_RUN_PERF=1 \
pg_retrieval_engine_PERF_ROWS=1000000 \
pg_retrieval_engine_PERF_DIM=768 \
pg_retrieval_engine_PERF_QUERIES=100 \
prove -I ./test/perl test/t/020_perf_cpu_vs_pgvector.pl

# GPU 性能 TAP（重负载）
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

详细参数与错误语义请见 API 参考文档。

## C++ 代码规范

- `src/faiss_in_pg/faiss_engine.cpp` / `src/faiss_in_pg/faiss_engine.hpp` 按 Google C++ 风格格式化。
- 规范配置文件：`contrib/pg_retrieval_engine/.clang-format`。
- 本地格式化：`make -C contrib/pg_retrieval_engine format`
- 本地检查：`make -C contrib/pg_retrieval_engine format-check`

## GitHub 项目网站

- 站点源码目录：`contrib/pg_retrieval_engine/site`
- 支持中英文切换，覆盖：项目目的、使用示例、性能数据、文档入口。
- GitHub Pages 工作流：`.github/workflows/pages.yml`

启用方式：
1. 推送到 GitHub 仓库（`main` 或 `master`）。
2. 在仓库 Settings -> Pages 中将 Source 设为 **GitHub Actions**。
3. 触发 `pg_retrieval_engine-pages` workflow 后即可获得站点 URL（`https://onelazyteam.github.io/pg_retrieval_engine/`）。
4. 在仓库首页 **About -> Website** 中填入该 URL，即可像 `pg_llm` 一样直接点击跳转。

## 文档

- API 参考：[docs/api.zh.md](docs/api.zh.md)
- 设计文档：[docs/design.zh.md](docs/design.zh.md)
- 使用文档：[docs/usage.zh.md](docs/usage.zh.md)
