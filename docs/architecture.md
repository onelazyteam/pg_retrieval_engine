# Architecture (Planned)

This repository is evolving into a PostgreSQL-native hybrid retrieval engine.

Current implemented module:
- `src/faiss_in_pg`: FAISS-based ANN inside PostgreSQL (available now)

Planned modules (directory scaffold only in current phase):
- `src/disk_graph`: Disk-based HNSW for large-scale vectors
- `src/fts_rerank`: Sparse/BM25 rerank path
- `src/rrf_sql`: RRF fusion logic in PostgreSQL

Integration target:
- Combine `pgvector` vector retrieval with PostgreSQL full-text search (`tsvector`) and fuse rankings via RRF.
