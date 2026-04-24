extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/hsearch.h"
#include "utils/json.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/tuplestore.h"
}

#include "faiss_engine.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#ifdef USE_FAISS_GPU
#include <faiss/gpu/GpuCloner.h>
#endif

extern "C" {
PG_MODULE_MAGIC;
}

extern "C" {
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_create);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_train);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_add);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_search);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_search_batch);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_search_filtered);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_search_batch_filtered);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_save);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_load);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_autotune);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_metrics_reset);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_stats);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_index_drop);
PG_FUNCTION_INFO_V1(pg_retrieval_engine_reset);
}

typedef struct PgVector {
  int32 vl_len_;
  int16 dim;
  int16 unused;
  float x[FLEXIBLE_ARRAY_MEMBER];
} PgVector;

static HTAB* pg_retrieval_engine_registry = NULL;

static inline void ensure_registry(void) {
  if (pg_retrieval_engine_registry == NULL) {
    HASHCTL ctl;

    memset(&ctl, 0, sizeof(ctl));
    ctl.keysize = pg_retrieval_engine_MAX_INDEX_NAME;
    ctl.entrysize = sizeof(PgRetrievalEngineIndexEntry);
    ctl.hcxt = TopMemoryContext;

    pg_retrieval_engine_registry =
        hash_create("pg_retrieval_engine index registry", 128, &ctl, HASH_ELEM | HASH_STRINGS | HASH_CONTEXT);

    if (pg_retrieval_engine_registry == NULL)
      ereport(ERROR,
              (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("failed to create pg_retrieval_engine registry")));
  }
}

static inline const char* metric_name(int metric) {
  switch (metric) {
    case pg_retrieval_engine_METRIC_L2:
      return "l2";
    case pg_retrieval_engine_METRIC_IP:
      return "ip";
    case pg_retrieval_engine_METRIC_COSINE:
      return "cosine";
    default:
      return "unknown";
  }
}

static inline const char* index_type_name(int index_type) {
  switch (index_type) {
    case pg_retrieval_engine_INDEX_HNSW:
      return "hnsw";
    case pg_retrieval_engine_INDEX_IVF_FLAT:
      return "ivfflat";
    case pg_retrieval_engine_INDEX_IVF_PQ:
      return "ivfpq";
    default:
      return "unknown";
  }
}

static inline const char* device_name(int device) {
  return device == pg_retrieval_engine_DEVICE_GPU ? "gpu" : "cpu";
}

static inline const char* autotune_mode_name(int mode) {
  switch (mode) {
    case pg_retrieval_engine_AUTOTUNE_LATENCY:
      return "latency";
    case pg_retrieval_engine_AUTOTUNE_RECALL:
      return "recall";
    case pg_retrieval_engine_AUTOTUNE_BALANCED:
    default:
      return "balanced";
  }
}

static inline void reset_runtime_stats(PgRetrievalEngineIndexEntry* entry, bool reset_tuning) {
  entry->train_calls = 0;
  entry->add_calls = 0;
  entry->add_vectors_total = 0;
  entry->search_single_calls = 0;
  entry->search_batch_calls = 0;
  entry->search_filtered_calls = 0;
  entry->search_query_total = 0;
  entry->search_result_total = 0;
  entry->save_calls = 0;
  entry->load_calls = 0;
  entry->autotune_calls = 0;
  entry->error_calls = 0;
  entry->search_single_ms_total = 0.0;
  entry->search_batch_ms_total = 0.0;
  entry->search_filtered_ms_total = 0.0;
  if (reset_tuning) {
    entry->last_candidate_k = 0;
    entry->last_batch_size = 0;
    entry->last_autotune_mode = pg_retrieval_engine_AUTOTUNE_BALANCED;
    entry->preferred_batch_size = 256;
  }
}

static inline double elapsed_ms(const instr_time& start_time, const instr_time& end_time) {
  instr_time delta = end_time;

  INSTR_TIME_SUBTRACT(delta, start_time);
  return INSTR_TIME_GET_MILLISEC(delta);
}

static inline void record_error(PgRetrievalEngineIndexEntry* entry) {
  if (entry != NULL) entry->error_calls++;
}

static inline void initialize_entry_defaults(PgRetrievalEngineIndexEntry* entry) {
  reset_runtime_stats(entry, true);
  if (entry->device == pg_retrieval_engine_DEVICE_GPU) entry->preferred_batch_size = 1024;
}

static inline faiss::MetricType to_faiss_metric(int metric) {
  if (metric == pg_retrieval_engine_METRIC_L2) return faiss::METRIC_L2;

  return faiss::METRIC_INNER_PRODUCT;
}

static inline PgRetrievalEngineIndexEntry* lookup_entry(const char* name) {
  if (pg_retrieval_engine_registry == NULL) return NULL;

  return (PgRetrievalEngineIndexEntry*)hash_search(pg_retrieval_engine_registry, name, HASH_FIND, NULL);
}

static inline faiss::Index* unwrap_idmap(faiss::Index* index) {
  faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(index);

  if (idmap != NULL) return idmap->index;

  return index;
}

static inline void normalize_one(float* vec, int dim) {
  float norm = 0.0f;

  for (int i = 0; i < dim; i++) norm += vec[i] * vec[i];

  norm = sqrtf(norm);

  if (norm > 0.0f) {
    for (int i = 0; i < dim; i++) vec[i] /= norm;
  }
}

static inline void normalize_many(float* vecs, int64 n, int dim) {
  for (int64 i = 0; i < n; i++) normalize_one(&vecs[i * dim], dim);
}

static inline faiss::Index* active_index(PgRetrievalEngineIndexEntry* entry) {
#ifdef USE_FAISS_GPU
  if (entry->device == pg_retrieval_engine_DEVICE_GPU) {
    if (entry->gpu_index == NULL)
      ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                      errmsg("GPU index for \"%s\" is not initialized", entry->name)));

    return entry->gpu_index;
  }
#endif

  if (entry->cpu_index == NULL)
    ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                    errmsg("CPU index for \"%s\" is not initialized", entry->name)));

  return entry->cpu_index;
}

#ifdef USE_FAISS_GPU
static void rebuild_gpu_index(PgRetrievalEngineIndexEntry* entry) {
  if (entry->device != pg_retrieval_engine_DEVICE_GPU) return;

  if (entry->gpu_resources == NULL) entry->gpu_resources = new faiss::gpu::StandardGpuResources();

  if (entry->gpu_index != NULL) {
    delete entry->gpu_index;
    entry->gpu_index = NULL;
  }

  entry->gpu_index =
      faiss::gpu::index_cpu_to_gpu(entry->gpu_resources, entry->gpu_device, entry->cpu_index);
}

static void sync_gpu_to_cpu(PgRetrievalEngineIndexEntry* entry) {
  if (entry->device != pg_retrieval_engine_DEVICE_GPU || entry->gpu_index == NULL) return;

  faiss::Index* new_cpu = faiss::gpu::index_gpu_to_cpu(entry->gpu_index);

  if (entry->cpu_index != NULL) delete entry->cpu_index;

  entry->cpu_index = new_cpu;
}
#endif

static inline void free_entry_resources(PgRetrievalEngineIndexEntry* entry) {
#ifdef USE_FAISS_GPU
  if (entry->gpu_index != NULL) {
    delete entry->gpu_index;
    entry->gpu_index = NULL;
  }

  if (entry->gpu_resources != NULL) {
    delete entry->gpu_resources;
    entry->gpu_resources = NULL;
  }
#endif

  if (entry->cpu_index != NULL) {
    delete entry->cpu_index;
    entry->cpu_index = NULL;
  }
}

static JsonbValue* jsonb_find_key(Jsonb* json, const char* key) {
  JsonbValue key_value;

  if (json == NULL) return NULL;

  key_value.type = jbvString;
  key_value.val.string.val = const_cast<char*>(key);
  key_value.val.string.len = strlen(key);

  return findJsonbValueFromContainer(&json->root, JB_FOBJECT, &key_value);
}

static bool jsonb_get_int32(Jsonb* json, const char* key, int32* out) {
  JsonbValue* value = jsonb_find_key(json, key);

  if (value == NULL) return false;

  if (value->type == jbvNumeric) {
    *out = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(value->val.numeric)));
    return true;
  }

  if (value->type == jbvString) {
    std::string text(value->val.string.val, value->val.string.len);

    try {
      *out = std::stoi(text);
      return true;
    } catch (const std::exception&) {
      ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                      errmsg("invalid integer value for option \"%s\"", key)));
    }
  }

  ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                  errmsg("option \"%s\" must be an integer", key)));

  return false;
}

static bool jsonb_get_float8(Jsonb* json, const char* key, double* out) {
  JsonbValue* value = jsonb_find_key(json, key);

  if (value == NULL) return false;

  if (value->type == jbvNumeric) {
    *out = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(value->val.numeric)));
    return true;
  }

  if (value->type == jbvString) {
    std::string text(value->val.string.val, value->val.string.len);

    try {
      *out = std::stod(text);
      return true;
    } catch (const std::exception&) {
      ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                      errmsg("invalid float value for option \"%s\"", key)));
    }
  }

  ereport(ERROR,
          (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("option \"%s\" must be a float", key)));

  return false;
}

static int32 jsonb_option_int32(Jsonb* json, const char* key, int32 default_value, int32 min_value,
                                int32 max_value) {
  int32 value = default_value;

  if (!jsonb_get_int32(json, key, &value)) return default_value;

  if (value < min_value || value > max_value)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("option \"%s\" out of range (%d..%d): %d", key, min_value, max_value, value)));

  return value;
}

static int parse_autotune_mode(const char* mode) {
  if (pg_strcasecmp(mode, "balanced") == 0) return pg_retrieval_engine_AUTOTUNE_BALANCED;
  if (pg_strcasecmp(mode, "latency") == 0) return pg_retrieval_engine_AUTOTUNE_LATENCY;
  if (pg_strcasecmp(mode, "recall") == 0) return pg_retrieval_engine_AUTOTUNE_RECALL;

  ereport(ERROR,
          (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unknown autotune mode: %s", mode)));
  return pg_retrieval_engine_AUTOTUNE_BALANCED;
}

static inline PgVector* datum_to_pgvector(Datum datum) {
  return (PgVector*)PG_DETOAST_DATUM(datum);
}

static void read_vector_array(ArrayType* arr, int expected_dim, std::vector<float>& out,
                              int64* num_vectors) {
  Datum* elements = NULL;
  bool* nulls = NULL;
  int nelems = 0;

  if (ARR_NDIM(arr) != 1)
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("vector[] argument must be one-dimensional")));

  deconstruct_array(arr, ARR_ELEMTYPE(arr), -1, false, 'i', &elements, &nulls, &nelems);

  if (nelems <= 0)
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("vector[] argument must not be empty")));

  out.resize((size_t)nelems * (size_t)expected_dim);

  for (int i = 0; i < nelems; i++) {
    PgVector* vec;

    if (nulls[i])
      ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                      errmsg("vector[] argument must not contain NULL values")));

    vec = datum_to_pgvector(elements[i]);

    if (vec->dim != expected_dim)
      ereport(ERROR,
              (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
               errmsg("vector dimension mismatch: expected %d, got %d", expected_dim, vec->dim)));

    memcpy(&out[(size_t)i * (size_t)expected_dim], vec->x, sizeof(float) * expected_dim);
  }

  pfree(elements);
  pfree(nulls);

  *num_vectors = nelems;
}

static void read_ids_array(ArrayType* arr, std::vector<faiss::idx_t>& ids) {
  Datum* elements = NULL;
  bool* nulls = NULL;
  int nelems = 0;

  if (ARR_NDIM(arr) != 1)
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("ids argument must be one-dimensional bigint[]")));

  if (ARR_ELEMTYPE(arr) != INT8OID)
    ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("ids argument must be bigint[]")));

  deconstruct_array(arr, INT8OID, 8, true, 'd', &elements, &nulls, &nelems);

  if (nelems <= 0)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("ids argument must not be empty")));

  ids.resize(nelems);

  for (int i = 0; i < nelems; i++) {
    if (nulls[i])
      ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                      errmsg("ids argument must not contain NULL values")));

    ids[i] = (faiss::idx_t)DatumGetInt64(elements[i]);
  }

  pfree(elements);
  pfree(nulls);
}

static void read_optional_ids_array(ArrayType* arr, std::unordered_set<faiss::idx_t>* out,
                                    bool* has_filter) {
  std::vector<faiss::idx_t> ids;

  if (arr == NULL) {
    *has_filter = false;
    return;
  }

  read_ids_array(arr, ids);
  out->reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) out->insert(ids[i]);

  *has_filter = true;
}

typedef struct SearchExecutionOptions {
  int32 candidate_k;
  int32 batch_size;
  int old_ef_search;
  int old_nprobe;
  bool changed_ef_search;
  bool changed_nprobe;
} SearchExecutionOptions;

static void apply_search_params(PgRetrievalEngineIndexEntry* entry, faiss::Index* index, Jsonb* search_params,
                                int32 effective_k, bool widen_candidate,
                                SearchExecutionOptions* options) {
  faiss::Index* base = unwrap_idmap(index);
  int32 candidate_k_default = effective_k;

  if (widen_candidate) {
    candidate_k_default =
        std::min<int64>(std::max<int64>((int64)effective_k * 8, effective_k), index->ntotal);
  }

  options->changed_ef_search = false;
  options->changed_nprobe = false;
  options->old_ef_search = 0;
  options->old_nprobe = 0;
  options->candidate_k =
      jsonb_option_int32(search_params, "candidate_k", candidate_k_default, effective_k, 1000000);
  options->batch_size =
      jsonb_option_int32(search_params, "batch_size", entry->preferred_batch_size, 1, 1000000);

  if (options->candidate_k < effective_k) options->candidate_k = effective_k;
  options->candidate_k = std::min<int64>(options->candidate_k, index->ntotal);
  if (options->batch_size <= 0) options->batch_size = entry->preferred_batch_size;

  if (entry->index_type == pg_retrieval_engine_INDEX_HNSW) {
    faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(base);

    if (hnsw != NULL) {
      int32 ef_search = entry->hnsw_ef_search;

      if (jsonb_get_int32(search_params, "ef_search", &ef_search)) {
        if (ef_search < 1 || ef_search > 1000000)
          ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                          errmsg("ef_search must be in range 1..1000000")));
      }

      options->old_ef_search = hnsw->hnsw.efSearch;
      hnsw->hnsw.efSearch = ef_search;
      options->changed_ef_search = true;
    }
  }

  if (entry->index_type == pg_retrieval_engine_INDEX_IVF_FLAT || entry->index_type == pg_retrieval_engine_INDEX_IVF_PQ) {
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(base);

    if (ivf != NULL) {
      int32 nprobe = entry->ivf_nprobe;

      if (jsonb_get_int32(search_params, "nprobe", &nprobe)) {
        if (nprobe < 1 || nprobe > 1000000)
          ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                          errmsg("nprobe must be in range 1..1000000")));
      }

      options->old_nprobe = ivf->nprobe;
      ivf->nprobe = nprobe;
      options->changed_nprobe = true;
    }
  }
}

static void restore_search_params(faiss::Index* index, const SearchExecutionOptions* options) {
  faiss::Index* base = unwrap_idmap(index);

  if (options->changed_ef_search) {
    faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(base);

    if (hnsw != NULL) hnsw->hnsw.efSearch = options->old_ef_search;
  }

  if (options->changed_nprobe) {
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(base);

    if (ivf != NULL) ivf->nprobe = options->old_nprobe;
  }
}

static void append_search_row(Tuplestorestate* tupstore, TupleDesc tupdesc, int query_no,
                              bool include_query_no, int64 id, float distance) {
  if (include_query_no) {
    Datum values[3];
    bool nulls[3] = {false, false, false};

    values[0] = Int32GetDatum(query_no);
    values[1] = Int64GetDatum(id);
    values[2] = Float4GetDatum(distance);
    tuplestore_putvalues(tupstore, tupdesc, values, nulls);
    return;
  }

  Datum values[2];
  bool nulls[2] = {false, false};

  values[0] = Int64GetDatum(id);
  values[1] = Float4GetDatum(distance);
  tuplestore_putvalues(tupstore, tupdesc, values, nulls);
}

static void materialize_result_begin(FunctionCallInfo fcinfo, Tuplestorestate** tupstore,
                                     TupleDesc* tupdesc, ReturnSetInfo** rsinfo) {
  MemoryContext oldcontext;

  *rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;

  if (*rsinfo == NULL || !IsA(*rsinfo, ReturnSetInfo))
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("set-valued function called in context that cannot accept a set")));

  if (!((*rsinfo)->allowedModes & SFRM_Materialize))
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("materialize mode required")));

  if (get_call_result_type(fcinfo, NULL, tupdesc) != TYPEFUNC_COMPOSITE)
    ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("return type must be a row type")));

  oldcontext = MemoryContextSwitchTo((*rsinfo)->econtext->ecxt_per_query_memory);
  *tupstore = tuplestore_begin_heap(true, false, work_mem);
  MemoryContextSwitchTo(oldcontext);

  BlessTupleDesc(*tupdesc);
}

static void materialize_result_end(ReturnSetInfo* rsinfo, Tuplestorestate* tupstore,
                                   TupleDesc tupdesc) {
  rsinfo->returnMode = SFRM_Materialize;
  rsinfo->setResult = tupstore;
  rsinfo->setDesc = tupdesc;
}

static void write_metadata_file(const PgRetrievalEngineIndexEntry* entry, const char* path) {
  std::ofstream out(std::string(path) + ".meta", std::ios::trunc);

  if (!out.is_open())
    ereport(ERROR, (errcode(ERRCODE_IO_ERROR),
                    errmsg("could not open metadata file \"%s.meta\" for write", path)));

  out << "version=" << pg_retrieval_engine_VERSION << "\n";
  out << "metric=" << metric_name(entry->metric) << "\n";
  out << "index_type=" << index_type_name(entry->index_type) << "\n";
  out << "dim=" << entry->dim << "\n";
  out << "hnsw_m=" << entry->hnsw_m << "\n";
  out << "hnsw_ef_construction=" << entry->hnsw_ef_construction << "\n";
  out << "hnsw_ef_search=" << entry->hnsw_ef_search << "\n";
  out << "ivf_nlist=" << entry->ivf_nlist << "\n";
  out << "ivf_nprobe=" << entry->ivf_nprobe << "\n";
  out << "ivfpq_m=" << entry->ivfpq_m << "\n";
  out << "ivfpq_bits=" << entry->ivfpq_bits << "\n";
  out << "gpu_device=" << entry->gpu_device << "\n";

  out.close();
}

static std::unordered_map<std::string, std::string> read_metadata_file(const char* path) {
  std::unordered_map<std::string, std::string> data;
  std::ifstream in(std::string(path) + ".meta");
  std::string line;

  if (!in.is_open()) return data;

  while (std::getline(in, line)) {
    size_t pos = line.find('=');

    if (pos == std::string::npos) continue;

    data[line.substr(0, pos)] = line.substr(pos + 1);
  }

  return data;
}

static inline int parse_metric(const char* metric) {
  if (pg_strcasecmp(metric, "l2") == 0) return pg_retrieval_engine_METRIC_L2;
  if (pg_strcasecmp(metric, "ip") == 0 || pg_strcasecmp(metric, "inner_product") == 0)
    return pg_retrieval_engine_METRIC_IP;
  if (pg_strcasecmp(metric, "cosine") == 0) return pg_retrieval_engine_METRIC_COSINE;

  ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unknown metric: %s", metric)));

  return pg_retrieval_engine_METRIC_L2;
}

static inline int parse_index_type(const char* index_type) {
  if (pg_strcasecmp(index_type, "hnsw") == 0) return pg_retrieval_engine_INDEX_HNSW;
  if (pg_strcasecmp(index_type, "ivfflat") == 0 || pg_strcasecmp(index_type, "ivf_flat") == 0)
    return pg_retrieval_engine_INDEX_IVF_FLAT;
  if (pg_strcasecmp(index_type, "ivfpq") == 0 || pg_strcasecmp(index_type, "ivf_pq") == 0)
    return pg_retrieval_engine_INDEX_IVF_PQ;

  ereport(ERROR,
          (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unknown index_type: %s", index_type)));

  return pg_retrieval_engine_INDEX_HNSW;
}

static inline int parse_device(const char* device) {
  if (pg_strcasecmp(device, "cpu") == 0) return pg_retrieval_engine_DEVICE_CPU;
  if (pg_strcasecmp(device, "gpu") == 0) return pg_retrieval_engine_DEVICE_GPU;

  ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unknown device: %s", device)));

  return pg_retrieval_engine_DEVICE_CPU;
}

static faiss::Index* build_index(const PgRetrievalEngineIndexEntry* entry) {
  faiss::MetricType metric = to_faiss_metric(entry->metric);
  faiss::Index* base = NULL;

  if (entry->index_type == pg_retrieval_engine_INDEX_HNSW) {
    faiss::IndexHNSWFlat* index = new faiss::IndexHNSWFlat(entry->dim, entry->hnsw_m, metric);
    index->hnsw.efConstruction = entry->hnsw_ef_construction;
    index->hnsw.efSearch = entry->hnsw_ef_search;
    base = index;
  } else if (entry->index_type == pg_retrieval_engine_INDEX_IVF_FLAT) {
    faiss::IndexFlat* quantizer = new faiss::IndexFlat(entry->dim, metric);
    faiss::IndexIVFFlat* index =
        new faiss::IndexIVFFlat(quantizer, entry->dim, entry->ivf_nlist, metric);
    index->own_fields = true;
    index->nprobe = entry->ivf_nprobe;
    base = index;
  } else {
    faiss::IndexFlat* quantizer = new faiss::IndexFlat(entry->dim, metric);
    faiss::IndexIVFPQ* index = new faiss::IndexIVFPQ(quantizer, entry->dim, entry->ivf_nlist,
                                                     entry->ivfpq_m, entry->ivfpq_bits, metric);
    index->own_fields = true;
    index->nprobe = entry->ivf_nprobe;
    base = index;
  }

  return new faiss::IndexIDMap2(base);
}

extern "C" Datum pg_retrieval_engine_index_create(PG_FUNCTION_ARGS) {
  text* name_text = PG_GETARG_TEXT_PP(0);
  int32 dim = PG_GETARG_INT32(1);
  text* metric_text = PG_GETARG_TEXT_PP(2);
  text* index_type_text = PG_GETARG_TEXT_PP(3);
  Jsonb* options = PG_GETARG_JSONB_P(4);
  text* device_text = PG_GETARG_TEXT_PP(5);

  char* name = text_to_cstring(name_text);
  char* metric = text_to_cstring(metric_text);
  char* index_type = text_to_cstring(index_type_text);
  char* device = text_to_cstring(device_text);
  bool found = false;
  PgRetrievalEngineIndexEntry* entry;

  if (dim < 1 || dim > pg_retrieval_engine_MAX_DIMENSIONS)
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("dim must be in range 1..%d", pg_retrieval_engine_MAX_DIMENSIONS)));

  if (strlen(name) >= pg_retrieval_engine_MAX_INDEX_NAME)
    ereport(ERROR, (errcode(ERRCODE_NAME_TOO_LONG),
                    errmsg("index name too long (max %d)", pg_retrieval_engine_MAX_INDEX_NAME - 1)));

  ensure_registry();

  entry = (PgRetrievalEngineIndexEntry*)hash_search(pg_retrieval_engine_registry, name, HASH_ENTER, &found);

  if (found)
    ereport(ERROR,
            (errcode(ERRCODE_DUPLICATE_OBJECT), errmsg("index \"%s\" already exists", name)));

  memset(entry, 0, sizeof(PgRetrievalEngineIndexEntry));
  strlcpy(entry->name, name, sizeof(entry->name));
  entry->dim = dim;
  entry->metric = parse_metric(metric);
  entry->index_type = parse_index_type(index_type);
  entry->device = parse_device(device);
#ifndef USE_FAISS_GPU
  if (entry->device == pg_retrieval_engine_DEVICE_GPU)
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("pg_retrieval_engine was built without GPU support")));
#endif

  entry->hnsw_m = jsonb_option_int32(options, "m", pg_retrieval_engine_DEFAULT_HNSW_M, 2, 256);
  entry->hnsw_ef_construction = jsonb_option_int32(
      options, "ef_construction", pg_retrieval_engine_DEFAULT_HNSW_EF_CONSTRUCTION, 4, 1000000);
  entry->hnsw_ef_search =
      jsonb_option_int32(options, "ef_search", pg_retrieval_engine_DEFAULT_HNSW_EF_SEARCH, 1, 1000000);
  entry->ivf_nlist = jsonb_option_int32(options, "nlist", pg_retrieval_engine_DEFAULT_IVF_NLIST, 1, 1000000);
  entry->ivf_nprobe =
      jsonb_option_int32(options, "nprobe", pg_retrieval_engine_DEFAULT_IVF_NPROBE, 1, 1000000);
  entry->ivfpq_m = jsonb_option_int32(options, "pq_m", pg_retrieval_engine_DEFAULT_IVFPQ_M, 1, 4096);
  entry->ivfpq_bits = jsonb_option_int32(options, "pq_bits", pg_retrieval_engine_DEFAULT_IVFPQ_BITS, 1, 16);
  entry->gpu_device = jsonb_option_int32(options, "gpu_device", 0, 0, 128);
  initialize_entry_defaults(entry);

  try {
    entry->cpu_index = build_index(entry);
    entry->is_trained = (entry->index_type == pg_retrieval_engine_INDEX_HNSW);
    entry->num_vectors = 0;

#ifdef USE_FAISS_GPU
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) rebuild_gpu_index(entry);
#endif
  } catch (const std::exception& e) {
    hash_search(pg_retrieval_engine_registry, name, HASH_REMOVE, NULL);
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS create error: %s", e.what())));
  }

  pfree(name);
  pfree(metric);
  pfree(index_type);
  pfree(device);

  PG_RETURN_VOID();
}

extern "C" Datum pg_retrieval_engine_index_train(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  ArrayType* vectors_arr = PG_GETARG_ARRAYTYPE_P(1);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  std::vector<float> vectors;
  int64 n = 0;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  read_vector_array(vectors_arr, entry->dim, vectors, &n);

  if (entry->metric == pg_retrieval_engine_METRIC_COSINE) normalize_many(vectors.data(), n, entry->dim);

  try {
    entry->train_calls++;
    faiss::Index* index = active_index(entry);
    index->train(n, vectors.data());
    entry->is_trained = index->is_trained;
    entry->num_vectors = index->ntotal;
#ifdef USE_FAISS_GPU
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) sync_gpu_to_cpu(entry);
#endif
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS train error: %s", e.what())));
  }

  pfree(name);
  PG_RETURN_VOID();
}

extern "C" Datum pg_retrieval_engine_index_add(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  ArrayType* ids_arr = PG_GETARG_ARRAYTYPE_P(1);
  ArrayType* vectors_arr = PG_GETARG_ARRAYTYPE_P(2);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  std::vector<faiss::idx_t> ids;
  std::vector<float> vectors;
  int64 n = 0;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  read_ids_array(ids_arr, ids);
  read_vector_array(vectors_arr, entry->dim, vectors, &n);

  if ((int64)ids.size() != n)
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("ids count (%lld) and vectors count (%lld) must match",
                           (long long)ids.size(), (long long)n)));

  if (entry->metric == pg_retrieval_engine_METRIC_COSINE) normalize_many(vectors.data(), n, entry->dim);

  try {
    entry->add_calls++;
    faiss::Index* index = active_index(entry);
    faiss::IndexIDMap2* idmap = dynamic_cast<faiss::IndexIDMap2*>(index);

    if (idmap == NULL)
      ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                      errmsg("internal index is not an ID map")));

    if (!index->is_trained)
      ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                      errmsg("index \"%s\" is not trained", name)));

    idmap->add_with_ids(n, vectors.data(), ids.data());
    entry->num_vectors = index->ntotal;
    entry->add_vectors_total += n;
#ifdef USE_FAISS_GPU
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) sync_gpu_to_cpu(entry);
#endif
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR,
            (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION), errmsg("FAISS add error: %s", e.what())));
  }

  pfree(name);
  PG_RETURN_INT64(n);
}

extern "C" Datum pg_retrieval_engine_index_search(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  PgVector* query = datum_to_pgvector(PG_GETARG_DATUM(1));
  int32 k = PG_GETARG_INT32(2);
  Jsonb* search_params = PG_GETARG_JSONB_P(3);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  ReturnSetInfo* rsinfo;
  Tuplestorestate* tupstore;
  TupleDesc tupdesc;
  int64 emitted_rows = 0;
  instr_time started_at;
  instr_time finished_at;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  if (query->dim != entry->dim)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("query dimension mismatch: expected %d, got %d", entry->dim, query->dim)));

  if (k <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k must be > 0")));

  materialize_result_begin(fcinfo, &tupstore, &tupdesc, &rsinfo);
  INSTR_TIME_SET_CURRENT(started_at);

  try {
    faiss::Index* index = active_index(entry);
    int32 effective_k = std::min<int64>(k, index->ntotal);

    if (effective_k > 0) {
      std::vector<float> query_buf(entry->dim);
      SearchExecutionOptions options;
      std::vector<float> distances;
      std::vector<faiss::idx_t> labels;
      bool params_applied = false;

      memcpy(query_buf.data(), query->x, sizeof(float) * entry->dim);
      if (entry->metric == pg_retrieval_engine_METRIC_COSINE) normalize_one(query_buf.data(), entry->dim);

      apply_search_params(entry, index, search_params, effective_k, false, &options);
      params_applied = true;
      distances.resize(options.candidate_k);
      labels.resize(options.candidate_k);

      try {
        index->search(1, query_buf.data(), options.candidate_k, distances.data(), labels.data());
      } catch (...) {
        if (params_applied) restore_search_params(index, &options);
        throw;
      }
      restore_search_params(index, &options);

      for (int i = 0; i < options.candidate_k && emitted_rows < effective_k; ++i) {
        float distance;

        if (labels[i] < 0) continue;
        distance = distances[i];
        if (entry->metric == pg_retrieval_engine_METRIC_COSINE) distance = 1.0f - distance;
        append_search_row(tupstore, tupdesc, 0, false, (int64)labels[i], distance);
        emitted_rows++;
      }

      entry->last_candidate_k = options.candidate_k;
      entry->last_batch_size = 1;
    }
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS search error: %s", e.what())));
  }

  INSTR_TIME_SET_CURRENT(finished_at);
  entry->search_single_calls++;
  entry->search_query_total++;
  entry->search_result_total += emitted_rows;
  entry->search_single_ms_total += elapsed_ms(started_at, finished_at);
  materialize_result_end(rsinfo, tupstore, tupdesc);

  pfree(name);
  PG_RETURN_NULL();
}

static int64 emit_batch_results(Tuplestorestate* tupstore, TupleDesc tupdesc, int32 global_q_start,
                                int32 effective_k, int32 candidate_k, int64 chunk_queries,
                                const std::vector<float>& distances,
                                const std::vector<faiss::idx_t>& labels, bool cosine_metric,
                                bool filtered, const std::unordered_set<faiss::idx_t>* filter_ids) {
  int64 emitted_rows = 0;

  for (int64 q = 0; q < chunk_queries; ++q) {
    int32 emitted_for_query = 0;

    for (int32 i = 0; i < candidate_k; ++i) {
      size_t offset = (size_t)q * (size_t)candidate_k + (size_t)i;
      faiss::idx_t id = labels[offset];
      float distance = distances[offset];

      if (id < 0) continue;
      if (filtered && filter_ids->find(id) == filter_ids->end()) continue;

      if (cosine_metric) distance = 1.0f - distance;
      append_search_row(tupstore, tupdesc, global_q_start + (int32)q + 1, true, (int64)id,
                        distance);

      emitted_rows++;
      emitted_for_query++;
      if (emitted_for_query >= effective_k) break;
    }
  }

  return emitted_rows;
}

static void run_batch_search(PgRetrievalEngineIndexEntry* entry, ArrayType* queries_arr, int32 k,
                             Jsonb* search_params, bool filtered,
                             const std::unordered_set<faiss::idx_t>* filter_ids,
                             Tuplestorestate* tupstore, TupleDesc tupdesc, int64* emitted_rows) {
  std::vector<float> queries;
  int64 num_queries = 0;
  faiss::Index* index = active_index(entry);
  int32 effective_k = std::min<int64>(k, index->ntotal);

  *emitted_rows = 0;
  read_vector_array(queries_arr, entry->dim, queries, &num_queries);
  if (entry->metric == pg_retrieval_engine_METRIC_COSINE)
    normalize_many(queries.data(), num_queries, entry->dim);
  if (effective_k <= 0) return;

  SearchExecutionOptions options;
  int64 offset = 0;
  bool params_applied = false;
  apply_search_params(entry, index, search_params, effective_k, filtered, &options);
  params_applied = true;
  entry->last_candidate_k = options.candidate_k;

  try {
    while (offset < num_queries) {
      int64 chunk_queries = std::min<int64>(options.batch_size, num_queries - offset);
      std::vector<float> distances((size_t)chunk_queries * (size_t)options.candidate_k);
      std::vector<faiss::idx_t> labels((size_t)chunk_queries * (size_t)options.candidate_k);
      const float* chunk_query_data = queries.data() + (size_t)offset * (size_t)entry->dim;

      index->search(chunk_queries, chunk_query_data, options.candidate_k, distances.data(),
                    labels.data());
      *emitted_rows += emit_batch_results(
          tupstore, tupdesc, (int32)offset, effective_k, options.candidate_k, chunk_queries,
          distances, labels, entry->metric == pg_retrieval_engine_METRIC_COSINE, filtered, filter_ids);
      offset += chunk_queries;
    }
  } catch (...) {
    if (params_applied) restore_search_params(index, &options);
    throw;
  }

  restore_search_params(index, &options);
  entry->last_batch_size = options.batch_size;
}

extern "C" Datum pg_retrieval_engine_index_search_batch(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  ArrayType* queries_arr = PG_GETARG_ARRAYTYPE_P(1);
  int32 k = PG_GETARG_INT32(2);
  Jsonb* search_params = PG_GETARG_JSONB_P(3);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  ReturnSetInfo* rsinfo;
  Tuplestorestate* tupstore;
  TupleDesc tupdesc;
  int64 emitted_rows = 0;
  int64 query_count = ArrayGetNItems(ARR_NDIM(queries_arr), ARR_DIMS(queries_arr));
  instr_time started_at;
  instr_time finished_at;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));
  if (k <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k must be > 0")));

  materialize_result_begin(fcinfo, &tupstore, &tupdesc, &rsinfo);
  INSTR_TIME_SET_CURRENT(started_at);

  try {
    run_batch_search(entry, queries_arr, k, search_params, false, NULL, tupstore, tupdesc,
                     &emitted_rows);
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS batch search error: %s", e.what())));
  }

  INSTR_TIME_SET_CURRENT(finished_at);
  entry->search_batch_calls++;
  entry->search_query_total += query_count;
  entry->search_result_total += emitted_rows;
  entry->search_batch_ms_total += elapsed_ms(started_at, finished_at);
  materialize_result_end(rsinfo, tupstore, tupdesc);

  pfree(name);
  PG_RETURN_NULL();
}

extern "C" Datum pg_retrieval_engine_index_search_filtered(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  PgVector* query = datum_to_pgvector(PG_GETARG_DATUM(1));
  int32 k = PG_GETARG_INT32(2);
  ArrayType* filter_ids_arr = PG_GETARG_ARRAYTYPE_P(3);
  Jsonb* search_params = PG_GETARG_JSONB_P(4);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  ReturnSetInfo* rsinfo;
  Tuplestorestate* tupstore;
  TupleDesc tupdesc;
  std::unordered_set<faiss::idx_t> filter_ids;
  bool has_filter = false;
  int64 emitted_rows = 0;
  instr_time started_at;
  instr_time finished_at;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));
  if (query->dim != entry->dim)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("query dimension mismatch: expected %d, got %d", entry->dim, query->dim)));
  if (k <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k must be > 0")));

  read_optional_ids_array(filter_ids_arr, &filter_ids, &has_filter);
  if (!has_filter)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("filter_ids must not be NULL")));

  materialize_result_begin(fcinfo, &tupstore, &tupdesc, &rsinfo);
  INSTR_TIME_SET_CURRENT(started_at);

  try {
    faiss::Index* index = active_index(entry);
    int32 effective_k = std::min<int64>(k, index->ntotal);

    if (effective_k > 0) {
      std::vector<float> query_buf(entry->dim);
      SearchExecutionOptions options;
      std::vector<float> distances;
      std::vector<faiss::idx_t> labels;
      bool params_applied = false;

      memcpy(query_buf.data(), query->x, sizeof(float) * entry->dim);
      if (entry->metric == pg_retrieval_engine_METRIC_COSINE) normalize_one(query_buf.data(), entry->dim);

      apply_search_params(entry, index, search_params, effective_k, true, &options);
      params_applied = true;
      distances.resize(options.candidate_k);
      labels.resize(options.candidate_k);
      try {
        index->search(1, query_buf.data(), options.candidate_k, distances.data(), labels.data());
      } catch (...) {
        if (params_applied) restore_search_params(index, &options);
        throw;
      }
      restore_search_params(index, &options);

      for (int32 i = 0; i < options.candidate_k && emitted_rows < effective_k; ++i) {
        float distance;
        faiss::idx_t id = labels[i];

        if (id < 0) continue;
        if (filter_ids.find(id) == filter_ids.end()) continue;
        distance = distances[i];
        if (entry->metric == pg_retrieval_engine_METRIC_COSINE) distance = 1.0f - distance;
        append_search_row(tupstore, tupdesc, 0, false, (int64)id, distance);
        emitted_rows++;
      }

      entry->last_candidate_k = options.candidate_k;
      entry->last_batch_size = 1;
    }
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS filtered search error: %s", e.what())));
  }

  INSTR_TIME_SET_CURRENT(finished_at);
  entry->search_filtered_calls++;
  entry->search_query_total++;
  entry->search_result_total += emitted_rows;
  entry->search_filtered_ms_total += elapsed_ms(started_at, finished_at);
  materialize_result_end(rsinfo, tupstore, tupdesc);

  pfree(name);
  PG_RETURN_NULL();
}

extern "C" Datum pg_retrieval_engine_index_search_batch_filtered(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  ArrayType* queries_arr = PG_GETARG_ARRAYTYPE_P(1);
  int32 k = PG_GETARG_INT32(2);
  ArrayType* filter_ids_arr = PG_GETARG_ARRAYTYPE_P(3);
  Jsonb* search_params = PG_GETARG_JSONB_P(4);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  ReturnSetInfo* rsinfo;
  Tuplestorestate* tupstore;
  TupleDesc tupdesc;
  std::unordered_set<faiss::idx_t> filter_ids;
  bool has_filter = false;
  int64 emitted_rows = 0;
  int64 query_count = ArrayGetNItems(ARR_NDIM(queries_arr), ARR_DIMS(queries_arr));
  instr_time started_at;
  instr_time finished_at;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));
  if (k <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k must be > 0")));

  read_optional_ids_array(filter_ids_arr, &filter_ids, &has_filter);
  if (!has_filter)
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("filter_ids must not be NULL")));

  materialize_result_begin(fcinfo, &tupstore, &tupdesc, &rsinfo);
  INSTR_TIME_SET_CURRENT(started_at);

  try {
    run_batch_search(entry, queries_arr, k, search_params, true, &filter_ids, tupstore, tupdesc,
                     &emitted_rows);
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS filtered batch search error: %s", e.what())));
  }

  INSTR_TIME_SET_CURRENT(finished_at);
  entry->search_filtered_calls++;
  entry->search_query_total += query_count;
  entry->search_result_total += emitted_rows;
  entry->search_filtered_ms_total += elapsed_ms(started_at, finished_at);
  materialize_result_end(rsinfo, tupstore, tupdesc);

  pfree(name);
  PG_RETURN_NULL();
}

extern "C" Datum pg_retrieval_engine_index_save(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  char* path = text_to_cstring(PG_GETARG_TEXT_PP(1));
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  try {
    entry->save_calls++;
#ifdef USE_FAISS_GPU
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) sync_gpu_to_cpu(entry);
#endif

    faiss::write_index(entry->cpu_index, path);
    write_metadata_file(entry, path);
    strlcpy(entry->index_path, path, sizeof(entry->index_path));
  } catch (const std::exception& e) {
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS save error: %s", e.what())));
  }

  pfree(name);
  pfree(path);
  PG_RETURN_VOID();
}

extern "C" Datum pg_retrieval_engine_index_load(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  char* path = text_to_cstring(PG_GETARG_TEXT_PP(1));
  char* device = text_to_cstring(PG_GETARG_TEXT_PP(2));
  int parsed_device = parse_device(device);
  bool found = false;
  PgRetrievalEngineIndexEntry* entry;
  std::unordered_map<std::string, std::string> meta;
  faiss::Index* loaded_index = NULL;
  faiss::Index* base_index = NULL;

#ifndef USE_FAISS_GPU
  if (parsed_device == pg_retrieval_engine_DEVICE_GPU)
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("pg_retrieval_engine was built without GPU support")));
#endif

  ensure_registry();

  entry = (PgRetrievalEngineIndexEntry*)hash_search(pg_retrieval_engine_registry, name, HASH_ENTER, &found);

  if (found)
    ereport(ERROR,
            (errcode(ERRCODE_DUPLICATE_OBJECT), errmsg("index \"%s\" already exists", name)));

  memset(entry, 0, sizeof(PgRetrievalEngineIndexEntry));
  strlcpy(entry->name, name, sizeof(entry->name));
  initialize_entry_defaults(entry);

  try {
    loaded_index = faiss::read_index(path);

    if (dynamic_cast<faiss::IndexIDMap2*>(loaded_index) == NULL)
      loaded_index = new faiss::IndexIDMap2(loaded_index);

    meta = read_metadata_file(path);

    entry->cpu_index = loaded_index;
    entry->dim = loaded_index->d;
    entry->metric =
        loaded_index->metric_type == faiss::METRIC_L2 ? pg_retrieval_engine_METRIC_L2 : pg_retrieval_engine_METRIC_IP;
    entry->index_type = pg_retrieval_engine_INDEX_HNSW;
    entry->device = parsed_device;
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) entry->preferred_batch_size = 1024;
    base_index = unwrap_idmap(loaded_index);

    if (dynamic_cast<faiss::IndexHNSW*>(base_index) != NULL)
      entry->index_type = pg_retrieval_engine_INDEX_HNSW;
    else if (dynamic_cast<faiss::IndexIVFPQ*>(base_index) != NULL)
      entry->index_type = pg_retrieval_engine_INDEX_IVF_PQ;
    else
      entry->index_type = pg_retrieval_engine_INDEX_IVF_FLAT;

    if (meta.find("metric") != meta.end()) entry->metric = parse_metric(meta["metric"].c_str());
    if (meta.find("index_type") != meta.end())
      entry->index_type = parse_index_type(meta["index_type"].c_str());
    if (meta.find("hnsw_m") != meta.end())
      entry->hnsw_m = std::stoi(meta["hnsw_m"]);
    else
      entry->hnsw_m = pg_retrieval_engine_DEFAULT_HNSW_M;
    if (meta.find("hnsw_ef_construction") != meta.end())
      entry->hnsw_ef_construction = std::stoi(meta["hnsw_ef_construction"]);
    else
      entry->hnsw_ef_construction = pg_retrieval_engine_DEFAULT_HNSW_EF_CONSTRUCTION;
    if (meta.find("hnsw_ef_search") != meta.end())
      entry->hnsw_ef_search = std::stoi(meta["hnsw_ef_search"]);
    else
      entry->hnsw_ef_search = pg_retrieval_engine_DEFAULT_HNSW_EF_SEARCH;
    if (meta.find("ivf_nlist") != meta.end())
      entry->ivf_nlist = std::stoi(meta["ivf_nlist"]);
    else
      entry->ivf_nlist = pg_retrieval_engine_DEFAULT_IVF_NLIST;
    if (meta.find("ivf_nprobe") != meta.end())
      entry->ivf_nprobe = std::stoi(meta["ivf_nprobe"]);
    else
      entry->ivf_nprobe = pg_retrieval_engine_DEFAULT_IVF_NPROBE;
    if (meta.find("ivfpq_m") != meta.end())
      entry->ivfpq_m = std::stoi(meta["ivfpq_m"]);
    else
      entry->ivfpq_m = pg_retrieval_engine_DEFAULT_IVFPQ_M;
    if (meta.find("ivfpq_bits") != meta.end())
      entry->ivfpq_bits = std::stoi(meta["ivfpq_bits"]);
    else
      entry->ivfpq_bits = pg_retrieval_engine_DEFAULT_IVFPQ_BITS;
    if (meta.find("gpu_device") != meta.end())
      entry->gpu_device = std::stoi(meta["gpu_device"]);
    else
      entry->gpu_device = 0;

    entry->num_vectors = loaded_index->ntotal;
    entry->is_trained = loaded_index->is_trained;
    entry->load_calls = 1;
    strlcpy(entry->index_path, path, sizeof(entry->index_path));

#ifdef USE_FAISS_GPU
    if (entry->device == pg_retrieval_engine_DEVICE_GPU) rebuild_gpu_index(entry);
#endif
  } catch (const std::exception& e) {
    hash_search(pg_retrieval_engine_registry, name, HASH_REMOVE, NULL);
    record_error(entry);
    ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                    errmsg("FAISS load error: %s", e.what())));
  }

  pfree(name);
  pfree(path);
  pfree(device);
  PG_RETURN_VOID();
}

static int32 clamp_int32(int64 value, int32 min_value, int32 max_value) {
  if (value < min_value) return min_value;
  if (value > max_value) return max_value;
  return (int32)value;
}

extern "C" Datum pg_retrieval_engine_index_autotune(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  char* mode_text = text_to_cstring(PG_GETARG_TEXT_PP(1));
  Jsonb* options = PG_GETARG_JSONB_P(2);
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  int mode = parse_autotune_mode(mode_text);
  double target_recall = 0.95;
  int32 min_batch_size = 32;
  int32 max_batch_size = 4096;
  int32 old_hnsw_ef_search = 0;
  int32 old_ivf_nprobe = 0;
  int32 old_batch_size = 0;
  StringInfoData json;
  Datum result;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  old_hnsw_ef_search = entry->hnsw_ef_search;
  old_ivf_nprobe = entry->ivf_nprobe;
  old_batch_size = entry->preferred_batch_size;
  jsonb_get_float8(options, "target_recall", &target_recall);
  min_batch_size = jsonb_option_int32(options, "min_batch_size", 32, 1, 65536);
  max_batch_size = jsonb_option_int32(options, "max_batch_size", 4096, min_batch_size, 65536);

  if (entry->index_type == pg_retrieval_engine_INDEX_HNSW) {
    double multiplier = 1.0;
    int64 suggested = (int64)(sqrt((double)std::max<int64>(entry->num_vectors, 1)) * 8.0);

    if (mode == pg_retrieval_engine_AUTOTUNE_LATENCY) multiplier = 0.75;
    if (mode == pg_retrieval_engine_AUTOTUNE_RECALL) multiplier = 1.75;
    if (target_recall > 0.98) multiplier *= 1.25;

    suggested = (int64)((double)suggested * multiplier);
    entry->hnsw_ef_search = clamp_int32(suggested, 16, 4096);
  }

  if (entry->index_type == pg_retrieval_engine_INDEX_IVF_FLAT || entry->index_type == pg_retrieval_engine_INDEX_IVF_PQ) {
    double multiplier = 1.0;
    int64 suggested = (int64)(sqrt((double)std::max(entry->ivf_nlist, 1)));

    if (mode == pg_retrieval_engine_AUTOTUNE_LATENCY) multiplier = 0.75;
    if (mode == pg_retrieval_engine_AUTOTUNE_RECALL) multiplier = 2.0;
    if (target_recall > 0.98) multiplier *= 1.2;

    suggested = (int64)((double)suggested * multiplier);
    entry->ivf_nprobe = clamp_int32(suggested, 1, std::max(entry->ivf_nlist, 1));
  }

  {
    int64 base_batch = (entry->device == pg_retrieval_engine_DEVICE_GPU) ? 2048 : 256;
    int64 dim_penalty = std::max(entry->dim, 1) / 16;

    if (mode == pg_retrieval_engine_AUTOTUNE_LATENCY) base_batch = base_batch / 2;
    if (mode == pg_retrieval_engine_AUTOTUNE_RECALL) base_batch = base_batch * 2;

    base_batch = base_batch - dim_penalty;
    entry->preferred_batch_size = clamp_int32(base_batch, min_batch_size, max_batch_size);
  }

  {
    faiss::Index* base = unwrap_idmap(active_index(entry));
    faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(base);
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(base);

    if (hnsw != NULL) hnsw->hnsw.efSearch = entry->hnsw_ef_search;
    if (ivf != NULL) ivf->nprobe = entry->ivf_nprobe;
  }

  entry->autotune_calls++;
  entry->last_autotune_mode = mode;

  initStringInfo(&json);
  appendStringInfoChar(&json, '{');
  appendStringInfo(&json, "\"name\":\"%s\",", entry->name);
  appendStringInfo(&json, "\"mode\":\"%s\",", autotune_mode_name(mode));
  appendStringInfo(&json, "\"target_recall\":%.4f,", target_recall);
  appendStringInfo(&json, "\"hnsw_ef_search\":{\"old\":%d,\"new\":%d},", old_hnsw_ef_search,
                   entry->hnsw_ef_search);
  appendStringInfo(&json, "\"ivf_nprobe\":{\"old\":%d,\"new\":%d},", old_ivf_nprobe,
                   entry->ivf_nprobe);
  appendStringInfo(&json, "\"preferred_batch_size\":{\"old\":%d,\"new\":%d}", old_batch_size,
                   entry->preferred_batch_size);
  appendStringInfoChar(&json, '}');
  result = DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data));

  pfree(name);
  pfree(mode_text);
  PG_RETURN_DATUM(result);
}

extern "C" Datum pg_retrieval_engine_metrics_reset(PG_FUNCTION_ARGS) {
  HASH_SEQ_STATUS status;
  PgRetrievalEngineIndexEntry* entry;

  ensure_registry();

  if (PG_ARGISNULL(0)) {
    hash_seq_init(&status, pg_retrieval_engine_registry);
    while ((entry = (PgRetrievalEngineIndexEntry*)hash_seq_search(&status)) != NULL)
      reset_runtime_stats(entry, false);
  } else {
    char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));

    entry = lookup_entry(name);
    if (entry == NULL)
      ereport(ERROR,
              (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

    reset_runtime_stats(entry, false);
    pfree(name);
  }

  PG_RETURN_VOID();
}

extern "C" Datum pg_retrieval_engine_index_stats(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  PgRetrievalEngineIndexEntry* entry = lookup_entry(name);
  StringInfoData json;
  Datum result;

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  initStringInfo(&json);

  appendStringInfoChar(&json, '{');

  appendStringInfoString(&json, "\"name\":");
  escape_json(&json, entry->name);
  appendStringInfoString(&json, ",");

  appendStringInfo(&json, "\"version\":\"%s\",", pg_retrieval_engine_VERSION);
  appendStringInfo(&json, "\"dim\":%d,", entry->dim);
  appendStringInfo(&json, "\"metric\":\"%s\",", metric_name(entry->metric));
  appendStringInfo(&json, "\"index_type\":\"%s\",", index_type_name(entry->index_type));
  appendStringInfo(&json, "\"device\":\"%s\",", device_name(entry->device));
  appendStringInfo(&json, "\"num_vectors\":%lld,", (long long)entry->num_vectors);
  appendStringInfo(&json, "\"is_trained\":%s,", entry->is_trained ? "true" : "false");
  appendStringInfo(&json, "\"hnsw\":{\"m\":%d,\"ef_construction\":%d,\"ef_search\":%d},",
                   entry->hnsw_m, entry->hnsw_ef_construction, entry->hnsw_ef_search);
  appendStringInfo(&json, "\"ivf\":{\"nlist\":%d,\"nprobe\":%d},", entry->ivf_nlist,
                   entry->ivf_nprobe);
  appendStringInfo(&json, "\"ivfpq\":{\"m\":%d,\"bits\":%d},", entry->ivfpq_m, entry->ivfpq_bits);
  appendStringInfo(
      &json,
      "\"runtime\":{\"train_calls\":%lld,\"add_calls\":%lld,\"add_vectors_total\":%lld,"
      "\"search_single_calls\":%lld,\"search_batch_calls\":%lld,\"search_filtered_calls\":%lld,"
      "\"search_query_total\":%lld,\"search_result_total\":%lld,\"save_calls\":%lld,"
      "\"load_calls\":%lld,\"autotune_calls\":%lld,\"error_calls\":%lld,"
      "\"search_single_ms_total\":%.4f,\"search_batch_ms_total\":%.4f,"
      "\"search_filtered_ms_total\":%.4f,\"search_single_ms_avg\":%.4f,"
      "\"search_batch_ms_avg\":%.4f,\"search_filtered_ms_avg\":%.4f,"
      "\"last_candidate_k\":%d,\"last_batch_size\":%d,\"preferred_batch_size\":%d,"
      "\"last_autotune_mode\":\"%s\"},",
      (long long)entry->train_calls, (long long)entry->add_calls,
      (long long)entry->add_vectors_total, (long long)entry->search_single_calls,
      (long long)entry->search_batch_calls, (long long)entry->search_filtered_calls,
      (long long)entry->search_query_total, (long long)entry->search_result_total,
      (long long)entry->save_calls, (long long)entry->load_calls, (long long)entry->autotune_calls,
      (long long)entry->error_calls, entry->search_single_ms_total, entry->search_batch_ms_total,
      entry->search_filtered_ms_total,
      entry->search_single_calls > 0
          ? entry->search_single_ms_total / (double)entry->search_single_calls
          : 0.0,
      entry->search_batch_calls > 0
          ? entry->search_batch_ms_total / (double)entry->search_batch_calls
          : 0.0,
      entry->search_filtered_calls > 0
          ? entry->search_filtered_ms_total / (double)entry->search_filtered_calls
          : 0.0,
      entry->last_candidate_k, entry->last_batch_size, entry->preferred_batch_size,
      autotune_mode_name(entry->last_autotune_mode));
  appendStringInfo(&json, "\"index_path\":");
  escape_json(&json, entry->index_path);

  appendStringInfoChar(&json, '}');

  result = DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data));

  pfree(name);

  PG_RETURN_DATUM(result);
}

extern "C" Datum pg_retrieval_engine_index_drop(PG_FUNCTION_ARGS) {
  char* name = text_to_cstring(PG_GETARG_TEXT_PP(0));
  PgRetrievalEngineIndexEntry* entry;

  ensure_registry();
  entry = lookup_entry(name);

  if (entry == NULL)
    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("index \"%s\" does not exist", name)));

  free_entry_resources(entry);
  hash_search(pg_retrieval_engine_registry, name, HASH_REMOVE, NULL);

  pfree(name);
  PG_RETURN_VOID();
}

extern "C" Datum pg_retrieval_engine_reset(PG_FUNCTION_ARGS) {
  HASH_SEQ_STATUS status;
  PgRetrievalEngineIndexEntry* entry;

  if (pg_retrieval_engine_registry != NULL) {
    hash_seq_init(&status, pg_retrieval_engine_registry);
    while ((entry = (PgRetrievalEngineIndexEntry*)hash_seq_search(&status)) != NULL)
      free_entry_resources(entry);

    hash_destroy(pg_retrieval_engine_registry);
    pg_retrieval_engine_registry = NULL;
  }

  ensure_registry();

  PG_RETURN_VOID();
}
