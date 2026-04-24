EXTENSION = pg_retrieval_engine
EXTVERSION = 0.2.0

MODULE_big = pg_retrieval_engine

OBJS = src/faiss_in_pg/faiss_engine.o
HEADERS = src/faiss_in_pg/faiss_engine.hpp

DATA = $(wildcard sql/*--*.sql)

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test
TAP_TESTS = 1

PG_CPPFLAGS += -I$(shell pg_config --includedir-server) -std=c++17
SHLIB_LINK += -lfaiss

USE_FAISS_GPU ?= 0
ifeq ($(USE_FAISS_GPU),1)
	PG_CPPFLAGS += -DUSE_FAISS_GPU
	SHLIB_LINK += $(FAISS_GPU_LIBS)
endif

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

# for Postgres < 15
PROVE_FLAGS += -I ./test/perl

CLANG_FORMAT ?= clang-format
FORMAT_FILES = src/faiss_in_pg/faiss_engine.cpp src/faiss_in_pg/faiss_engine.hpp

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

format:
	$(CLANG_FORMAT) -style=file -i $(FORMAT_FILES)

format-check:
	$(CLANG_FORMAT) -style=file --dry-run --Werror $(FORMAT_FILES)

.PHONY: prove_installcheck format format-check
