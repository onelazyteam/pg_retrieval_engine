use strict;
use warnings FATAL => 'all';
use Time::HiRes qw(time);
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

if (!$ENV{pg_retrieval_engine_RUN_PERF})
{
    plan skip_all => 'set pg_retrieval_engine_RUN_PERF=1 to run CPU performance comparison';
}

my $dim = $ENV{pg_retrieval_engine_PERF_DIM} // 768;
my $rows = $ENV{pg_retrieval_engine_PERF_ROWS} // 1_000_000;
my $queries = $ENV{pg_retrieval_engine_PERF_QUERIES} // 100;
my $k = $ENV{pg_retrieval_engine_PERF_K} // 10;
my $recall_target = 0.95;
my $speedup_target = 5.0;

my $array_sql = join(',', ('random()') x $dim);
my $query_stride = int($rows / $queries);
$query_stride = 1 if $query_stride < 1;

my $node = PostgreSQL::Test::Cluster->new('pg_retrieval_engine_perf_cpu');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', 'CREATE EXTENSION pg_retrieval_engine;');

$node->safe_psql('postgres', qq(
    SELECT setseed(0.42);
    CREATE TABLE items (id bigint PRIMARY KEY, embedding vector($dim));
    INSERT INTO items
    SELECT i, ARRAY[$array_sql]::vector($dim)
    FROM generate_series(1, $rows) AS i;
));

$node->safe_psql('postgres', qq(
    CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_l2_ops);
    CREATE INDEX items_ivf ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 4096);
));

$node->safe_psql('postgres', qq(
    SELECT pg_retrieval_engine_index_create(
        'faiss_hnsw',
        $dim,
        'l2',
        'hnsw',
        '{"m":32,"ef_construction":200,"ef_search":64}'::jsonb,
        'cpu'
    );
    SELECT pg_retrieval_engine_index_add(
        'faiss_hnsw',
        (SELECT array_agg(id ORDER BY id) FROM items),
        (SELECT array_agg(embedding ORDER BY id) FROM items)
    );

    SELECT pg_retrieval_engine_index_create(
        'faiss_ivf',
        $dim,
        'l2',
        'ivfflat',
        '{"nlist":4096,"nprobe":32}'::jsonb,
        'cpu'
    );
    SELECT pg_retrieval_engine_index_train('faiss_ivf', (SELECT array_agg(embedding ORDER BY id) FROM items));
    SELECT pg_retrieval_engine_index_add(
        'faiss_ivf',
        (SELECT array_agg(id ORDER BY id) FROM items),
        (SELECT array_agg(embedding ORDER BY id) FROM items)
    );
));

my @query_vectors = split(/\n/, $node->safe_psql('postgres', qq(
    SELECT embedding::text
    FROM items
    WHERE id % $query_stride = 0
    ORDER BY id
    LIMIT $queries;
)));

sub overlap_ratio
{
    my ($expected, $actual) = @_;
    my %actual_set = map { $_ => 1 } @$actual;
    my $hit = 0;

    for my $id (@$expected)
    {
        $hit++ if $actual_set{$id};
    }

    return @$expected ? ($hit / scalar(@$expected)) : 0;
}

sub run_plan
{
    my ($name, $query_sql_cb) = @_;
    my $total_time = 0.0;
    my $total_recall = 0.0;

    for my $query (@query_vectors)
    {
        my @expected = split(/\n/, $node->safe_psql('postgres', qq(
            SELECT id
            FROM items
            ORDER BY embedding <-> '$query'::vector
            LIMIT $k;
        )));

        my $sql = $query_sql_cb->($query);

        my $start = time();
        my @actual = split(/\n/, $node->safe_psql('postgres', $sql));
        my $elapsed = time() - $start;

        $total_time += $elapsed;
        $total_recall += overlap_ratio(\@expected, \@actual);
    }

    my $avg_recall = @query_vectors ? ($total_recall / scalar(@query_vectors)) : 0;

    diag(sprintf('%s avg_recall=%.4f total_time=%.4fs', $name, $avg_recall, $total_time));

    return ($avg_recall, $total_time);
}

my ($recall_pgvector_hnsw, $time_pgvector_hnsw) = run_plan('pgvector_hnsw', sub {
    my ($query) = @_;
    return qq(
        SET enable_seqscan = off;
        SET hnsw.ef_search = 64;
        SELECT id
        FROM items
        ORDER BY embedding <-> '$query'::vector
        LIMIT $k;
    );
});

my ($recall_faiss_hnsw, $time_faiss_hnsw) = run_plan('pg_retrieval_engine_hnsw', sub {
    my ($query) = @_;
    return qq(
        SELECT id
        FROM pg_retrieval_engine_index_search(
            'faiss_hnsw',
            '$query'::vector,
            $k,
            '{"ef_search":64}'::jsonb
        )
        ORDER BY distance, id;
    );
});

my ($recall_pgvector_ivf, $time_pgvector_ivf) = run_plan('pgvector_ivfflat', sub {
    my ($query) = @_;
    return qq(
        SET enable_seqscan = off;
        SET ivfflat.probes = 32;
        SELECT id
        FROM items
        ORDER BY embedding <-> '$query'::vector
        LIMIT $k;
    );
});

my ($recall_faiss_ivf, $time_faiss_ivf) = run_plan('pg_retrieval_engine_ivfflat', sub {
    my ($query) = @_;
    return qq(
        SELECT id
        FROM pg_retrieval_engine_index_search(
            'faiss_ivf',
            '$query'::vector,
            $k,
            '{"nprobe":32}'::jsonb
        )
        ORDER BY distance, id;
    );
});

cmp_ok($recall_faiss_hnsw, '>=', $recall_target, 'pg_retrieval_engine HNSW recall target met');
cmp_ok($recall_faiss_ivf, '>=', $recall_target, 'pg_retrieval_engine IVFFlat recall target met');

my $speedup_hnsw = $time_pgvector_hnsw / $time_faiss_hnsw;
my $speedup_ivf = $time_pgvector_ivf / $time_faiss_ivf;

diag(sprintf('HNSW speedup=%.4fx, IVFFlat speedup=%.4fx', $speedup_hnsw, $speedup_ivf));

cmp_ok($speedup_hnsw, '>=', $speedup_target, "HNSW speedup >= ${speedup_target}x");
cmp_ok($speedup_ivf, '>=', $speedup_target, "IVFFlat speedup >= ${speedup_target}x");

done_testing();
