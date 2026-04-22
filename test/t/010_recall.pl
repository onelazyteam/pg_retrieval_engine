use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = $ENV{pg_retrieval_engine_RECALL_DIM} // 32;
my $rows = $ENV{pg_retrieval_engine_RECALL_ROWS} // 20000;
my $queries = $ENV{pg_retrieval_engine_RECALL_QUERIES} // 30;
my $k = 10;
my $recall_target = 0.95;

my $array_sql = join(',', ('random()') x $dim);
my $node = PostgreSQL::Test::Cluster->new('pg_retrieval_engine_recall');
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
    SELECT pg_retrieval_engine_index_create(
        'recall_hnsw',
        $dim,
        'l2',
        'hnsw',
        '{"m":32,"ef_construction":200,"ef_search":128}'::jsonb,
        'cpu'
    );
    SELECT pg_retrieval_engine_index_add(
        'recall_hnsw',
        (SELECT array_agg(id ORDER BY id) FROM items),
        (SELECT array_agg(embedding ORDER BY id) FROM items)
    );
));

my @query_vectors = split(/\n/, $node->safe_psql('postgres', qq(
    SELECT embedding::text
    FROM items
    WHERE id % GREATEST(1, ($rows / $queries)) = 0
    ORDER BY id
    LIMIT $queries;
)));

my $correct = 0;
my $total = 0;

for my $query (@query_vectors)
{
    my @expected = split(/\n/, $node->safe_psql('postgres', qq(
        SELECT id
        FROM items
        ORDER BY embedding <-> '$query'::vector
        LIMIT $k;
    )));

    my @actual = split(/\n/, $node->safe_psql('postgres', qq(
        SELECT id
        FROM pg_retrieval_engine_index_search(
            'recall_hnsw',
            '$query'::vector,
            $k,
            '{"ef_search":128}'::jsonb
        )
        ORDER BY distance, id;
    )));

    my %seen = map { $_ => 1 } @actual;

    for my $id (@expected)
    {
        $correct++ if $seen{$id};
        $total++;
    }
}

my $recall = $total > 0 ? ($correct / $total) : 0;
cmp_ok($recall, '>=', $recall_target, sprintf('recall@%d >= %.2f (actual=%.4f)', $k, $recall_target, $recall));

done_testing();
