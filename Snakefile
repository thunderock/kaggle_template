COMPETITION = "child-mind-institute-problematic-internet-use"

rule all:
    input:
        "dag.pdf",
        "data/features/train_wide.csv",

rule combine_features:
    input:
        train_features="data/features/train_features.csv",
        test_features="data/features/test_features.csv",
        train_timeseries="data/features/train_timeseries.csv",
        test_timeseries="data/features/test_timeseries.csv"
    output:
        train_wide_df="data/features/train_wide.csv",
        test_wide_df="data/features/test_wide.csv"
    threads: 1
    shell:
        "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/combine_features.py"

rule generate_timeseries:
    input:
        train=directory("data/input/series_train.parquet"),
        test=directory("data/input/series_test.parquet")
    output:
        train="data/features/train_timeseries.csv",
        test="data/features/test_timeseries.csv",
    threads: 12
    shell:
        "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/timeseries.py"

rule download_data:
    output:
        zip="data/input/{competition}.zip".format(competition=COMPETITION),
        train="data/input/train.csv",
        test="data/input/test.csv",
        timeseries_train=directory("data/input/series_train.parquet"),
        timeseries_test=directory("data/input/series_test.parquet")
    params:
        competition=COMPETITION
    threads: 1
    shell:
        "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/download_dataset.py"

rule generate_features:
    input:
        train="data/input/train.csv",
        test="data/input/test.csv"
    output:
        train="data/features/train_features.csv",
        test="data/features/test_features.csv"
    threads: 1
    shell:
        "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/scaled.py"

rule generate_dag:
    output:
        "dag.pdf"
    threads: 1
    shell:
        "snakemake --dag | dot -Tpdf > dag.pdf"
