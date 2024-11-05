COMPETITION = "child-mind-institute-problematic-internet-use"

rule all:
    input:
        "data/features/train_features.csv"

rule download_data:
    output:
        zip="data/input/{competition}.zip".format(competition=COMPETITION),
        train="data/input/train.csv",
        test="data/input/test.csv"
    params:
        competition=COMPETITION
    threads: 1
    script: "kaggle_template/scripts/download_dataset.py"

rule generate_features:
    input:
        train="data/input/train.csv",
        test="data/input/test.csv"
    output:
        train="data/features/train_features.csv",
        test="data/features/test_features.csv"
    threads: 1
    script: "kaggle_template/scripts/scaled.py"

rule generate_dag:
    output:
        "dag.pdf"
    threads: 1
    shell:
        "snakemake --dag | dot -Tpdf > dag.pdf"
