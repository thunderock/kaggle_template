from kaggle_template.utils.run_utils import GPU_CORES, CPU_CORES
from os.path import join as j
import time
NUM_CORES = workflow.cores
print(GPU_CORES, CPU_CORES, NUM_CORES)
COMPETITION = "child-mind-institute-problematic-internet-use"
TRAILS = 2
FEATURE_SELECTION_THRESHOLD = 0.7
train_files = ["train_features", "train_wide_features"]
base_data_path = config.get("base_data_path", "data")
base_script_path = config.get("base_script_path", "kaggle_template/scripts")
models = {
    "catboost": j(base_data_path, "models/catboost_{train_file}.json",),
    "xgb": j(base_data_path, "models/xgb_{train_file}.json", ),
    "rf": j(base_data_path, "models/rf_{train_file}.json",),
    "lgbm": j(base_data_path, "models/lgbm_{train_file}.json",),
}

rule all:
    input:
        expand(
            j(base_data_path, "models/{model}_{train_file}.json"),
            model=models.keys(),
            train_file=train_files,
        ),
        j(base_data_path, "output/dag.pdf",),

rule combine_features:
    input:
        train_features=j(base_data_path, "features/train_features.csv"),
        test_features=j(base_data_path, "features/test_features.csv"),
        train_timeseries=j(base_data_path, "features/train_timeseries.csv"),
        test_timeseries=j(base_data_path, "features/test_timeseries.csv"),
    output:
        train_wide_df=j(base_data_path, "features/train_wide_features.csv"),
        test_wide_df=j(base_data_path, "features/test_wide_features.csv"),
    threads: 1
    script: j(base_script_path, "combine_features.py")
    # shell:
    #     "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/combine_features.py"

rule generate_timeseries:
    input:
        train=directory(j(base_data_path, "input/series_train.parquet")),
        test=directory(j(base_data_path, "input/series_test.parquet")),
    output:
        train=j(base_data_path, "features/train_timeseries.csv"),
        test=j(base_data_path, "features/test_timeseries.csv"),
    threads: 12
    script: j(base_script_path, "timeseries.py")
    # shell:
    #     "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/timeseries.py"


rule generate_features:
    input:
        train=j(base_data_path, "input/train.csv"),
        test=j(base_data_path, "input/test.csv"),
    output:
        train=j(base_data_path, "features/train_features.csv"),
        test=j(base_data_path, "features/test_features.csv"),
    threads: 1
    script: j(base_script_path, "scaled.py")
    #  shell:
    #      "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/scaled.py"


rule tune_model:
    input:
        train=j(base_data_path, "features/train_features.csv"),
        train_wide=j(base_data_path, "features/train_wide_features.csv"),
    output:
        output_path=j(base_data_path, "models/{model}_train_features.json"),
        output_wide_path=j(base_data_path, "models/{model}_train_wide_features.json"),
    params:
        trials=TRAILS,
        seed=42,
        model="{model}",
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD,
    threads: NUM_CORES // 2
    script: j(base_script_path, "tune_model.py")

rule tune_meta_model:
    input:
        train=j(base_data_path, "features/train_features.csv"),
        train_wide=j(base_data_path, "features/train_wide_features.csv"),
    output:
        meta_model=j(base_data_path, "models/meta_model.json"),
    params:
        trials=TRAILS,
        seed=42,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD,
    threads: NUM_CORES // 2
    script: j(base_script_path, "tune_meta_model.py")

rule submission:
    input:
        train_wide=j(base_data_path, "features/train_wide_features.csv"),
        train=j(base_data_path, "features/train_features.csv"),
        test=j(base_data_path, "features/test_features.csv"),
        test_wide=j(base_data_path, "features/test_wide_features.csv"),
        catboost=j(base_data_path, "models/catboost_train_features.json"),
        catboost_wide=j(base_data_path, "models/catboost_train_wide_features.json"),
        xgb=j(base_data_path, "models/xgb_train_features.json"),
        xgb_wide=j(base_data_path, "models/xgb_train_wide_features.json"),
        rf=j(base_data_path, "models/rf_train_features.json"),
        rf_wide=j(base_data_path, "models/rf_train_wide_features.json"),
        lgbm=j(base_data_path, "models/lgbm_train_features.json"),
        lgbm_wide=j(base_data_path, "models/lgbm_train_wide_features.json"),
        meta_model=j(base_data_path, "models/meta_model.json"),
    params:
        seed=42,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD,
    threads: NUM_CORES
    output:
        analyze=j(base_data_path, "output/analyze.csv"),
        predictions="submission.csv",
    script: j(base_script_path, "submission.py")


rule upload_data_generate_dag:
    input:
        "submission.csv",
    output:
        dag=j(base_data_path, "output/dag.pdf"),
        dag_filegraph=j(base_data_path, "output/dag_filegraph.pdf"),
    threads: 1
    run:
        shell("snakemake --dag | sed '1d' | dot -Tpdf > {output.dag}")
        shell("snakemake --filegraph | sed '1d' | dot -Tpdf > {output.dag_filegraph}")
        shell("rm -rf temp; mkdir temp; cp dataset-metadata.json temp/")
        shell("zip -r temp/kaggle_template.zip Snakefile Makefile kaggle_template kaggle_packages.mp4 submission.csv data/models data/output data/features")
        if not os.getenv('KAGGLE_URL_BASE'):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            shell("poetry run kaggle datasets version  -p temp -m 'Updated at {timestamp}'")
        shell("rm -rf temp")

rule download_data:
    output:
        train=j(base_data_path, "input/train.csv"),
        test=j(base_data_path, "input/test.csv"),
        timeseries_train=directory(j(base_data_path, "input/series_train.parquet")),
        timeseries_test=directory(j(base_data_path, "input/series_test.parquet")),
    params:
        competition=COMPETITION
    threads: 1
    script: j(base_script_path, "download_dataset.py")
    #  shell:
    #      "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/download_dataset.py"
