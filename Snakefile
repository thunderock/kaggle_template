from kaggle_template.utils.run_utils import GPU_CORES, CPU_CORES
from os.path import join as j
import time
from subprocess import check_output
NUM_CORES = workflow.cores
print(GPU_CORES, CPU_CORES, NUM_CORES)
COMPETITION = "child-mind-institute-problematic-internet-use"
TRIALS = config.get("trials", 1)
FEATURE_SELECTION_THRESHOLD = 1.0
train_files = ["train_features", "train_wide_features"]
base_data_path = config.get("base_data_path", "data")
base_script_path = config.get("base_script_path", "kaggle_template/scripts")
KFOLD = config.get("kfold", 2)
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
        trials=TRIALS,
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
        trials=TRIALS,
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
        kfold=KFOLD,
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
        if not os.getenv('KAGGLE_URL_BASE'):
            shell("rm -rf temp; mkdir temp; cp dataset-metadata.json temp/")
            # shell("pip download --prefer-binary --dest kaggle_packages --platform any pytest-runner Cython wheel snakemake==7.32.4 pulp==2.7.0 kaleido==0.1.0 setuptools==42 tomli scikit-learn==1.3.0 ") # scikit-learn==1.3.0
            # shell("pip download  --no-binary=:none: --dest kaggle_packages    --ignore-requires-python --no-cache-dir pytest-runner Cython wheel snakemake==8.25.5 pulp==2.7.0 kaleido==0.1.0 setuptools==42 tomli scikit-learn==1.3.0")
            # shell("zip -r kaggle_packages.zip kaggle_packages ; mv kaggle_packages.zip kaggle_packages.mp4")
            shell("zip -r temp/kaggle_template.zip Snakefile Makefile kaggle_template submission.csv data/models data/output data/features")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            commit_sha = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('utf-8').strip()
            shell("kaggle datasets version  -p temp -m 'Updated at: {timestamp}, git commit: {commit_sha}'")
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
