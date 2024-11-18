from kaggle_template.utils.run_utils import GPU_CORES, CPU_CORES
print(GPU_CORES, CPU_CORES)
COMPETITION = "child-mind-institute-problematic-internet-use"
train_files = ["train_features", "train_wide"]
models = {
    "catboost": "data/models/catboost_{train_file}.pkl",
    "xgb": "data/models/xgb_{train_file}.pkl",
    "rf": "data/models/rf_{train_file}.pkl",
    "lgbm": "data/models/lgbm_{train_file}.pkl",
}


rule all:
    input:
        expand("data/models/{model}_{train_file}.pkl", model=models.keys(), train_file=train_files),
        "data/models/meta_model.pkl"

rule combine_features:
    input:
        train_features="data/features/train_features.csv",
        test_features="data/features/test_features.csv",
        train_timeseries="data/features/train_timeseries.csv",
        test_timeseries="data/features/test_timeseries.csv",
    output:
        train_wide_df="data/features/train_wide.csv",
        test_wide_df="data/features/test_wide.csv",
    threads: 1
    script: "kaggle_template/scripts/combine_features.py"
    # shell:
    #     "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/combine_features.py"

rule generate_timeseries:
    input:
        train=directory("data/input/series_train.parquet"),
        test=directory("data/input/series_test.parquet"),
    output:
        train="data/features/train_timeseries.csv",
        test="data/features/test_timeseries.csv",
    threads: 12
    script: "kaggle_template/scripts/timeseries.py"
    # shell:
    #     "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/timeseries.py"

rule download_data:
    output:
        zip="data/input/{competition}.zip".format(competition=COMPETITION),
        train="data/input/train.csv",
        test="data/input/test.csv",
        timeseries_train=directory("data/input/series_train.parquet"),
        timeseries_test=directory("data/input/series_test.parquet"),
    params:
        competition=COMPETITION
    threads: 1
    script: "kaggle_template/scripts/download_dataset.py"
    #  shell:
    #      "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/download_dataset.py"

rule generate_features:
    input:
        train="data/input/train.csv",
        test="data/input/test.csv",
    output:
        train="data/features/train_features.csv",
        test="data/features/test_features.csv",
    threads: 1
    script: "kaggle_template/scripts/scaled.py"
    #  shell:
    #      "CUDA_VISIBLE_DEVICES=0,2 python kaggle_template/scripts/scaled.py"


rule tune_model:
    input:
        train=expand("data/features/{train_file}.csv", train_file=train_files),
    output:
        model_path=expand("data/models/{model}_{train_file}.pkl", model=models.keys(), train_file=train_files),
    params:
        trials=100,
        seed=42,
        model=expand("{model}", model=models.keys()),
    script: "kaggle_template/scripts/tune_model.py"

rule tune_meta_model:
    input:
        train="data/features/train_wide.csv",
        train_wide="data/features/train_wide.csv",
    output:
        meta_model="data/models/meta_model.pkl",
    params:
        trails=100,
        seed=42,
    script: "kaggle_template/scripts/tune_meta_model.py"
# rule tune_stack_regression_and_predict:
#     input:
#         train_wide="data/features/train_wide.csv",
#         train="data/features/train_features.csv",
#         test="data/features/test_features.csv",
#         test_wide="data/features/test_wide.csv",
#         catboost_wide="data/models/catboost_train_wide.pkl",
#         xgb_wide="data/models/xgb_train_wide.pkl",
#         rf_wide="data/models/rf_train_wide.pkl",
#         lgbm_wide="data/models/lgbm_train_wide.pkl",
#         catboost="data/models/catboost_train_features.pkl",
#         xgb="data/models/xgb_train_features.pkl",
#         rf="data/models/rf_train_features.pkl",
#         lgbm="data/models/lgbm_train_features.pkl",
#     params:
#         trails=100
#     output:
#         submission="data/submissions/submission.csv"
#     script: "kaggle_template/scripts/tune_stack_regression_and_predict.py"


# rule generate_dag:
#     output:
#         "dag.pdf"
#     threads: 1
#     shell:
#         "snakemake --dag | dot -Tpdf > dag.pdf"
