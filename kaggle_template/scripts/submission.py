import importlib.util
import json
import os
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBRegressor

if os.getenv("KAGGLE_URL_BASE"):
    current_dir = os.path.dirname(__file__)
    target_path = os.path.join(current_dir, "../../kaggle_template/utils/run_utils.py")
    target_path = os.path.abspath(target_path)

    spec = importlib.util.spec_from_file_location("run_utils", target_path)
    run_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_utils)

    get_dframe_with_features_by_threshold = (
        run_utils.get_dframe_with_features_by_threshold
    )
else:
    from kaggle_template.utils.run_utils import get_dframe_with_features_by_threshold

warnings.filterwarnings("ignore")


def read_dictionary(json_file):
    if isinstance(json_file, dict):
        return json_file
    with open(json_file, "r") as file:
        return json.load(file)


RF_TRAIN_PARAMS = "data/models/rf_train_features.json"
RF_TRAIN_WIDE_PARAMS = "data/models/rf_train_wide_features.json"
CATBOOST_TRAIN_PARAMS = "data/models/catboost_train_features.json"
CATBOOST_TRAIN_WIDE_PARAMS = "data/models/catboost_train_wide_features.json"
XGB_TRAIN_PARAMS = "data/models/xgb_train_features.json"
XGB_TRAIN_WIDE_PARAMS = "data/models/xgb_train_wide_features.json"
LGBM_TRAIN_PARAMS = "data/models/lgbm_train_features.json"
LGBM_TRAIN_WIDE_PARAMS = "data/models/lgbm_train_wide_features.json"
META_MODEL = "data/models/meta_model.json"
TRAIN_CSV = "data/features/train_features.csv"
TRAIN_WIDE_CSV = "data/features/train_wide_features.csv"
TEST_CSV = "data/features/test_features.csv"
TEST_WIDE_CSV = "data/features/test_wide_features.csv"
PREDICTIONS_TO_ANALYZE = "predictions_to_analyze.csv"
PREDICTIONS_TO_SUBMIT = "predictions_to_submit.csv"
RANDOM_STATE = 42
THREADS = 4
FEATURE_SELECTION_THRESHOLD = 0.7
KFOLD = 10

if "snakemake" in sys.modules:
    # RF_TRAIN_PARAMS = snakemake.input.rf
    # RF_TRAIN_WIDE_PARAMS = snakemake.input.rf_wide
    CATBOOST_TRAIN_PARAMS = snakemake.input.catboost
    CATBOOST_TRAIN_WIDE_PARAMS = snakemake.input.catboost_wide
    XGB_TRAIN_PARAMS = snakemake.input.xgb
    XGB_TRAIN_WIDE_PARAMS = snakemake.input.xgb_wide
    LGBM_TRAIN_PARAMS = snakemake.input.lgbm
    LGBM_TRAIN_WIDE_PARAMS = snakemake.input.lgbm_wide
    META_MODEL = snakemake.input.meta_model
    TRAIN_CSV = snakemake.input.train
    TRAIN_WIDE_CSV = snakemake.input.train_wide
    TEST_CSV = snakemake.input.test
    TEST_WIDE_CSV = snakemake.input.test_wide
    PREDICTIONS_TO_ANALYZE = snakemake.output.analyze
    PREDICTIONS_TO_SUBMIT = snakemake.output.predictions
    RANDOM_STATE = snakemake.params.seed
    THREADS = snakemake.threads
    FEATURE_SELECTION_THRESHOLD = snakemake.params.feature_selection_threshold
    KFOLD = snakemake.params.kfold


CATBOOST_TRAIN_PARAMS: dict[str:Any] = {
    "learning_rate": 0.052335599837526434,
    "depth": 6,
    "iterations": 352,
    "l2_leaf_reg": 15,
    "random_seed": 42,
    "verbose": False,
}
CATBOOST_TRAIN_WIDE_PARAMS: dict[str:Any] = {
    "learning_rate": 0.07739082107142103,
    "depth": 4,
    "iterations": 254,
    "l2_leaf_reg": 10,
    "random_seed": 42,
    "verbose": False,
}
LGBM_TRAIN_PARAMS: dict[str:Any] = {
    "max_depth": 14,
    "learning_rate": 0.05266538108576567,
    "num_leaves": 439,
    "min_data_in_leaf": 11,
    "bagging_fraction": 0.5372487838046379,
    "bagging_freq": 1,
    "lambda_l1": 5.861462412806318,
    "lambda_l2": 0.07292117586631806,
    "random_state": 42,
    "verbosity": -1,
}
LGBM_TRAIN_WIDE_PARAMS: dict[str:Any] = {
    "max_depth": 17,
    "learning_rate": 0.061251438480778755,
    "num_leaves": 117,
    "min_data_in_leaf": 14,
    "bagging_fraction": 0.6245504220583282,
    "bagging_freq": 5,
    "lambda_l1": 6.453079863609765,
    "lambda_l2": 0.012166685442452725,
    "random_state": 42,
    "verbosity": -1,
}
META_MODEL: dict[str:Any] = {
    "alpha": 0.8861230517534572,
    "wide_alpha": 0.09192897203020362,
    "wide_weight": 0.4221394292247178,
}
RF_TRAIN_PARAMS: dict[str:Any] = {
    "n_estimators": 400,
    "max_depth": 40,
    "min_samples_split": 4,
    "min_samples_leaf": 5,
    "max_features": 0.8099200178604307,
    "random_state": 42,
    "bootstrap": True,
}

RF_TRAIN_WIDE_PARAMS: dict[str:Any] = {
    "n_estimators": 85,
    "max_depth": 13,
    "min_samples_split": 15,
    "min_samples_leaf": 4,
    "max_features": 0.8851446675024941,
    "random_state": 42,
    "bootstrap": True,
}
XGB_TRAIN_PARAMS: dict[str:Any] = {
    "learning_rate": 0.026291134093558096,
    "max_depth": 5,
    "n_estimators": 740,
    "subsample": 0.7116355627180203,
    "colsample_bytree": 0.635298144423355,
    "gamma": 2,
    "reg_alpha": 3,
    "reg_lambda": 2,
    "random_state": 42,
    "verbosity": 0,
}

XGB_TRAIN_WIDE_PARAMS: dict[str:Any] = {
    "learning_rate": 0.07065769767808697,
    "max_depth": 10,
    "n_estimators": 563,
    "subsample": 0.6045349329921547,
    "colsample_bytree": 0.760205453222906,
    "gamma": 2,
    "reg_alpha": 3,
    "reg_lambda": 1,
    "random_state": 42,
    "verbosity": 0,
}

print("DEBUG LOGGING: ")
print("RF_TRAIN_PARAMS: ", read_dictionary(RF_TRAIN_PARAMS))
print("RF_TRAIN_WIDE_PARAMS: ", read_dictionary(RF_TRAIN_WIDE_PARAMS))
print("CATBOOST_TRAIN_PARAMS: ", read_dictionary(CATBOOST_TRAIN_PARAMS))
print("CATBOOST_TRAIN_WIDE_PARAMS: ", read_dictionary(CATBOOST_TRAIN_WIDE_PARAMS))
print("XGB_TRAIN_PARAMS: ", read_dictionary(XGB_TRAIN_PARAMS))
print("XGB_TRAIN_WIDE_PARAMS: ", read_dictionary(XGB_TRAIN_WIDE_PARAMS))
print("LGBM_TRAIN_PARAMS: ", read_dictionary(LGBM_TRAIN_PARAMS))
print("LGBM_TRAIN_WIDE_PARAMS: ", read_dictionary(LGBM_TRAIN_WIDE_PARAMS))
print("META_MODEL: ", read_dictionary(META_MODEL))

base_models = [
    # ("rf", RandomForestRegressor(**read_dictionary(RF_TRAIN_PARAMS))),
    ("catboost", CatBoostRegressor(**read_dictionary(CATBOOST_TRAIN_PARAMS))),
    ("xgb", XGBRegressor(**read_dictionary(XGB_TRAIN_PARAMS))),
    ("lgbm", LGBMRegressor(**read_dictionary(LGBM_TRAIN_PARAMS))),
]

base_wide_models = [
    # ("rf", RandomForestRegressor(**read_dictionary(RF_TRAIN_WIDE_PARAMS))),
    ("catboost", CatBoostRegressor(**read_dictionary(CATBOOST_TRAIN_WIDE_PARAMS))),
    ("xgb", XGBRegressor(**read_dictionary(XGB_TRAIN_WIDE_PARAMS))),
    ("lgbm", LGBMRegressor(**read_dictionary(LGBM_TRAIN_WIDE_PARAMS))),
]


def get_base_model(name):
    for model_name, model in base_models:
        if model_name == name:
            return model
    raise ValueError(f"Model {name} not found in base models")


meta_dict = read_dictionary(META_MODEL)
meta_model = Ridge(alpha=meta_dict["alpha"], random_state=RANDOM_STATE)
meta_wide_model = Ridge(alpha=meta_dict["wide_alpha"], random_state=RANDOM_STATE)
wide_model_weight = meta_dict["wide_weight"]

kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)


def custom_cohen_kappa_scorer(y_true, y_pred):
    initial_thresholds = [0.5, 1.5, 2.5]

    def objective(thresholds):
        thresholds = np.sort(thresholds)
        y_pred_classes = np.digitize(y_pred, thresholds)
        return -cohen_kappa_score(y_true, y_pred_classes, weights="quadratic")

    result = minimize(objective, initial_thresholds, method="Nelder-Mead")
    best_thresholds = np.sort(result.x)
    y_pred_classes = np.digitize(y_pred, best_thresholds)
    return (
        cohen_kappa_score(y_true, y_pred_classes, weights="quadratic"),
        best_thresholds,
        y_pred_classes,
    )


train_df = get_dframe_with_features_by_threshold(
    pd.read_csv(TRAIN_CSV), FEATURE_SELECTION_THRESHOLD
)
train_wide_df = get_dframe_with_features_by_threshold(
    pd.read_csv(TRAIN_WIDE_CSV), FEATURE_SELECTION_THRESHOLD
)
train_df = train_df.set_index("id", drop=False)
train_wide_df = train_wide_df.set_index("id", drop=False)
test_df = pd.read_csv(TEST_CSV).set_index("id", drop=False)
test_wide_df = pd.read_csv(TEST_WIDE_CSV).set_index("id", drop=False)
train_df.index.name = None
train_wide_df.index.name = None
test_df.index.name = None
test_wide_df.index.name = None

predictions_to_analyze = pd.DataFrame(
    {
        "id": train_df["id"].values,
        "original": train_df["sii"].values,
    }
).set_index("id")
predictions_to_submit = pd.DataFrame({"id": test_df["id"].values})

model_scores: dict[str : list[float]] = {name: [] for name, _ in base_models}
wide_model_scores: dict[str : list[float]] = {name: [] for name, _ in base_wide_models}
X, y = train_df.drop("sii", axis=1), train_df["sii"]


def generate_test_predictions(narrow_df, wide_df, wide_score_weight):
    # Ensure both dataframes have the required columns
    assert all(col in narrow_df.columns for col in ["id", "pred"])
    assert all(col in wide_df.columns for col in ["id", "pred"])
    # Ensure `wide_df` IDs are a subset of `narrow_df` IDs
    assert set(wide_df["id"]).issubset(
        set(narrow_df["id"])
    ), f"{set(wide_df['id'])} not subset of {set(narrow_df['id'])}"
    assert narrow_df.shape[0] >= wide_df.shape[0]

    # Create a dictionary for faster lookups of wide predictions
    wide_pred_dict = dict(zip(wide_df["id"], wide_df["pred"]))
    narrow_pred_dict = dict(zip(narrow_df["id"], narrow_df["pred"]))

    # Generate predictions by vectorized processing
    def compute_score(row):
        id_ = row["id"]
        if id_ in wide_pred_dict:
            return (
                wide_score_weight * wide_pred_dict[id_]
                + (1 - wide_score_weight) * narrow_pred_dict[id_]
            )
        return narrow_pred_dict[id_]

    narrow_df["pred"] = narrow_df.apply(compute_score, axis=1)
    return narrow_df[["id", "pred"]]


stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_wide_model = StackingRegressor(
    estimators=base_wide_models, final_estimator=meta_wide_model
)

for idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(X, y))):
    train_ids, val_ids = set(X["id"].iloc[train_idx]), set(X["id"].iloc[test_idx])
    val_wide_ids = set(train_wide_df["id"]) & val_ids

    ## TRAINING DATA
    # narrow data to train
    train_df_ = X.iloc[train_idx].drop(columns="id").reset_index(drop=True)
    train_y_ = y.iloc[train_idx].reset_index(drop=True)

    # wide data to train
    train_wide_df_ = (
        train_wide_df[train_wide_df.index.isin(train_ids)]
        .drop(columns=["id", "sii"])
        .reset_index(drop=True)
    )
    train_wide_y_ = train_wide_df[train_wide_df.index.isin(train_ids)][
        "sii"
    ].reset_index(drop=True)

    ## VALIDATION DATA
    # narrow data to val
    val_df_ = X.iloc[test_idx].reset_index(drop=True)
    val_y_ = y.iloc[test_idx].reset_index(drop=True)

    # wide data to val to be combined with weights
    val_wide_df_ = train_wide_df[train_wide_df.index.isin(val_ids)].sort_values("id")
    val_wide_y_ = (
        train_wide_df[train_wide_df.index.isin(val_ids)]
        .sort_values("id")
        .drop(columns="id")
        .reset_index(drop=True)[["sii"]]
    )

    val_wide_df_from_narrow_ = (
        X[X.index.isin(val_wide_ids)].sort_values("id").reset_index(drop=True)
    )
    val_wide_y_from_narrow_ = (
        y[y.index.isin(val_wide_ids)]
        .rename_axis("id")
        .reset_index()
        .sort_values("id")
        .drop(columns="id")
        .reset_index(drop=True)
    )
    assert (
        val_wide_y_from_narrow_.values == val_wide_y_.values
    ).all(), f"{val_wide_y_from_narrow_} != {val_wide_y_}"
    narrow_columns = train_df_.columns
    wide_columns = train_wide_df_.columns

    for name, model in base_models:
        model.fit(train_df_, train_y_)
        y_pred = model.predict(val_df_[narrow_columns])
        score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
            val_y_, y_pred
        )
        model_scores[name].append(score)
        print(f"Model: {name}, Score: {score}")

    for name, wide_model in base_wide_models:
        wide_model.fit(train_wide_df_, train_wide_y_)
        y_pred = wide_model.predict(val_wide_df_[wide_columns])
        score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
            val_wide_y_from_narrow_, y_pred
        )
        wide_model_scores[name].append(score)
        print(f"Wide Model: {name}, Score: {score}")

    stacking_model.fit(train_df_, train_y_)
    stacking_wide_model.fit(train_wide_df_, train_wide_y_)

    # predictions to see if method works

    y_pred = generate_test_predictions(
        narrow_df=pd.DataFrame(
            {
                "id": val_df_["id"],
                "pred": stacking_model.predict(val_df_[narrow_columns]),
            }
        ),
        wide_df=pd.DataFrame(
            {
                "id": val_wide_df_["id"],
                "pred": stacking_wide_model.predict(val_wide_df_[wide_columns]),
            }
        ),
        wide_score_weight=wide_model_weight,
    )
    score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
        val_y_, y_pred.pred.values
    )
    y_pred["class"] = pred_classes

    print(f"Stacking Model: Score: {score}")
    predictions_to_analyze[f"pred_{idx}"] = pd.Series(
        [pd.NA] * len(predictions_to_analyze), dtype="Int32"
    )
    # fill values for predictions to analyze

    y_pred = y_pred.set_index("id")
    predictions_to_analyze[f"pred_{idx}"].update(y_pred["class"])
    # final predictions
    test_predictions = generate_test_predictions(
        narrow_df=pd.DataFrame(
            {
                "id": test_df["id"],
                "pred": stacking_model.predict(test_df[narrow_columns]),
            }
        ),
        wide_df=pd.DataFrame(
            {
                "id": test_wide_df["id"],
                "pred": stacking_wide_model.predict(test_wide_df[wide_columns]),
            }
        ),
        wide_score_weight=wide_model_weight,
    )
    test_predictions["class"] = test_predictions["pred"].apply(
        lambda x: np.digitize(x, model_thresholds)
    )

    predictions_to_submit[f"pred_{idx}"] = test_predictions["class"].values

# predictions to analyse df, id with 10 columns
predictions_to_analyze.reset_index().to_csv(PREDICTIONS_TO_ANALYZE, index=False)
predictions_to_submit["sii"] = (
    predictions_to_submit[[f"pred_{i}" for i in range(KFOLD)]].mean(axis=1).astype(int)
)
predictions_to_submit[["id", "sii"]].to_csv(PREDICTIONS_TO_SUBMIT, index=False)

print("Model Scores: ", model_scores)
print("Wide Model Scores: ", wide_model_scores)
