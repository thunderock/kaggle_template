import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBRegressor


def read_dictionary(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ['rf_train_features.pkl',
#  'meta_model.pkl',
#  'rf_train_wide_features.pkl',
#  'catboost_train_wide_features.pkl',
#  'xgb_train_wide_features.pkl',
#  'catboost_train_features.pkl',
#  'lgbm_train_wide_features.pkl',
#  'xgb_train_features.pkl',
#  'lgbm_train_features.pkl']
RF_TRAIN_PARAMS = "data/rf_train_features.pkl"
RF_TRAIN_WIDE_PARAMS = "data/rf_train_wide_features.pkl"
CATBOOST_TRAIN_PARAMS = "data/catboost_train_features.pkl"
CATBOOST_TRAIN_WIDE_PARAMS = "data/catboost_train_wide_features.pkl"
XGB_TRAIN_PARAMS = "data/xgb_train_features.pkl"
XGB_TRAIN_WIDE_PARAMS = "data/xgb_train_wide_features.pkl"
LGBM_TRAIN_PARAMS = "data/lgbm_train_features.pkl"
LGBM_TRAIN_WIDE_PARAMS = "data/lgbm_train_wide_features.pkl"
META_MODEL = "data/meta_model.pkl"
RANDOM_STATE = 42
TRAIN_CSV = "data/features/train_features.csv"
TRAIN_WIDE_CSV = "data/features/train_wide_features.csv"
TEST_CSV = "data/features/test_features.csv"
TEST_WIDE_CSV = "data/features/test_wide_features.csv"


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

base_models = {
    "rf": RandomForestRegressor(**read_dictionary(RF_TRAIN_PARAMS)),
    "catboost": CatBoostRegressor(**read_dictionary(CATBOOST_TRAIN_PARAMS)),
    "xgb": XGBRegressor(**read_dictionary(XGB_TRAIN_PARAMS)),
    "lgbm": LGBMRegressor(**read_dictionary(LGBM_TRAIN_PARAMS)),
}

base_wide_models = {
    "rf": RandomForestRegressor(**read_dictionary(RF_TRAIN_WIDE_PARAMS)),
    "catboost": CatBoostRegressor(**read_dictionary(CATBOOST_TRAIN_WIDE_PARAMS)),
    "xgb": XGBRegressor(**read_dictionary(XGB_TRAIN_WIDE_PARAMS)),
    "lgbm": LGBMRegressor(**read_dictionary(LGBM_TRAIN_WIDE_PARAMS)),
}
meta_dict = read_dictionary(META_MODEL)
meta_model = Ridge(alpha=meta_dict["alpha"], random_state=RANDOM_STATE)
meta_wide_model = Ridge(alpha=meta_dict["wide_alpha"], random_state=RANDOM_STATE)
wide_model_weight = meta_dict["wide_weight"]

stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_wide_model = StackingRegressor(
    estimators=base_wide_models, final_estimator=meta_wide_model
)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


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


train_df = pd.read_csv(TRAIN_CSV).set_index("id", drop=False)
train_wide_df = pd.read_csv(TRAIN_WIDE_CSV).set_index("id", drop=False)
test_df = pd.read_csv(TEST_CSV).set_index("id", drop=False)
test_wide_df = pd.read_csv(TEST_WIDE_CSV).set_index("id", drop=False)
train_df.index.name = None
train_wide_df.index.name = None
test_df.index.name = None
test_wide_df.index.name = None

predictions_to_analyze = np.zeros(train_df.shape[0])
stack_regressor_scores = []
wide_stack_regressor_scores = []
predictions_to_submit = np.zeros(test_df.shape[0])

model_scores: dict[str : list[float]] = {name: [] for name in base_models.keys()}
wide_model_scores: dict[str : list[float]] = {
    name: [] for name in base_wide_models.keys()
}
X, y = train_df.drop("sii", axis=1), train_df["sii"]


def generate_test_predictions(
    narrow_df,
    wide_df,
    wide_score_weight,
):
    pass


for idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    train_ids, val_ids = set(X["id"].iloc[train_idx]), set(X["id"].iloc[test_idx])
    val_wide_ids = set(train_wide_df["id"]) & val_ids

    ## TRAINING DATA
    # narrow data to train
    train_df_ = X.iloc[train_idx].drop(columns="id").reset_index(drop=True)
    train_y_ = y.iloc[train_idx].reset_index(drop=True)

    # wide data to train
    train_wide_df_ = (
        train_wide_df[train_wide_df.index.isin(train_ids)]
        .drop(columns="id")
        .reset_index(drop=True)
    )
    train_wide_y_ = train_wide_df[train_wide_df.index.isin(train_ids)][
        "sii"
    ].reset_index(drop=True)

    ## VALIDATION DATA
    # narrow data to val
    val_df_ = X.iloc[test_idx].drop(columns="id").reset_index(drop=True)
    val_y_ = y.iloc[test_idx].reset_index(drop=True)

    # wide data to val to be combined with weights
    val_wide_df_ = (
        train_wide_df[train_wide_df.index.isin(val_ids)]
        .sort_values("id")
        .drop(columns="id")
        .reset_index(drop=True)
    )
    val_wide_y_ = (
        train_wide_df[train_wide_df.index.isin(val_ids)]
        .sort_values("id")
        .drop(columns="id")
        .reset_index(drop=True)[["sii"]]
    )

    val_wide_df_from_narrow_ = (
        X[X.index.isin(val_wide_ids)]
        .sort_values("id")
        .drop(columns="id")
        .reset_index(drop=True)
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
    ).all(), f"{val_wide_df_from_narrow_} != {val_wide_df_}"

    for name, model in base_models:
        model.fit(train_df_, train_y_)
        y_pred = model.predict(val_df_)
        score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
            val_y_, y_pred
        )
        model_scores[name].append(score)
        print(f"Model: {name}, Score: {score}")

    for name, wide_model in base_wide_models:
        wide_model.fit(train_wide_df_, train_wide_y_)
        model = base_models[name]
        model.fit(train_df_, train_y_)
        y_pred = wide_model_weight * wide_model.predict(val_wide_df_) + (
            1 - wide_model_weight
        ) * model.predict(val_wide_df_from_narrow_)
        score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
            val_wide_y_from_narrow_, y_pred
        )
        wide_model_scores[name].append(score)
        print(f"Wide Model: {name}, Score: {score}")

    stacking_model.fit(train_df_, train_y_)
    stacking_wide_model.fit(train_wide_df_, train_wide_y_)

    y_pred = stacking_model.predict(val_df_)
    score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(val_y_, y_pred)
    stack_regressor_scores.append(score)
    print(f"Stacking Model: Score: {score}")

    # 1. narrow_test_df
    # 2. narrow_test_df which has ids from wide_test_df
    # 3. wide_test_df
    narrow_columns = train_df_.columns
    wide_columns = train_wide_df_.columns

    y_pred = wide_model_weight * stacking_wide_model.predict(val_wide_df_) + (
        1 - wide_model_weight
    ) * stacking_model.predict(val_wide_df_from_narrow_)
    score, model_thresholds, pred_classes = custom_cohen_kappa_scorer(
        val_wide_y_from_narrow_, y_pred
    )
    wide_stack_regressor_scores.append(score)
    print(f"Wide Stacking Model: Score: {score}")
