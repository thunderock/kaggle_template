# %%
import pickle
import sys

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

# %%

TRAIN_DF = "data/features/train_features.csv"
TRAIN_WIDE_DF = "data/features/train_wide.csv"
# TEST_DF = "data/features/test_features.csv"
# TEST_WIDE_DF = "data/features/test_wide.csv"
# CATBOOST_MODEL = "data/models/catboost_train_features.pkl"
# XGB_MODEL = "data/models/xgb_train_features.pkl"
# RF_MODEL = "data/models/rf_train_features.pkl"
# LGBM_MODEL = "data/models/lgbm_train_features.pkl"
# CATBOOST_WIDE_MODEL = "data/models/catboost_train_wide.pkl"
# XGB_WIDE_MODEL = "data/models/xgb_train_wide.pkl"
# RF_WIDE_MODEL = "data/models/rf_train_wide.pkl"
# LGBM_WIDE_MODEL = "data/models/lgbm_train_wide.pkl"
META_MODEL = "data/models/meta_model.pkl"
TRAILS = 2
SEED = 42
THREADS = 2
if "snakemake" in sys.modules:
    TRAIN_DF = snakemake.input.train
    TRAIN_WIDE_DF = snakemake.input.train_wide
    # CATBOOST = snakemake.output.catboost
    # XGB = snakemake.output.xgb
    # RF = snakemake.output.rf
    # LGBM = snakemake.output.lgbm
    # TEST_DF = snakemake.input.test
    # CATBOOST_WIDE = snakemake.output.catboost_wide
    # XGB_WIDE = snakemake.output.xgb_wide
    # RF_WIDE = snakemake.output.rf_wide
    # LGBM_WIDE = snakemake.output.lgbm_wide
    # TEST_WIDE_DF = snakemake.input.test_wide
    TRAILS = snakemake.params.trails
    META_MODEL = snakemake.output.meta_model
    SEED = snakemake.params.seed
    THREADS = snakemake.threads

train_df = pd.read_csv(TRAIN_DF).set_index("id", drop=False)
train_wide_df = pd.read_csv(TRAIN_WIDE_DF).set_index("id", drop=False)
train_df.index.name = None
train_wide_df.index.name = None

X, y = train_df.drop("sii", axis=1), train_df["sii"]


# %%
def get_dictionary(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def write_dictionary(file_name, dictionary):
    with open(file_name, "wb") as f:
        pickle.dump(dictionary, f)


# base_models = [
#     ("catboost", CatBoostRegressor(**get_dictionary(CATBOOST_MODEL))),
#     ("xgb", XGBRegressor(**get_dictionary(XGB_MODEL))),
#     ("rf", RandomForestRegressor(**get_dictionary(RF_MODEL))),
#     ("lgbm", LGBMRegressor(**get_dictionary(LGBM_MODEL))),
#     ("catboost_wide", CatBoostRegressor(**get_dictionary(CATBOOST_WIDE_MODEL))),
#     ("xgb_wide", XGBRegressor(**get_dictionary(XGB_WIDE_MODEL))),
#     ("rf_wide", RandomForestRegressor(**get_dictionary(RF_WIDE_MODEL))),
#     ("lgbm_wide", LGBMRegressor(**get_dictionary(LGBM_WIDE_MODEL))),
# ]


# %%
def custom_cohen_kappa_scorer(y_true, y_pred):
    initial_thresholds = [0.5, 1.5, 2.5]

    def objective(thresholds):
        thresholds = np.sort(thresholds)
        y_pred_classes = np.digitize(y_pred, thresholds)
        return -cohen_kappa_score(y_true, y_pred_classes, weights="quadratic")

    result = minimize(objective, initial_thresholds, method="Nelder-Mead")
    best_thresholds = np.sort(result.x)

    y_pred_classes = np.digitize(y_pred, best_thresholds)

    return cohen_kappa_score(y_true, y_pred_classes, weights="quadratic")


def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
    wide_alpha = trial.suggest_float("wide_alpha", 1e-4, 10, log=True)
    wide_weight = trial.suggest_float("wide_weight", 0, 1)

    score = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in kf.split(X, y):
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

        meta_model = Ridge(alpha=alpha)
        wide_meta_model = Ridge(alpha=wide_alpha)

        meta_model.fit(train_df_, train_y_)
        wide_meta_model.fit(train_wide_df_, train_wide_y_)

        meta_preds = meta_model.predict(val_df_)
        weight_meta_preds = meta_model.predict(val_wide_df_from_narrow_)
        wide_meta_preds = wide_meta_model.predict(val_wide_df_)

        narrow_score = custom_cohen_kappa_scorer(val_y_, meta_preds)
        wide_score_from_narrow = custom_cohen_kappa_scorer(
            val_wide_y_from_narrow_, weight_meta_preds
        )
        wide_score = custom_cohen_kappa_scorer(val_wide_y_, wide_meta_preds)
        wide_valid_ratio = len(val_wide_ids) / len(val_ids)
        f_score = (
            wide_valid_ratio
            * (wide_score * wide_weight + wide_score_from_narrow * (1 - wide_weight))
            + (1 - wide_valid_ratio) * narrow_score
        )
        score.append(f_score)

    return np.mean(score)


sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    direction="maximize", sampler=sampler, study_name="meta_model"
)
study.optimize(objective, n_trials=TRAILS, show_progress_bar=True, n_jobs=THREADS)

params = study.best_params
print("Best params for meta model:", params)
write_dictionary(META_MODEL, params)
