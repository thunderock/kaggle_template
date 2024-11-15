# %%
import pickle
import sys
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBRegressor

# %%

TRAIN_DF = "data/features/train_features.csv"
TRAIN_WIDE_DF = "data/features/train_wide.csv"
TEST_DF = "data/features/test_features.csv"
TEST_WIDE_DF = "data/features/test_wide.csv"
CATBOOST_MODEL = "data/models/catboost_train_features.pkl"
XGB_MODEL = "data/models/xgb_train_features.pkl"
RF_MODEL = "data/models/rf_train_features.pkl"
LGBM_MODEL = "data/models/lgbm_train_features.pkl"
CATBOOST_WIDE_MODEL = "data/models/catboost_train_wide.pkl"
XGB_WIDE_MODEL = "data/models/xgb_train_wide.pkl"
RF_WIDE_MODEL = "data/models/rf_train_wide.pkl"
LGBM_WIDE_MODEL = "data/models/lgbm_train_wide.pkl"
TRAILS = 2
SUBMISSION_FILE = "data/submissions/submission.csv"
if "snakemake" in sys.modules:
    TRAIN_DF = snakemake.input.train
    CATBOOST = snakemake.output.catboost
    XGB = snakemake.output.xgb
    RF = snakemake.output.rf
    LGBM = snakemake.output.lgbm
    TEST_DF = snakemake.input.test
    CATBOOST_WIDE = snakemake.output.catboost_wide
    XGB_WIDE = snakemake.output.xgb_wide
    RF_WIDE = snakemake.output.rf_wide
    LGBM_WIDE = snakemake.output.lgbm_wide
    TEST_WIDE_DF = snakemake.input.test_wide
    TRAILS = snakemake.params.trails
    SUBMISSION_FILE = snakemake.output.submission

SEED = 42
df = pd.read_csv(TRAIN_DF)
wide_df = pd.read_csv(TRAIN_WIDE_DF)

X, y = df.drop(columns=["sii"]), df["sii"]
X_wide, y_wide = wide_df.drop(columns=["sii"]), wide_df["sii"]
ids, wide_ids = set(df["id"]), set(wide_df["id"])


# %%
def get_dictionary(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


base_models = [
    ("catboost", CatBoostRegressor(**get_dictionary(CATBOOST_MODEL))),
    ("xgb", XGBRegressor(**get_dictionary(XGB_MODEL))),
    ("rf", RandomForestRegressor(**get_dictionary(RF_MODEL))),
    ("lgbm", LGBMRegressor(**get_dictionary(LGBM_MODEL))),
    ("catboost_wide", CatBoostRegressor(**get_dictionary(CATBOOST_WIDE_MODEL))),
    ("xgb_wide", XGBRegressor(**get_dictionary(XGB_WIDE_MODEL))),
    ("rf_wide", RandomForestRegressor(**get_dictionary(RF_WIDE_MODEL))),
    ("lgbm_wide", LGBMRegressor(**get_dictionary(LGBM_WIDE_MODEL))),
]


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
    weight = 1.0 - wide_weight

    score = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in kf.split(X, y):
        train_ids, val_ids = set(X["id"].iloc[train_idx]), set(X["id"].iloc[test_idx])
        train_df = X.iloc[train_idx].drop(columns=["id"])  # all ids
        train_wide_df = X_wide[X_wide["id"].isin(train_ids)].drop(columns=["id"])
        y_train = y.iloc[train_idx]
        y_train_wide = y_wide[X_wide["id"].isin(train_ids)]

        ### validation data
        # validation only with ids that are not wide
        idxs = X[~X["id"].isin(wide_ids)].index
        val_df = X.iloc[test_idx & idxs].drop(columns=["id"])
        y_val = y.iloc[test_idx & idxs]
        # validation qith ids that are not wide but are in the train set
        idxs = X[X["id"].isin(wide_ids)].index
        val_weight_df = X.iloc[test_idx & idxs].drop(columns=["id"])
        y_val_weight = y.iloc[test_idx & idxs]
        # validation with wide ids
        val_wide_df = X_wide[X_wide["id"].isin(val_ids)].drop(columns=["id"])
        y_val_wide = y_wide[X_wide["id"].isin(val_ids)]

        meta_model = Ridge(alpha=alpha)
        wide_meta_model = Ridge(alpha=wide_alpha)

        meta_model.fit(train_df, y_train)
        wide_meta_model.fit(train_wide_df, y_train_wide)

        meta_preds = meta_model.predict(val_df)
        weight_meta_preds = meta_model.predict(val_weight_df)
        wide_meta_preds = wide_meta_model.predict(val_wide_df)

        scores = custom_cohen_kappa_scorer(y_val, meta_preds)
        weight_scores = custom_cohen_kappa_scorer(y_val_weight, weight_meta_preds)
        wide_scores = custom_cohen_kappa_scorer(y_val_wide, wide_meta_preds)
        score.append

    return score.mean()


sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    direction="maximize", sampler=sampler, study_name=f"CatBoost_{uuid4()}"
)
study.optimize(catboost_objective, n_trials=TRAILS)

params = study.best_params
params["random_seed"] = SEED
params["verbose"] = False
print("Best params for CatBoost:", params)

# train and save model
# model = CatBoostRegressor(**params)
# model.fit(X, y)
# model.save_model(CATBOOST_MODEL)
with open(CATBOOST_MODEL, "wb") as f:
    pickle.dump(params, f)


# %%


def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "random_state": SEED,
    }

    model = XGBRegressor(**params)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    score = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring=make_scorer(custom_cohen_kappa_scorer, greater_is_better=True),
        n_jobs=-1,
        verbose=2,
    )
    return score.mean()


study = optuna.create_study(
    direction="maximize", sampler=sampler, study_name=f"XGB_{uuid4()}"
)
study.optimize(xgb_objective, n_trials=TRAILS)

# train and save model
params = study.best_params
params["random_state"] = SEED
print("Best params for XGB:", params)
# model = XGBRegressor(**params)
# model.fit(X, y)
# model.save_model(XGB_MODEL)
#
with open(XGB_MODEL, "wb") as f:
    pickle.dump(params, f)


# %%
def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "random_state": SEED,
    }

    model = RandomForestRegressor(**params)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    score = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring=make_scorer(custom_cohen_kappa_scorer, greater_is_better=True),
        n_jobs=-1,
        verbose=2,
    )
    return score.mean()


study = optuna.create_study(
    direction="maximize", sampler=sampler, study_name=f"RF_{uuid4()}"
)
study.optimize(rf_objective, n_trials=TRAILS)

params = study.best_params
params["random_state"] = SEED

print("Best params for RF:", params)

# train and save model
# model = RandomForestRegressor(**params)
# model.fit(X, y)
#
# with open(RF_MODEL, "wb") as f:
#     cPickle.dump(model, f)

with open(RF_MODEL, "wb") as f:
    pickle.dump(params, f)
# %%


def lgbm_objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 8, 17),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_leaves": trial.suggest_int("n_leaves", 10, 1000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": -1,
    }

    model = LGBMRegressor(**params)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    score = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring=make_scorer(custom_cohen_kappa_scorer, greater_is_better=True),
        n_jobs=-1,
        verbose=2,
    )
    return score.mean()


study = optuna.create_study(
    direction="maximize", sampler=sampler, study_name=f"LGBM_{uuid4()}"
)
study.optimize(lgbm_objective, n_trials=TRAILS)

# %%
params = study.best_params
params["random_state"] = SEED
params["verbosity"] = -1
params["n_jobs"] = -1
print("Best params for LGBM:", study.best_params)
# model = LGBMRegressor(**params)
# model.fit(X, y)
# model.booster_.save_model(LGBM_MODEL)

with open(LGBM_MODEL, "wb") as f:
    pickle.dump(params, f)
