# %%
import sys
from uuid import uuid4

import _pickle as cPickle
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBRegressor

# %%

TRAIN_DF = "data/features/train_wide.csv"
CATBOOST_MODEL = "data/models/catboost_model.pkl"
XGB_MODEL = "data/models/xgb_model.pkl"
RF_MODEL = "data/models/rf_model.pkl"
LGBM_MODEL = "data/models/lgbm_model.pkl"
TRAILS = 2
if "snakemake" in sys.modules:
    TRAIN_DF = snakemake.input.train
    CATBOOST_MODEL = snakemake.output.catboost
    XGB_MODEL = snakemake.output.xgb
    RF_MODEL = snakemake.output.rf
    LGBM_MODEL = snakemake.output.lgbm
    TRAILS = snakemake.params.trails

SEED = 42
df = pd.read_csv(TRAIN_DF)
X, y = df.drop(columns=["sii", "id"]), df["sii"]
# test_df = pd.read_csv(TEST_LONG_DF)
df.head()

# %%
df[list(set(df.columns) - set(["sii", "id"]))].dtypes.value_counts(dropna=False)

# %%
df.sii.value_counts()


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


def catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 10, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "random_seed": SEED,
        "verbose": False,
    }

    model = CatBoostRegressor(**params)

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


gaussian_sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    direction="maximize", sampler=gaussian_sampler, study_name=f"CatBoost_{uuid4()}"
)
study.optimize(catboost_objective, n_trials=TRAILS)

print("Best params for CatBoost:", study.best_params)

# train and save model
params = study.best_params
params["random_seed"] = SEED
params["verbose"] = False
model = CatBoostRegressor(**params)
model.fit(X, y)
model.save_model(CATBOOST_MODEL)


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
    direction="maximize", sampler=gaussian_sampler, study_name=f"XGB_{uuid4()}"
)
study.optimize(xgb_objective, n_trials=TRAILS)
print("Best params for XGB:", study.best_params)

# train and save model
params = study.best_params
params["random_state"] = SEED
model = XGBRegressor(**params)
model.fit(X, y)
model.save_model(XGB_MODEL)


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
    direction="maximize", sampler=gaussian_sampler, study_name=f"RF_{uuid4()}"
)
study.optimize(rf_objective, n_trials=TRAILS)
print("Best params for RF:", study.best_params)

# train and save model
params = study.best_params
params["random_state"] = SEED
model = RandomForestRegressor(**params)
model.fit(X, y)

with open(RF_MODEL, "wb") as f:
    cPickle.dump(model, f)

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
    direction="maximize", sampler=gaussian_sampler, study_name=f"LGBM_{uuid4()}"
)
study.optimize(lgbm_objective, n_trials=TRAILS)
print("Best params for LGBM:", study.best_params)

# %%
params = study.best_params
params["random_state"] = SEED
params["verbosity"] = -1
model = LGBMRegressor(**params)
model.fit(X, y)
model.booster_.save_model(LGBM_MODEL)
