import pickle
import sys
from abc import ABC, abstractmethod

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

# Common configurations
TRAIN_DF = "data/features/train_wide_features.csv"
THREADS = -1
TRIALS = 20
SEED = 42
SELECTED_MODEL = "catboost"
OUTPUT_PATH = "data/models/catboost_model.pkl"
if "snakemake" in sys.modules:
    TRAIN_DF = snakemake.input.train
    TRIALS = snakemake.params.trials
    SEED = snakemake.params.seed
    SELECTED_MODEL = snakemake.params.model
    OUTPUT_PATH = snakemake.output.output_path
    THREADS = snakemake.threads
print("debugging: ")
print("TRAIN_DF: ", TRAIN_DF)
print("TRIALS: ", TRIALS)
print("SEED: ", SEED)
print("SELECTED_MODEL: ", SELECTED_MODEL)
print("OUTPUT_PATH: ", OUTPUT_PATH)


df = pd.read_csv(TRAIN_DF)
X, y = df.drop(columns=["sii", "id"]), df["sii"]


# Common scorer function
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


class ModelTrainer(ABC):
    def __init__(self, X, y, threads=THREADS, seed=SEED):
        self.X = X
        self.y = y
        self.threads = threads
        self.seed = seed

    @abstractmethod
    def get_model(self, params):
        pass

    @abstractmethod
    def suggest_params(self, trial):
        pass

    @abstractmethod
    def get_fixed_params(self):
        """Return fixed parameters that are not optimized."""
        pass

    def objective(self, trial):
        params = self.suggest_params(trial)
        params.update(self.get_fixed_params())

        model = self.get_model(params)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        score = cross_val_score(
            model,
            self.X,
            self.y,
            cv=kf,
            scoring=make_scorer(custom_cohen_kappa_scorer, greater_is_better=True),
            n_jobs=self.threads,
        )
        return score.mean()

    def optimize(self, n_trials):
        sampler = optuna.samplers.TPESampler(multivariate=True)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name=SELECTED_MODEL
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=self.threads,
        )

        # Get the best parameters and update with fixed params
        best_params = study.best_params
        best_params.update(self.get_fixed_params())
        return best_params


class CatBoostTrainer(ModelTrainer):
    def get_model(self, params):
        return CatBoostRegressor(**params)

    def suggest_params(self, trial):
        return {
            "iterations": trial.suggest_int("iterations", 10, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    def get_fixed_params(self):
        return {"random_seed": self.seed, "verbose": False}


class XGBTrainer(ModelTrainer):
    def get_model(self, params):
        return XGBRegressor(**params)

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    def get_fixed_params(self):
        return {"random_state": self.seed}


class RandomForestTrainer(ModelTrainer):
    def get_model(self, params):
        return RandomForestRegressor(**params)

    def suggest_params(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.0, 1.0),
        }

    def get_fixed_params(self):
        return {"random_state": self.seed, "bootstrap": True}


class LGBMTrainer(ModelTrainer):
    def get_model(self, params):
        return LGBMRegressor(**params)

    def suggest_params(self, trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 8, 18),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 600),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 20),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.5),
        }

    def get_fixed_params(self):
        return {"random_state": self.seed, "verbosity": -1}


# Usage
trainers = {
    "catboost": CatBoostTrainer(X, y),
    "xgb": XGBTrainer(X, y),
    "rf": RandomForestTrainer(X, y),
    "lgbm": LGBMTrainer(X, y),
}


def get_trainer(name):
    for key, trainer in trainers.items():
        if name == f"{key}_train" or name == f"{key}_train_wide":
            return trainer
    raise ValueError(f"Trainer {name} not found")


trainer = get_trainer(SELECTED_MODEL)
best_params = trainer.optimize(n_trials=TRIALS)

print(f"Best params for {SELECTED_MODEL}:", best_params)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(best_params, f)
