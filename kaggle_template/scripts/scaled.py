# %%
from numpy import number
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
from tqdm import tqdm
import sys

warnings.filterwarnings("ignore")

if "snakemake" in sys.modules:
    TRAIN_INPUT = snakemake.input.train
    TEST_INPUT = snakemake.input.test
    TRAIN_OUTPUT = snakemake.output.train
    TEST_OUTPUT = snakemake.output.test
else:
    TRAIN_INPUT = "data/input/train.csv"
    TEST_INPUT = "data/input/test.csv"
    TRAIN_OUTPUT = "data/features/train_encoded.csv"
    TEST_OUTPUT = "data/features/test_encoded.csv"


# %%
def process_file(filename, dirname):
    data = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
    data.drop("step", axis=1, inplace=True)
    return data.describe().values.reshape(-1), filename.split("=")[1]


def load_time_series(dirname):
    ids = os.listdir(dirname)
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda fname: process_file(fname, dirname), ids),
                total=len(ids),
            )
        )
    stats, indexes = zip(*results)
    data = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    data["id"] = indexes
    return data


# %%


train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

print(
    "columns not in test data: ",
    [f for f in train_df.columns if f not in test_df.columns],
)

# %%

# %%

# %%
BASE_FEATURES = test_df.drop("id", axis=1).columns
TEST_IDS = test_df["id"]

# %% [markdown]
# # analysis

# %%


# %%
train_df.dtypes.groupby(train_df.dtypes).size()

# %%
train_df.dtypes.groupby(train_df[BASE_FEATURES].dtypes).size()

# %%
test_df.dtypes.groupby(test_df[BASE_FEATURES].dtypes).size()

# %%
CATEGORICAL_FEATURES = train_df.select_dtypes(include="object").columns
NUMERICAL_FEATURES = train_df.select_dtypes(exclude="object").columns
RANDOM_STATE = 42
CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# %%

# categorical features with missing values
train_df[CATEGORICAL_FEATURES].isnull().sum().sort_values(ascending=False)

# %%
train_df[CATEGORICAL_FEATURES] = train_df[CATEGORICAL_FEATURES].fillna("missing")
test_df[[col for col in CATEGORICAL_FEATURES if col in test_df.columns]] = test_df[
    [col for col in CATEGORICAL_FEATURES if col in test_df.columns]
].fillna("missing")

# %%
train_df[NUMERICAL_FEATURES].isnull().sum().sort_values(ascending=False)

# %%
from sklearn.impute import KNNImputer, SimpleImputer


def fill_numerical_features(df):
    for feature in tqdm(NUMERICAL_FEATURES):
        if feature in df.columns and df[feature].isnull().sum() > 0:
            df["{feature}_median"] = df[feature].fillna(df[feature].median())
            df["{feature}_knn"] = KNNImputer(n_neighbors=5).fit_transform(
                df[feature].values.reshape(-1, 1)
            )
            df[feature] = SimpleImputer(strategy="mean").fit_transform(
                df[feature].values.reshape(-1, 1)
            )
    return df


train_df = fill_numerical_features(train_df)
test_df = fill_numerical_features(test_df)
print(train_df.isnull().sum(), test_df.isnull().sum())

# %%
from sklearn.preprocessing import LabelEncoder

# train_df.to_csv("../data/input/train_original_imputed.csv", index=False)
# test_df.to_csv("../data/input/test_original_imputed.csv", index=False)


def encode_categorical_features(feature, df, tdf):
    le = LabelEncoder()
    combined_data = df[feature].tolist()
    if feature in tdf.columns:
        combined_data += tdf[feature].tolist()
    le.fit(combined_data)

    df[feature] = le.transform(df[feature]).astype(int)
    if feature in tdf.columns:
        tdf[feature] = le.transform(tdf[feature]).astype(int)
    else:
        print(f"{feature} not in test data")
    return df, tdf


for feature in CATEGORICAL_FEATURES:
    train_df, test_df = encode_categorical_features(feature, train_df, test_df)


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_base_features = [
    f
    for f in tqdm(
        [feature for feature in BASE_FEATURES]
        + [f"{feature}_median" for feature in BASE_FEATURES]
        + [f"{feature}_knn" for feature in BASE_FEATURES]
    )
    if f in test_df.columns
]
train_df[numerical_base_features] = scaler.fit_transform(
    train_df[numerical_base_features]
)
test_df[numerical_base_features] = scaler.transform(test_df[numerical_base_features])
categorical_base_features = [f for f in CATEGORICAL_FEATURES if f in test_df.columns]
train_df = train_df[
    numerical_base_features + categorical_base_features + ["id"] + ["sii"]
]
test_df = test_df[numerical_base_features + categorical_base_features + ["id"]]
train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

# print base features
print("Base features: ", BASE_FEATURES)
print(
    "Categorical features not in test data: ",
    [f for f in CATEGORICAL_FEATURES if f not in test_df.columns],
)
print(
    "Numerical features not in test data: ",
    [f for f in NUMERICAL_FEATURES if f not in test_df.columns],
)

print("final categorical features: ", categorical_base_features)
print("final numerical features: ", numerical_base_features)
