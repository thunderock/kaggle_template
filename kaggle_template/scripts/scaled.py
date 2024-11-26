# %%
import sys

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

TRAIN_INPUT = "data/input/train.csv"
TEST_INPUT = "data/input/test.csv"
TRAIN_OUTPUT = "data/features/train_encoded.csv"
TEST_OUTPUT = "data/features/test_encoded.csv"

if "snakemake" in sys.modules:
    TRAIN_INPUT = snakemake.input.train
    TEST_INPUT = snakemake.input.test
    TRAIN_OUTPUT = snakemake.output.train
    TEST_OUTPUT = snakemake.output.test

train_df = pd.read_csv(TRAIN_INPUT).dropna(subset=["sii"])
test_df = pd.read_csv(TEST_INPUT)

print(
    "columns not in test data: ",
    [f for f in train_df.columns if f not in test_df.columns],
)

# %%

# %%


# %%
def feature_engineering(df):
    df["BMI_Age"] = df["Physical-BMI"] * df["Basic_Demos-Age"]
    df["Internet_Hours_Age"] = (
        df["PreInt_EduHx-computerinternet_hoursday"] * df["Basic_Demos-Age"]
    )
    df["BMI_Internet_Hours"] = (
        df["Physical-BMI"] * df["PreInt_EduHx-computerinternet_hoursday"]
    )
    df["BFP_BMI"] = df["BIA-BIA_Fat"] / df["BIA-BIA_BMI"]
    df["FFMI_BFP"] = df["BIA-BIA_FFMI"] / df["BIA-BIA_Fat"]
    df["FMI_BFP"] = df["BIA-BIA_FMI"] / df["BIA-BIA_Fat"]
    df["LST_TBW"] = df["BIA-BIA_LST"] / df["BIA-BIA_TBW"]
    df["BFP_BMR"] = df["BIA-BIA_Fat"] * df["BIA-BIA_BMR"]
    df["BFP_DEE"] = df["BIA-BIA_Fat"] * df["BIA-BIA_DEE"]
    df["BMR_Weight"] = df["BIA-BIA_BMR"] / df["Physical-Weight"]
    df["DEE_Weight"] = df["BIA-BIA_DEE"] / df["Physical-Weight"]
    df["SMM_Height"] = df["BIA-BIA_SMM"] / df["Physical-Height"]
    df["Muscle_to_Fat"] = df["BIA-BIA_SMM"] / df["BIA-BIA_FMI"]
    df["Hydration_Status"] = df["BIA-BIA_TBW"] / df["Physical-Weight"]
    df["ICW_TBW"] = df["BIA-BIA_ICW"] / df["BIA-BIA_TBW"]
    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
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
ID = "id"
TARGET = "sii"
CATEGORICAL_FEATURES = train_df.select_dtypes(include="object").columns.drop([ID])
print(train_df.select_dtypes(exclude="object").columns.drop([TARGET]))
NUMERICAL_FEATURES = train_df.select_dtypes(exclude="object").columns.drop([TARGET])
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


def fill_numerical_features(df):
    for feature in tqdm(NUMERICAL_FEATURES):
        if feature in df.columns and df[feature].isnull().sum() > 0:
            df[f"{feature}_median"] = df[feature].fillna(df[feature].median())
            df[f"{feature}_knn"] = KNNImputer(n_neighbors=5).fit_transform(
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

# encoding sii
train_df[TARGET] = train_df[TARGET].astype(int)


# %%

scaler = StandardScaler()
numerical_base_features = [
    f
    for f in test_df.columns
    if f in NUMERICAL_FEATURES
    or f[:-7] in NUMERICAL_FEATURES
    or f[:-4] in NUMERICAL_FEATURES
]
print(NUMERICAL_FEATURES, test_df.columns, numerical_base_features)
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
