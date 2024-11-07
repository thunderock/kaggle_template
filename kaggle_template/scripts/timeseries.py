# %%
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import warnings
import os
from tqdm import tqdm
import sys

warnings.filterwarnings("ignore")


# %%
if "snakemake" in sys.modules:
    TRAIN_INPUT = snakemake.input.train
    TEST_INPUT = snakemake.input.test
    TRAIN_OUTPUT = snakemake.output.train
    TEST_OUTPUT = snakemake.output.test
else:
    TRAIN_INPUT = "data/input/series_train.parquet"
    TEST_INPUT = "data/input/series_test.parquet"
    TRAIN_OUTPUT = "data/features/train_time_series.csv"
    TEST_OUTPUT = "data/features/test_time_series.csv"


print(TRAIN_INPUT)


def process_file(filename, dirname):
    data = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
    data = data.sort_values(by="step", ascending=True)
    data = data.drop("step", axis=1)
    return (
        data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).values.reshape(-1),
        filename.split("=")[1],
    )


def load_time_series(dirname):
    ids = [fname for fname in os.listdir(dirname) if fname.startswith("id=")]
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
train_parquet = load_time_series(TRAIN_INPUT)
test_parquet = load_time_series(TEST_INPUT)

# %%

print(train_parquet.shape, test_parquet.shape)

# %%
print(train_parquet.isna().sum().sum(), test_parquet.isna().sum().sum())


# %%
print(train_parquet.dtypes.value_counts(), test_parquet.dtypes.value_counts())

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_parquet.loc[:, train_parquet.columns != "id"] = scaler.fit_transform(
    train_parquet.loc[:, train_parquet.columns != "id"]
)
test_parquet.loc[:, test_parquet.columns != "id"] = scaler.transform(
    test_parquet.loc[:, test_parquet.columns != "id"]
)
print(train_parquet.dtypes.value_counts(), test_parquet.dtypes.value_counts())

# %%
print(train_parquet.head())

# %%
train_parquet.to_csv(TRAIN_OUTPUT, index=False)
test_parquet.to_csv(TEST_OUTPUT, index=False)
