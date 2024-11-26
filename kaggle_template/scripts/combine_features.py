# %%
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# %%
TRAIN_FEATURE_INPUT = "data/features/train_features.csv"
TRAIN_TIMESERIES = "data/features/train_timeseries.csv"
TEST_FEATURE_INPUT = "data/features/test_features.csv"
TEST_TIMESERIES = "data/features/test_timeseries.csv"
TRAIN_WIDE_DF = "data/features/train_wide_df.csv"
TEST_WIDE_DF = "data/features/test_wide_df.csv"

if "snakemake" in sys.modules:
    TRAIN_FEATURE_INPUT = snakemake.input.train_features
    TRAIN_TIMESERIES = snakemake.input.train_timeseries
    TEST_FEATURE_INPUT = snakemake.input.test_features
    TEST_TIMESERIES = snakemake.input.test_timeseries
    TRAIN_WIDE_DF = snakemake.output.train_wide_df
    TEST_WIDE_DF = snakemake.output.test_wide_df

# %%
train_df = pd.read_csv(TRAIN_FEATURE_INPUT)
train_ts = pd.read_csv(TRAIN_TIMESERIES)
test_df = pd.read_csv(TEST_FEATURE_INPUT)
test_ts = pd.read_csv(TEST_TIMESERIES)

# %%
train_df.shape, test_df.shape, train_ts.shape, test_ts.shape

# %%
train_df.head()

# %%

all([i in train_df.id.tolist() for i in train_ts.id.tolist()]), all(
    [i in test_df.id.tolist() for i in test_ts.id.tolist()]
)

# %%

# two train datasets, one with features and one with features + timeseries
train_wide_df = pd.merge(train_df, train_ts, on="id")
test_wide_df = pd.merge(test_df, test_ts, on="id")

train_df.shape, test_df.shape, train_wide_df.shape, test_wide_df.shape

# %%

train_wide_df.to_csv(TRAIN_WIDE_DF, index=False)
test_wide_df.to_csv(TEST_WIDE_DF, index=False)
