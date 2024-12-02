import os

import torch

GPU_CORES = list(range(torch.cuda.device_count()))
CPU_CORES = list(range(os.cpu_count()))


def get_dframe_with_features_by_threshold(dframe, sample_threshold):
    corr = dframe.drop(["id"], axis=1).corr()
    features = dframe.drop(["id", "sii"], axis=1).columns
    sii_corr = (
        corr["sii"]
        .sort_values(ascending=False, key=abs)
        .head(int(sample_threshold * len(features) + 1))
    )
    return dframe[list(set(sii_corr.index.tolist() + ["sii", "id"]))]
