import os
import shutil
import sys
import zipfile

import kaggle

COMPETITION_NAME = "child-mind-institute-problematic-internet-use"
TRAIN_FILE = "data/input/train.csv"
# Check if running within Snakemake
if "snakemake" in sys.modules:
    COMPETITION_NAME = snakemake.params.competition
    TRAIN_FILE = snakemake.output.train

DOWNLOAD_DIR = os.path.dirname(TRAIN_FILE)
if os.environ.get("KAGGLE_URL_BASE") is not None:
    src = "/kaggle/input/child-mind-institute-problematic-internet-use/"
    shutil.copy(src, DOWNLOAD_DIR)
else:
    kaggle.api.competition_download_files(
        competition=COMPETITION_NAME, path=DOWNLOAD_DIR, force=False, quiet=False
    )

    ZIP_FILE_NAME = os.path.join(DOWNLOAD_DIR, f"{COMPETITION_NAME}.zip")
    with zipfile.ZipFile(ZIP_FILE_NAME, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)
