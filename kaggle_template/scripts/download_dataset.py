import kaggle
import zipfile
from os.path import join as j
import os
import sys

COMPETITION_NAME = "child-mind-institute-problematic-internet-use"
DOWNLOAD_DIR = "data"
if "snakemake" in sys.modules:
    COMPETITION_NAME = snakemake.params.competition_name
    DOWNLOAD_DIR = snakemake.params.download_dir

ZIP_FILE_NAME = j(DOWNLOAD_DIR, f"{COMPETITION_NAME}.zip")
