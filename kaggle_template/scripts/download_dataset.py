import kaggle
import zipfile
from os.path import join as j
import sys
import os

# Check if running within Snakemake
if "snakemake" in sys.modules:
    COMPETITION_NAME = snakemake.params.competition
    ZIP_FILE_NAME = snakemake.output[0]
# else:
#     COMPETITION_NAME = "child-mind-institute-problematic-internet-use"
#     ZIP_FILE_NAME = f"../data/raw/{COMPETITION_NAME}.zip"  # Fixed directory to match Snakefile

DOWNLOAD_DIR = os.path.dirname(ZIP_FILE_NAME)

# Download competition data using Kaggle API
kaggle.api.competition_download_files(
    competition=COMPETITION_NAME, path=DOWNLOAD_DIR, force=False, quiet=False
)

# Extract the downloaded ZIP file
with zipfile.ZipFile(ZIP_FILE_NAME, "r") as zip_ref:
    zip_ref.extractall(DOWNLOAD_DIR)
