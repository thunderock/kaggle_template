import os
import sys
import zipfile

import kaggle

# Check if running within Snakemake
if "snakemake" in sys.modules:
    COMPETITION_NAME = snakemake.params.competition
    ZIP_FILE_NAME = snakemake.output.zip
else:
    COMPETITION_NAME = "child-mind-institute-problematic-internet-use"
    ZIP_FILE_NAME = f"../data/input/{COMPETITION_NAME}.zip"

DOWNLOAD_DIR = os.path.dirname(ZIP_FILE_NAME)

kaggle.api.competition_download_files(
    competition=COMPETITION_NAME, path=DOWNLOAD_DIR, force=False, quiet=False
)

with zipfile.ZipFile(ZIP_FILE_NAME, "r") as zip_ref:
    zip_ref.extractall(DOWNLOAD_DIR)
