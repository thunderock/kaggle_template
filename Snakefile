COMPETITION = "child-mind-institute-problematic-internet-use"

rule download_data:
    output:
        "data/input/{competition}.zip".format(competition=COMPETITION)
    params:
        competition=COMPETITION
    threads: 1
    script: "kaggle_template/scripts/download_dataset.py"