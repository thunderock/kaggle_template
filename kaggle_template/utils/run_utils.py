import json
import os

import torch

GPU_CORES = list(range(torch.cuda.device_count()))
CPU_CORES = list(range(os.cpu_count()))


def read_dictionary_from_json(json_file):
    with open(json_file, "r") as file:
        return json.load(file)


def write_dictionary_to_json(json_file, dictionary):
    with open(json_file, "w") as file:
        json.dump(dictionary, file, indent=4)
