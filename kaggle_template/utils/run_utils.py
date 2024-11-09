import os

import torch

GPU_CORES = list(range(torch.cuda.device_count()))
CPU_CORES = list(range(os.cpu_count()))
