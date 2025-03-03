import os
import gc

import torch
from dotenv import load_dotenv
from transformers import set_seed
from huggingface_hub import login


seed = 42

def init_env():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_dotenv()
    set_seed(seed)
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
