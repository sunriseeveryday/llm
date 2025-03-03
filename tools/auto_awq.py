import os

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from core.env_loader import init_env


init_env()
model_path = 'google/gemma-2-9b-it'
model = AutoAWQForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

quant_path = os.getenv("HF_NAME") + "/" + model_path.split("/")[-1]
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
