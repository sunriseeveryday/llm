import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model, PeftModel


def get_model(model_name: str, bnb_config: BitsAndBytesConfig=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=os.getenv("DEVICE"),
        trust_remote_code=True,
        torch_dtype=getattr(torch, os.getenv("TORCH_DTYPE")),
        quantization_config=bnb_config
    )
    return tokenizer, model


def get_8bit_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=True
    )
    tokenizer, model = get_model(model_name, bnb_config)
    return tokenizer, model


def get_4bit_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, os.getenv("TORCH_DTYPE"))
    )
    tokenizer, model = get_model(model_name, bnb_config)
    return tokenizer, model


def get_k_bit_model_for_training(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def load_peft_model(model):
    peft_config = LoraConfig(
        r=int(os.getenv("LORA_R")),
        lora_alpha=int(os.getenv("LORA_ALPHA")),
        target_modules=['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'],
        bias="none",
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.config.use_cache = False
    return peft_model


def load_exist_peft_model(model_name, peft_name: str, bit: int=16):
    if type(model_name) != str:
        tokenizer, model = None, model_name
    elif bit == 16:
        tokenizer, model = get_model(model_name)
    elif bit == 8:
        tokenizer, model = get_8bit_model(model_name)
    elif bit == 4:
        tokenizer, model = get_4bit_model(model_name)
    else:
        raise ValueError("bit must be 4, 8 or 16")

    peft_model = PeftModel.from_pretrained(model, peft_name, device=os.getenv("DEVICE"),
                                           trust_remote_code=True, torch_dtype=getattr(torch, os.getenv("TORCH_DTYPE")))
    peft_model.config.use_cache = True
    return tokenizer, peft_model


if __name__ == "__main__":
    pass
