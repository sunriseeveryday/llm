import os
import sys
sys.path.append("./")
import logging

from core.env_loader import init_env
from prepare_dataset import prepare_dataset
from core.model_trainer import get_model_trainer
from core.model_getter import get_4bit_model, get_k_bit_model_for_training, load_peft_model


init_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tokenizer, model = get_4bit_model(os.getenv('MODEL_NAME'))
dataset = prepare_dataset(tokenizer, model)

model = get_k_bit_model_for_training(model)
peft_model = load_peft_model(model)

trainer = get_model_trainer(tokenizer, peft_model, dataset)
trainer.train()

trainer.push_to_hub(os.getenv('WANDB_NAME'))
tokenizer.push_to_hub(os.getenv('WANDB_NAME'))
