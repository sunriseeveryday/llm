import os

from datasets import load_dataset

from core.env_loader import init_env
from core.model_trainer import get_model_trainer
from core.model_getter import get_4bit_model, get_k_bit_model_for_training, load_peft_model
from qlora_on_microsoft_phi_2_and_neil_code_dialogsum_test.data_preprocessor import prepare_dataset


init_env()
tokenizer, model = get_4bit_model(os.getenv('MODEL_NAME'))
dataset = load_dataset(os.getenv('DATASET_NAME'))
train_dataset, eval_dataset, test_dataset = prepare_dataset(tokenizer, model, dataset)
model = get_k_bit_model_for_training(model)
peft_model = load_peft_model(model)
trainer = get_model_trainer(tokenizer, peft_model, train_dataset, eval_dataset)
trainer.train()
trainer.push_to_hub(os.getenv('WANDB_NAME'))
tokenizer.push_to_hub(os.getenv('WANDB_NAME'))
