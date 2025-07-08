import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
from datasets import Dataset
import json


def train_prompt_tuning():
    """Prompt Tuning 训练"""
    model_name = "Qwen/Qwen3-32B"

    # 加载tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prompt Tuning 配置
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=50,  # 虚拟token数量
        prompt_tuning_init_text="Translate the following Czech text to English:",
        tokenizer_name_or_path=model_name,
    )

    # 应用Prompt Tuning
    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    # 加载数据
    train_data = load_data('wmt14_cs_en_train.json')
    val_data = load_data('wmt14_cs_en_val.json')

    # 转换为Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 预处理数据
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen3-32b-prompt-tuning",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-3,  # Prompt tuning通常使用更高的学习率
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="qwen3-32b-prompt-tuning-run",
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained("./qwen3-32b-prompt-tuning")


if __name__ == "__main__":
    train_prompt_tuning()