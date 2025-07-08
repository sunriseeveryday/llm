import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json


def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_function(examples, tokenizer, max_length=512):
    """数据预处理函数"""
    inputs = []
    targets = []

    for i in range(len(examples['instruction'])):
        # 构建输入文本
        input_text = f"<|im_start|>system\n你是一个专业的翻译助手。<|im_end|>\n<|im_start|>user\n{examples['instruction'][i]}\n{examples['input'][i]}<|im_end|>\n<|im_start|>assistant\n"
        target_text = f"{examples['output'][i]}<|im_end|>"

        inputs.append(input_text)
        targets.append(input_text + target_text)

    # 分词
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding=False)

    # 设置labels
    for i in range(len(model_inputs['input_ids'])):
        input_len = len(model_inputs['input_ids'][i])
        labels['input_ids'][i] = [-100] * input_len + labels['input_ids'][i][input_len:]

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def train_lora_plus():
    """LoRA+ 训练"""
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

    # LoRA+ 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # LoRA rank
        lora_alpha=128,  # LoRA alpha
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
        # LoRA+ 特有参数
        use_rslora=True,  # 使用 Rank-Stabilized LoRA
        use_dora=False,  # 是否使用 DoRA
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
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
        output_dir="./qwen3-32b-lora-plus",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
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
    tokenizer.save_pretrained("./qwen3-32b-lora-plus")


if __name__ == "__main__":
    train_lora_plus()