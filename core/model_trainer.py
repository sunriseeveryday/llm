import os

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


def get_model_trainer(tokenizer, model, train_dataset, eval_dataset=None):
    training_args = TrainingArguments(
        logging_strategy="steps",
        logging_steps=50,

        output_dir=os.getenv("MODEL_OUTPUT") + "/" + os.getenv("WANDB_NAME"),
        overwrite_output_dir=True,
        save_strategy="epoch",

        report_to=["wandb"],
        run_name=os.getenv("WANDB_NAME"),

        per_device_train_batch_size=5,
        gradient_accumulation_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        warmup_steps=200,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True if os.getenv("TORCH_DTYPE") == "bfloat16" else False,
        fp16=True if os.getenv("TORCH_DTYPE") == "float16" else False,

        eval_strategy="no" if eval_dataset is None else "epoch",
        per_device_eval_batch_size=5
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    return trainer
