from functools import partial

from core.env_loader import seed


def create_prompt_formats(sample):
    blurb = "\nBelow is an instruction that describes a task. Write a response that appropriately completes the request."
    instruction = "### Instruct: Summarize the below conversation."
    input_context = sample['dialogue'] if sample["dialogue"] else None
    response = f"### Output:\n{sample['summary']}"
    end = "### End"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample


def get_max_length(model):
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer, model, dataset):
    dataset = dataset.map(create_prompt_formats)

    max_length = get_max_length(model)
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],
    )

    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def prepare_dataset(tokenizer, model, dataset):
    train_dataset = preprocess_dataset(tokenizer, model, dataset['train'])
    eval_dataset = preprocess_dataset(tokenizer, model, dataset['validation'])
    test_dataset = preprocess_dataset(tokenizer, model, dataset['test'])
    return train_dataset, eval_dataset, test_dataset
