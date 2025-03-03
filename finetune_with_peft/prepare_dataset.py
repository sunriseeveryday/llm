import os
import json
import random
import logging
from functools import partial
from typing import Dict, List

from jinja2 import Template
from datasets import Dataset, DatasetDict, enable_caching, load_dataset

from core.env_loader import init_env, seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
init_env()
# precompiled_template = Template("{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}")
precompiled_template = Template("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n")
enable_caching()


def get_dataset_dict(path: str="my_qa_extractor/data/questions.json") -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf8") as f:
        ds: List[Dict[str, str]] = json.loads(f.read())
    questions = []
    answers = []
    chapters = []
    for item in ds:
        questions.append(item["question"])
        answers.append(item["answer"])
        chapters.append(item["source"].split("/")[-1].split(".")[0])
    return {"question": questions, "answer": answers, "chapter": chapters}


def split_dataset(ds: Dataset, train_size: float=0.8, validation_size: float=0.1) -> DatasetDict:
    test_size = 1 - train_size - validation_size
    if test_size < 0:
        raise ValueError("train_size + validation_size should be less than 1")

    temp = ds.train_test_split(train_size=train_size)
    train_ds = temp["train"]
    temp = temp["test"].train_test_split(train_size=validation_size / (validation_size + test_size))
    validation_ds = temp["train"]
    test_ds = temp["test"]
    return DatasetDict({
        "train": train_ds,
        "validation": validation_ds,
        "test": test_ds
    })


def match_specific_format(sample: Dict[str, str], tokenizer, probability: float=0.8) -> Dict[str, str]:
    if random.random() < probability:
        user_content = "在" + sample["chapter"] + "中，" + sample["question"]
    else:
        user_content = sample["question"]
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["answer"]}
    ]

    sample["text"] = precompiled_template.render(messages=messages, bos_token=tokenizer.bos_token, add_generation_prompt=False)
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


def prepare_ds(tokenizer, model, ds: Dataset) -> Dataset:
    _match_specific_format = partial(match_specific_format, tokenizer=tokenizer)
    ds = ds.map(_match_specific_format)
    max_length = get_max_length(model)
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], max_length=max_length, truncation=True),
        batched=True,
        remove_columns=["question", "answer", "chapter"],
    )
    ds = ds.shuffle(seed=seed)
    return ds


def prepare_dataset(tokenizer, model) -> Dataset:
    ds = load_dataset(os.getenv("DATASET_NAME"))
    ds = prepare_ds(tokenizer, model, ds["train"])
    return ds


def push_to_huggingface():
    ds = get_dataset_dict()
    ds = Dataset.from_dict(ds)
    ds.push_to_hub(os.getenv("DATASET_NAME").split("/")[-1])


def push_raw_to_huggingface(path: str):
    with open(path, "r", encoding="utf8") as f:
        ds = f.read()
    ds = {"raw": [ds]}
    ds = Dataset.from_dict(ds)
    ds.push_to_hub(os.getenv("DATASET_NAME").split("/")[-1] + "Raw")


if __name__ == "__main__":
    # push_to_huggingface()
    push_raw_to_huggingface("my_qa_extractor/data/docs/Wu Zhi Zhuan Sheng  ~Zai Yi Shi - Wei Zhi.txt")
