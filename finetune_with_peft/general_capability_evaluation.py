import os
import sys
sys.path.append("./")
import logging

from jinja2 import Template
from datasets import load_dataset
from tqdm import tqdm
import evaluate

from core.env_loader import init_env
from core.model_getter import load_exist_peft_model
from core.inference import do_inference


init_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
precompiled_template = Template("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n")
accuracy = evaluate.load("accuracy")


def format_sample(sample):
    question = sample["question"]
    choices = sample["choices"]
    options = "\n".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
    msg = [
        {"role": "user", "content": f"Question: {question}\nOptions:\n{options}\nAnswer:"}
    ]
    return precompiled_template.render(messages=msg, tools=None, add_generation_prompt=False)


def evaluate_mmlu(dataset):
    correct = []
    for item in tqdm(dataset, desc="Eval"):
        prompt = format_sample(item)
        prediction = do_inference(tokenizer, peft_model, prompt)

        predicted_index = None
        for choice_idx in range(len(item["choices"])):
            if str(choice_idx) in prediction:
                predicted_index = choice_idx
                break

        if predicted_index == item["answer"]:
            correct.append(1)
        else:
            correct.append(0)
        acc = accuracy.compute(predictions=correct)
        logging.info(f"Accuracy: {acc['accuracy']}")


ds = load_dataset("cais/mmlu", "all")["test"]  # "stem", "humanities", "social_sciences", "other"
logging.info(f"Evaluating {len(ds)} datasets")
tokenizer, peft_model = load_exist_peft_model(
    model_name=os.getenv("MODEL_NAME"),
    peft_name=os.getenv("HF_NAME") + "/" + os.getenv("WANDB_NAME"),
    bit=4
)
evaluate_mmlu(ds)
