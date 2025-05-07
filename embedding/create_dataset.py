import json
from tqdm import tqdm
from model_api.llm import run_llm

with open("model_api/dataset/raw_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = []
prompt = "针对{}取一个别名，仅输出别名，要求为中文"
for item in tqdm(data):
    p = prompt.format(item[3])
    a = run_llm([{"role": "user", "content": p}])
    dataset_item = {"text": "schema_field_" + item[3], "alias": a}
    dataset.append(dataset_item)

with open("model_api/dataset/dataset.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(dataset, ensure_ascii=False, indent=4))
