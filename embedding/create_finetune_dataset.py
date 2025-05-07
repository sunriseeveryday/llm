import json
import uuid

with open("model_api/dataset/dataset.json", "r", encoding="utf8") as f:
    dataset = json.loads(f.read())

finetune_dataset = {
    "queries": {},
    "corpus": {},
    "relevant_docs": {}
}

for item in dataset:
    a = item["text"]
    q = item["alias"]
    a_id = str(uuid.uuid4())
    q_id = str(uuid.uuid4())
    finetune_dataset["queries"][q_id] = q
    finetune_dataset["corpus"][a_id] = a
    finetune_dataset["relevant_docs"][q_id] = [a_id]

with open("model_api/dataset/finetune_dataset.json", "w", encoding="utf8") as f:
    f.write(json.dumps(finetune_dataset, ensure_ascii=False, indent=4))
