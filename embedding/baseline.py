import json
from tqdm import tqdm
from model_api.search_dataset import run_search

with open("model_api/dataset/dataset.json", "r", encoding="utf8") as f:
    dataset = json.loads(f.read())

right_count = 0
dataset_length = len(dataset)
for item in tqdm(dataset):
    a = run_search(item["alias"])
    if not a:
        continue
    t = item["text"]
    a = a[0]
    if t in a:
        right_count += 1
print(right_count)
print(right_count / dataset_length)
# 36 0.3076
