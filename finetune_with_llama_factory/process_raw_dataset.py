import json


def format_output(output: dict) -> str:
    return "<think>\n\n</think>\n" + json.dumps(output, ensure_ascii=False)


def add_additional_instruction(instruction: str) -> str:
    return instruction + "结果仅输出一个JSON。KEY为实体类型，VALUE为实体值，且VALUE格式为列表。"


data = []
buffer = ""
with open("raw_criminal_case_ner.jsonl", mode="r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        buffer += line
        try:
            obj = json.loads(buffer)
            data.append(obj)
            buffer = ""  # 重置缓冲区，准备下一个对象
        except json.JSONDecodeError:
            continue  # JSON 不完整，继续读下一行


for i, _ in enumerate(data):
    data[i]["output"] = format_output(data[i]["output"])
    data[i]["instruction"] = add_additional_instruction(data[i]["instruct"])
    del data[i]["instruct"]

with open("criminal_case_ner.json", mode="w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
