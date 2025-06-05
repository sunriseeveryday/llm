with open("ddl.txt", "r", encoding="utf8") as f:
    ddl = f.read()

from hive_ddl_parser import parse
parsed_dict = parse(ddl)

with open("full_can_signals.json", "r", encoding="utf8") as f:
    import json
    full_can_signals = json.loads(f.read())
    full_can_signals = {signal.replace("schema_field_", ""): ["", ""] for signal in full_can_signals.keys()}

dataset = full_can_signals | parsed_dict
with open("dataset.json", "w", encoding="utf8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
