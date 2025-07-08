import json
from datasets import load_dataset


def prepare_wmt14_data():
    """准备WMT14 cs-en数据集"""
    dataset = load_dataset("wmt/wmt14", "cs-en")

    def format_translation_data(examples):
        """将翻译数据格式化为instruction格式"""
        formatted_data = []
        for i in range(len(examples['translation'])):
            cs_text = examples['translation'][i]['cs']
            en_text = examples['translation'][i]['en']

            # 格式化为instruction-response格式
            formatted_item = {
                "instruction": "请将以下捷克语翻译成英语：",
                "input": cs_text,
                "output": en_text
            }
            formatted_data.append(formatted_item)
        return formatted_data

    # 处理训练集
    train_data = []
    for batch in dataset['train']:
        train_data.extend(format_translation_data({'translation': [batch['translation']]}))

    # 处理验证集
    val_data = []
    for batch in dataset['validation']:
        val_data.extend(format_translation_data({'translation': [batch['validation']]}))

    # 限制数据量（可选）
    train_data = train_data[:10000]  # 取前10000条
    val_data = val_data[:1000]  # 取前1000条

    # 保存数据
    with open('wmt14_cs_en_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open('wmt14_cs_en_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(val_data)} 条")


if __name__ == "__main__":
    prepare_wmt14_data()