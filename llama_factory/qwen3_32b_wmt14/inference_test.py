import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def test_model(model_path, peft_path=None):
    """测试微调后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if peft_path:
        # 加载PEFT模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        # 加载SFT模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    # 测试翻译
    test_text = "Dobrý den, jak se máte?"
    prompt = f"<|im_start|>system\n你是一个专业的翻译助手。<|im_end|>\n<|im_start|>user\n请将以下捷克语翻译成英语：\n{test_text}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {test_text}")
    print(f"输出: {response}")


if __name__ == "__main__":
    # 测试LoRA+模型
    print("测试LoRA+模型:")
    test_model("Qwen/Qwen3-32B", "./qwen3-32b-lora-plus")

    # 测试Prompt Tuning模型
    print("\n测试Prompt Tuning模型:")
    test_model("Qwen/Qwen3-32B", "./qwen3-32b-prompt-tuning")

    # 测试SFT模型
    print("\n测试SFT模型:")
    test_model("./qwen3-32b-sft")