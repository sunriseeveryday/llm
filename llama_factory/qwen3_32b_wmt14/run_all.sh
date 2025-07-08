echo "开始微调Qwen3-32B模型..."

# 1. 准备数据
echo "1. 准备WMT14数据..."
python prepare_wmt14_data.py

# 2. 运行LoRA+微调
echo "2. 开始LoRA+微调..."
python lora_plus_train.py

# 3. 运行Prompt Tuning
echo "3. 开始Prompt Tuning..."
python prompt_tuning_train.py

# 4. 运行SFT (需要多GPU)
echo "4. 开始SFT微调..."
# 单GPU运行
python sft_train.py

# 或者多GPU运行
# torchrun --nproc_per_node=4 sft_train.py

echo "所有微调任务完成！"