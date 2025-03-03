deepspeed --num_gpus=1 \
    --deepspeed_config deepspeed_config.json \
    python -m vllm.entrypoints.api_server \
    --model <AWQ_MODEL> \
    --quantization awq \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --ds-offload \
    --ds-offload-device nvme \
    --host 0.0.0.0 \
    --port <PORT>