export VLLM_USE_MODELSCOPE=True
vllm serve Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8001 \
  --uvicorn-log-level "info" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 1024 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --tensor_parallel_size 2