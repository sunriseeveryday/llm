export VLLM_USE_MODELSCOPE=True
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8003 \
  --uvicorn-log-level "info" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 1024 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --tensor_parallel_size 4