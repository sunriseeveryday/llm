export VLLM_USE_MODELSCOPE=True
export CUDA_VISIBLE_DEVICES=6,7
vllm serve Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8002 \
  --uvicorn-log-level "info" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --tensor_parallel_size 2