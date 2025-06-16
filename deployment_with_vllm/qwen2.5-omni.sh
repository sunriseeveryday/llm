export VLLM_USE_MODELSCOPE=True
vllm serve Qwen/Qwen2.5-Omni-7B \
  --host 0.0.0.0 \
  --port 8000 \
  --uvicorn-log-level "info" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max-model-len 10240