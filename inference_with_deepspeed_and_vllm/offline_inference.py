from vllm import LLM, SamplingParams

from core.env_loader import init_env


init_env()
model_name = ""
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
llm = LLM(model=model_name)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
