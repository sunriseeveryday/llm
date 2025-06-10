from openai import OpenAI
from pydantic import BaseModel


"""
vllm不兼容responses.parse接口
"""

class FriendlyResponse(BaseModel):
    inner_thought: str
    answer: str


api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
model_name = "Qwen/Qwen3-0.6B"

client = OpenAI(api_key=api_key, base_url=base_url)

messages = [
    {"role": "user", "content": "你是谁？"},
]

response = client.responses.parse(
    model=model_name,
    input=messages,  # type: ignore
    text_format=FriendlyResponse
)
print()
