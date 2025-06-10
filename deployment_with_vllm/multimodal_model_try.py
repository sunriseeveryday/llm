from openai import OpenAI


api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
model_name = "Qwen/Qwen2.5-Omni-7B"

client = OpenAI(api_key=api_key, base_url=base_url)

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            }
        },
    ]},
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,  # type: ignore
    timeout=600
)
print()
