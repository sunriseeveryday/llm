from openai import OpenAI


api_key = ""
base_url = ""
model_name = ""
client = OpenAI(api_key=api_key, base_url=base_url)
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "How are you?"},
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
