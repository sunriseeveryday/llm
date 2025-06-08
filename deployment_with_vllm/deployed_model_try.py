from openai import OpenAI
from pydantic import BaseModel


class FriendlyResponse(BaseModel):
    inner_thought: str
    answer: str


TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "get_current_temperature",
      "description": "Get current temperature at a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
          }
        },
        "required": [
          "location"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_temperature_date",
      "description": "Get temperature at a location and date.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
          },
          "date": {
            "type": "string",
            "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
          }
        },
        "required": [
          "location",
          "date"
        ]
      }
    }
  }
]


api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
model_name = "Qwen/Qwen3-0.6B"

client = OpenAI(api_key=api_key, base_url=base_url)

messages = [
    {"role": "user", "content": "现在的天气是什么？"},
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,  # type: ignore
    tools=TOOLS,  # type: ignore
    tool_choice="auto",
    logprobs=True,
    top_logprobs=5,
    extra_body={
        # "guided_json": FriendlyResponse.model_json_schema(),
        "chat_template_kwargs": {"enable_thinking": True},
        "echo": True,
    }
)
content = response.choices[0].message.content
reasoning_content = response.choices[0].message.model_extra["reasoning_content"]
print()
