import jinja2


chat_template = """
{%- if tools %}\n {{- '<|im_start|>system\\n' }}\n {%- if messages[0].role == 'system' %}\n {{- messages[0].content + '\\n\\n' }}\n {%- endif %}\n {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n {%- for tool in tools %}\n {{- \"\\n\" }}\n {{- tool | tojson }}\n {%- endfor %}\n {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0].role == 'system' %}\n {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n {%- set index = (messages|length - 1) - loop.index0 %}\n {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n {%- set ns.multi_step_tool = false %}\n {%- set ns.last_query_index = index %}\n {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n {%- elif message.role == \"assistant\" %}\n {%- set content = message.content %}\n {%- set reasoning_content = '' %}\n {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n {%- set reasoning_content = message.reasoning_content %}\n {%- else %}\n {%- if '</think>' in message.content %}\n {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n {%- endif %}\n {%- endif %}\n {%- if loop.index0 > ns.last_query_index %}\n {%- if loop.last or (not loop.last and reasoning_content) %}\n {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n {%- else %}\n {{- '<|im_start|>' + message.role + '\\n' + content }}\n {%- endif %}\n {%- else %}\n {{- '<|im_start|>' + message.role + '\\n' + content }}\n {%- endif %}\n {%- if message.tool_calls %}\n {%- for tool_call in message.tool_calls %}\n {%- if (loop.first and content) or (not loop.first) %}\n {{- '\\n' }}\n {%- endif %}\n {%- if tool_call.function %}\n {%- set tool_call = tool_call.function %}\n {%- endif %}\n {{- '<tool_call>\\n{\"name\": \"' }}\n {{- tool_call.name }}\n {{- '\", \"arguments\": ' }}\n {%- if tool_call.arguments is string %}\n {{- tool_call.arguments }}\n {%- else %}\n {{- tool_call.arguments | tojson }}\n {%- endif %}\n {{- '}\\n</tool_call>' }}\n {%- endfor %}\n {%- endif %}\n {{- '<|im_end|>\\n' }}\n {%- elif message.role == \"tool\" %}\n {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n {{- '<|im_start|>user' }}\n {%- endif %}\n {{- '\\n<tool_response>\\n' }}\n {{- message.content }}\n {{- '\\n</tool_response>' }}\n {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n {{- '<|im_end|>\\n' }}\n {%- endif %}\n {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n {{- '<|im_start|>assistant\\n' }}\n {%- if enable_thinking is defined and enable_thinking is false %}\n {{- '<think>\\n\\n</think>\\n\\n' }}\n {%- endif %}\n{%- endif %}
""".strip()

"""
==>input:
- tools: list[dict]
- messages: list[dict]
  - role
  - content
  - reasoning_content or <think>...</think> in content
  - tool_calls: list[dict]
    - function
      - name
      - arguments

==>return:
-->type 1 for user/system
<|im_start|>user/system
{{content}}<|im_end|>

-->type 2 for assistant
<|im_start|>assistant
<think>
{{reasoning_content}}
</think>
{{content}}
<tool_call>
{"name": "{{name}}", "arguments": {{arguments}}}
</tool_call><|im_end|>

-->type 3: for tool
<|im_start|>user
<tool_response>
{{content}}
</tool_response><|im_end|>

-->add_generation_prompt
<|im_start|>assistant
<think>

</think>
"""

template = jinja2.Template(chat_template)
print(chat_template)
print("\n\n\n")
print("-" * 50)

render_str = template.render(
    {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "I need to find the capital of France.",
                "reasoning_content": "What is the capital of France?",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_capital",
                            "arguments": {"country": "France"},
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": "The capital of France is Paris."
            },
            {
                "role": "assistant",
                "content": "The capital of France is Paris."
            }
        ],
        "tools": [
            {
                "name": "get_capital",
                "description": "Get the capital of a country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The name of the country.",
                        }
                    },
                    "required": ["country"],
                },
            }
        ]
    }
)
print(render_str)