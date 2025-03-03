import asyncio
import traceback
from typing import List

from transformers import AutoModel, AutoTokenizer

from core.env_loader import init_env


async def download(model_name: str) -> None:
    try:
        await asyncio.to_thread(AutoModel.from_pretrained, model_name)
        await asyncio.to_thread(AutoTokenizer.from_pretrained, model_name)
        print(f"Model {model_name} is downloaded.")
    except Exception as e:
        print(f"Error occurred while downloading {model_name}: {e}")
        traceback.print_exc()


async def download_models(model_names: List[str]) -> None:
    tasks = [download(model_name) for model_name in model_names]
    await asyncio.gather(*tasks)


init_env()
model_names = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]
asyncio.run(download_models(model_names))
