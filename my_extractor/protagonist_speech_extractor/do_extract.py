import os
import sys
sys.path.append("./")
import json
import asyncio
from pathlib import Path
from typing import List

from tqdm import tqdm
from datasets import Dataset

from core.env_loader import init_env
from model_api.llm import run_llm
from my_extractor.splitter.wu_zhi_zhuan_sheng_novel_splitter import WuZhiZhuanShengNovelSplitter


init_env()

def create_speech_extraction_messages(text: str) -> List[dict]:
    return [
        {"role": "user", "content": text}
    ]


class ProtagonistSpeechExtractor:
    def __init__(self, novel_path):
        self.novel_path = novel_path
        splitter = WuZhiZhuanShengNovelSplitter(novel_path)
        asyncio.run(splitter.load_novel())
        titles, self.chapters = asyncio.run(splitter.split_chapter())
        self.titles = [" ".join(title) for title in titles]
        self.protagonist = "鲁迪乌斯"

    def extract(self):
        existed_titles = []
        for _, _, files in os.walk(f"{self.novel_path}.speech"):
            existed_titles = [file.split(".")[0] for file in files]
        for title, chapter in tqdm(zip(self.titles, self.chapters), desc="Extracting", total=len(self.titles)):
            if title in existed_titles:
                continue
            self.extract_chapter(title, chapter)

    def extract_chapter(self, title: str, chapter: str):
        prompt = (
            f"Extract the protagonist {self.protagonist}'s inner monologue and dialogue in the following format: [\"(Inner monologue)<insert specific inner monologue>(Dialogue)<insert specific dialogue>\", ...]\n"
            "If none exist, leave it empty: [].\n"
            "Nothing else can be output except for this list.\n"
            f"<chapter_name>{title}</chapter_name>\n"
            f"<chapter_content>{chapter}</chapter_content>"
        )
        pairs = run_llm(create_speech_extraction_messages(prompt))
        print("==> For Checking: " + pairs)
        try:
            pairs = json.loads(pairs)
        except json.JSONDecodeError:
            return

        Path(f"{self.novel_path}.speech").mkdir(parents=True, exist_ok=True)
        text = []
        for pair in pairs:
            text.append(json.dumps({"pair": pair}, ensure_ascii=False))
        with open(f"{self.novel_path}.speech/{title}.speech", "w", encoding="utf8") as f:
            f.write("\n".join(text))

    def make_dataset_and_push(self):
        ds = []
        for _, _, files in os.walk(f"{self.novel_path}.speech"):
            for file in files:
                title = file.split(".")[0]
                with open(f"{self.novel_path}.speech/{file}", "r", encoding="utf8") as f:
                    title_ds = [json.loads(item) for item in f.readlines()]
                    title_ds = [{"title": title, "text": item["pair"]} for item in title_ds]
                    ds.extend(title_ds)
        titles = []
        texts = []
        for item in ds:
            titles.append(item["title"])
            texts.append(item["text"])
        ds = {"title": titles, "text": texts}
        ds = Dataset.from_dict(ds)
        ds.push_to_hub(os.getenv("DATASET_NAME").split("/")[-1] + "Speech")


def test():
    novel_path = "my_qa_extractor/data/docs/Wu Zhi Zhuan Sheng  ~Zai Yi Shi - Wei Zhi.txt"
    protagonist_speech_extractor = ProtagonistSpeechExtractor(novel_path)
    protagonist_speech_extractor.extract()


def push():
    novel_path = "my_qa_extractor/data/docs/Wu Zhi Zhuan Sheng  ~Zai Yi Shi - Wei Zhi.txt"
    protagonist_speech_extractor = ProtagonistSpeechExtractor(novel_path)
    protagonist_speech_extractor.make_dataset_and_push()


if __name__ == "__main__":
    push()
