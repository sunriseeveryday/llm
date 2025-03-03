import os
import re
from pathlib import Path
from typing import List, Tuple, Any


# should not be .split and .json
allow_suffixes = [".txt", ".md"]

extraction_system_prompt="You are an expert user extracting information to quiz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text. {extraction_num}"
answering_system_prompt="You are an expert user answering questions. You will be passed a page extracted from a documentation and a question. Generate a comprehensive and informative answer to the question based *solely* on the given text."
extraction_num = "extraction_num = {num}"
extraction_inf = "While ensuring quality, extract as many questions as possible."


def create_extraction_conversation_messages(text: str, num: int) -> List[dict]:
    extraction_content = extraction_system_prompt.format(extraction_num=extraction_num.format(num) if num >= 0 else extraction_inf)
    content = (
        f"<system>{extraction_content}</system>\n"
        f"<user>{text}</user>"
    )
    return [
        {"role": "user", "content": content}
    ]


def create_answering_conversation_messages(question: str, text: str) -> List[dict]:
    content = (
        f"<system>{answering_system_prompt}</system>\n"
        f"<user>{text}\n{question}</user>"
    )
    return [
        {"role": "user", "content": content}
    ]


def load_plain_text_files_from_directory(directory: Path) -> List[Tuple[str, str]]:
    files_data = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if any(file_name.endswith(suffix) for suffix in allow_suffixes):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    file_content = file.read()
                files_data.append((file_path, file_content))
    return files_data


def extract_questions_from_output(output: str) -> List[str]:
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
    questions = question_pattern.findall(output)
    return questions


def flatten_nested_lists(nested_lists: List[List[Any]]) -> List[Any]:
    flattened_list = []
    for sublist in nested_lists:
        flattened_list.extend(sublist)
    return flattened_list
