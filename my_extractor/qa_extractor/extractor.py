import os
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Dict

import aiofiles
from tqdm.asyncio import tqdm as async_tqdm

from core.rate_limiter import RateLimiter
from my_extractor.splitter.wu_zhi_zhuan_sheng_novel_splitter import WuZhiZhuanShengNovelSplitter
from model_api.llm import run_llm
from my_extractor.qa_extractor.utils import (create_extraction_conversation_messages,
                                             create_answering_conversation_messages,
                                             load_plain_text_files_from_directory, extract_questions_from_output,
                                             flatten_nested_lists, allow_suffixes)


lock = asyncio.Lock()

def extract_questions_from_directory(input_folder: Path, verbose: bool=True) -> List[Dict[str, str]]:
    if verbose:
        logging.info(f"Loading files from '{input_folder}'.")
    files = load_plain_text_files_from_directory(input_folder)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_files(files, verbose=verbose))

    if verbose:
        logging.info(f"Done, {len(results)} question/answer pairs have been generated!")
    return results


async def process_files(files: List[Tuple[str, str]], verbose=True) -> List[Dict[str, str]]:
    nb_files = len(files)
    progress_counter = {'nb_files': nb_files, 'nb_files_done': 0}

    tasks = []
    for file_path, text in files:
        task = process_file(file_path, text, progress_counter, verbose=verbose)
        tasks.append(task)

    tasks_outputs = await asyncio.gather(*tasks)
    return flatten_nested_lists(tasks_outputs)


async def process_file(file_path: str, text: str, progress_counter: Dict[str, int], verbose=True, max_qa_pairs: int=-1) -> List[Dict[str, str]]:
    if verbose:
        logging.info(f"Processing file '{file_path}'.")
    questions = await generate_q(file_path, text, max_qa_pairs)
    result = await generate_a(file_path, questions)

    progress_counter['nb_files_done'] += 1
    if verbose:
        logging.info(f"{progress_counter['nb_files_done']}/{progress_counter['nb_files']}: File '{file_path}' done!")

    return result


async def generate_q(file_path: str, text: str, max_qa_pairs: int=-1) -> List[Tuple[str, str, str, str]]:
    questions_file_name = f"{file_path}.questions.json"
    if Path(questions_file_name).is_file():
        async with aiofiles.open(questions_file_name, 'r', encoding="utf-8") as input_file:
            questions = json.loads(await input_file.read())
        for question in questions:
            async with aiofiles.open(question[1], 'r', encoding="utf-8") as input_file:
                question.insert(2, await input_file.read())

    else:
        questions = await extract_questions_from_text(file_path, text, max_qa_pairs)

        async with aiofiles.open(questions_file_name, 'w', encoding="utf-8") as output_file:
            save_questions = [(question[0], question[1], question[3]) for question in questions]
            await output_file.write(json.dumps(save_questions, indent=4, ensure_ascii=False))
    return questions


async def generate_a(file_path: str, questions: List[Tuple[str, str, str, str]]) -> List[Dict[str, str]]:
    Path(file_path + ".answer").mkdir(parents=True, exist_ok=True)
    already_answer_ids = []
    for root, dirs, files in os.walk(file_path + ".answer"):
        already_answer_ids = [file.replace(".answer", "") for file in files if file.endswith('.answer')]

    tasks = []
    tasks_outputs = []
    tasks_durations = []
    for question_id, sub_file_path, sub_text, question in questions:
        if question_id in already_answer_ids:
            async with aiofiles.open(f"{file_path}.answer/{question_id}.answer", 'r', encoding="utf-8") as input_file:
                tasks_outputs.append({'source': sub_file_path, 'question': question, 'answer': await input_file.read()})
        else:
            task = generate_answer(question, sub_text, sub_file_path, question_id, len(questions), tasks_outputs, tasks_durations, file_path)
            tasks.append(task)

    logging.info(f"Processing Answers For Questions, num: {len(tasks)}.")
    await asyncio.gather(*tasks)
    return tasks_outputs


async def extract_questions_from_text(file_path: str, text: str, max_qa_pairs: int) -> List[Tuple[str, str, str, str]]:
    splitter = WuZhiZhuanShengNovelSplitter(file_path, text.strip())
    await splitter.load_novel()
    titles, chapters = await splitter.split_chapter()

    questions = []
    for title, chapter in async_tqdm(zip(titles, chapters), total=len(titles), desc="Processing Questions For Chapters"):
        title = " ".join(title)
        messages = create_extraction_conversation_messages("章节标题:" + title + "\n内容:" + chapter, max_qa_pairs)
        output = RateLimiter().call(run_llm, messages)

        for allow_suffix in allow_suffixes:
            if file_path.endswith(allow_suffix):
                file_path = file_path.replace(allow_suffix, "")
        Path(file_path).mkdir(parents=True, exist_ok=True)

        sub_file_path = file_path + "/" + title + ".split"
        async with aiofiles.open(sub_file_path, 'w', encoding="utf-8") as output_file:
            await output_file.write(chapter)

        questions.extend([(sub_file_path, chapter, question.strip()) for question in extract_questions_from_output(output)])
    questions = [(str(i), ) + question for i, question in enumerate(questions)]
    return questions


async def generate_answer(question: str, sub_text: str, sub_file_path: str, question_id: str, total_tasks_num: int,
                          tasks_outputs: List[Dict[str, str]], tasks_durations: List[float], file_path: str) -> str:
    start_time = time.time()
    messages = create_answering_conversation_messages(question, sub_text)
    answer = RateLimiter().call(run_llm, messages)
    end_time = time.time()
    duration = end_time - start_time

    async with lock:
        tasks_outputs.append({'source': sub_file_path, 'question': question, 'answer': answer})
        tasks_durations.append(duration)

    with open(f"{file_path}.answer/{question_id}.answer", 'w', encoding="utf-8") as output_file:
        output_file.write(answer)

    still_need_time = (total_tasks_num - len(tasks_outputs)) * sum(tasks_durations) / len(tasks_durations)
    logging.info(f"{len(tasks_outputs)} tasks finished, this task took {duration:.2f}s, still need {str(timedelta(seconds=still_need_time))}.")
    return answer
