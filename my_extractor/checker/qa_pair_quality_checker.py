import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from model_api.embeddings import run_embedding


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_questions(file_name: str) -> Dict[str, list]:
    with open(f"data/docs/{file_name}.questions.json", 'r', encoding="utf8") as f:
        questions = json.loads(f.read())
    questions_with_chapter = {}
    for question in tqdm(questions, desc="Loading questions"):
        chapter_record = question[1]
        if chapter_record in questions_with_chapter:
            questions_with_chapter[chapter_record].append(question[2])
        else:
            questions_with_chapter[chapter_record] = [question[2]]
    return questions_with_chapter


def get_id_from_question(file_name: str, question: str) -> Optional[str]:
    with open(f"data/docs/{file_name}.questions.json", 'r', encoding="utf8") as f:
        qs: List[List[str]] = json.loads(f.read())
    for q in qs:
        if q[2] == question:
            return q[0]
    logging.warning(f"Question {question} not found in file {file_name}.questions.json")
    return None


def delete_question_in_file(file_name: str, question: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    with open(f"data/docs/{file_name}.questions.json", 'r', encoding="utf8") as f:
        qs: List[List[str]] = json.loads(f.read())
    for q in qs:
        if q[2] == question:
            qs.remove(q)
            return qs, q
    logging.warning(f"Question {question} not found in file {file_name}.questions.json")
    return qs, None


def get_embeddings(questions) -> Dict[str, np.ndarray]:
    embeddings = {}
    for chapter, qs in tqdm(questions.items(), desc="Embedding chapters"):
        embeddings[chapter] = np.array([run_embedding(q) for q in tqdm(qs, desc=f"Embedding questions")])
        break  # TODO: remove this line
    return embeddings


def get_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -1)
    similarity_matrix = np.triu(similarity_matrix, k=1)
    return similarity_matrix


def get_high_similarity_pairs(similarity_matrix: np.ndarray, threshold: float=0.8) -> List[Tuple[int, int]]:
    high_similarity_pairs = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                high_similarity_pairs.append((i, j))
    return high_similarity_pairs


def refresh_bad_questions(file_name: str, bad_question: List[str]) -> None:
    if Path(f"data/docs/{file_name}.bad_questions.json").exists():
        with open(f"data/docs/{file_name}.bad_questions.json", 'r', encoding="utf8") as f:
            bad_questions: List[List[str]] = json.loads(f.read())
    else:
        bad_questions = []
    bad_questions.append(bad_question)

    with open(f"data/docs/{file_name}.bad_questions.json", 'w', encoding="utf8") as f:
        f.write(json.dumps(bad_questions, ensure_ascii=False, indent=4))
        logging.info(f"Put bad question: {bad_question[2]} in bad_questions.json")


def delete_question(file_name: str, questions: Dict[str, list], chapter: str, question_id: int) -> None:
    qs, q = delete_question_in_file(file_name, questions[chapter][question_id])
    if q:
        with open(f"data/docs/{file_name}.questions.json", 'w', encoding="utf8") as f:
            f.write(json.dumps(qs, ensure_ascii=False, indent=4))
        refresh_bad_questions(file_name, q)
        logging.info(f"Deleted question {question_id} from chapter {chapter}")


def check_similarity(file_name: str, questions: Dict[str, list]) -> None:
    embeddings = get_embeddings(questions)

    for chapter, qs in embeddings.items():
        logging.info(f"Calculating similarity matrix for chapter {chapter}")
        similarity_matrix = get_similarity_matrix(qs)
        high_similarity_pairs = get_high_similarity_pairs(similarity_matrix)

        if not high_similarity_pairs:
            logging.info(f"No high similarity pairs found in chapter {chapter}")
            continue

        for i, j in high_similarity_pairs:
            logging.info(f"Question {i}: {questions[chapter][i]}")
            logging.info(f"Question {j}: {questions[chapter][j]}")
            logging.info(f"Similarity: {similarity_matrix[i, j]}")
            question_id = input("Delete all, or one question, or none...")
            while question_id != str(i) and question_id != str(j) and question_id != "none" and question_id != "all":
                logging.info("Invalid input. Please enter 'all', 'none', or the question number.")
                question_id = input("Delete all, or one question, or none...")
            if question_id == "all":
                delete_question(file_name, questions, chapter, i)
                delete_question(file_name, questions, chapter, j)
            elif question_id == "none":
                continue
            else:
                delete_question(file_name, questions, chapter, int(question_id))
            logging.info("")


def human_check(file_name: str, questions: Dict[str, list], max_sample: int=3) -> None:
    for chapter, qs in questions.items():
        logging.info(f"Chapter {chapter}")
        sample_n = random.randint(0, max_sample)
        sampled_qs = random.sample(qs, sample_n)
        for q in sampled_qs:
            question_id = get_id_from_question(file_name, q)
            if question_id:
                with open(f"data/docs/{file_name}.answer/{question_id}.answer", 'r', encoding="utf8") as f:
                    answer = f.read()
                logging.info(f"Question {question_id}: {q}")
                logging.info(f"Answer: {answer}")
                msg = input("Delete or remain...")
                while msg != "delete" and msg != "remain":
                    logging.info("Invalid input. Please enter 'delete' or 'remain'.")
                    msg = input("Delete or remain...")
                if msg == "delete":
                    delete_question(file_name, questions, chapter, int(question_id))
                else:
                    continue
            else:
                continue


def interact_cli(file_name: str) -> None:
    questions = load_questions(file_name)
    check_similarity(file_name, questions)
    human_check(file_name, questions, max_sample=3)


def main():
    interact_cli("Wu Zhi Zhuan Sheng  ~Zai Yi Shi - Wei Zhi.txt")


if __name__ == "__main__":
    main()
