import json
import logging
from pathlib import Path

from extractor import extract_questions_from_directory


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

input_directory = Path('./data/docs')
output_filepath = Path('./data/questions.json')
extracted_questions = extract_questions_from_directory(input_directory)
with open(output_filepath, 'w', encoding="utf-8") as output_file:
    output_file.write(json.dumps(extracted_questions, indent=4, ensure_ascii=False))
    logging.info(f"Results have been saved to {output_filepath}.")
