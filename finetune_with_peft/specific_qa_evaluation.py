import os

import sacrebleu
from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm

from core.model_getter import load_exist_peft_model
from core.env_loader import init_env
from core.inference import do_inference


init_env()

tokenizer, model = load_exist_peft_model(os.getenv("MODEL_NAME"), os.getenv("HF_NAME") + "/" + os.getenv("WANDB_NAME"), bit=4)
ds = load_dataset(os.getenv("DATASET_NAME"))["train"]

predictions = []
references = []
for question, expected_answer in tqdm(zip(ds["question"], ds["answer"]), total=len(ds), desc="Generating answers"):
    generated_answer = do_inference(tokenizer, model, question)
    predictions.append(generated_answer)
    references.append(expected_answer)

bleu = sacrebleu.corpus_bleu(predictions, [references]).score

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

print(f"BLEU: {bleu:.2f}")
print(f"ROUGE-1: {rouge1:.2f}")
print(f"ROUGE-L: {rougeL:.2f}")
