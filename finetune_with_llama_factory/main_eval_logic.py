import sys
import json
import logging
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(handler)

def item_recall_eval(pred: str, true: str) -> dict[str, dict[str, int | float]]:
    try:
        if pred.startswith("<think>"):
            pred = pred.split("</think>")[1].strip()
        if "```json" in pred:
            pred = pred.split("```json")[1].split("```")[0]
        parsed_pred = json.loads(pred)
        parsed_pred = {k.lower(): v for k, v in parsed_pred.items()}
        parsed_pred = {k.split("（")[0].strip(): v for k, v in parsed_pred.items()}

    except json.decoder.JSONDecodeError:
        logger.warning(f"Could not parse {pred} as JSON")
        parsed_pred = {}

    true = json.loads(true.split("</think>")[1].strip())
    true = {k.lower(): v for k, v in true.items()}

    logger.info(f"\nparsed_pred: {json.dumps(parsed_pred, ensure_ascii=False)}\ntrue: {json.dumps(true, ensure_ascii=False)}")

    # do evaluation
    all_types = set(parsed_pred.keys()).union(set(true.keys()))
    metrics = {}

    for label in all_types:
        pred_ents = set(parsed_pred.get(label, []))
        if type(pred_ents) == str:
            pred_ents = {pred_ents}
        true_ents = set(true.get(label, []))

        tp = len(pred_ents & true_ents)
        fp = len(pred_ents - true_ents)
        fn = len(true_ents - pred_ents)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        logger.info("Label: %s, Precision: %.4f, Recall: %.4f, F1: %.4f, TP: %d, FP: %d, FN: %d", label, round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn)
    return metrics


def do_eval():
    true = load_eval_data()
    # Micro
    total_tp = total_fp = total_fn = 0
    # Each type micro
    tp_total = defaultdict(int)
    fp_total = defaultdict(int)
    fn_total = defaultdict(int)
    # Macro
    precisions = []
    recalls = []
    f1s = []
    for t in tqdm(true, desc="Evaluating", total=len(true)):
        msgs = [
            {
                "role": "user",
                "content": t["instruction"] + "\n" + t["input"]
            }
        ]
        completion = generate_completion(msgs)
        metrics = item_recall_eval(completion, t["output"])

        # Micro
        for v in metrics.values():
            total_tp += v["tp"]
            total_fp += v["fp"]
            total_fn += v["fn"]

        # Each type micro
        for label in metrics.keys():
            tp_total[label] += metrics[label]["tp"]
            fp_total[label] += metrics[label]["fp"]
            fn_total[label] += metrics[label]["fn"]

        # Macro
        per_sample_precision = sum([v["precision"] for v in metrics.values()]) / len(metrics)
        per_sample_recall = sum([v["recall"] for v in metrics.values()]) / len(metrics)
        per_sample_f1 = sum([v["f1"] for v in metrics.values()]) / len(metrics)
        precisions.append(per_sample_precision)
        recalls.append(per_sample_recall)
        f1s.append(per_sample_f1)

    logger.info("=== Evaluation Results ===")

    # Micro
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)
    logger.info(f"Micro Precision: {precision:.4f}, Macro Recall: {recall:.4f}, Macro F1: {f1:.4f}")

    logger.info("=== Evaluation Results ===")

    # Macro
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)
    logger.info(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")

    logger.info("=== Evaluation Results for Each Type ===")

    # Each type micro
    for label in tp_total.keys():
        tp = tp_total[label]
        fp = fp_total[label]
        fn = fn_total[label]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        logger.info(f"Type: {label}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, TP: {tp}, FP: {fp}, FN: {fn}")


def load_eval_data():
    eval_data_path = "LLaMA-Factory/data/criminal_case_ner.json"
    with open(eval_data_path, mode="r", encoding="utf8") as f:
        eval_data = json.loads(f.read())
    return eval_data


def generate_completion(msgs: list[dict[str, str]]) -> str:
    api_key = "fake"
    base_url = "http://localhost:8000/v1"
    model_name = "criminal_case_ner"
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=msgs,  # type: ignore
        temperature=0.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    do_eval()
    logger.info("Evaluation completed.")
