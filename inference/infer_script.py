#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import logging
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import nltk
import csv
import ast
from datetime import datetime
from rouge_score import rouge_scorer

import os
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM inference over OntoBench splits")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset ID (e.g., XiaoZhang98/OntoBench)"
    )
    parser.add_argument(
        "--split_index",
        type=str,
        required=True,
        help="OntoBench split (e.g., 1_1, 1_2, 2_1, 2_2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or Hugging Face ID of the model"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Template file with few-shot examples and '|||'-separated question template"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Base directory for JSONL results"
    )
    parser.add_argument(
        "--max_batched_tokens",
        type=int,
        default=4096,
        help="vLLM max_num_batched_tokens and max_model_len"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Max tokens to generate per example"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling probability"
    )
    parser.add_argument(
        "--stop",
        nargs="+",
        default=None,
        help="Stop sequences for generation (optional)"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.max_batched_tokens <= 0:
        raise ValueError("max_batched_tokens must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not (0 <= args.top_p <= 1):
        raise ValueError("top_p must be in [0, 1]")
    return args

def load_prompt_template(prompt_path):
    """
    Load the prompt file, splitting into examples block and question template.
    The file should contain two parts separated by '|||'.
    Returns (examples_block, template).
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        parts = raw.split("|||")
        if len(parts) != 2:
            raise ValueError(f"Prompt file '{prompt_path}' must contain exactly one '|||' separator")
        examples_block, template = parts
        if not examples_block.strip() or not template.strip():
            raise ValueError(f"Prompt file '{prompt_path}' has empty examples or template")
        return examples_block, template
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_path}' not found")

def build_vllm_engine(model, max_batched_tokens):
    """
    Initialize the vLLM engine with given model and batching settings.
    """
    try:
        num_gpus = torch.cuda.device_count()
        llm = LLM(
            model=model,
            tensor_parallel_size=max(1, num_gpus),
            # model_impl="transformers",  # for gemma
            max_num_batched_tokens=max_batched_tokens,
            max_model_len=max_batched_tokens,
            trust_remote_code=True,
            tokenizer_mode="auto",
            enforce_eager=True,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vLLM with model '{model}': {str(e)}")

def get_task_type(split_index: str) -> str:
    """
    Determine the task type based on split_index:
      - Multiple choice (A/B/C/D) -> 'mc'
      - True/False questions      -> 'bool'
      - Open-ended text (BLEU-evaluated) -> 'open_text'
      - Open-ended triple extraction, order-sensitive -> 'open_triple'
      - Open-ended tuple extraction, order-insensitive -> 'open_tuple'
    Only split '3_5' is open_tuple; splits 3_2,3_3,3_4 are open_triple.
    """
    task_map = {
        "1_1": "mc", "1_2": "mc", "1_3": "mc", "1_4": "mc", "1_5": "mc",
        "2_1": "mc", "2_2": "mc", "2_3": "mc", "2_4": "mc", "2_5": "bool",
        "3_1": "open_text",
        "3_2": "open_triple", "3_3": "open_triple", "3_4": "open_triple",
        "3_5": "open_tuple",
    }
    if split_index not in task_map:
        logger.warning(f"Unknown split '{split_index}', defaulting to 'mc' task type")
        return "mc"
    return task_map[split_index]

def parse_tuples_ast(raw: str):
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []

    def collect(o):
        out = []
        if isinstance(o, (list, tuple)):
            if len(o) >= 2 and all(not isinstance(x, (list, tuple)) for x in o):
                out.append(tuple(str(x).strip() for x in o))
            else:
                for x in o:
                    out.extend(collect(x))
        return out

    return collect(obj)

def parse_tuples_regex(raw: str):
    s = raw.strip()
    s = s.replace('[', '(').replace('{', '(')
    s = s.replace(']', ')').replace('}', ')')
    s = re.sub(r'^[\(\)]+', '', s)
    s = re.sub(r'[\(\)]+$', '', s)
    tuples = []
    for span in re.findall(r'\(\s*([^\)]+?)\s*\)', s):
        parts = [p.strip(" '\"") for p in span.split(',') if p.strip()]
        if len(parts) >= 2:
            tuples.append(tuple(parts))
    return tuples

def extract_prediction(raw: str, task_type: str):
    """
    Extract the model's prediction from raw generated text,对 `open_triple` 和 `open_tuple` 使用更鲁棒的解析。
    """
    raw = raw.strip()
    if task_type == "mc":
        m = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
        return m.group(1).upper() if m else raw
    elif task_type == "bool":
        m = re.search(r"\b(true|false)\b", raw, re.IGNORECASE)
        return m.group(1).lower() if m else raw.lower()
    elif task_type == "open_text":
        return raw
    elif task_type in ("open_triple", "open_tuple"):
        parsed = parse_tuples_ast(raw)
        if not parsed:
            parsed = parse_tuples_regex(raw)
        if task_type == "open_triple":
            return [t for t in parsed if len(t) == 3]
        else:
            return parsed
    else:
        return raw

def compute_open_f1(preds: list, refs: list):
    """
    Compute micro-Precision, micro-Recall, and micro-F1 for open_tuple tasks (order-insensitive).
    """
    tp = fp = fn = 0
    for pred, ref in zip(preds, refs):
        pred_set = set(pred)
        ref_set = set(ref)
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def compute_open_triple_metrics(preds: list[list], refs: list[list]):
    """
    Compute precision, recall, F1 for open_triple tasks (order-sensitive).
    Matches only count at the same positions.
    """
    tp = fp = fn = 0
    for pred_seq, ref_seq in zip(preds, refs):
        matches = sum(1 for i, tup in enumerate(pred_seq) if i < len(ref_seq) and tup == ref_seq[i])
        tp += matches
        fp += max(0, len(pred_seq) - matches)
        fn += max(0, len(ref_seq) - matches)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def compute_text_metrics(preds: list[str], refs: list[str]):
    """
    Compute BLEU and ROUGE scores for open-ended text predictions.
    Returns a dictionary with BLEU, ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, ref in zip(preds, refs):
        # Compute BLEU
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        except ZeroDivisionError:
            logger.warning(f"Zero BLEU score for pred: '{pred}', ref: '{ref}'")
            bleu = 0.0
        bleu_scores.append(bleu)

        # Compute ROUGE
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougel_scores.append(scores['rougeL'].fmeasure)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougel = sum(rougel_scores) / len(rougel_scores) if rougel_scores else 0.0

    return {
        "bleu": avg_bleu,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougel
    }

def append_metrics_to_csv(csv_path, row, header):
    """Append a row of metrics to a CSV file, creating it with header if it doesn't exist."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    """Run vLLM inference on OntoBench and evaluate results."""
    args = parse_args()

    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{args.dataset}': {str(e)}")

    ds = None
    for index in dataset.keys():
        if args.split_index in index:
            ds = dataset[index]
    if not ds:
        raise ValueError(f"Split '{args.split_index}' not found. Available splits: {list(dataset.keys())}")

    n = len(ds)
    logger.info(f"Processing split '{args.split_index}' with {n} examples")

    # Load prompt template
    examples_block, template = load_prompt_template(args.prompt_path)

    # Initialize vLLM
    llm = build_vllm_engine(args.model, args.max_batched_tokens)

    # Set up generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop if args.stop else None
    )

    # Build prompts
    prompts = []
    for ex in ds:
        if ("1_" in args.split_index or "2_" in args.split_index) and args.split_index != "2_5":
            question_block = ex["question"].strip() + "\n" + ex.get("options", "").strip().replace("\n\n", "\n")
        else:
            question_block = ex["question"].strip() + "\n"
        p = examples_block + template.format(question_block)
        prompts.append(p)

    # Generate outputs
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        raise RuntimeError(f"vLLM generation failed: {str(e)}")

    # Sort outputs by request_id
    sorted_outputs = sorted(outputs, key=lambda o: int(o.request_id))

    # Evaluate predictions
    task_type = get_task_type(args.split_index)
    preds, refs = [], []
    correct = 0
    results = []

    for i, (ex, out) in enumerate(zip(ds, sorted_outputs)):
        raw = out.outputs[0].text.strip()
        pred = extract_prediction(raw, task_type)

        # Retrieve or generate identifier
        identifier = ex.get("identifier", ex.get("id", f"example_{i}"))

        if task_type in ("mc", "bool"):
            ref = ex["answer"].strip()
            is_match = (pred == ref)
            correct += int(is_match)
            preds.append(pred)
            refs.append(ref)
        elif task_type == "open_tuple":
            ref_tuples = extract_prediction(ex["answer"], task_type)
            preds.append(pred)
            refs.append(ref_tuples)
            is_match = None
        elif task_type == "open_triple":
            ref_seq = extract_prediction(ex["answer"], task_type)
            preds.append(pred)
            refs.append(ref_seq)
            is_match = None
        else:  # open_text
            ref = ex["answer"].strip()
            preds.append(pred)
            refs.append(ref)
            is_match = None

        # Store raw log-probability
        logp = out.outputs[0].cumulative_logprob
        toks = len(out.outputs[0].token_ids) or 1
        results.append({
            "identifier": identifier,
            "question": ex["question"],
            "options": ex.get("options", ""),
            "response": raw,
            "answer": ex["answer"].strip(),
            "prediction": pred,
            "log_probability": logp,
            "token_count": toks,
            "match": is_match
        })

    # Save results
    model_safe = args.model.replace("/", "_").replace("-", "_")
    out_dir = os.path.join(args.output_dir, model_safe)
    os.makedirs(out_dir, exist_ok=True)

    pattern = r'(zero|one|two|three|four)_shot'
    shot_match = re.search(pattern, os.path.basename(args.prompt_path), re.IGNORECASE)
    jsonl_path = os.path.join(out_dir, f"{args.split_index}_{shot_match.group(1)}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fl:
        for rec in results:
            fl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Log metrics and append to CSV summary
    csv_path = os.path.join(args.output_dir, f"{model_safe}_results.csv")
    timestamp = datetime.now().isoformat()
    header = ["timestamp", "dataset", "split", "model", "prompt_path", "task_type", "accuracy", "precision", "recall", "f1", "bleu", "rouge1", "rouge2", "rougeL"]
    row = {h: None for h in header}
    row.update({
        "timestamp": timestamp,
        "dataset": args.dataset,
        "split": args.split_index,
        "model": args.model,
        "prompt_path": args.prompt_path,
        "task_type": task_type
    })
    if task_type in ("mc", "bool"):
        accuracy = correct / n if n > 0 else 0.0
        logger.info(f"Split '{args.split_index}' [{task_type}] Accuracy: {accuracy:.4f}")
        row["accuracy"] = f"{accuracy:.4f}"
    elif task_type == "open_tuple":
        metrics = compute_open_f1(preds, refs)
        logger.info(
            f"Split '{args.split_index}' [open_tuple] "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        row["precision"] = f"{metrics['precision']:.4f}"
        row["recall"] = f"{metrics['recall']:.4f}"
        row["f1"] = f"{metrics['f1']:.4f}"
    elif task_type == "open_triple":
        metrics = compute_open_triple_metrics(preds, refs)
        logger.info(
            f"Split '{args.split_index}' [open_triple] "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        row["precision"] = f"{metrics['precision']:.4f}"
        row["recall"] = f"{metrics['recall']:.4f}"
        row["f1"] = f"{metrics['f1']:.4f}"
    else:  # open_text
        metrics = compute_text_metrics(preds, refs)
        logger.info(
            f"Split '{args.split_index}' [open_text] "
            f"BLEU: {metrics['bleu']:.4f}, "
            f"ROUGE-1: {metrics['rouge1']:.4f}, "
            f"ROUGE-2: {metrics['rouge2']:.4f}, "
            f"ROUGE-L: {metrics['rougeL']:.4f}"
        )
        row["bleu"] = f"{metrics['bleu']:.4f}"
        row["rouge1"] = f"{metrics['rouge1']:.4f}"
        row["rouge2"] = f"{metrics['rouge2']:.4f}"
        row["rougeL"] = f"{metrics['rougeL']:.4f}"

    # Append metrics
    append_metrics_to_csv(csv_path, row, header)

if __name__ == "__main__":
    main()