#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild metrics from existing inference JSONL files without loading vLLM.

This script is intentionally separate from inference. It lets expensive GPU
generation finish independently, then computes or recomputes metrics from the
saved JSONL outputs. Use it for BERTScore and for recovering metrics JSON files
after a metric dependency failure.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METRICS_HEADER = [
    "timestamp", "dataset", "split", "model", "prompt_path", "task_type",
    "accuracy", "precision", "recall", "f1", "entity_f1", "class_f1",
    "property_f1", "bleu", "rouge1", "rouge2", "rougeL",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "bertscore_model", "bertscore_error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild metrics from existing JSONL inference outputs.")
    parser.add_argument("--output_dir", required=True, help="Inference output directory, e.g. output_ontourl_standard.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model-safe directory names to include.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split IDs to include, e.g. 3_2 3_3.")
    parser.add_argument("--shots", nargs="*", default=None, help="Optional shot names, e.g. zero two four.")
    parser.add_argument("--compute_bertscore", action="store_true", help="Compute BERTScore for open_text outputs.")
    parser.add_argument("--bertscore_model", default="roberta-large")
    parser.add_argument("--bertscore_device", default=None, help="Optional BERTScore device, e.g. cuda or cpu.")
    parser.add_argument("--bertscore_batch_size", type=int, default=64)
    parser.add_argument("--overwrite_metrics", action="store_true", help="Recompute metrics even when metrics JSON exists.")
    parser.add_argument("--fail_on_bertscore_error", action="store_true")
    parser.add_argument(
        "--no_reparse_responses",
        action="store_true",
        help="Use stored prediction/match fields instead of reparsing saved model responses.",
    )
    return parser.parse_args()


PROPP_MC_TASKS = {
    "propp_verbalisation_classification",
    "propp_verbalisation_reverse",
    "propp_sequence_statement",
    "propp_sequence_symbol",
}


def is_local_propp_record(record: dict) -> bool:
    dataset = str(record.get("dataset") or "")
    hf_split = str(record.get("hf_split") or "")
    prompt_path = str(record.get("prompt_path") or "")
    return "propp" in dataset.lower() or "propp" in hf_split.lower() or "propp_csv" in prompt_path


def get_task_type(split_index: str, record: dict | None = None) -> str:
    if split_index in PROPP_MC_TASKS:
        return "mc"
    if record and is_local_propp_record(record):
        if split_index in {"3_2", "3_4"}:
            return "open_triple"
        if split_index.startswith(("1_", "2_")):
            return "mc"
    if split_index in {"1_1", "1_2", "1_3", "1_4", "1_5", "2_1", "2_2", "2_3", "2_4"}:
        return "mc"
    if split_index == "2_5":
        return "bool"
    if split_index == "3_1":
        return "entity_extraction"
    if split_index == "3_2":
        return "open_text"
    if split_index in {"3_3", "3_4", "3_5"}:
        return "open_triple"
    return "open_text"


def split_and_shot_from_path(jsonl_path: Path) -> tuple[str, str]:
    stem = jsonl_path.stem
    for shot in ("zero", "one", "two", "three", "four"):
        suffix = f"_{shot}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], shot
    parts = stem.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (stem, "")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def normalize_tuple_value(x) -> str:
    return normalize_text(str(x).strip(" '\"\t\n\r"))


def normalize_tuple(item) -> tuple[str, ...]:
    if isinstance(item, (list, tuple)):
        return tuple(normalize_tuple_value(x) for x in item)
    parts = [p for p in re.split(r"\s*,\s*", str(item).strip("()[] ")) if p]
    return tuple(normalize_tuple_value(p) for p in parts)


def parse_sequence(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError):
        triples = re.findall(r"\(([^()]+)\)", text)
        return [tuple(part.strip() for part in t.split(",")) for t in triples]


def strip_thinking(raw: str) -> tuple[str, str]:
    raw = raw or ""
    think_blocks = re.findall(r"(?is)<think>\s*(.*?)\s*</think>", raw)
    if "</think>" in raw.lower():
        content = re.split(r"(?is)</think>", raw)[-1].strip()
        return content, "\n\n".join(block.strip() for block in think_blocks)
    return raw.strip(), "\n\n".join(block.strip() for block in think_blocks)


def clean_answer_scope(raw: str) -> str:
    scope, _thinking = strip_thinking(raw or "")
    scope = scope.strip()
    scope = re.sub(r"(?is)^\s*(?:<\|assistant\|>|assistant\s*[:：])\s*", "", scope)
    scope = re.sub(r"(?is)^\s*(?:\[/INST\]|\[/?INST\])\s*", "", scope)
    return scope.strip()


def parse_mc_prediction(raw: str) -> str:
    scope = clean_answer_scope(raw)
    if not scope:
        return ""

    boxed = re.findall(r"\\boxed\{\s*([A-D])\s*\}", scope, re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()

    final = re.findall(
        r"(?im)(?:^|\n)\s*(?:final\s+answer|answer)\s*[:：]\s*(?:option\s*)?[\(\[]?\s*([A-D])\s*[\)\].:]?\b",
        scope,
    )
    if final:
        return final[-1].upper()

    # Common non-chat-template leakage: "[/INST] A. answer text".
    leading = re.search(
        r"(?is)^\s*(?:\[/INST\]|\[/?INST\])?\s*(?:the\s+answer\s+is\s+|answer\s*[:：]\s*|option\s*)?"
        r"[\(\[]?\s*([A-D])\s*[\)\].:]?\b",
        scope,
    )
    if leading:
        return leading.group(1).upper()

    line_letters = re.findall(
        r"(?im)^\s*(?:option\s*)?[\(\[]?\s*([A-D])\s*[\)\].:]?\s*$",
        scope,
    )
    if line_letters:
        return line_letters[-1].upper()

    explanatory = re.findall(
        r"(?is)(?:correct\s+(?:answer|option|choice)\s*(?:is|:)\s*|"
        r"(?:the\s+)?answer\s+(?:is|should\s+be)\s*)"
        r"(?:option\s*)?[\(\[]?\s*([A-D])\s*[\)\].:]?\b",
        scope,
    )
    if explanatory:
        return explanatory[-1].upper()
    return ""


def parse_bool_prediction(raw: str) -> str:
    scope = clean_answer_scope(raw)
    if not scope:
        return ""
    boxed = re.findall(r"\\boxed\{\s*(true|false)\s*\}", scope, re.IGNORECASE)
    if boxed:
        return boxed[-1].lower()
    final = re.findall(r"(?im)(?:^|\n)\s*(?:final\s+answer|answer)\s*[:：]\s*(true|false)\b", scope)
    if final:
        return final[-1].lower()
    tf = re.findall(r"(?i)\b(true|false)\b", scope)
    return tf[-1].lower() if tf else ""


def split_terms(text: str) -> list[str]:
    text = re.sub(r"(?im)^\s*[-*]\s*", "", text or "")
    parts = re.split(r";|,|\n", text)
    return [part.strip(" .;\t\r\n") for part in parts if part.strip(" .;\t\r\n")]


def parse_term_extraction_output(raw: str) -> dict[str, list[str]]:
    scope = clean_answer_scope(raw)
    classes = ""
    properties = ""
    class_match = re.search(
        r"(?is)(?:^|\n)\s*(?:classes|class|concepts)\s*[:：]\s*(.*?)(?=\n\s*(?:properties|property|relations)\s*[:：]|\Z)",
        scope,
    )
    prop_match = re.search(
        r"(?is)(?:^|\n)\s*(?:properties|property|relations)\s*[:：]\s*(.*?)(?=\n\s*(?:classes|class|concepts)\s*[:：]|\Z)",
        scope,
    )
    if class_match:
        classes = class_match.group(1)
    if prop_match:
        properties = prop_match.group(1)
    class_terms = [x for x in split_terms(classes) if normalize_term(x) not in {"none", "n/a", "na"}]
    property_terms = [x for x in split_terms(properties) if normalize_term(x) not in {"none", "n/a", "na"}]
    return {"classes": class_terms, "properties": property_terms}


def extract_prediction(raw: str, task_type: str):
    scope = clean_answer_scope(raw)
    if task_type == "mc":
        return parse_mc_prediction(scope)
    if task_type == "bool":
        return parse_bool_prediction(scope)
    if task_type == "entity_extraction":
        return parse_term_extraction_output(scope)
    if task_type == "open_text":
        return scope
    if task_type == "open_triple":
        return parse_sequence(scope)
    return scope


def prediction_parse_success(pred, task_type: str) -> bool:
    if task_type in {"mc", "bool", "open_text"}:
        return bool(str(pred or "").strip())
    if task_type == "entity_extraction":
        return bool(typed_terms(pred))
    if task_type == "open_triple":
        return bool(parse_sequence(pred))
    return pred not in (None, [], "")


def parse_failures_for_records(records: list[dict], task_type: str, no_reparse: bool = False) -> int:
    if no_reparse:
        return sum(1 for rec in records if not rec.get("parse_success"))
    failures = 0
    for rec in records:
        pred = extract_prediction(rec.get("answer_content") or rec.get("response") or rec.get("prediction") or "", task_type)
        if not prediction_parse_success(pred, task_type):
            failures += 1
    return failures


def compute_open_triple_metrics(preds, refs):
    tp = fp = fn = 0
    for pred, ref in zip(preds, refs):
        pred_set = {normalize_tuple(x) for x in parse_sequence(pred) if normalize_tuple(x)}
        ref_set = {normalize_tuple(x) for x in parse_sequence(ref) if normalize_tuple(x)}
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_term(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[_-]+", " ", s)
    s = re.sub(r"['\"`]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def typed_terms(value) -> set[tuple[str, str]]:
    if isinstance(value, dict):
        classes = value.get("classes", []) or []
        properties = value.get("properties", []) or []
    else:
        parsed = parse_term_extraction_output(str(value or ""))
        classes = parsed.get("classes", []) or []
        properties = parsed.get("properties", []) or []
    return {("class", normalize_term(x)) for x in classes if normalize_term(x)} | {
        ("property", normalize_term(x)) for x in properties if normalize_term(x)
    }


def f1_for_sets(pred_set, ref_set):
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_entity_f1(preds, refs):
    all_pred = set()
    all_ref = set()
    pred_classes = set()
    ref_classes = set()
    pred_props = set()
    ref_props = set()
    for pred, ref in zip(preds, refs):
        pset = typed_terms(pred)
        rset = typed_terms(ref)
        all_pred |= pset
        all_ref |= rset
        pred_classes |= {x for x in pset if x[0] == "class"}
        ref_classes |= {x for x in rset if x[0] == "class"}
        pred_props |= {x for x in pset if x[0] == "property"}
        ref_props |= {x for x in rset if x[0] == "property"}
    precision, recall, entity_f1 = f1_for_sets(all_pred, all_ref)
    _, _, class_f1 = f1_for_sets(pred_classes, ref_classes)
    _, _, property_f1 = f1_for_sets(pred_props, ref_props)
    return {
        "precision": precision,
        "recall": recall,
        "entity_f1": entity_f1,
        "class_f1": class_f1,
        "property_f1": property_f1,
    }


def compute_text_metrics(
    preds,
    refs,
    compute_bertscore=False,
    bertscore_model="roberta-large",
    bertscore_device=None,
    bertscore_batch_size=64,
    fail_on_error=False,
):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = re.findall(r"\w+", str(pred).lower())
        ref_tokens = re.findall(r"\w+", str(ref).lower())
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        except ZeroDivisionError:
            bleu = 0.0
        scores = scorer.score(str(ref), str(pred))
        bleu_scores.append(bleu)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougel_scores.append(scores["rougeL"].fmeasure)
    metrics = {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougel_scores) / len(rougel_scores) if rougel_scores else 0.0,
    }
    if compute_bertscore:
        try:
            from bert_score import score as bertscore_score

            bert_preds = [str(x).strip() or "." for x in preds]
            bert_refs = [str(x).strip() or "." for x in refs]
            precision, recall, f1 = bertscore_score(
                bert_preds,
                bert_refs,
                model_type=bertscore_model,
                lang="en",
                verbose=False,
                rescale_with_baseline=False,
                use_fast_tokenizer=True,
                device=bertscore_device,
                batch_size=bertscore_batch_size,
            )
            metrics.update({
                "bertscore_precision": float(precision.mean().item()),
                "bertscore_recall": float(recall.mean().item()),
                "bertscore_f1": float(f1.mean().item()),
                "bertscore_model": bertscore_model,
            })
        except Exception as exc:
            if fail_on_error:
                raise
            logger.exception("BERTScore failed for offline metrics; keeping BLEU/ROUGE only.")
            metrics["bertscore_error"] = str(exc)
    return metrics


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics_json_has_bertscore(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("metrics", {}).get("bertscore_f1"))


def upsert_csv_row(csv_path: Path, row: dict):
    rows = []
    key = (row["dataset"], row["split"], row["model"], row["prompt_path"], row["task_type"])
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                old_key = (old.get("dataset"), old.get("split"), old.get("model"), old.get("prompt_path"), old.get("task_type"))
                if old_key != key:
                    rows.append({h: old.get(h) for h in METRICS_HEADER})
    rows.append({h: row.get(h) for h in METRICS_HEADER})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def build_row(records, jsonl_path: Path, args):
    first = records[0]
    split = first["split"]
    task_type = get_task_type(split, first)
    if args.no_reparse_responses:
        preds = [rec.get("prediction") for rec in records]
    else:
        preds = [
            extract_prediction(rec.get("answer_content") or rec.get("response") or rec.get("prediction") or "", task_type)
            for rec in records
        ]
    refs = [rec.get("answer") for rec in records]
    row = {h: None for h in METRICS_HEADER}
    row.update({
        "timestamp": datetime.now().isoformat(),
        "dataset": first.get("dataset"),
        "split": split,
        "model": first.get("model"),
        "prompt_path": first.get("prompt_path"),
        "task_type": task_type,
    })
    if task_type in {"mc", "bool"}:
        matches = [str(pred).strip().lower() == str(ref).strip().lower() for pred, ref in zip(preds, refs)]
        row["accuracy"] = f"{sum(matches) / len(matches):.4f}" if matches else "0.0000"
    elif task_type == "entity_extraction":
        metrics = compute_entity_f1(preds, refs)
        for key in ("precision", "recall", "entity_f1", "class_f1", "property_f1"):
            row[key] = f"{metrics[key]:.4f}"
    elif task_type == "open_triple":
        metrics = compute_open_triple_metrics(preds, refs)
        for key in ("precision", "recall", "f1"):
            row[key] = f"{metrics[key]:.4f}"
    else:
        metrics = compute_text_metrics(
            preds,
            refs,
            compute_bertscore=args.compute_bertscore,
            bertscore_model=args.bertscore_model,
            bertscore_device=args.bertscore_device,
            bertscore_batch_size=args.bertscore_batch_size,
            fail_on_error=args.fail_on_bertscore_error,
        )
        for key in ("bleu", "rouge1", "rouge2", "rougeL", "bertscore_precision", "bertscore_recall", "bertscore_f1"):
            if key in metrics:
                row[key] = f"{metrics[key]:.4f}"
        if "bertscore_model" in metrics:
            row["bertscore_model"] = metrics["bertscore_model"]
        if "bertscore_error" in metrics:
            row["bertscore_error"] = metrics["bertscore_error"]
    return row


def metrics_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.stem + "_metrics.json")


def iter_jsonl_paths(args):
    root = Path(args.output_dir)
    model_dirs = [root / model for model in args.models] if args.models else [p for p in root.iterdir() if p.is_dir()]
    split_filter = set(args.splits or [])
    shot_filter = set(args.shots or [])
    for model_dir in sorted(model_dirs):
        if not model_dir.exists():
            logger.warning("Model directory does not exist: %s", model_dir)
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            split, shot = split_and_shot_from_path(jsonl_path)
            if split_filter and split not in split_filter:
                continue
            if shot_filter and shot not in shot_filter:
                continue
            yield jsonl_path


def main():
    args = parse_args()
    rebuilt = skipped = 0
    for jsonl_path in iter_jsonl_paths(args):
        metrics_json_path = metrics_path_for(jsonl_path)
        if metrics_json_path.exists() and not args.overwrite_metrics:
            split, _shot = split_and_shot_from_path(jsonl_path)
            needs_bertscore = (
                args.compute_bertscore
                and get_task_type(split) == "open_text"
                and not metrics_json_has_bertscore(metrics_json_path)
            )
            if not needs_bertscore:
                skipped += 1
                continue
        records = read_jsonl(jsonl_path)
        if not records:
            logger.warning("Empty JSONL, skipping: %s", jsonl_path)
            skipped += 1
            continue
        row = build_row(records, jsonl_path, args)
        task_type = get_task_type(records[0]["split"], records[0])
        metrics_payload = {
            "timestamp": row["timestamp"],
            "dataset": row["dataset"],
            "split": row["split"],
            "n": len(records),
            "metrics": {
                key: value for key, value in row.items()
                if key not in {"timestamp", "dataset", "split", "model", "prompt_path", "task_type"}
                and value is not None
            },
            "parse_failures": parse_failures_for_records(records, task_type, args.no_reparse_responses),
            "jsonl_path": str(jsonl_path),
            "source": "offline_evaluator",
        }
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        csv_path = Path(args.output_dir) / f"{jsonl_path.parent.name}_results.csv"
        upsert_csv_row(csv_path, row)
        logger.info("Rebuilt metrics: %s", metrics_json_path)
        rebuilt += 1
    logger.info("Done: rebuilt=%s skipped=%s", rebuilt, skipped)


if __name__ == "__main__":
    main()
