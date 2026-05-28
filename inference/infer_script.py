#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import argparse
import logging
import hashlib

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import torch
from datasets import load_dataset
from datasets import Dataset
from vllm import LLM, SamplingParams
from nltk.translate.bleu_score import sentence_bleu
import csv
import ast
from datetime import datetime
from pathlib import Path
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM inference over OntoURL splits")
    parser.add_argument(
        "--dataset",
        type=str,
        default="XiaoZhang98/OntoURL",
        help="Hugging Face dataset ID (default: XiaoZhang98/OntoURL)"
    )
    parser.add_argument(
        "--local_csv",
        type=str,
        default=None,
        help="Optional local CSV file to evaluate instead of loading a Hugging Face dataset."
    )
    parser.add_argument(
        "--local_csv_kind",
        choices=["auto", "ontourl", "propp"],
        default="auto",
        help="Schema for --local_csv. Auto infers from filename; use ontourl for final_benchmark CSVs."
    )
    parser.add_argument(
        "--split_index",
        type=str,
        default=None,
        help="OntoURL split (e.g., 1_1, 1_2, 2_1, 2_2)"
    )
    parser.add_argument(
        "--split_indices",
        nargs="+",
        default=None,
        help="Batch mode: run multiple OntoURL splits in one model load."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or Hugging Face ID of the model"
    )
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="auto",
        help="vLLM tokenizer_mode. Mistral official checkpoints may require 'mistral'."
    )
    parser.add_argument(
        "--config_format",
        type=str,
        default="auto",
        help="vLLM config_format. Mistral official checkpoints may require 'mistral'."
    )
    parser.add_argument(
        "--load_format",
        type=str,
        default="auto",
        help="vLLM load_format. Mistral official checkpoints may require 'mistral'."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Template file with few-shot examples and '|||'-separated question template"
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Batch mode: prompt root containing bench_<split>/<shot>.txt files."
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        default=None,
        help="Batch mode: shot files to run, e.g. zero_shot two_shot four_shot."
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
        help="vLLM max_num_batched_tokens"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM max_model_len. Defaults to --max_batched_tokens for backwards compatibility."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Max tokens to generate per example"
    )
    parser.add_argument("--max_tokens_mc", type=int, default=128)
    parser.add_argument("--max_tokens_l1", type=int, default=512)
    parser.add_argument("--max_tokens_text", type=int, default=512)
    parser.add_argument("--max_tokens_triple", type=int, default=1024)
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
        "--min_tokens",
        type=int,
        default=0,
        help="Minimum generated tokens. Useful for legacy base-style models that emit EOS immediately."
    )
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Continue generation even if EOS is sampled. Use only for compatibility smoke/full runs."
    )
    parser.add_argument(
        "--choice_decoding",
        action="store_true",
        help="For MC/boolean tasks, constrain generation to valid answer tokens."
    )
    parser.add_argument(
        "--stop",
        nargs="+",
        default=None,
        help="Stop sequences for generation (optional)"
    )
    parser.add_argument(
        "--mc_select",
        choices=["first", "last"],
        default="last",
        help="For multiple-choice questions, returns the first or last matching option."
    )
    parser.add_argument(
        "--task_type",
        choices=["mc", "bool", "entity_extraction", "open_text", "open_triple", "open_tuple"],
        default=None,
        help="Override task type. Useful for local legacy CSVs such as Propp final."
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap each prompt with the model tokenizer's chat template when available."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are evaluating ontology benchmark questions. Follow the task instruction exactly. "
            "Return the final answer in the requested format."
        ),
        help="System message used when --use_chat_template is enabled."
    )
    parser.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False to tokenizers that support it, such as Qwen3/Qwen3.5."
    )
    parser.add_argument(
        "--append_no_think",
        action="store_true",
        help="Append /no_think to the user prompt as an extra soft switch for Qwen-style models."
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing JSONL result file.")
    parser.add_argument("--save_prompts", action="store_true", help="Store full rendered prompts in JSONL.")
    parser.add_argument(
        "--compute_bertscore",
        action="store_true",
        help=(
            "Compute BERTScore for open-text tasks such as L2 definition generation. "
            "For full GPU inference runs, prefer the offline evaluator instead so "
            "metric dependency failures cannot terminate vLLM jobs."
        )
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="roberta-large",
        help="Model name passed to bert-score when --compute_bertscore is enabled."
    )
    args = parser.parse_args()

    # Validate arguments
    if args.max_batched_tokens <= 0:
        raise ValueError("max_batched_tokens must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    for name in ("max_tokens_mc", "max_tokens_l1", "max_tokens_text", "max_tokens_triple"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not (0 <= args.top_p <= 1):
        raise ValueError("top_p must be in [0, 1]")
    if args.split_indices:
        if args.prompt_path and len(args.split_indices) > 1:
            raise ValueError("--prompt_path is only valid for a single split; use --prompt_dir with --split_indices")
        if not args.prompt_dir and not args.prompt_path:
            raise ValueError("Batch mode requires --prompt_dir, unless exactly one --prompt_path is provided")
    elif not args.split_index:
        raise ValueError("Provide --split_index for single mode or --split_indices for batch mode")
    if args.split_index and not args.prompt_path and not args.prompt_dir:
        raise ValueError("Single mode requires --prompt_path, or --prompt_dir plus --shots")
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

def build_vllm_engine(
    model,
    max_batched_tokens,
    max_model_len=None,
    tensor_parallel_size=None,
    gpu_memory_utilization=0.90,
    seed=0,
    tokenizer_mode="auto",
    config_format="auto",
    load_format="auto",
):
    """
    Initialize the vLLM engine with given model and batching settings.
    """
    try:
        num_gpus = torch.cuda.device_count()
        tp_size = tensor_parallel_size or max(1, num_gpus)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            # model_impl="transformers",  # for gemma
            max_num_batched_tokens=max_batched_tokens,
            max_model_len=max_model_len or max_batched_tokens,
            trust_remote_code=True,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            load_format=load_format,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vLLM with model '{model}': {str(e)}")

def get_task_type(split_index: str) -> str:
    """
    Determine the task type based on split_index:
      - Multiple choice (A/B/C/D) -> 'mc'
      - True/False questions      -> 'bool'
      - Ontology term extraction -> 'entity_extraction'
      - Open-ended text -> 'open_text'
      - Open-ended triple extraction, order-sensitive -> 'open_triple'
      - Open-ended tuple extraction, order-insensitive -> 'open_tuple'
    """
    task_map = {
        "1_1": "mc", "1_2": "mc", "1_3": "mc", "1_4": "mc", "1_5": "mc",
        "2_1": "mc", "2_2": "mc", "2_3": "mc", "2_4": "mc", "2_5": "bool",
        "3_1": "entity_extraction",
        "3_2": "open_text",
        "3_3": "open_triple", "3_4": "open_triple", "3_5": "open_triple",
        "propp_verbalisation_classification": "mc",
        "propp_verbalisation_reverse": "mc",
        "propp_sequence_statement": "mc",
        "propp_sequence_symbol": "mc",
    }
    if split_index not in task_map:
        logger.warning(f"Unknown split '{split_index}', defaulting to 'mc' task type")
        return "mc"
    return task_map[split_index]

def infer_local_csv_kind(local_csv: str | None, explicit_kind: str = "auto") -> str:
    if explicit_kind != "auto":
        return explicit_kind
    name = Path(local_csv or "").name.lower()
    if "propp" in name:
        return "propp"
    return "ontourl"


def get_local_csv_task_type(split_index: str, local_csv_kind: str = "ontourl") -> str:
    if local_csv_kind == "propp" and split_index in {"3_2", "3_4"}:
        return "open_triple"
    return get_task_type(split_index)

MAX_AST_TUPLE_PARSE_CHARS = 20000
MAX_AST_TUPLE_PARSE_BRACKETS = 2000
MAX_PARSED_TUPLES = 10000

def parse_tuples_ast(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []
    if len(raw) > MAX_AST_TUPLE_PARSE_CHARS:
        return []
    if raw[0] not in {"[", "("}:
        return []
    if raw.count("(") + raw.count("[") > MAX_AST_TUPLE_PARSE_BRACKETS:
        return []
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError, SystemError, OverflowError):
        return []

    result = []

    def collect(o):
        if len(result) >= MAX_PARSED_TUPLES:
            return
        if isinstance(o, (list, tuple)):
            if len(o) >= 2 and all(not isinstance(x, (list, tuple)) for x in o):
                result.append(tuple(str(x).strip() for x in o))
            else:
                for x in o:
                    collect(x)
                    if len(result) >= MAX_PARSED_TUPLES:
                        break

    collect(obj)
    return result

def parse_tuples_regex(raw: str):
    s = raw.strip()
    tuples = []

    def iter_tuple_contents(text: str):
        start = None
        depth = 0
        quote = None
        escape = False
        for idx, ch in enumerate(text):
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                continue
            if ch == "(":
                if depth == 0:
                    start = idx + 1
                depth += 1
            elif ch == ")" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:idx]
                    start = None

    def split_tuple_content(content: str):
        parts = []
        current = []
        quote = None
        escape = False
        nested = 0
        for ch in content:
            if quote:
                current.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                current.append(ch)
                continue
            if ch in "([{":
                nested += 1
                current.append(ch)
                continue
            if ch in ")]}" and nested:
                nested -= 1
                current.append(ch)
                continue
            if ch == "," and nested == 0:
                item = "".join(current).strip(" '\"")
                if item:
                    parts.append(item)
                current = []
                continue
            current.append(ch)
        item = "".join(current).strip(" '\"")
        if item:
            parts.append(item)
        return parts

    for content in iter_tuple_contents(s):
        elements = split_tuple_content(content)
        if len(elements) >= 2:
            tuples.append(tuple(elements))
        if len(tuples) >= MAX_PARSED_TUPLES:
            break
    return tuples

def strip_thinking(raw: str) -> tuple[str, str]:
    """Return (answer_content, hidden_reasoning_content) from model output."""
    raw = raw or ""
    think_blocks = re.findall(r"(?is)<think>\s*(.*?)\s*</think>", raw)
    if "</think>" in raw.lower():
        # Qwen-style outputs can include an empty or long think block followed by
        # the actual answer. Parse only the post-thinking answer content.
        content = re.split(r"(?is)</think>", raw)[-1].strip()
        return content, "\n\n".join(block.strip() for block in think_blocks)
    return raw.strip(), "\n\n".join(block.strip() for block in think_blocks)

def extract_prediction(raw: str, task_type: str, mc_select: str="first") -> str:
    """
    Extract the model's prediction from raw generated text.
    Prefer concise answer-only outputs; keep boxed-output parsing only for
    backward compatibility with older runs.
    对 `open_triple` 和 `open_tuple` 使用 AST/正则并带回退。
    """
    raw = (raw or "").strip()
    answer_scope, _thinking = strip_thinking(raw)
    final_answer_match = re.search(r"(?is)(?:^|\n)\s*final\s+answer\s*[:：]\s*(.*)$", answer_scope)
    if final_answer_match:
        answer_scope = final_answer_match.group(1).strip()
    content = None  # 兼容旧变量名，始终为 None（不使用 <ans> 范围）

    def _extract_boxed(text: str):
        # 抓取所有 \boxed{...} 的内部内容（跨行）
        return [m.strip() for m in re.findall(r"\\boxed\{\s*([\s\S]*?)\s*\}", text)]

    # 解析 MC（A-D）
    if task_type == "mc":
        scope = content if content else answer_scope
        scope = re.sub(r"(?is)^\s*(?:\[/INST\]|\[/?INST\])\s*", "", scope).strip()
        select_index = 0 if mc_select == "first" else -1
        # 兼容旧的 boxed 输出，但新 prompt 不再要求 boxed 格式。
        boxes = re.findall(r"\\boxed\{\s*([A-D])\s*\}", scope, re.IGNORECASE)
        if boxes:
            return boxes[select_index].upper()
        # 其次解析裸字母（行内或行尾），尽量避免误匹配：独立字母或 'Answer: A' 形式
        letter = re.findall(r"(?im)^\s*(?:final\s+answer\s*[:：]\s*|answer\s*[:：]\s*)?([A-D])\s*[\).]?\s*$", scope)
        if letter:
            return letter[select_index].upper()
        letter = re.findall(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*([A-D])\b", scope)
        if letter:
            return letter[select_index].upper()
        leading = re.search(
            r"(?is)^\s*(?:the\s+answer\s+is\s+|answer\s*[:：]\s*|option\s*)?"
            r"[\(\[]?\s*([A-D])\s*[\)\].:]?\b",
            scope,
        )
        if leading:
            return leading.group(1).upper()
        explanatory = re.findall(
            r"(?is)(?:correct\s+(?:answer|option|choice)\s*(?:is|:)\s*)"
            r"(?:option\s*)?([A-D])\b",
            scope,
        )
        if explanatory:
            return explanatory[select_index].upper()
        # Some instruction-tuned models explain despite "answer only" prompts.
        # Recover clear same-sentence declarations such as "Option B ... correct"
        # while avoiding "Option A is not correct" / "incorrect".
        explanatory = []
        for sent in re.split(r"(?<=[.!?])\s+|\n+", scope):
            if re.search(r"(?i)\bincorrect\b|\bnot\b[^.!?\n]{0,80}\bcorrect\b", sent):
                continue
            if not re.search(r"(?i)\bcorrect\b", sent):
                continue
            m = re.search(r"(?i)\b(?:option\s*)?([A-D])\b", sent)
            if m:
                explanatory.append(m.group(1))
        if explanatory:
            return explanatory[select_index].upper()
        # 再回退到全文（若之前只看了 block）
        if scope is not raw:
            boxes = re.findall(r"\\boxed\{\s*([A-D])\s*\}", raw, re.IGNORECASE)
            if boxes:
                return boxes[select_index].upper()
            letter = re.findall(r"(?im)^\s*(?:final\s+answer\s*[:：]\s*|answer\s*[:：]\s*)?([A-D])\s*[\).]?\s*$", raw)
            if letter:
                return letter[select_index].upper()
            explanatory = []
            for sent in re.split(r"(?<=[.!?])\s+|\n+", raw):
                if re.search(r"(?i)\bincorrect\b|\bnot\b[^.!?\n]{0,80}\bcorrect\b", sent):
                    continue
                if not re.search(r"(?i)\bcorrect\b", sent):
                    continue
                m = re.search(r"(?i)\b(?:option\s*)?([A-D])\b", sent)
                if m:
                    explanatory.append(m.group(1))
            if explanatory:
                return explanatory[select_index].upper()
        return ""

    # 解析 True/False
    if task_type == "bool":
        scope = content if content else answer_scope
        boxes = re.findall(r"\\boxed\{\s*(true|false)\s*\}", scope, re.IGNORECASE)
        if boxes:
            return boxes[-1].lower()
        tf = re.findall(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*(true|false)\b", scope)
        if tf:
            return tf[-1].lower()
        tf = re.findall(r"(?i)\b(true|false)\b", scope)
        if tf:
            return tf[-1].lower()
        if scope is not raw:
            boxes = re.findall(r"\\boxed\{\s*(true|false)\s*\}", raw, re.IGNORECASE)
            if boxes:
                return boxes[-1].lower()
            tf = re.findall(r"(?i)\b(true|false)\b", raw)
            if tf:
                return tf[-1].lower()
        return ""

    # 解析开放文本
    if task_type == "open_text":
        # 兼容旧的 boxed 输出；新 prompt 直接返回正文。
        boxed_segments = _extract_boxed(answer_scope)
        if boxed_segments:
            return " ".join(boxed_segments)
        return answer_scope

    if task_type == "entity_extraction":
        boxed_segments = _extract_boxed(answer_scope)
        return " ".join(boxed_segments) if boxed_segments else answer_scope

    # 解析三元组/元组
    if task_type in ("open_triple", "open_tuple"):
        # 兼容旧的 boxed 输出；新 prompt 直接输出 triples/tuples。
        boxed_segments = _extract_boxed(answer_scope)
        if boxed_segments:
            # 将所有 boxed 内容连接起来，用空格分隔
            joined = " ".join(boxed_segments)
            parse_scope_candidates = [joined, answer_scope, raw]
        else:
            parse_scope_candidates = [answer_scope, raw]

        parsed = []
        for scope in parse_scope_candidates:
            parsed = parse_tuples_ast(scope)
            if not parsed:
                parsed = parse_tuples_regex(scope)
            if parsed:
                break

        if task_type == "open_triple":
            triples = [t for t in parsed if len(t) == 3]
            return triples
        else:
            return parsed

    # 其他情况直接返回全文
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

def normalize_tuple_item(value: str) -> str:
    value = str(value or "").strip().strip("'\"`")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.lower()

def normalize_tuple(tup) -> tuple[str, ...]:
    return tuple(normalize_tuple_item(item) for item in tup)

def compute_open_triple_metrics(preds: list[list], refs: list[list]):
    """
    Compute micro precision, recall, and F1 for open_triple tasks.
    Triple order is not semantically meaningful, so matching is set-based.
    """
    tp = fp = fn = 0
    for pred_seq, ref_seq in zip(preds, refs):
        pred_set = {normalize_tuple(tup) for tup in pred_seq if len(tup) == 3}
        ref_set = {normalize_tuple(tup) for tup in ref_seq if len(tup) == 3}
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def compute_text_metrics(
    preds: list[str],
    refs: list[str],
    compute_bertscore: bool = False,
    bertscore_model: str = "roberta-large",
):
    """
    Compute open-ended text metrics for definition-generation tasks.
    BLEU/ROUGE are kept as auxiliary lexical scores; BERTScore is optional
    because it loads an extra model and should be requested explicitly.
    """
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for pred, ref in zip(preds, refs):
        # Compute BLEU
        pred_tokens = re.findall(r"\w+", pred.lower())
        ref_tokens = re.findall(r"\w+", ref.lower())
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

    metrics = {
        "bleu": avg_bleu,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougel
    }

    if compute_bertscore:
        try:
            from bert_score import score as bertscore_score

            precision, recall, f1 = bertscore_score(
                preds,
                refs,
                model_type=bertscore_model,
                lang="en",
                verbose=False,
                rescale_with_baseline=False,
                use_fast_tokenizer=True,
            )
            metrics.update({
                "bertscore_precision": float(precision.mean().item()),
                "bertscore_recall": float(recall.mean().item()),
                "bertscore_f1": float(f1.mean().item()),
                "bertscore_model": bertscore_model,
            })
        except Exception as exc:
            # BERTScore is an auxiliary metric that loads a second model and is
            # sensitive to cache and dependency versions. Do not let it kill a
            # paid vLLM inference job after generations have already succeeded.
            logger.exception("BERTScore failed; writing BLEU/ROUGE metrics only.")
            metrics["bertscore_error"] = str(exc)

    return metrics

def normalize_term(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[_-]+", " ", s)
    s = re.sub(r"['\"`]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def split_terms(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw or raw.lower() in {"none", "n/a", "na", "null"}:
        return []
    parts = re.split(r";|,|\n|(?:^|\n)\s*[-*]\s+", raw)
    return [part.strip(" .:;,'\"\t") for part in parts if part.strip(" .:;,'\"\t")]

def extract_section(text: str, headings: list[str]) -> str:
    pattern = "|".join(re.escape(h.rstrip(":")) for h in headings)
    match = re.search(rf"(?is)\b({pattern})\s*:\s*(.*)", text or "")
    if not match:
        return ""
    body = match.group(2)
    stop = re.search(r"(?is)\n\s*(classes?|concepts?|properties?|relations?)\s*:\s*", body)
    return body[:stop.start()].strip() if stop else body.strip()

def parse_term_extraction_output(text: str) -> dict[str, set[str]]:
    classes_raw = extract_section(text, ["Classes", "Class", "Concepts", "Concept"])
    props_raw = extract_section(text, ["Properties", "Property", "Relations", "Relation"])
    return {
        "classes": {normalize_term(term) for term in split_terms(classes_raw) if normalize_term(term)},
        "properties": {normalize_term(term) for term in split_terms(props_raw) if normalize_term(term)},
    }

def _typed_prf(pred: set[str], gold: set[str]) -> dict[str, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_entity_f1(preds: list[dict[str, set[str]]], refs: list[dict[str, set[str]]]) -> dict[str, float]:
    pred_class = set()
    gold_class = set()
    pred_prop = set()
    gold_prop = set()
    pred_all = set()
    gold_all = set()
    for index, (pred, ref) in enumerate(zip(preds, refs)):
        pc = {(index, term) for term in pred["classes"]}
        gc = {(index, term) for term in ref["classes"]}
        pp = {(index, term) for term in pred["properties"]}
        gp = {(index, term) for term in ref["properties"]}
        pred_class |= pc
        gold_class |= gc
        pred_prop |= pp
        gold_prop |= gp
        pred_all |= {("class",) + item for item in pc} | {("property",) + item for item in pp}
        gold_all |= {("class",) + item for item in gc} | {("property",) + item for item in gp}
    overall = _typed_prf(pred_all, gold_all)
    class_scores = _typed_prf(pred_class, gold_class)
    property_scores = _typed_prf(pred_prop, gold_prop)
    return {
        "precision": overall["precision"],
        "recall": overall["recall"],
        "entity_f1": overall["f1"],
        "class_f1": class_scores["f1"],
        "property_f1": property_scores["f1"],
    }

def append_metrics_to_csv(csv_path, row, header):
    """Append a row of metrics to a CSV file, creating it with header if it doesn't exist."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def load_local_csv(local_csv: str, split_index: str) -> tuple[Dataset, str]:
    rows = []
    with open(local_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split_index == "all" or row.get("task_label") == split_index:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows for split/task_label '{split_index}' in local CSV: {local_csv}")
    return Dataset.from_list(rows), f"local_csv:{Path(local_csv).name}:{split_index}"

def infer_shot_name(prompt_path: str) -> str:
    pattern = r'(zero|one|two|three|four)_shot'
    shot_match = re.search(pattern, os.path.basename(prompt_path), re.IGNORECASE)
    if shot_match:
        return shot_match.group(1)
    return Path(prompt_path).stem

def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

def build_chat_prompt(tokenizer, user_content: str, system_prompt: str, disable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False

    def apply(messages):
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    try:
        return apply(messages)
    except Exception as exc:
        if system_prompt:
            merged_user_content = f"{system_prompt.strip()}\n\n{user_content.strip()}"
            try:
                return apply([{"role": "user", "content": merged_user_content}])
            except Exception as retry_exc:
                logger.warning("Chat template retry without system role failed: %s", retry_exc)
        logger.warning("Falling back to raw prompt; chat template failed: %s", exc)
        return user_content

def load_chat_tokenizer(args):
    if args.use_chat_template:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if getattr(tokenizer, "chat_template", None) is None:
                logger.warning(
                    "Tokenizer for %s has no chat_template; using raw prompts for this model.",
                    args.model,
                )
                return None
            return tokenizer
        except Exception as exc:
            logger.warning("Could not load tokenizer for chat template; using raw prompts: %s", exc)
            return None
    return None

def build_prompts(ds, args, examples_block: str, template: str, tokenizer=None) -> list[str]:
    prompts = []
    for ex in ds:
        options = ex.get("options", "").strip().replace("\n\n", "\n")
        if options:
            question_block = ex["question"].strip() + "\n" + options
        else:
            question_block = ex["question"].strip() + "\n"
        prompt_content = examples_block + template.replace("{}", question_block)
        if args.append_no_think:
            prompt_content = prompt_content.rstrip() + "\n/no_think"
        if tokenizer is not None:
            prompt_content = build_chat_prompt(
                tokenizer,
                user_content=prompt_content,
                system_prompt=args.system_prompt,
                disable_thinking=args.disable_thinking,
            )
        prompts.append(prompt_content)
    return prompts

def allowed_answer_token_ids(tokenizer, task_type: str) -> list[int] | None:
    if tokenizer is None:
        return None
    if task_type == "mc":
        labels = ["A", "B", "C", "D"]
    elif task_type == "bool":
        labels = ["true", "false", "True", "False"]
    else:
        return None
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    ids = sorted(set(ids))
    return ids or None

def model_safe_name(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")

def shot_file_name(shot: str) -> str:
    if shot.endswith(".txt"):
        return shot
    if shot.endswith("_shot"):
        return f"{shot}.txt"
    return f"{shot}_shot.txt"

def max_tokens_for_split(args, split_index: str) -> int:
    if split_index == "3_1":
        return args.max_tokens_l1
    if args.local_csv and split_index in {"3_2", "3_4"}:
        return args.max_tokens_triple
    if split_index == "3_2":
        return args.max_tokens_text
    if split_index in {"3_3", "3_4", "3_5"}:
        return args.max_tokens_triple
    return args.max_tokens_mc if args.split_indices else args.max_tokens

def result_paths(args, split_index: str, prompt_path: str) -> tuple[str, str, str]:
    model_safe = model_safe_name(args.model)
    out_dir = os.path.join(args.output_dir, model_safe)
    shot_name = infer_shot_name(prompt_path)
    jsonl_path = os.path.join(out_dir, f"{split_index}_{shot_name}.jsonl")
    metrics_json_path = os.path.join(out_dir, f"{split_index}_{shot_name}_metrics.json")
    return out_dir, jsonl_path, metrics_json_path

def build_run_plan(args) -> list[dict]:
    splits = args.split_indices or [args.split_index]
    plan = []
    for split_index in splits:
        if args.prompt_dir:
            shots = args.shots or ["zero_shot", "two_shot", "four_shot"]
            for shot in shots:
                if args.local_csv:
                    prompt_path = os.path.join(args.prompt_dir, split_index, shot_file_name(shot))
                    if not os.path.exists(prompt_path):
                        fallback = os.path.join(args.prompt_dir, f"bench_{split_index}", shot_file_name(shot))
                        if os.path.exists(fallback):
                            prompt_path = fallback
                else:
                    prompt_path = os.path.join(args.prompt_dir, f"bench_{split_index}", shot_file_name(shot))
                plan.append({
                    "split_index": split_index,
                    "prompt_path": prompt_path,
                    "max_tokens": max_tokens_for_split(args, split_index),
                })
        else:
            plan.append({
                "split_index": split_index,
                "prompt_path": args.prompt_path,
                "max_tokens": args.max_tokens,
            })
    return plan

def select_split_dataset(args, dataset, split_index: str) -> tuple[Dataset, str, str]:
    if args.local_csv:
        ds, selected_split = load_local_csv(args.local_csv, split_index)
        dataset_name = f"local_csv:{Path(args.local_csv).name}"
    else:
        dataset_name = args.dataset
        ds = None
        selected_split = None
        for index in dataset.keys():
            if index == split_index or index.startswith(f"{split_index}_"):
                ds = dataset[index]
                selected_split = index
                break
        if ds is None:
            raise ValueError(f"Split '{split_index}' not found. Available splits: {list(dataset.keys())}")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    return ds, selected_split, dataset_name

METRICS_HEADER = [
    "timestamp", "dataset", "split", "model", "prompt_path", "task_type",
    "accuracy", "precision", "recall", "f1", "entity_f1", "class_f1",
    "property_f1", "bleu", "rouge1", "rouge2", "rougeL",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "bertscore_model", "bertscore_error"
]

def result_complete(args, split_index: str, prompt_path: str) -> tuple[bool, bool, str, str]:
    _, jsonl_path, metrics_json_path = result_paths(args, split_index, prompt_path)
    has_jsonl = os.path.exists(jsonl_path)
    has_metrics = os.path.exists(metrics_json_path)
    return has_jsonl and has_metrics, has_jsonl and not has_metrics, jsonl_path, metrics_json_path

def run_one_job(args, llm, tokenizer, dataset, job: dict) -> bool:
    split_index = job["split_index"]
    prompt_path = job["prompt_path"]
    max_tokens = job["max_tokens"]

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found for split '{split_index}': {prompt_path}")

    out_dir, jsonl_path, metrics_json_path = result_paths(args, split_index, prompt_path)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(jsonl_path) and os.path.exists(metrics_json_path) and not args.overwrite:
        logger.info("Result and metrics exist; skipping: %s", jsonl_path)
        return False
    if os.path.exists(jsonl_path) and not os.path.exists(metrics_json_path) and not args.overwrite:
        logger.warning(
            "JSONL exists but metrics are missing; skipping vLLM rerun to avoid GPU waste. "
            "Use inference/evaluate_existing_outputs.py to rebuild metrics: %s",
            jsonl_path,
        )
        return False

    ds, selected_split, dataset_name = select_split_dataset(args, dataset, split_index)
    n = len(ds)
    logger.info(
        "Processing split '%s' (%s) with %s examples, prompt=%s, max_tokens=%s",
        split_index, selected_split, n, prompt_path, max_tokens,
    )
    local_csv_kind = infer_local_csv_kind(args.local_csv, args.local_csv_kind) if args.local_csv else "ontourl"
    task_type = args.task_type or (
        get_local_csv_task_type(split_index, local_csv_kind) if args.local_csv else get_task_type(split_index)
    )
    examples_block, template = load_prompt_template(prompt_path)

    allowed_token_ids = allowed_answer_token_ids(tokenizer, task_type) if args.choice_decoding else None
    effective_max_tokens = 1 if allowed_token_ids and task_type in {"mc", "bool"} else max_tokens
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=effective_max_tokens,
        min_tokens=args.min_tokens,
        ignore_eos=args.ignore_eos,
        allowed_token_ids=allowed_token_ids,
        stop=args.stop if args.stop else None,
    )
    prompts = build_prompts(ds, args, examples_block, template, tokenizer=tokenizer)

    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        raise RuntimeError(f"vLLM generation failed for split '{split_index}': {str(e)}")

    sorted_outputs = sorted(outputs, key=lambda o: int(o.request_id))
    preds, refs = [], []
    correct = 0
    results = []

    for i, (ex, out) in enumerate(zip(ds, sorted_outputs)):
        raw = out.outputs[0].text.strip()
        answer_content, thinking_content = strip_thinking(raw)
        pred = extract_prediction(raw, task_type, args.mc_select)
        identifier = ex.get("identifier", ex.get("id", f"example_{i}"))

        if task_type == "mc":
            ref = ex["answer"].strip().upper()
            if isinstance(pred, str):
                pred = pred.strip().upper()
            is_match = (pred == ref)
            correct += int(is_match)
            preds.append(pred)
            refs.append(ref)
        elif task_type == "bool":
            ref = ex["answer"].strip().lower()
            if isinstance(pred, str):
                pred = pred.strip().lower()
            is_match = (pred == ref)
            correct += int(is_match)
            preds.append(pred)
            refs.append(ref)
        elif task_type == "entity_extraction":
            ref = parse_term_extraction_output(ex["answer"])
            parsed_pred = parse_term_extraction_output(pred)
            preds.append(parsed_pred)
            refs.append(ref)
            is_match = None
        elif task_type == "open_tuple":
            try:
                ref_tuples = ast.literal_eval(ex["answer"].strip())
                if not isinstance(ref_tuples, list):
                    ref_tuples = [ref_tuples]
            except (ValueError, SyntaxError):
                ref_tuples = extract_prediction(ex["answer"], task_type)
            preds.append(pred)
            refs.append(ref_tuples)
            is_match = None
        elif task_type == "open_triple":
            try:
                ref_seq = ast.literal_eval(ex["answer"].strip())
                if not isinstance(ref_seq, list):
                    ref_seq = [ref_seq]
            except (ValueError, SyntaxError):
                ref_seq = extract_prediction(ex["answer"], task_type)
            preds.append(pred)
            refs.append(ref_seq)
            is_match = None
        else:
            ref = ex["answer"].strip()
            preds.append(pred)
            refs.append(ref)
            is_match = None

        logp = out.outputs[0].cumulative_logprob
        toks = len(out.outputs[0].token_ids) or 1
        results.append({
            "identifier": identifier,
            "dataset": dataset_name,
            "split": split_index,
            "hf_split": selected_split,
            "model": args.model,
            "prompt_path": prompt_path,
            "prompt_hash": prompt_hash(prompts[i]),
            "question": ex["question"],
            "options": ex.get("options", ""),
            "response": raw,
            "answer_content": answer_content,
            "thinking_content": thinking_content,
            "answer": ex["answer"].strip(),
            "prediction": pred,
            "parse_success": bool(pred) if task_type in ("mc", "bool") else pred not in (None, [], ""),
            "log_probability": logp,
            "token_count": toks,
            "match": is_match,
            "finish_reason": getattr(out.outputs[0], "finish_reason", None),
            "stop_reason": getattr(out.outputs[0], "stop_reason", None),
            "prompt_token_count": len(out.prompt_token_ids or []),
        })
        if args.save_prompts:
            results[-1]["prompt"] = prompts[i]

    with open(jsonl_path, "w", encoding="utf-8") as fl:
        for rec in results:
            fl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = os.path.join(args.output_dir, f"{model_safe_name(args.model)}_results.csv")
    timestamp = datetime.now().isoformat()
    row = {h: None for h in METRICS_HEADER}
    row.update({
        "timestamp": timestamp,
        "dataset": dataset_name,
        "split": split_index,
        "model": args.model,
        "prompt_path": prompt_path,
        "task_type": task_type,
    })
    if task_type in ("mc", "bool"):
        accuracy = correct / n if n > 0 else 0.0
        logger.info("Split '%s' [%s] Accuracy: %.4f", split_index, task_type, accuracy)
        row["accuracy"] = f"{accuracy:.4f}"
    elif task_type == "entity_extraction":
        metrics = compute_entity_f1(preds, refs)
        logger.info(
            "Split '%s' [entity_extraction] Precision: %.4f, Recall: %.4f, Entity-F1: %.4f, Class-F1: %.4f, Property-F1: %.4f",
            split_index, metrics["precision"], metrics["recall"], metrics["entity_f1"],
            metrics["class_f1"], metrics["property_f1"],
        )
        row["precision"] = f"{metrics['precision']:.4f}"
        row["recall"] = f"{metrics['recall']:.4f}"
        row["entity_f1"] = f"{metrics['entity_f1']:.4f}"
        row["class_f1"] = f"{metrics['class_f1']:.4f}"
        row["property_f1"] = f"{metrics['property_f1']:.4f}"
    elif task_type == "open_tuple":
        metrics = compute_open_f1(preds, refs)
        logger.info(
            "Split '%s' [open_tuple] Precision: %.4f, Recall: %.4f, F1: %.4f",
            split_index, metrics["precision"], metrics["recall"], metrics["f1"],
        )
        row["precision"] = f"{metrics['precision']:.4f}"
        row["recall"] = f"{metrics['recall']:.4f}"
        row["f1"] = f"{metrics['f1']:.4f}"
    elif task_type == "open_triple":
        metrics = compute_open_triple_metrics(preds, refs)
        logger.info(
            "Split '%s' [open_triple] Precision: %.4f, Recall: %.4f, F1: %.4f",
            split_index, metrics["precision"], metrics["recall"], metrics["f1"],
        )
        row["precision"] = f"{metrics['precision']:.4f}"
        row["recall"] = f"{metrics['recall']:.4f}"
        row["f1"] = f"{metrics['f1']:.4f}"
    else:
        metrics = compute_text_metrics(
            preds,
            refs,
            compute_bertscore=args.compute_bertscore,
            bertscore_model=args.bertscore_model,
        )
        logger.info(
            "Split '%s' [open_text] BLEU: %.4f, ROUGE-1: %.4f, ROUGE-2: %.4f, ROUGE-L: %.4f",
            split_index, metrics["bleu"], metrics["rouge1"], metrics["rouge2"], metrics["rougeL"],
        )
        row["bleu"] = f"{metrics['bleu']:.4f}"
        row["rouge1"] = f"{metrics['rouge1']:.4f}"
        row["rouge2"] = f"{metrics['rouge2']:.4f}"
        row["rougeL"] = f"{metrics['rougeL']:.4f}"
        if args.compute_bertscore and "bertscore_error" not in metrics:
            logger.info(
                "Split '%s' [open_text] BERTScore-P: %.4f, BERTScore-R: %.4f, BERTScore-F1: %.4f (%s)",
                split_index, metrics["bertscore_precision"], metrics["bertscore_recall"],
                metrics["bertscore_f1"], metrics["bertscore_model"],
            )
            row["bertscore_precision"] = f"{metrics['bertscore_precision']:.4f}"
            row["bertscore_recall"] = f"{metrics['bertscore_recall']:.4f}"
            row["bertscore_f1"] = f"{metrics['bertscore_f1']:.4f}"
            row["bertscore_model"] = metrics["bertscore_model"]
        elif "bertscore_error" in metrics:
            row["bertscore_error"] = metrics["bertscore_error"]

    append_metrics_to_csv(csv_path, row, METRICS_HEADER)
    metrics_payload = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "hf_split": selected_split,
        "split": split_index,
        "model": args.model,
        "prompt_path": prompt_path,
        "shot": infer_shot_name(prompt_path),
        "task_type": task_type,
        "n": n,
        "args": vars(args),
        "metrics": {
            key: value for key, value in row.items()
            if key not in {"timestamp", "dataset", "split", "model", "prompt_path", "task_type"}
            and value is not None
        },
        "parse_failures": sum(1 for rec in results if not rec["parse_success"]),
        "jsonl_path": jsonl_path,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSONL: %s", jsonl_path)
    logger.info("Saved metrics JSON: %s", metrics_json_path)
    return True

def main():
    """Run vLLM inference over one or many OntoURL split/shot jobs."""
    args = parse_args()
    plan = build_run_plan(args)

    pending = []
    for job in plan:
        complete, incomplete, jsonl_path, metrics_json_path = result_complete(args, job["split_index"], job["prompt_path"])
        if complete and not args.overwrite:
            logger.info("Will skip existing complete result before model load: %s", jsonl_path)
            continue
        if incomplete and not args.overwrite:
            logger.warning(
                "Will not load vLLM for incomplete result with missing metrics. "
                "Rebuild metrics offline with inference/evaluate_existing_outputs.py: jsonl=%s metrics=%s",
                jsonl_path,
                metrics_json_path,
            )
            continue
        pending.append(job)
    if not pending:
        logger.info("No pending split/shot jobs; exiting before model load.")
        return

    if args.local_csv:
        dataset = None
    else:
        try:
            dataset = load_dataset(args.dataset)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{args.dataset}': {str(e)}")

    tokenizer = load_chat_tokenizer(args)
    llm = build_vllm_engine(
        args.model,
        args.max_batched_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        tokenizer_mode=args.tokenizer_mode,
        config_format=args.config_format,
        load_format=args.load_format,
    )

    logger.info("Loaded model once; running %s pending split/shot job(s).", len(pending))
    completed = 0
    for job in pending:
        completed += int(run_one_job(args, llm, tokenizer, dataset, job))
    logger.info("Completed %s new split/shot job(s); skipped %s existing job(s).", completed, len(plan) - len(pending))

if __name__ == "__main__":
    main()
