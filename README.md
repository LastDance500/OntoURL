# OntoURL

OntoURL is a benchmark for evaluating large language models on symbolic
ontological understanding, reasoning, and learning.

This repository contains the OntoURL v1.1 benchmark release, inference scripts,
prompt templates, and evaluation utilities. The construction pipeline is
maintained in the companion repository:
https://github.com/LastDance500/Bench_Construct

Paper: https://arxiv.org/abs/2505.11031  
Dataset: https://huggingface.co/datasets/XiaoZhang98/OntoURL

## Version 1.1

OntoURL v1.1 contains 36,159 benchmark instances generated from 43 formal
ontologies across 8 domains. The release covers 15 tasks:

- Understanding: explicit ontology content such as class definitions, class
  relations, property semantics, and instances.
- Reasoning: inferred relations, property constraints, inferred instance
  classes, SWRL-style rules, and description logic.
- Learning: ontology term extraction from text, definition generation,
  hierarchy construction, property relation construction, and constraint
  construction.

The v1.1 release adds L1 Ontology Term Extraction from Text and refreshes the
task/domain distribution to match the revised paper.

## Dataset

Install the datasets library and load the Hugging Face release:

```python
from datasets import load_dataset

dataset = load_dataset("XiaoZhang98/OntoURL")
```

The same CSV files are also available in `benchmark/` for local use. The
directory includes `manifest.json`, `task_counts.csv`, `domain_counts.csv`, and
`capability_counts.csv` for release-level metadata.

## Task Suite

| ID | Task | Format | Metric | Instances |
| --- | --- | --- | --- | ---: |
| U1 | Class Definition Understanding | MCQ | Accuracy | 3,000 |
| U2 | Class Relation Understanding | MCQ | Accuracy | 3,000 |
| U3 | Property Semantics Understanding | MCQ | Accuracy | 2,862 |
| U4 | Instance Class Understanding | MCQ | Accuracy | 3,116 |
| U5 | Instance Description Understanding | MCQ | Accuracy | 1,776 |
| R1 | Inferred Class Relation Reasoning | MCQ | Accuracy | 2,968 |
| R2 | Property Constraint Reasoning | MCQ | Accuracy | 2,814 |
| R3 | Inferred Instance Class Reasoning | MCQ | Accuracy | 2,415 |
| R4 | SWRL-based Rule Reasoning | MCQ | Accuracy | 3,000 |
| R5 | Description Logic Reasoning | True/False | Accuracy | 2,535 |
| L1 | Ontology Term Extraction | Generation | Entity-F1 | 1,799 |
| L2 | Class Definition Generation | Generation | BERTScore F1 | 3,000 |
| L3 | Class Hierarchy Construction | Generation | Triple-F1 | 2,997 |
| L4 | Property Relation Construction | Generation | Triple-F1 | 264 |
| L5 | Constraint Construction | Generation | Triple-F1 | 613 |

Domain distribution:

| Domain | Instances |
| --- | ---: |
| Business and Finance | 10,771 |
| Health and Medicine | 6,354 |
| Arts, Media, and Entertainment | 4,850 |
| Sciences | 4,510 |
| Earth and Environment | 3,554 |
| Food and Agriculture | 2,736 |
| Legal | 2,429 |
| Human and Society | 955 |

## Repository Layout

```text
benchmark/                  OntoURL v1.1 CSV splits and metadata
inference/infer_script.py    vLLM inference runner
inference/evaluate_existing_outputs.py
                             offline metric recomputation utility
inference/prompt/            standard zero/few-shot prompts
inference/cot_prompt_new/    chain-of-thought prompt variants
inference/model_registry.json
                             model metadata used by the experiments
```

## Quick Start

Install the evaluation dependencies:

```bash
pip install -r requirements.txt
```

Run a local smoke test on one split before launching a larger batch:

```bash
cd inference
python infer_script.py \
  --dataset XiaoZhang98/OntoURL \
  --split_index 1_1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt_path ./prompt/bench_1_1/zero_shot.txt \
  --output_dir ./output_smoke \
  --max_batched_tokens 8192 \
  --max_model_len 8192 \
  --max_tokens 128 \
  --temperature 0.0 \
  --top_p 1.0 \
  --limit 20
```

Generation tasks use longer outputs. For example:

```bash
cd inference
python infer_script.py \
  --dataset XiaoZhang98/OntoURL \
  --split_index 3_3 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt_path ./prompt/bench_3_3/zero_shot.txt \
  --output_dir ./output_smoke \
  --max_batched_tokens 8192 \
  --max_model_len 8192 \
  --max_tokens_triple 1024 \
  --temperature 0.0 \
  --top_p 1.0 \
  --limit 10
```

Use `evaluate_existing_outputs.py` to recompute metrics from saved JSONL files,
especially for BERTScore on L2 definition generation:

```bash
cd inference
python evaluate_existing_outputs.py \
  --output_dir ./output_smoke \
  --compute_bertscore \
  --bertscore_model roberta-large
```

## Main Results

The table below reports zero-shot OntoURL v1.1 scores from the revised paper.
Metrics are task-specific: U/R columns are accuracy, L1 is Entity-F1, L2 is
BERTScore F1, and L3-L5 are Triple-F1.

| Model | U1 | U2 | U3 | U4 | U5 | R1 | R2 | R3 | R4 | R5 | L1 | L2 | L3 | L4 | L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma-4-E4B | 78.7 | 80.3 | 60.6 | 79.6 | 75.3 | 83.3 | 82.2 | 69.4 | 70.3 | 69.3 | 79.0 | 87.8 | 40.9 | 6.1 | 0.3 |
| Qwen2.5-7B | 84.2 | 81.3 | 59.8 | 80.7 | 80.1 | 88.4 | 90.1 | 74.0 | 75.3 | 53.0 | 84.8 | 86.1 | 38.3 | 4.1 | 13.0 |
| Qwen3-8B | 83.0 | 81.8 | 57.8 | 82.9 | 78.3 | 86.9 | 88.3 | 72.4 | 74.3 | 64.4 | 83.0 | 86.2 | 45.7 | 6.4 | 14.8 |
| Ollm-wiki-7B | 38.3 | 34.2 | 30.6 | 44.3 | 42.2 | 29.5 | 39.2 | 31.5 | 26.8 | 47.9 | 18.3 | 89.5 | 0.8 | 0.0 | 0.1 |
| Ollm-arxiv-7B | 49.8 | 41.7 | 33.5 | 56.0 | 51.7 | 33.9 | 46.6 | 38.6 | 28.4 | 47.9 | 10.6 | 89.5 | 6.5 | 0.8 | 0.2 |
| Aya-8B | 77.2 | 74.8 | 50.3 | 77.6 | 74.9 | 74.4 | 70.0 | 71.2 | 62.4 | 72.6 | 74.5 | 85.0 | 40.6 | 6.1 | 0.5 |
| InternLM3-8B | 83.9 | 79.6 | 60.4 | 79.1 | 80.2 | 86.8 | 85.2 | 73.5 | 76.9 | 85.2 | 75.9 | 84.4 | 33.3 | 3.8 | 11.5 |
| Phi-4 | 85.5 | 84.3 | 60.3 | 81.2 | 82.7 | 87.0 | 87.7 | 73.9 | 75.1 | 83.1 | 85.8 | 85.4 | 51.6 | 8.9 | 31.6 |
| Mistral-Small-24B | 88.1 | 85.9 | 64.6 | 84.2 | 84.4 | 91.3 | 85.9 | 73.3 | 78.5 | 83.4 | 81.2 | 86.0 | 49.0 | 9.7 | 33.8 |
| Gemma-4-26B | 87.4 | 83.6 | 65.2 | 81.5 | 83.0 | 88.0 | 84.2 | 69.7 | 79.5 | 83.5 | 79.0 | 83.5 | 55.2 | 9.8 | 4.5 |
| Qwen2.5-32B | 88.8 | 85.0 | 65.3 | 82.7 | 86.2 | 92.6 | 95.0 | 78.0 | 81.7 | 54.5 | 87.3 | 86.1 | 56.8 | 9.4 | 31.1 |
| Qwen3-32B | 88.9 | 85.2 | 63.2 | 84.5 | 85.9 | 90.4 | 92.2 | 75.8 | 80.9 | 82.6 | 91.5 | 86.2 | 54.9 | 9.1 | 35.2 |
| GLM-4-32B | 88.0 | 84.4 | 62.2 | 83.0 | 86.1 | 90.2 | 89.9 | 73.0 | 79.1 | 69.0 | 80.4 | 87.4 | 50.8 | 6.9 | 30.1 |
| Aya-32B | 82.8 | 83.5 | 59.6 | 82.2 | 80.2 | 87.9 | 88.4 | 73.0 | 73.2 | 76.1 | 82.0 | 84.8 | 46.3 | 8.9 | 11.1 |
| LLaMA3.3-70B | 89.2 | 85.3 | 61.7 | 84.0 | 87.4 | 90.6 | 89.3 | 75.4 | 76.8 | 88.2 | 79.4 | 86.3 | 56.0 | 11.3 | 32.0 |
| Qwen2.5-72B | 90.4 | 87.1 | 65.9 | 85.6 | 87.7 | 92.4 | 94.7 | 77.2 | 83.2 | 83.2 | 86.9 | 86.0 | 55.7 | 12.0 | 34.1 |

## Citation

```bibtex
@article{zhang2025ontourl,
  title={OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning},
  author={Zhang, Xiao and Lai, Huiyuan and Meng, Qianru and Bos, Johan},
  journal={arXiv preprint arXiv:2505.11031},
  year={2025}
}
```

## Availability and License

Code is released under the MIT License. Generated benchmark data are released
under CC BY 4.0 where permitted by source ontology licenses; ontology-specific
license metadata are provided in the dataset card.

