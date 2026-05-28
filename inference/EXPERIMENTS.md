# OntoURL Slurm Inference Notes

## Recommended command style

Run from `inference/`.

Before submitting a full experiment matrix, run one representative smoke job
first and inspect its Slurm log, output JSONL/CSV, metrics file, parser behavior,
and exit code. This is mandatory after changing prompts, inference code, metric
code, dependencies, cache locations, or Slurm settings. Do not submit all models
or all tasks at once until that smoke job has produced a valid result; otherwise
a single configuration issue can waste many GPU jobs.

Use `submit_one_model.sh` for the clean experiment path. It exposes only the
two real axes: dataset (`ontourl`, `propp`) and mode (`standard`, `cot`). The
jobs are incremental by default: if `OUTPUT_DIR/<model>/<split>_<shot>.jsonl`
and the matching metrics file already exist, that split/shot is skipped before
vLLM loads the model. Do not set `OVERWRITE_FLAG=--overwrite` unless you
intentionally want to rerun finished results.

## Four experiment blocks

Run from `inference/`. Submit one model first, inspect the log and output, then
expand to more models.

### OntoURL standard

```bash
CONFIRM_SUBMIT=1 \
MODE=standard DATASET_KIND=ontourl \
MODEL=Qwen/Qwen3-8B GPUS=1 MEM=180G TIME_ONTOURL=01:30:00 \
./submit_one_model.sh
```

Uses:

- dataset: `XiaoZhang98/OntoURL`
- prompts: `prompt/bench_*`
- output: `output_ontourl_standard`
- token defaults: MC `128`, L1 `512`, text `512`, triples `1024`

### OntoURL CoT

```bash
CONFIRM_SUBMIT=1 \
MODE=cot DATASET_KIND=ontourl \
MODEL=Qwen/Qwen3-8B GPUS=1 MEM=180G TIME_ONTOURL=02:00:00 \
./submit_one_model.sh
```

Uses:

- dataset: `XiaoZhang98/OntoURL`
- prompts: `prompt_cot/bench_*`
- output: `output_ontourl_cot`
- token defaults: MC `1024`, L1 `768`, text `768`, triples `2048`

### Propp standard

```bash
CONFIRM_SUBMIT=1 \
MODE=standard DATASET_KIND=propp \
MODEL=Qwen/Qwen3-8B GPUS=1 MEM=180G TIME_PROPP=00:30:00 \
./submit_one_model.sh
```

Uses:

- dataset: `../data/propp_final.csv`
- prompts: `prompt/propp_csv/*`
- output: `output_propp_standard`
- token defaults: MC `256`, text `512`, triples `1536`

### Propp CoT

```bash
CONFIRM_SUBMIT=1 \
MODE=cot DATASET_KIND=propp \
MODEL=Qwen/Qwen3-8B GPUS=1 MEM=180G TIME_PROPP=00:45:00 \
./submit_one_model.sh
```

Uses:

- dataset: `../data/propp_final.csv`
- prompts: `prompt/propp_csv_cot/*`
- output: `output_propp_cot`
- token defaults: MC `1024`, text `768`, triples `2048`

You can set `DATASET_KIND=both` only after the single-dataset run is verified.

## Legacy submit scripts

The older submit scripts remain for reproducibility, but the clean path above
is preferred for new experiments.

For the selected Qwen family set, the dedicated legacy scripts are:

Smoke-test the five selected Qwen models:

```bash
CONFIRM_SUBMIT=1 ./submit_qwen_smoke.sh
```

Run the selected Qwen family on the main benchmark:

```bash
CONFIRM_SUBMIT=1 ./submit_qwen_family.sh
```

This Qwen family set is:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-32B`

The Qwen2.5 jobs do not use `/no_think`. Qwen3 answer-only jobs use
`--append_no_think`; when `PROMPT_DIR=prompt_cot`, the Qwen script disables that
soft no-think suffix so visible CoT can be generated.

## Added Legacy-Coverage Baselines

The current final model set already replaces several old-table models with
newer or more stable alternatives. For explicit legacy coverage, use
`submit_added_baselines.sh`. It submits only these four add-on models:

- `microsoft/Phi-4-mini-instruct`
- `internlm/internlm3-8b-instruct`
- `andylolu24/ollm-wikipedia`
- `andylolu24/ollm-arxiv`

The script covers all four blocks by default: OntoURL standard, OntoURL CoT,
Propp standard, and Propp CoT. It uses the same incremental skip/resume behavior
as `submit_one_model.sh`.

When `LIMIT` is set for smoke testing, this script writes to
`output_smoke_added_ontourl_<mode>` and `output_smoke_added_propp_<mode>` by
default so partial smoke files do not block later full runs.

Phi-4-mini and InternLM3 use normal chat templates. The two OLLM checkpoints
are legacy Mistral-style models and can emit EOS instead of an answer on
answer-only tasks. For MC/true-false tasks, OLLM uses `--choice_decoding`,
which constrains the first generated token to valid answer labels (`A/B/C/D` or
`true/false`). Open-generation tasks are still generated normally.

Smoke one block first:

```bash
CONFIRM_SUBMIT=1 DRY_RUN=1 \
LIMIT=20 SHOTS=zero_shot \
DATASET_KINDS=ontourl MODES=standard \
./submit_added_baselines.sh
```

After inspecting the printed commands, submit the actual smoke:

```bash
CONFIRM_SUBMIT=1 \
LIMIT=20 SHOTS=zero_shot \
DATASET_KINDS=ontourl MODES=standard \
./submit_added_baselines.sh
```

If the smoke outputs are parseable and metrics are generated, submit the full
matrix:

```bash
CONFIRM_SUBMIT=1 ./submit_added_baselines.sh
```

Do not add old Qwen variants here. The Qwen coverage is handled by the selected
Qwen2.5/Qwen3 family set above.

```bash
sbatch \
  --job-name=qwen3_8b \
  --gres=gpu:h100:1 \
  --mem=180G \
  --time=08:00:00 \
  --export=ALL,MODEL=Qwen/Qwen3-8B,DATASET=XiaoZhang98/OntoURL,PROMPT_DIR=prompt,OUTPUT_DIR=output_final,TP_SIZE=1,APPEND_NO_THINK_FLAG=--append_no_think \
  slurm/run_model.sbatch
```

For 70B-class dense models:

```bash
sbatch \
  --job-name=llama33_70b \
  --gres=gpu:h100:4 \
  --mem=720G \
  --time=14:00:00 \
  --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,DATASET=XiaoZhang98/OntoURL,PROMPT_DIR=prompt,OUTPUT_DIR=output_final,TP_SIZE=4 \
  slurm/run_model.sbatch
```

You can also edit and run:

```bash
CONFIRM_SUBMIT=1 APPEND_NO_THINK_FLAG=--append_no_think ./submit_recommended.sh
```

The previous experiment suite contained many more legacy/domain baselines. To
reproduce those baselines on the current OntoURL data, run:

```bash
CONFIRM_SUBMIT=1 APPEND_NO_THINK_FLAG=--append_no_think ./submit_legacy_baselines.sh
```

This legacy script includes the old Llama 3/3.1, Qwen2.5, QwQ, Gemma 3,
Phi 3/4, current Mistral replacements, InternLM, GLM, Aya, BioMistral,
AdaptLLM, SaulLM, and OLLM baselines. `mistralai/Ministral-8B-Instruct-2410`
is no longer submitted by default because it produced severe parse/format
instability on OntoURL and Propp. ERNIE/RAG/ultra are listed in
`model_registry.json` but not submitted by default because they are not standard
chat/instruct causal LMs for this vLLM pipeline.

## Optional Propp CSV

The curated local Propp file can be run as a separate auxiliary experiment
without mixing it into the main OntoURL benchmark:

```bash
sbatch \
  --job-name=propp_qwen3_8b \
  --gres=gpu:h100:1 \
  --mem=180G \
  --time=03:00:00 \
  --export=ALL,MODEL=Qwen/Qwen3-8B,LOCAL_CSV=../data/propp_final.csv,OUTPUT_DIR=output_propp_final,TP_SIZE=1,APPEND_NO_THINK_FLAG=--append_no_think \
  slurm/run_propp_csv.sbatch
```

This script uses Propp-specific prompts sampled from `propp_final.csv` itself.
It treats the legacy Propp `3_2` rows as structure-triple construction and
legacy `3_4` rows as constraint construction, using explicit `--task_type`
overrides so they are not confused with the current OntoURL L2/L4 numbering.
Tiny Propp splits such as `1_3` and `1_5` intentionally use no few-shot
examples for all shot labels to avoid leaking their test answers.

## GPU sizing

- `<10B`: usually 1 H100.
- `10B-20B`: usually 1 H100, sometimes lower batch size if the model has large context defaults.
- `27B-35B`: use 2 H100 for stable vLLM batching at `MAX_MODEL_LEN=8192`.
- `70B dense`: use 4 H100.
- Large MoE models: use the full-weight size, not only active parameters, when estimating memory. Start with 4 H100 unless using FP8/quantized weights.

For official Mistral checkpoints that require the Mistral vLLM loader, pass:

```bash
VLLM_TOKENIZER_MODE=mistral
VLLM_CONFIG_FORMAT=mistral
VLLM_LOAD_FORMAT=mistral
```

This is configured automatically in the submission scripts for
`mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

## Thinking models

The inference script uses chat templates by default and passes `enable_thinking=False` when tokenizers support it. This is important for Qwen3/Qwen3.5-style models so standard runs do not waste tokens on hidden or visible reasoning traces.

The cleaned experiment path does not include CoT prompt ablations. Keep standard
runs answer-only; add a new reviewed prompt directory only if a separate CoT
ablation is intentionally designed.

## CoT Ablation

CoT prompts are kept as a separate, current-task ablation in `prompt_cot/`.
They use the same 15 OntoURL tasks and examples as `prompt/`, but ask the model
for concise reasoning followed by a parseable `Final answer:` section. Do not
reuse old CoT prompt directories from earlier task numbering.

For the final CoT ablation, do not constrain the reasoning to be brief. The
prompt asks the model to reason step by step, while the runner gives enough
generation budget for both reasoning and a parseable final answer:

- `MAX_TOKENS_MC=1024`
- `MAX_TOKENS_L1=768`
- `MAX_TOKENS_TEXT=768`
- `MAX_TOKENS_TRIPLE=2048`

```bash
sbatch \
  --job-name=qwen3_8b_cot \
  --gres=gpu:h100:1 \
  --mem=180G \
  --time=08:00:00 \
  --export=ALL,MODEL=Qwen/Qwen3-8B,DATASET=XiaoZhang98/OntoURL,PROMPT_DIR=prompt_cot,OUTPUT_DIR=output_cot,TP_SIZE=1,MAX_TOKENS_MC=1024,MAX_TOKENS_L1=768,MAX_TOKENS_TEXT=768,MAX_TOKENS_TRIPLE=2048 \
  slurm/run_model.sbatch
```

For thinking models, the script still passes tokenizer-level
`enable_thinking=False` where supported. The CoT reasoning here is visible,
prompt-requested reasoning, not uncontrolled hidden thinking.

## Outputs

For each model and split/shot, the script writes:

- `OUTPUT_DIR/<model_safe>/<split>_<shot>.jsonl`
- `OUTPUT_DIR/<model_safe>/<split>_<shot>_metrics.json`
- `OUTPUT_DIR/<model_safe>_results.csv`

Each JSONL record includes raw response, post-thinking answer content, hidden thinking content if present, parsed prediction, parse success, reference answer, match for MC/TF tasks, token counts, prompt hash, dataset, split, model, and prompt path.

By default, existing split/shot JSONL files are skipped. Use `OVERWRITE_FLAG=--overwrite` when resubmitting intentionally.
Do not compute BERTScore inside vLLM inference jobs. BERTScore loads a second
transformer model and can fail because of dependency or cache issues after
generations have already succeeded. Instead, rebuild metrics from the saved
JSONL files after inference:

```bash
python evaluate_existing_outputs.py \
  --output_dir output_final \
  --splits 3_2 \
  --shots zero two four \
  --compute_bertscore \
  --overwrite_metrics
```

If BERTScore fails, the offline evaluator preserves BLEU/ROUGE metrics and
records the BERTScore error instead of wasting vLLM GPU time.
