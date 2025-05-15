#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G
#SBATCH --output=log/onto_internlm3_8b_all.log

# Activate env
source ~/.bashrc
conda activate onto

# Task 1 and 2
splits=(1_1 1_2 1_3 1_4 1_5 2_1 2_2 2_3 2_4 2_5)
shots=(zero_shot two_shot four_shot)

for split in "${splits[@]}"; do
    for shot in "${shots[@]}"; do
        echo "▶ Running split=$split, prompt=$shot"
        prompt_path="./prompt/bench_${split}/${shot}.txt"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_script.py \
              --dataset XiaoZhang98/OntoURL \
              --split_index "$split" \
              --model internlm/internlm3-8b-instruct \
              --prompt_path "$prompt_path" \
              --output_dir ./output \
              --max_batched_tokens 8192 \
              --max_tokens 128 \
              --temperature 0.0 \
              --top_p 1.0
    done
done

# Task 3
splits=(3_1 3_2 3_3 3_4 3_5)
shots=(zero_shot two_shot four_shot)

for split in "${splits[@]}"; do
    for shot in "${shots[@]}"; do
        echo "▶ Running split=$split, prompt=$shot"
        prompt_path="./prompt/bench_${split}/${shot}.txt"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_script.py \
              --dataset XiaoZhang98/OntoURL \
              --split_index "$split" \
              --model internlm/internlm3-8b-instruct \
              --prompt_path "$prompt_path" \
              --output_dir ./output \
              --max_batched_tokens 8192 \
              --max_tokens 512 \
              --temperature 0.0 \
              --top_p 1.0
    done
done