#!/bin/bash

for i in {0..3}
do
    USE_FP8_ATTENTION=$i
    echo "USE_FP8_ATTENTION: $USE_FP8_ATTENTION"
    CUDA_VISIBLE_DEVICES=0 USE_FP8_ATTENTION=$i  python scripts/eval_harness.py --architecture=llama --variant=3-8b --model_source=hf --model_path="/home/sijiac/oss/hf-models/Meta-Llama-3-8B/" --tokenizer="/home/sijiac/oss/hf-models/Meta-Llama-3-8B/"  --tasks=hellaswag --num_fewshot=0 --device_type=cuda
done
