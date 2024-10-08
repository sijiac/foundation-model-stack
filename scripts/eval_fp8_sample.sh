#!/bin/bash

for i in {0..3}
do
    USE_FP8_ATTENTION=$i
    echo "USE_FP8_ATTENTION: $USE_FP8_ATTENTION"
    CUDA_VISIBLE_DEVICES=0 USE_FP8_ATTENTION=$i python scripts/inference.py --architecture=llama --variant=3-8b --tokenizer="/home/sijiac/oss/hf-models/Meta-Llama-3-8B/" --model_path="/home/sijiac/oss/hf-models/Meta-Llama-3-8B/" --device_type=cuda --model_source=hf
done
