#!/bin/bash

# This script runs various tasks with different values of USE_FP8_ATTENTION.
# Usage examples:
#
# 1. Run all tasks for all USE_FP8_ATTENTION modes (0 to 3):
#    ./run_combined_tasks.sh
#
# 2. Run all tasks for a specific USE_FP8_ATTENTION mode (e.g., mode 2):
#    ./run_combined_tasks.sh 2
#
# 3. Run a specific task (e.g., task 'mmlu') for all USE_FP8_ATTENTION modes (0 to 3):
#    ./run_combined_tasks.sh "" mmlu
#
# 4. Run a specific task (e.g., task 'perplexity') for a specific USE_FP8_ATTENTION mode (e.g., mode 1):
#    ./run_combined_tasks.sh 1 perplexity
#
# 5. Run for multiple modes (e.g., 0,1,3) with a specific task:
#    ./scripts/eval_fp8.sh "0,1,3" perplexity

# Define tasks and the associated commands
declare -A TASKS
TASKS=(
  ["mmlu"]="python scripts/eval_harness.py --architecture=llama --variant=3-8b --model_source=hf --model_path=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tokenizer=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tasks=mmlu --num_fewshot=0 --device_type=cuda"
  ["hellaswag"]="python scripts/eval_harness.py --architecture=llama --variant=3-8b --model_source=hf --model_path=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tokenizer=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tasks=hellaswag --num_fewshot=0 --device_type=cuda"
  ["perplexity"]="python scripts/eval_harness.py --architecture=llama --variant=3-8b --model_source=hf --model_path=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tokenizer=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tasks=perplexity --device_type=cuda"
  ["sample"]="python scripts/inference.py --architecture=llama --variant=3-8b --model_source=hf --model_path=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --tokenizer=/home/sijiac/oss/hf-models/Meta-Llama-3-8B/ --device_type=cuda"
)

# Function to validate task input
function validate_task {
  if [[ ! -v TASKS[$1] ]]; then
    echo "Invalid task: $1. Available tasks are: ${!TASKS[@]}"
    exit 1
  fi
}

# Parse mode and task arguments
modes=""
task=""

if [ "$#" -ge 1 ] && [ -n "$1" ]; then
  modes=$1
  # Validate modes (accept comma-separated list, e.g., "0,1,2")
  IFS=',' read -r -a mode_array <<< "$modes"
  for mode in "${mode_array[@]}"; do
    if ! [[ "$mode" =~ ^[0-6]$ ]]; then
      echo "Invalid mode: $mode. Please pass values between 0 and 3."
      exit 1
    fi
  done
fi

if [ "$#" -ge 2 ]; then
  task=$2
  validate_task "$task"
fi

# Function to run tasks based on the current mode and task
function run_tasks {
  for task_name in "${!TASKS[@]}"
  do
    if [ -n "$task" ] && [ "$task_name" != "$task" ]; then
      continue
    fi
    echo "Executing task: $task_name with USE_FP8_ATTENTION=$1"
    CUDA_VISIBLE_DEVICES=0 USE_FP8_ATTENTION=$1 USE_HDT=1 ${TASKS[$task_name]}
    echo "Completed task: $task_name with USE_FP8_ATTENTION: $1"
    echo "----------------------------------------"
  done
}

# If modes are specified, run tasks for each specified mode
if [ -n "$modes" ]; then
  for mode in "${mode_array[@]}"; do
    echo "Running with USE_FP8_ATTENTION: $mode"
    run_tasks "$mode"
  done
else
  # If no modes are specified, default to running for all modes (0..3)
  for i in {0..3}; do
    echo "Running with USE_FP8_ATTENTION: $i"
    run_tasks "$i"
  done
fi
