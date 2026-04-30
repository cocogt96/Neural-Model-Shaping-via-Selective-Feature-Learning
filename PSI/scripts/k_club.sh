#!/bin/bash

cd ../public/ || exit

TRAINER="K_CLUB_EXP"
PRUNING_STYLE="k_club"

L_HSIC=500
L_CLUB=3.0
MODETAG="${PRUNING_STYLE}(${L_HSIC}-${L_CLUB})"

DATASET_PATH="../../dataset"
ADV_METHOD="ml-leaks"
BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

SPLIT_LAYER=4
PRUNING_RATIO=0.64
GPUS=(0 1 2 3 4)

MAX_JOBS_PER_GPU=2

TASKS=(
    "fairface gender race"
    "fairface gender age"
    "celeba Smiling Mustache"
    "celeba Smiling Male"
    "celeba Smiling Big_Nose"
    "cifar10 living class"
)

RESULT_FILE="../../output/results_${TRAINER}.txt"
mkdir -p "../../output/"
echo "Model Target Privacy Seed Task_Acc Privacy_Acc" > "$RESULT_FILE"

NUM_GPUS=${#GPUS[@]}
declare -a GPU_TASK_LISTS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_TASK_LISTS[$i]=""
done

echo "🔍 Scanning existing experiments, filtering and assigning tasks to be executed..."
task_counter=0
total_tasks=0

for SEED in 1 2 3; do
    for MODEL_NAME in resnet18 resnet34 resnet50; do
        for task in "${TASKS[@]}"; do
            read -r DATASET PREDICTION_ATTR PROTECTED_ATTR <<< "$task"
            ((total_tasks++))

            exp_path="../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
            log_file="${exp_path}/train.log"

            if [ -d "$exp_path" ]; then
                echo "⏭️  [Skip] Already exists: $MODEL_NAME | $DATASET | $PREDICTION_ATTR vs $PROTECTED_ATTR | Seed $SEED"

                last_log=$(grep "Epoch $TOTAL_EPOCHS" "$log_file" | tail -n 1)
                task_acc=$(echo "$last_log" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
                priv_acc=$(echo "$last_log" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')
                echo "${MODEL_NAME} ${PREDICTION_ATTR} ${PROTECTED_ATTR} ${SEED} ${task_acc:-'Err'} ${priv_acc:-'Err'}" >> "$RESULT_FILE"
            else
                gpu_idx=$(( task_counter % NUM_GPUS ))
                GPU_TASK_LISTS[$gpu_idx]+="${DATASET}|${PREDICTION_ATTR}|${PROTECTED_ATTR}|${MODEL_NAME}|${SEED}|${exp_path} "
                ((task_counter++))
            fi

        done
    done
done

echo "✅ Scanning complete. Skipped $((total_tasks - task_counter)) experiments; $task_counter experiments remain to be executed."

if [ "$task_counter" -eq 0 ]; then
    echo "🎉 All K_CLUB experiments have been completed. Please see the summary of results below: $RESULT_FILE"
    exit 0
fi

run_cmd() {
    local dataset=$1; local pred_attr=$2; local prot_attr=$3; local model_name=$4; local seed=$5; local exp_path=$6; local gpu=$7
    local log_file="${exp_path}/train.log"

    mkdir -p "$exp_path"
    echo "▶️ GPU $gpu | Mode $model_name | $dataset: $pred_attr vs $prot_attr | Seed: $seed"

    CUDA_VISIBLE_DEVICES=$gpu python chap.py \
        --dataset "$dataset" \
        --dataset_path "$DATASET_PATH" \
        --prediction_attribute "$pred_attr" \
        --protected_attribute "$prot_attr" \
        --model_name "$model_name" \
        --split_layer "$SPLIT_LAYER" \
        --pruning_style "$PRUNING_STYLE" \
        --pruning_ratio "$PRUNING_RATIO" \
        --batch_size "$BATCH_SIZE" \
        --total_epochs "$TOTAL_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --seed "$seed" \
        --gpu_devices 0 \
        --experiment_path "$exp_path" \
        --lambda_hsic "$L_HSIC" \
        --lambda_club "$L_CLUB" \
        --adv_pred_method "$ADV_METHOD" > "$log_file" 2>&1

    # shellcheck disable=SC2155
    local last_log=$(grep "average test loss" "$log_file" | tail -n 1)
    # shellcheck disable=SC2155
    local task_acc=$(echo "$last_log" | sed -n 's/.*pred_accuracy \([0-9.]*\).*/\1/p')
    # shellcheck disable=SC2155
    local priv_acc=$(echo "$last_log" | sed -n 's/.*privacy_accuracy \([0-9.]*\).*/\1/p')

#    # shellcheck disable=SC2155
#    local last_log=$(grep "Epoch $TOTAL_EPOCHS" "$log_file" | tail -n 1)
#    # shellcheck disable=SC2155
#    local task_acc=$(echo "$last_log" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
#    # shellcheck disable=SC2155
#    local priv_acc=$(echo "$last_log" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')

    echo "${model_name} ${pred_attr} ${prot_attr} ${seed} ${task_acc:-'Err'} ${priv_acc:-'Err'}" >> "$RESULT_FILE"
}

for g_idx in "${!GPUS[@]}"; do
    (
        gpu_id=${GPUS[$g_idx]}
        job_count=0

        for task_str in ${GPU_TASK_LISTS[$g_idx]}; do

            IFS='|' read -r dataset pred_attr prot_attr model_name seed exp_path <<< "$task_str"

            run_cmd "$dataset" "$pred_attr" "$prot_attr" "$model_name" "$seed" "$exp_path" "$gpu_id" &
            ((job_count++))

            if [ "$job_count" -ge "$MAX_JOBS_PER_GPU" ]; then
                wait -n
                ((job_count--))
            fi
        done

        wait
    ) &
done

wait
echo "🎉 All K_CLUB experiments have been completed. Please see the summary of results below: $RESULT_FILE"