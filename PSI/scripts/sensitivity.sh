#!/bin/bash

cd ../public/ || exit

TRAINER="SENSITIVITY"
DATASET="fairface"
DATASET_PATH="../../dataset"
PREDICTION_ATTR="gender"
PROTECTED_ATTR="race"
MODEL_NAME="resnet18"
GPUS=(0 1 2 3)

BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

# range
HSIC_VALS=(500 750 1000 2000 5000)
CLUB_VALS=(0.5 1.0 2.0 3.0 4.0)

MAX_JOBS_PER_GPU=3

RESULT_FILE="../../output/results_sensitivity.txt"
mkdir -p "../../output/"
echo "Mode Seed Lambda_HSIC Lambda_CLUB MODETAG Task_Acc Privacy_Acc" > "$RESULT_FILE"

NUM_GPUS=${#GPUS[@]}
declare -a GPU_TASK_LISTS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_TASK_LISTS[$i]=""
done

echo "🔍 Scanning existing experiments, filtering and assigning tasks to be executed..."
task_counter=0

for SEED in 1 2 3
do
    for h in "${HSIC_VALS[@]}"; do
        for c in "${CLUB_VALS[@]}"; do

            mode="k_club"
            modetag="${mode}(${h}-${c})"
            exp_path="../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${modetag}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
            log_file="${exp_path}/train.log"

            if [ -d "$exp_path" ]; then
                echo "⏭️  [Skip] Already exists: Seed $SEED | Tag $modetag"

                last_log=$(grep "Epoch $TOTAL_EPOCHS" "$log_file" | tail -n 1)
                task_acc=$(echo "$last_log" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
                priv_acc=$(echo "$last_log" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')
                echo "$mode $SEED $h $c $modetag ${task_acc:-'Err'} ${priv_acc:-'Err'}" >> "$RESULT_FILE"
            else
                gpu_idx=$(( task_counter % NUM_GPUS ))

                GPU_TASK_LISTS[$gpu_idx]+="${mode}|${SEED}|${h}|${c}|${modetag}|${exp_path} "
                ((task_counter++))
            fi

        done
    done
done

echo "✅ Scanning complete. A total of $((${#HSIC_VALS[@]} * ${#CLUB_VALS[@]} * 3 - task_counter)) experiments have been skipped; $task_counter experiments remain to be executed."

if [ "$task_counter" -eq 0 ]; then
    echo "🎉 The sensitivity analysis has been completed. Please see the summary of results below: $RESULT_FILE"
    exit 0
fi

run_cmd() {
    local mode=$1; local seed=$2; local l_hsic=$3; local l_club=$4; local modetag=$5; local exp_path=$6; local gpu=$7
    local log_file="${exp_path}/train.log"

    mkdir -p "$exp_path"
    echo "▶️ GPU $gpu | Mode: $mode | Seed: $seed | Tag: $modetag"

    CUDA_VISIBLE_DEVICES=$gpu python chap.py \
        --dataset $DATASET \
        --dataset_path $DATASET_PATH \
        --prediction_attribute $PREDICTION_ATTR \
        --protected_attribute $PROTECTED_ATTR \
        --model_name $MODEL_NAME \
        --split_layer 4 \
        --pruning_style "$mode" \
        --pruning_ratio 0.64 \
        --batch_size $BATCH_SIZE \
        --total_epochs $TOTAL_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --seed "$seed" \
        --gpu_devices 0 \
        --experiment_path "$exp_path" \
        --lambda_hsic "$l_hsic" \
        --lambda_club "$l_club" > "$log_file" 2>&1


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

    echo "$mode $seed $l_hsic $l_club $modetag ${task_acc:-'Err'} ${priv_acc:-'Err'}" >> "$RESULT_FILE"
}

for g_idx in "${!GPUS[@]}"; do
    (
        gpu_id=${GPUS[$g_idx]}
        job_count=0

        for task_str in ${GPU_TASK_LISTS[$g_idx]}; do

            IFS='|' read -r mode seed h c modetag exp_path <<< "$task_str"

            run_cmd "$mode" "$seed" "$h" "$c" "$modetag" "$exp_path" "$gpu_id" &
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
echo "🎉 The sensitivity analysis has been completed. Please see the summary of results below: $RESULT_FILE"