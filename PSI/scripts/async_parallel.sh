#!/bin/bash

cd ../public/ || exit

TRAINER="K_CLUB_ASYNC_PARALLEL"
PRUNING_STYLE="k_club"

L_HSIC=500
L_CLUB=3.0
MODETAG="${PRUNING_STYLE}(${L_HSIC}-${L_CLUB})_async"

DATASET_PATH="../../dataset"
ADV_METHOD="ml-leaks"
BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01
SPLIT_LAYER=6
PRUNING_RATIO=0.64

DATASET="fairface"
PREDICTION_ATTR="gender"
PROTECTED_ATTR="race"
SEED=1

SERVER_IP="127.0.0.1"

MODELS=("resnet18" "resnet34" "resnet50")
GPUS=(1 2 3)
PORTS=(10080 10081 10082)

RESULT_FILE="../../output/results_${TRAINER}_${SPLIT_LAYER}.txt"
mkdir -p "../../output/"
echo "Model Target Privacy Seed Task_Acc Privacy_Acc" > "$RESULT_FILE"

echo "🚀 Starting PARALLEL Async Training for models: ${MODELS[*]}"

for i in "${!MODELS[@]}"; do
    MODEL_NAME=${MODELS[$i]}
    GPU_ID=${GPUS[$i]}
    PORT=${PORTS[$i]}

    EXP_PATH="../../output/${DATASET}/${TRAINER}_${SPLIT_LAYER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
    LOG_FILE_SERVER="${EXP_PATH}/server_train.log"
    LOG_FILE_CLIENT="${EXP_PATH}/client_train.log"

    if [ -d "$EXP_PATH" ]; then
        echo "⏭️  [Skip] Already exists: $MODEL_NAME on GPU $GPU_ID"
        continue
    fi

    mkdir -p "$EXP_PATH"
    echo "--------------------------------------------------------"
    echo "▶️ Launching $MODEL_NAME on GPU $GPU_ID with Port $PORT"

    CUDA_VISIBLE_DEVICES=$GPU_ID python chap.py \
        --dataset "$DATASET" \
        --dataset_path "$DATASET_PATH" \
        --prediction_attribute "$PREDICTION_ATTR" \
        --protected_attribute "$PROTECTED_ATTR" \
        --model_name "$MODEL_NAME" \
        --split_layer "$SPLIT_LAYER" \
        --pruning_style "$PRUNING_STYLE" \
        --pruning_ratio "$PRUNING_RATIO" \
        --batch_size "$BATCH_SIZE" \
        --total_epochs "$TOTAL_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --seed "$SEED" \
        --gpu_devices 0 \
        --experiment_path "$EXP_PATH" \
        --lambda_hsic "$L_HSIC" \
        --lambda_club "$L_CLUB" \
        --adv_pred_method "$ADV_METHOD" \
        --is_async 1 \
        --is_server 1 \
        --server_ip "$SERVER_IP" \
        --server_port "$PORT" > "$LOG_FILE_SERVER" 2>&1 &

    sleep 3

    CUDA_VISIBLE_DEVICES=$GPU_ID python chap.py \
        --dataset "$DATASET" \
        --dataset_path "$DATASET_PATH" \
        --prediction_attribute "$PREDICTION_ATTR" \
        --protected_attribute "$PROTECTED_ATTR" \
        --model_name "$MODEL_NAME" \
        --split_layer "$SPLIT_LAYER" \
        --pruning_style "$PRUNING_STYLE" \
        --pruning_ratio "$PRUNING_RATIO" \
        --batch_size "$BATCH_SIZE" \
        --total_epochs "$TOTAL_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --seed "$SEED" \
        --gpu_devices 0 \
        --experiment_path "$EXP_PATH" \
        --lambda_hsic "$L_HSIC" \
        --lambda_club "$L_CLUB" \
        --adv_pred_method "$ADV_METHOD" \
        --is_async 1 \
        --is_server 0 \
        --server_ip "$SERVER_IP" \
        --server_port "$PORT" > "$LOG_FILE_CLIENT" 2>&1 &

done

echo "⏳ All models have been launched in the background. Waiting for them to finish..."
wait

echo "✅ All parallel training tasks are complete!"

for i in "${!MODELS[@]}"; do
    MODEL_NAME=${MODELS[$i]}
    EXP_PATH="../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
    LOG_FILE_SERVER="${EXP_PATH}/server_train.log"

    if [ -f "$LOG_FILE_SERVER" ]; then
        last_log=$(grep "Epoch $TOTAL_EPOCHS" "$LOG_FILE_SERVER" | tail -n 1)
        task_acc=$(echo "$last_log" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
        priv_acc=$(echo "$last_log" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')

        echo "${MODEL_NAME} ${PREDICTION_ATTR} ${PROTECTED_ATTR} ${SEED} ${task_acc:-'Err'} ${priv_acc:-'Err'}" >> "$RESULT_FILE"
    fi
done

echo "🎉 Summary saved to $RESULT_FILE"
