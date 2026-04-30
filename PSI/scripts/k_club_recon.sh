#!/bin/bash

cd ../public/ || exit

TRAINER="K_CLUB_RECON"
DATASET="fairface"
DATASET_PATH="../../dataset"
PREDICTION_ATTR="gender"

PROTECTED_ATTR="data"

MODEL_NAME="resnet18"
SEED=1
GPU_ID=0

BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

PRUNING_STYLE="k_club"
L_HSIC=500
L_CLUB=3.0
MODETAG="${PRUNING_STYLE}(${L_HSIC}-${L_CLUB})"

ADV_GEN_METHOD="conv-transpose"

EXP_PATH="../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
LOG_FILE="${EXP_PATH}/train.log"

mkdir -p "$EXP_PATH"

echo "======================================================="
echo "▶️ Starting image-reconstruction level privacy erasure training (Continuous K-CLUB)"
echo "Model: $MODEL_NAME | Seed: $SEED | GPU: $GPU_ID"
echo "Task: $PREDICTION_ATTR | Adversarial Protection: $PROTECTED_ATTR"
echo "Generator Architecture: $ADV_GEN_METHOD"
echo "======================================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python chap.py \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --prediction_attribute "$PREDICTION_ATTR" \
    --protected_attribute "$PROTECTED_ATTR" \
    --model_name "$MODEL_NAME" \
    --split_layer 4 \
    --pruning_style "$PRUNING_STYLE" \
    --pruning_ratio 0.64 \
    --batch_size "$BATCH_SIZE" \
    --total_epochs "$TOTAL_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --gpu_devices 0 \
    --experiment_path "$EXP_PATH" \
    --lambda_hsic "$L_HSIC" \
    --lambda_club "$L_CLUB" \
    --adv_gen_method "$ADV_GEN_METHOD" > "$LOG_FILE" 2>&1

last_log=$(grep "Epoch $TOTAL_EPOCHS" "$LOG_FILE" | tail -n 1)
task_acc=$(echo "$last_log" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')

echo "✅ Training Complete!"
echo "Output Directory: $EXP_PATH"
echo "Main Task (Gender) Accuracy: ${task_acc:-'Error'}"
echo "Please check train.log for detailed training records."