#!/bin/bash

cd ../public/ || exit

TRAINER="MULTI_ATTR_K_CLUB"

DATASET_PATH="../../dataset"
BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

SPLIT_LAYER=4
GPUS=(0 1 2)

MODELS=("resnet18" "resnet34" "resnet50")
TASKS=("fairface gender race age")

LAMBDA_HSIC=250.0
LAMBDA_CLUB=1.5
MODE="k_club"

RESULT_FILE="../../output/results_${TRAINER}.txt"
mkdir -p "../../output/"
echo "Model Target Privacy1 Privacy2 Mode Seed Task_Acc Priv1_Acc Priv2_Acc" > "$RESULT_FILE"

echo "🚀 Start Multi-Attribute K-CLUB Ablation [ HSIC=${LAMBDA_HSIC}, CLUB=${LAMBDA_CLUB} ]"

for SEED in 1 2 3
do
  (
    GPU_ID=${GPUS[$((SEED-1))]}
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    for MODEL_NAME in "${MODELS[@]}"
    do
      for task in "${TASKS[@]}"
      do
        read -r DATASET PREDICTION_ATTR PROTECTED_ATTR_1 PROTECTED_ATTR_2 <<< "$task"

        MODETAG="${MODE}(${LAMBDA_HSIC}-${LAMBDA_CLUB})"
        EXPERIMENT_PATH="../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR_1}_${PROTECTED_ATTR_2}_seed${SEED}"
        LOG_FILE="${EXPERIMENT_PATH}/train.log"

        if [ -d "$EXPERIMENT_PATH" ]; then
          echo "⏭️  [GPU $GPU_ID] Oops! The results exist at ${EXPERIMENT_PATH} (so skip this job)"
        else
          mkdir -p "$EXPERIMENT_PATH"

          CMD="python chap.py \
              --dataset $DATASET \
              --dataset_path $DATASET_PATH \
              --prediction_attribute $PREDICTION_ATTR \
              --protected_attribute $PROTECTED_ATTR_1 $PROTECTED_ATTR_2 \
              --model_name $MODEL_NAME \
              --split_layer $SPLIT_LAYER \
              --pruning_style $MODE \
              --batch_size $BATCH_SIZE \
              --total_epochs $TOTAL_EPOCHS \
              --learning_rate $LEARNING_RATE \
              --seed $SEED \
              --gpu_devices 0 \
              --experiment_path $EXPERIMENT_PATH \
              --lambda_hsic $LAMBDA_HSIC \
              --lambda_club $LAMBDA_CLUB"

          echo "⏳ [GPU $GPU_ID] | Model: $MODEL_NAME | Seed: $SEED | Log: $LOG_FILE"

          $CMD > "$LOG_FILE" 2>&1
        fi

        LAST_LOG=$(grep "\[Test\]" "$LOG_FILE" | tail -n 1)

        TASK_ACC=$(echo "$LAST_LOG" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
        PRIV_ACC1=$(echo "$LAST_LOG" | sed -n 's/.*Privacy1(Acc): \([0-9.]*\).*/\1/p')
        PRIV_ACC2=$(echo "$LAST_LOG" | sed -n 's/.*Privacy2(Acc): \([0-9.]*\).*/\1/p')

        TASK_ACC=${TASK_ACC:-"Error"}
        PRIV_ACC1=${PRIV_ACC1:-"Error"}
        PRIV_ACC2=${PRIV_ACC2:-"Error"}

        echo "${MODEL_NAME} ${PREDICTION_ATTR} ${PROTECTED_ATTR_1} ${PROTECTED_ATTR_2} ${MODE} ${SEED} ${TASK_ACC} ${PRIV_ACC1} ${PRIV_ACC2}" >> "$RESULT_FILE"
      done
    done
  ) &
done

wait
echo "🎉 All multi-attribute experiments have completed! The results have been saved to: $RESULT_FILE"