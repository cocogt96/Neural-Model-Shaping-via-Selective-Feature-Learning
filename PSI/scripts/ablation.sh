#!/bin/bash

cd ../public/reconstruction || exit

TRAINER="ABLATION"

DATASET_PATH="../../../dataset"
BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

SPLIT_LAYER=4
PRUNING_RATIO=0.64
GPUS=(5 6 7)

MODELS=("resnet18")
TASKS=("fairface gender race")
MODES=("hsic" "club" "k_club")

RESULT_FILE="../../../output/results_${TRAINER}.txt"
mkdir -p "../../../output/"
echo "Model Target Privacy Mode Seed Task_Acc Privacy_Acc" > "$RESULT_FILE"

echo "🚀 Start to Ablation: conclude [ ${MODES[*]} ]"

for SEED in 1 2 3
do
  (
    GPU_ID=${GPUS[$((SEED-1))]}
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    for MODEL_NAME in "${MODELS[@]}"
    do
      for task in "${TASKS[@]}"
      do
        read -r DATASET PREDICTION_ATTR PROTECTED_ATTR <<< "$task"

        for MODE in "${MODES[@]}"
        do
          if [ "$MODE" == "hsic" ]; then
              LAMBDA_HSIC=500
              LAMBDA_CLUB=0.0
              MODETAG="$MODE($LAMBDA_HSIC)"
          elif [ "$MODE" == "club" ]; then
              LAMBDA_HSIC=0.0
              LAMBDA_CLUB=3.0
              MODETAG="$MODE($LAMBDA_CLUB)"
          elif [ "$MODE" == "k_club" ]; then
              LAMBDA_HSIC=500.0
              LAMBDA_CLUB=3.0
              MODETAG="$MODE($LAMBDA_HSIC-$LAMBDA_CLUB)"
          fi

          EXPERIMENT_PATH="../../../output/${DATASET}/${TRAINER}/${MODEL_NAME}_${MODETAG}/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
          LOG_FILE="${EXPERIMENT_PATH}/train.log"
          if [ -d "$EXPERIMENT_PATH" ]; then
            echo "⏭️  [GPU $GPU_ID] Oops! The results exist at ${EXPERIMENT_PATH} (so skip this job)"
          else
            mkdir -p "$EXPERIMENT_PATH"

            CMD="python chap.py \
                --dataset $DATASET \
                --dataset_path $DATASET_PATH \
                --prediction_attribute $PREDICTION_ATTR \
                --protected_attribute $PROTECTED_ATTR \
                --model_name $MODEL_NAME \
                --split_layer $SPLIT_LAYER \
                --pruning_style $MODE \
                --pruning_ratio $PRUNING_RATIO \
                --batch_size $BATCH_SIZE \
                --total_epochs $TOTAL_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --seed $SEED \
                --gpu_devices 0 \
                --experiment_path $EXPERIMENT_PATH \
                --lambda_hsic $LAMBDA_HSIC \
                --lambda_club $LAMBDA_CLUB"

            echo "⏳ [GPU $GPU_ID]  | Mode: $MODE | Seed: $SEED | Log: $LOG_FILE"

            $CMD > "$LOG_FILE" 2>&1
          fi

          LAST_LOG=$(grep "Epoch $TOTAL_EPOCHS" "$LOG_FILE" | tail -n 1)
          TASK_ACC=$(echo "$LAST_LOG" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
          PRIV_ACC=$(echo "$LAST_LOG" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')

          TASK_ACC=${TASK_ACC:-"Error"}
          PRIV_ACC=${PRIV_ACC:-"Error"}

          echo "${MODEL_NAME} ${PREDICTION_ATTR} ${PROTECTED_ATTR} ${MODE} ${SEED} ${TASK_ACC} ${PRIV_ACC}" >> "$RESULT_FILE"
        done
      done
    done
  ) &
done

wait
echo "🎉 All ablation experiments have completed! The results have been saved to: $RESULT_FILE"