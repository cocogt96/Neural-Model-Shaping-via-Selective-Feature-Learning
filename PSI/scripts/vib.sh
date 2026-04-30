#!/bin/bash

cd ../public/ || exit

# ==========================================
# 1. Experimental Methods and Labelling Arrangements
# ==========================================
TRAINER="VIB"
PRUNING_STYLE="vib"

# ==========================================
# 2. Basic hyperparameter configuration
# ==========================================
DATASET_PATH="../../dataset"
ADV_METHOD="ml-leaks"
BATCH_SIZE=64
TOTAL_EPOCHS=100
LEARNING_RATE=0.01

SPLIT_LAYER=4
LAMBDA_VIB=0.0002
GPUS=(0 1 2)

# ==========================================
# 3. Task Combination Mapping Table (Dataset Prediction_Attr Protected_Attr)
# ==========================================
TASKS=(
    "fairface gender race"
    "cifar10 living class"
)

RESULT_FILE="../../output/results_${TRAINER}.txt"
mkdir -p "../../output/"
echo "Model Target Privacy Seed Task_Acc Privacy_Acc" > "$RESULT_FILE"

# ==========================================
# 4. Experiment
# ==========================================
for SEED in 1 2 3
do
  (
    GPU_ID=${GPUS[$((SEED-1))]}
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    for MODEL_NAME in resnet18 resnet34 resnet50
    do
      for task in "${TASKS[@]}"
      do
        read -r DATASET PREDICTION_ATTR PROTECTED_ATTR <<< "$task"

        EXPERIMENT_PATH="../../output/${DATASET}/${TRAINER}/server_${MODEL_NAME}_adversarial_${ADV_METHOD}/${PREDICTION_ATTR}_${PROTECTED_ATTR}/seed${SEED}"
        LOG_FILE="${EXPERIMENT_PATH}/terminal.log"

        if [ -d "$EXPERIMENT_PATH" ]; then
          echo "⏭️  [GPU $GPU_ID] Oops! The results exist at ${EXPERIMENT_PATH} (so skip this job)"
        else
          echo "🚀  [GPU $GPU_ID] Start | Method: ${TRAINER} | Dataset: ${DATASET} | Model: ${MODEL_NAME} | Target: ${PREDICTION_ATTR} | Privacy: ${PROTECTED_ATTR} | Seed: ${SEED}"

          mkdir -p "$EXPERIMENT_PATH"

          CMD="python chap.py \
              --dataset $DATASET \
              --dataset_path $DATASET_PATH \
              --prediction_attribute $PREDICTION_ATTR \
              --protected_attribute $PROTECTED_ATTR \
              --model_name $MODEL_NAME \
              --split_layer $SPLIT_LAYER \
              --pruning_style $PRUNING_STYLE \
              --lambda_vib $LAMBDA_VIB \
              --batch_size $BATCH_SIZE \
              --total_epochs $TOTAL_EPOCHS \
              --learning_rate $LEARNING_RATE \
              --seed $SEED \
              --gpu_devices 0 \
              --experiment_path $EXPERIMENT_PATH \
              --adv_pred_method $ADV_METHOD"

          $CMD > "$LOG_FILE" 2>&1

        fi
        LAST_LOG=$(grep "Epoch $TOTAL_EPOCHS" "$LOG_FILE" | tail -n 1)
        TASK_ACC=$(echo "$LAST_LOG" | sed -n 's/.*Utility(Acc): \([0-9.]*\).*/\1/p')
        PRIV_ACC=$(echo "$LAST_LOG" | sed -n 's/.*Privacy(Acc): \([0-9.]*\).*/\1/p')

        TASK_ACC=${TASK_ACC:-"Error"}
        PRIV_ACC=${PRIV_ACC:-"Error"}

        echo "${MODEL_NAME} ${PREDICTION_ATTR} ${PROTECTED_ATTR} ${SEED} ${TASK_ACC} ${PRIV_ACC}" >> "$RESULT_FILE"
      done
    done
  ) &
done

GPU_LIST=$(IFS=', '; echo "${GPUS[*]}")
echo "⏳ All 3 seeds have been assigned to GPUs ${GPU_LIST} and are running simultaneously in the background..."
echo "⏳ You can use `tail -f ${RESULT_FILE}` to view the progress of the run in real time"

wait
echo "🎉 All experiments have completed! The results have been consolidated into: $RESULT_FILE"