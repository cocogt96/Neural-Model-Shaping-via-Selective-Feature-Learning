#!/bin/bash

# Ensure execution within the reconstruction directory
cd ../public/reconstruction/ || exit

# ==========================================
# 1. Basic Parameter Setup
# ==========================================
#GPUS=(0 1 2 3 4)                 # Use these 5 GPUs
GPUS=(0)
MAX_JOBS_PER_GPU=2             # Reconstruction consumes VRAM, recommend 2 processes per GPU

DATASET="fairface"
PREDICTION_ATTR="gender"
PROTECTED_ATTR="race"
SEED="1"

# Unified hyperparameters for reconstruction training
GOAL_DATA_DIR="../../../dataset/fairface/val/"
TASK="data"
BATCH_SIZE=32
NUM_EPOCHS=30                  # 30~50 epochs are sufficient for decoder convergence
LOSS_FN="l1"

# Methods and Models to iterate over (Restricted to resnet18)
MODELS=("resnet18")
METHODS=("DISCO" "ADV" "MAXENTROPY" "NOISE" "K_CLUB" "K_CLUB_RECON")
# ==========================================
# 2. Build Exclusive "Task Queue" for Each GPU
# ==========================================
NUM_GPUS=${#GPUS[@]}
declare -a GPU_TASK_LISTS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_TASK_LISTS[$i]=""
done

echo "🔍 Scanning output directories of each method, checking for client_model.pth and assigning tasks..."
task_counter=0

for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do

        # Core: Dynamically construct the path of the completed main task directory based on the method
        if [ "$METHOD" == "DISCO" ]; then
            SRC_PATH="../../../output/${DATASET}/DISCO/server_${MODEL}_adversarial_ml-leaks/${PREDICTION_ATTR}_${PROTECTED_ATTR}/seed${SEED}"
        elif [ "$METHOD" == "ADV" ]; then
            SRC_PATH="../../../output/${DATASET}/ADV/server_${MODEL}_adversarial_ml-leaks/${PREDICTION_ATTR}_${PROTECTED_ATTR}/seed${SEED}"
        elif [ "$METHOD" == "MAXENTROPY" ]; then
            SRC_PATH="../../../output/${DATASET}/MAXENTROPY/server_${MODEL}_adversarial_ml-leaks/${PREDICTION_ATTR}_${PROTECTED_ATTR}/seed${SEED}"
        elif [ "$METHOD" == "NOISE" ]; then
            SRC_PATH="../../../output/${DATASET}/NOISE/server_${MODEL}_adversarial_ml-leaks/${PREDICTION_ATTR}_${PROTECTED_ATTR}/seed${SEED}"
        elif [ "$METHOD" == "K_CLUB" ]; then
            # Compatible with K_CLUB's unique path naming format
            SRC_PATH="../../../output/${DATASET}/K_CLUB_EXP/${MODEL}_k_club(500-3.0)/${PREDICTION_ATTR}_${PROTECTED_ATTR}_seed${SEED}"
        elif [ "$METHOD" == "K_CLUB_RECON" ]; then
            SRC_PATH="../../../output/${DATASET}/K_CLUB_RECON/${MODEL}_k_club(500-3.0)/${PREDICTION_ATTR}_data_seed${SEED}"
        fi

        # Must verify the previous main experiment succeeded and client_model.pt exists before initiating reconstruction
        if [ ! -f "${SRC_PATH}/saved_models/client_model.pt" ]; then
            echo "⏭️  [Skip] Cannot find the original model, please confirm the main experiment is completed: ${SRC_PATH}/saved_models/client_model.pt"
            continue
        fi

        # Construct the architecture name for saving reconstruction results (e.g., recon_DISCO_resnet18)
        ARCH_TAG="recon_${METHOD}_${MODEL}"

        # Pack parameters and queue them (separated by |)
        gpu_idx=$(( task_counter % NUM_GPUS ))
        GPU_TASK_LISTS[$gpu_idx]+="${ARCH_TAG}|${SRC_PATH}|${MODEL} "
        ((task_counter++))

    done
done

echo "✅ Scanning complete. Assigned a total of $task_counter reconstruction tasks."

if [ "$task_counter" -eq 0 ]; then
    exit 0
fi

# ==========================================
# 3. Core Execution Function
# ==========================================
run_cmd() {
    local arch_tag=$1; local src_path=$2; local model_name=$3; local gpu=$4
    local log_file="./output/reconstruction/${arch_tag}/recon.log"

    mkdir -p "./output/reconstruction/${arch_tag}"
    echo "▶️ GPU $gpu | Method: $arch_tag | Initiating reconstruction attack..."

    # Launch reconstruction script
    # ⚠️ Ensure --experiment_path in main.py correctly receives $src_path
    CUDA_VISIBLE_DEVICES=$gpu python main.py \
        --architecture "$arch_tag" \
        --experiment_path "$src_path" \
        --tensor_data_dir "${src_path}challenge" \
        --split_layer 4 \
        --goal_data_dir "$GOAL_DATA_DIR" \
        --task "$TASK" \
        --img_fmt "jpg" \
        --tns_fmt "pt" \
        --train_split 0.95 \
        --loss_fn "$LOSS_FN" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --train_output_freq "$NUM_EPOCHS" \
        --test_output_freq "$NUM_EPOCHS" \
        --gpu_id 0 > "$log_file" 2>&1

    echo "✅ [Completed] $arch_tag reconstruction finished!"
}

# ==========================================
# 4. Start Exclusive Daemon Processes for Each GPU
# ==========================================
for g_idx in "${!GPUS[@]}"; do
    (
        gpu_id=${GPUS[$g_idx]}
        job_count=0

        for task_str in ${GPU_TASK_LISTS[$g_idx]}; do

            # Parse parameters using "|"
            IFS='|' read -r arch_tag src_path model_name <<< "$task_str"

            # Throw into background execution
            run_cmd "$arch_tag" "$src_path" "$model_name" "$gpu_id" &
            ((job_count++))

            # Strictly limit the maximum concurrency per card
            if [ "$job_count" -ge "$MAX_JOBS_PER_GPU" ]; then
                wait -n
                ((job_count--))
            fi
        done

        wait
    ) &
done

wait
echo "🎉 All image reconstruction tasks for all methods have been completed! Please check reconstruction/output/ for comparison images."