#!/bin/bash

# ----------------------------- #
#     Generation + Evaluation   #
# ----------------------------- #

# Configuration
GEN_SCRIPT="../code/inference_openai.py"
EVAL_SCRIPT="../code/evaluation_.py"
MODEL_NAME=""  # Set your model name here (e.g., "gpt-4o", "gpt-4.1-mini-2025-04-14")
SETTING="english" # Set your setting here (e.g., "english", "multilingual")
SUBSETS=("casa" "safeworld")
MAX_ATTEMPTS=20

# 🔒 Ensure MODEL_NAME is provided
if [ -z "$MODEL_NAME" ]; then
    echo "❗ ERROR: MODEL_NAME is not set. Please set the MODEL_NAME variable before running the script."
    exit 1
fi

# 📁 Create logs directory
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# ----------------------------- #
#        Generation Phase       #
# ----------------------------- #
for ROUND in 1 2 3; do
    for SUBSET in "${SUBSETS[@]}"; do
        START_INDEX=1
        PROGRESS_LOG="${LOG_DIR}/progress_gen_${SETTING}_${SUBSET}_${MODEL_NAME}_r${ROUND}.log"
        attempt=1

        if [ -f "$PROGRESS_LOG" ]; then
            LAST_INDEX=$(tail -n 1 "$PROGRESS_LOG")
            if [[ $LAST_INDEX =~ ^[0-9]+$ ]]; then
                START_INDEX=$((LAST_INDEX + 1))
                echo "🔄 Resuming generation from index $START_INDEX for subset '$SUBSET' (round $ROUND)"
            fi
        fi

        while [ $attempt -le $MAX_ATTEMPTS ]; do
            echo "🚀 Generation | Subset: '$SUBSET' | Round: $ROUND | Attempt: $attempt"

            SCRIPT="$GEN_SCRIPT_1"  # Standard version

            python3 "$SCRIPT" \
                --setting "$SETTING" \
                --subset "$SUBSET" \
                --model_name "$MODEL_NAME" \
                --round "$ROUND" \
                --start_index "$START_INDEX" \
                --progress_log "$PROGRESS_LOG"

            EXIT_CODE=$?

            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Generation Success | Subset: '$SUBSET' | Round: $ROUND"
                break
            else
                echo "❌ Generation Failure (exit code $EXIT_CODE). Retrying..."
                attempt=$((attempt + 1))
                sleep 2
                if [ -f "$PROGRESS_LOG" ]; then
                    LAST_INDEX=$(tail -n 1 "$PROGRESS_LOG")
                    if [[ $LAST_INDEX =~ ^[0-9]+$ ]]; then
                        START_INDEX=$((LAST_INDEX + 1))
                        echo "🔁 Updated start index to $START_INDEX after failure"
                    fi
                fi
            fi
        done

        if [ $attempt -gt $MAX_ATTEMPTS ]; then
            echo "🚨 Generation failed for subset '$SUBSET' (round $ROUND). Skipping..."
        fi
    done
done

# ----------------------------- #
#        Evaluation Phase       #
# ----------------------------- #
for ROUND in 1 2 3; do
    for SUBSET in "${SUBSETS[@]}"; do
        START_INDEX=1
        PROGRESS_LOG="${LOG_DIR}/progress_eval_${SETTING}_${SUBSET}_${MODEL_NAME}_r${ROUND}.log"
        attempt=1

        if [ -f "$PROGRESS_LOG" ]; then
            LAST_INDEX=$(tail -n 1 "$PROGRESS_LOG")
            if [[ $LAST_INDEX =~ ^[0-9]+$ ]]; then
                START_INDEX=$((LAST_INDEX + 1))
                echo "🔄 Resuming evaluation from index $START_INDEX for subset '$SUBSET' (round $ROUND)"
            fi
        fi

        while [ $attempt -le $MAX_ATTEMPTS ]; do
            echo "🔎 Evaluation | Subset: '$SUBSET' | Round: $ROUND | Attempt: $attempt"

            python3 "$EVAL_SCRIPT" \
                --setting "$SETTING" \
                --subset "$SUBSET" \
                --model_name "$MODEL_NAME" \
                --round "$ROUND" \
                --start_index "$START_INDEX" \
                --progress_log "$PROGRESS_LOG"

            EXIT_CODE=$?

            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Evaluation Success | Subset: '$SUBSET' | Round: $ROUND"
                break
            else
                echo "❌ Evaluation Failure (exit code $EXIT_CODE). Retrying..."
                attempt=$((attempt + 1))
                sleep 2
                if [ -f "$PROGRESS_LOG" ]; then
                    LAST_INDEX=$(tail -n 1 "$PROGRESS_LOG")
                    if [[ $LAST_INDEX =~ ^[0-9]+$ ]]; then
                        START_INDEX=$((LAST_INDEX + 1))
                        echo "🔁 Updated start index to $START_INDEX after failure"
                    fi
                fi
            fi
        done

        if [ $attempt -gt $MAX_ATTEMPTS ]; then
            echo "🚨 Evaluation failed for subset '$SUBSET' (round $ROUND). Skipping..."
        fi
    done
done
