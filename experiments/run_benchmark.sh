#!/bin/bash
set -e

# ==========================================
# FutureBench: Main Experiment Runner
# ==========================================

# Set Python Path to include 'src' directory
export PYTHONPATH=src

# Configuration
JUDGE_MODEL="openai/gpt-5.2-chat"
RUNS_DIR="runs"
DATA_DIR="data"

# List of Models (Configuration files must exist in src/tiebe_rag_conflict/configs/)
MODELS=(
    "gpt-5.1"
    "openai-gpt-4.1-mini"
    "openai-gpt-o4-mini-high"
    "claude-3.7-sonnet"
    "gemini-2.0-flash"
    "llama-4-maverick"
    "grok-4-fast"
    "deepseek-chat"
)

# Dataset Splits (Known vs Unknown Knowledge Cutoffs)
SPLITS=("known" "unknown")

echo "========================================================"
echo "🚀 Starting FutureBench Experiment Pipeline"
echo "   Models: ${#MODELS[@]}"
echo "   Splits: ${#SPLITS[@]}"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        
        # Define Input Data and Output Directory
        INPUT_FILE="${DATA_DIR}/en_us_${SPLIT}.jsonl"
        OUTPUT_DIR="${RUNS_DIR}/${MODEL}-${SPLIT}"
        CONFIG_FILE="src/tiebe_rag_conflict/configs/${MODEL}.yml"
        
        echo ">>> [Processing] Model: $MODEL | Split: $SPLIT"
        
        # 1. Run RAG Modes (Baseline & CoT)
        echo "    Step 1A: Running RAG evaluations (Baseline + CoT)..."
        python -m tiebe_rag_conflict.eval.run_eval \
            --items "$INPUT_FILE" \
            --config "$CONFIG_FILE" \
            --mode baseline cot \
            --ctx true false insuf \
            --output-dir "$OUTPUT_DIR"

        # 2. Run No-Context Baseline
        echo "    Step 1B: Running No-Context baseline..."
        python -m tiebe_rag_conflict.eval.run_eval \
            --items "$INPUT_FILE" \
            --config "$CONFIG_FILE" \
            --mode no_context \
            --ctx true \
            --output-dir "$OUTPUT_DIR"
            
        # 3. Run LLM Judge Tagging
        echo "    Step 2: Scoring results with LLM Judge ($JUDGE_MODEL)..."
        python experiments/scripts/tag_and_csi_llm.py \
            --runs-root "$RUNS_DIR" \
            --provider . \
            --model-dir "${MODEL}-${SPLIT}" \
            --judge-model "$JUDGE_MODEL" \
            --workers 5 \
            --out-suffix "_tagged.jsonl"
            
        echo "--------------------------------------------------------"
    done
done

echo "✅ ALL EXPERIMENTS COMPLETED!"
echo "   Results are saved in: $RUNS_DIR"
