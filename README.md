
# FutureBench: Evaluating RAG Robustness

This repository contains the code and data necessary to reproduce the experiments described in the paper "FutureBench: Evaluating RAG Accuracy and Robustness Beyond Knowledge Cutoffs".

## Repository Structure

- `src/`: Core evaluation logic.
- `experiments/`: Scripts to run the benchmark.
- `data/`: Contains the primary datasets (`World_Dec2024_to_Present.json`).
- `data_pipeline/`: Scripts used for data generation (`data_generation_pipeline.py`).
- `prompts/`: Detailed documentation of all prompts used (Data Gen, Eval, Judge).
- `analysis/`: Tools for generating plots and summarizing results.

## Setup

1. **Install Dependencies**:
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Create a `.env` file in the root directory (based on `.env.example`) and add your API keys:
   ```bash
   OPENAI_API_KEY=sk-...
   OPENROUTER_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-...
   GOOGLE_API_KEY=AIza...
   ```

## Running Experiments

To reproduce the main results (Baseline vs. CoT across all models):

```bash
bash experiments/run_benchmark.sh
```

This script will:
1. Load model configurations from `src/tiebe_rag_conflict/configs`.
2. Execute the evaluation pipeline for each model.
3. Save raw predictions to `runs/`.

## Analyzing Results

After running experiments, generate summary tables and plots:

1. **Summarize Metrics**:
   ```bash
   python analysis/summarize_results.py
   ```
   This generates `experiment_summary.csv`.

2. **Generate Plots**:
   ```bash
   python analysis/generate_plots.py
   ```
   This saves visualization figures to the root directory.

## Data Structure

The dataset uses a JSON format where each entry contains:
- `question`: The query text.
- `gold_answer`: The ground truth.
- `triplets`: A set of contexts ($C_T$, $C_F$, $C_I$) for testing robustness.

## License

This project is licensed under the MIT License.
