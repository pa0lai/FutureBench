# TiEBe RAG Conflict Benchmark

This project evaluates Retrieval-Augmented Generation (RAG) systems on the **TiEBe** news event dataset with an emphasis on _knowledge conflict_ behaviour. For every question we generate three contextual variants (True / False / Insufficient) and probe models in two answering modes (Open-QA / Evidence-Only) to compute the **Conflict Stubbornness Index (CSI)** and accuracy.

## Project layout

```
tiebe_rag_conflict/
  configs/              # YAML configuration files
  data_prep/            # Scripts to build evaluation items
  eval/                 # Prompt templates, evaluation loop, metrics, slice reports
  retriever/            # Simple FAISS-based retriever for counterfactual contexts
  utils/                # Shared utilities (I/O, text helpers)
  requirements.txt
  README.md
```

## Installation

```bash
pip install -r tiebe_rag_conflict/requirements.txt
```

The retriever relies on OpenAI's embeddings API and FAISS (`faiss-cpu`).

## 1. Build triplets (True / False / Insufficient contexts)

```bash
python -m tiebe_rag_conflict.data_prep.build_triplets \
  --config tiebe_rag_conflict/configs/en_us_2024.yml \
  --out data/en_us_2024_triplets.jsonl
```

* Downloads the TiEBe split defined in the config
* Applies the model-specific cutoff filtering (`year_min`/`year_max` plus optional `start_month`/`end_month`)
* Generates False contexts by retrieving counter examples within the same region/year or by minimally perturbing the gold answer
* Produces Insufficient contexts by removing answer-bearing sentences
* Every record includes `answer_false`

You can expect ≥1k items for active regions/years (e.g., US 2024+).

## 2. Run evaluation

```bash
python -m tiebe_rag_conflict.eval.run_eval \
  --config tiebe_rag_conflict/configs/en_us_2024.yml \
  --items data/en_us_2024_triplets.jsonl
```

The config specifies the model endpoint and decoding parameters. By default the evaluator reads `eval.modes` and `eval.ctx_types` from the config (e.g., Open-QA × True/False/Insufficient and Evidence-Only × True/False/Insufficient) and iterates through every combination, writing metrics/predictions for each run. You can override the grid from the command line, for example:

```bash
python -m tiebe_rag_conflict.eval.run_eval \
  --config tiebe_rag_conflict/configs/en_us_2024.yml \
  --items data/en_us_2024_triplets.jsonl \
  --mode openqa \
  --ctx false insuf
```

The script formats prompts for the selected answering mode, calls `ask_model`, and stores:

* `preds_<mode>_<ctx>.jsonl` – per-item predictions with metadata
* `metrics_<mode>_<ctx>.json` – Stick/Follow/Reject counts, CSI, and accuracy

Repeat for any combination of mode/context listed in `config['eval']`.

## 3. Slice-level analysis

```bash
python -m tiebe_rag_conflict.eval.slice_report \
  --run_dir runs/en_us_2024_llama31/ \
  --bucket 6
```

The slice report aggregates CSI/accuracy over `(mode, ctx_type, region, language, year_bucket)` and writes a CSV file (e.g., `slice_report_6m.csv`). Buckets group events into 6- or 12-month windows.

## Configuration tips

* **Language split** – set `split: "english"` (default) or the region's native language split for cross-lingual comparisons.
* **Year/month cutoff** – adjust `year_min`/`year_max` per model and optionally tighten the bounds with `start_month`/`end_month`. Months apply to the respective boundary years (e.g., `year_min: 2024` + `start_month: 1`, `year_max: 2025` + `end_month: 4` keeps Jan 2024 through Apr 2025). LLaMA-3.0 typically covers Apr 2023–Apr 2025 (`year_min: 2023`, `start_month: 4`, `year_max: 2025`, `end_month: 4`), while LLaMA-3.1 uses Jan 2024–Apr 2025.

* **Retriever** – configure `retriever.encoder` to one of OpenAI's embedding models (e.g., `text-embedding-3-small`) and supply an API key via `retriever.api_key`, `retriever.api_key_file`, or `retriever.api_key_env`. A more advanced retriever (e.g., Elasticsearch) can replace `SimpleRetriever`.
* **API keys** – export `OPENAI_API_KEY`/`OPENROUTER_API_KEY`, or point `retriever.api_key_file`/`model.api_key_file` to a plaintext secret (see `SECRETS.md`).
* **Model interface** – `ask_model` performs a JSON POST to `config['model']['endpoint']` with `prompt`, `max_tokens`, and `temperature`. Adapt it as needed for your serving stack.
* **OpenRouter** – alternatively set `model.provider: "openrouter"` and specify the OpenRouter `model` id, optional `system_prompt`, and `response_field` (defaults to `answer`/`output`/`text`). The evaluator now calls OpenRouter through the official `openai` Python SDK (`OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)`), so you can pass along optional `extra_headers` (e.g., `HTTP-Referer`, `X-Title`) and `extra_body` overrides (e.g., `top_p`). Provide the API key via `OPENROUTER_API_KEY` (or `model.api_key` / `model.api_key_env`), and reuse the usual `temperature`, `max_tokens`, `timeout`, and `max_retries` knobs.


## Extending the project

* Add alternative retrievers (ES, BM25) under `retriever/`
* Log richer decision traces (model outputs, rejection reasons)
* Explore multilingual experiments by building triplets for both English and native splits of the same region
* Simulate noise injection by mixing False contexts into retrieval top-k and plotting CSI/Acc vs. corruption ratio

