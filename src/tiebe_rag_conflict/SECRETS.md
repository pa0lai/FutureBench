# API key storage

This project talks to hosted services (OpenAI for embeddings, OpenRouter for LLMs). To keep secrets out of the repository, place them in plaintext files that are referenced by the configs, or export them as environment variables before running the scripts.

## OpenAI embeddings

1. Create a file such as `secrets/openai_api_key.txt` that contains only your OpenAI API key.
2. Update the config to point at the file:

   ```yaml
   retriever:
     provider: "openai"
     encoder: "text-embedding-3-small"
     api_key_file: "secrets/openai_api_key.txt"
   ```

   Alternatively export `OPENAI_API_KEY` (or set `retriever.api_key_env`) before running `python -m tiebe_rag_conflict.data_prep.build_triplets ...`.

## OpenRouter

1. Store the key in a file such as `secrets/openrouter_api_key.txt`.
2. Reference it from the model config:

   ```yaml
   model:
     provider: "openrouter"
     model: "meta/llama-3.1"
     api_key_file: "secrets/openrouter_api_key.txt"
   ```

   You can also export `OPENROUTER_API_KEY` or specify `model.api_key_env`.

Do **not** commit the plaintext secret files to version control.
