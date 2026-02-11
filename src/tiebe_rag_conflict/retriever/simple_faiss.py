from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tiebe_rag_conflict.utils.text import normalize_text


DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def _resolve_openai_api_key(
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_key_file: str | None = None,
) -> str:
    """Resolve the OpenAI API key from direct value, file, or environment."""

    if api_key:
        return str(api_key).strip()
    if api_key_file:
        path = Path(api_key_file).expanduser()
        if not path.exists():
            raise RuntimeError(f"OpenAI API key file not found: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"OpenAI API key file {path} is empty")
        return key
    env_name = api_key_env or DEFAULT_OPENAI_API_KEY_ENV
    key = os.environ.get(env_name)
    if key:
        return key.strip()
    raise RuntimeError(
        "OpenAI API key is missing. Provide `retriever.api_key`, set "
        f"`retriever.api_key_file`, or export {env_name}."
    )


class _OpenAIEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_key_file: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if client is None:
            try:  # pragma: no cover - exercised in integration usage
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - handled via requirements
                raise RuntimeError(
                    "The `openai` package is required for OpenAI embeddings. "
                    "Install it via `pip install openai`."
                ) from exc
            key = _resolve_openai_api_key(api_key, api_key_env, api_key_file)
            self.client = OpenAI(api_key=key)
        else:
            self.client = client
        self._last_dim: int | None = None

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        del show_progress_bar  # OpenAI API does not expose progress hooks
        if not sentences:
            if convert_to_numpy:
                dim = self._last_dim or 0
                return np.zeros((0, dim), dtype=np.float32)
            return []
        bs = max(1, int(batch_size or 1))
        vectors: list[np.ndarray] = []
        for start in range(0, len(sentences), bs):
            chunk = sentences[start : start + bs]
            response = self.client.embeddings.create(model=self.model_name, input=list(chunk))
            for row in response.data:
                vec = np.asarray(row.embedding, dtype=np.float32)
                vectors.append(vec)
        if not vectors:
            if convert_to_numpy:
                dim = self._last_dim or 0
                return np.zeros((0, dim), dtype=np.float32)
            return []
        mat = np.stack(vectors)
        self._last_dim = mat.shape[1]
        if normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        if convert_to_numpy:
            return mat
        return mat.tolist()


class SimpleRetriever:
    """A lightweight FAISS-backed semantic retriever for negative sampling."""

    def __init__(
        self,
        sentences: Sequence[str],
        metas: Sequence[dict] | None = None,
        answers: Sequence[str] | None = None,
        encoder_name: str = "text-embedding-3-small",
        provider: str | None = None,
        batch_size: int = 64,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_key_file: str | None = None,
        client: Any | None = None,
    ) -> None:
        if metas is None:
            metas = [{} for _ in sentences]
        if answers is None:
            answers = ["" for _ in sentences]
        if not (len(sentences) == len(metas) == len(answers)):
            raise ValueError("sentences, metas, and answers must have the same length")
        self.sentences = list(sentences)
        self.metas = list(metas)
        self.answers = list(answers)
        provider_name = (provider or "openai").lower()
        if provider_name != "openai":
            raise ValueError(f"Unsupported retriever provider: {provider}")
        self.encoder = _OpenAIEncoder(
            encoder_name,
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            client=client,
        )
        self.embeddings = self.encoder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.group_index: dict[tuple[str | None, int | None, int | None], list[int]] = defaultdict(list)
        for idx, meta in enumerate(self.metas):
            region = meta.get("region")
            year = meta.get("year")
            month = meta.get("month")
            key = (region, year, month)
            self.group_index[key].append(idx)
            if month is not None:
                self.group_index[(region, year, None)].append(idx)

    def nearest_with_diff_answer(
        self,
        q: str,
        gold_answer: str,
        meta: dict,
        topk: int = 50,
    ) -> tuple[str, str] | None:
        region = meta.get("region")
        year = meta.get("year")
        month = meta.get("month")
        keys: list[tuple[str | None, int | None, int | None]]
        if month is None:
            keys = [(region, year, None)]
        else:
            keys = [(region, year, month), (region, year, None)]

        candidate_indices: list[int] = []
        for key in keys:
            candidate_indices = self.group_index.get(key, [])
            if candidate_indices:
                break
        if not candidate_indices:
            return None

        query_emb = self.encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        gold_norm = normalize_text(gold_answer)
        exclude_doc_idx = meta.get("doc_idx")
        exclude_idx = meta.get("exclude_idx") if exclude_doc_idx is None else None
        for key in keys:
            candidate_indices = self.group_index.get(key, [])
            if not candidate_indices:
                continue
            subset_embs = self.embeddings[candidate_indices]
            sims = subset_embs @ query_emb
            ranked = np.argsort(-sims)
            for rank in ranked[:topk]:
                idx = candidate_indices[int(rank)]
                candidate_meta = self.metas[idx] if idx < len(self.metas) else {}
                candidate_doc_idx = candidate_meta.get("doc_idx")
                if exclude_doc_idx is not None and candidate_doc_idx == exclude_doc_idx:
                    continue
                if exclude_idx is not None and idx == exclude_idx:
                    continue
                candidate_answer = self.answers[idx]
                if gold_norm and normalize_text(candidate_answer) == gold_norm:
                    continue
                return self.sentences[idx], candidate_answer
        return None

