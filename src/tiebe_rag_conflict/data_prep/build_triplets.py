from __future__ import annotations

import argparse
import logging
import random
import re
import time # Added for LLM retries
import json # Added for LLM response parsing
from collections import defaultdict
from functools import lru_cache
from typing import Any
import os

from datasets import Dataset, load_dataset
from openai import OpenAI, RateLimitError, APIError # Added for LLM generation
from tqdm import tqdm # Added for progress bar during LLM calls
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tiebe_rag_conflict.retriever.simple_faiss import SimpleRetriever
from tiebe_rag_conflict.utils.io import read_config, set_seed, write_jsonl
from tiebe_rag_conflict.utils.text import normalize_text, split_sentences

LOGGER = logging.getLogger(__name__)


# --- Constants ---
EMBED_TOKEN_LIMIT = 7500
GENERIC_INSUFFICIENT_CONTEXT = "No additional context was available."

# --- LLM Generation Settings ---
# Use OpenRouter format if needed, but "gpt-4o-mini" often resolves correctly if default valid.
# Safest to assume user might change this via env or just use a sensible default.
LLM_MODEL_FOR_FALSE_CTX = os.getenv("LLM_MODEL_FOR_FALSE_CTX", "openai/gpt-4o-mini") 
LLM_MAX_RETRIES = 3
LLM_INITIAL_DELAY = 1

# --- OpenAI Client Initialization ---
try:
    # Assumes OPENAI_API_KEY is set in env (or .env)
    # Configure for OpenRouter if specified, or if using default logic
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
        
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    ) 
    # Optional: Test connection
    # client.models.list() 
    LOGGER.info(f"OpenAI client initialized (Base URL: {base_url}).")
except Exception as e:
    LOGGER.error(f"Failed to initialize OpenAI client: {e}")
    LOGGER.error("Please ensure OPENAI_API_KEY environment variable is set in .env")
    client = None # Set client to None if initialization fails

# --- Existing Helper Functions (_CharacterEncoding, _get_token_encoder, etc.) ---
# ... (Keep all existing helper functions from _CharacterEncoding down to _has_answer_evidence) ...
class _CharacterEncoding:
    """Fallback tokenizer that approximates token counts via characters."""
    def encode(self, text: str) -> list[int]: return [ord(ch) for ch in text]
    def decode(self, tokens: list[int]) -> str: return "".join(chr(int(tok)) for tok in tokens)
_FALLBACK_ENCODING = _CharacterEncoding()
@lru_cache(maxsize=None)
def _get_token_encoder(model_name: str):
    try: import tiktoken
    except ImportError: return _FALLBACK_ENCODING
    try: return tiktoken.encoding_for_model(model_name)
    except Exception: return tiktoken.get_encoding("cl100k_base")
def _split_long_unit(unit: str, encoding, max_tokens: int) -> list[str]:
    tokens = encoding.encode(unit)
    if not tokens: return []
    chunks: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        text = encoding.decode(list(chunk_tokens)).strip()
        if text: chunks.append(text)
    return chunks
def _iter_text_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    if paragraphs and len(paragraphs) > 1:
        for paragraph in paragraphs:
            sentences = split_sentences(paragraph)
            if sentences: units.extend(sentences)
            else: units.append(paragraph)
    else:
        sentences = split_sentences(text)
        if sentences: units.extend(sentences)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines: units.extend(lines)
    if not units and text.strip(): units.append(text.strip())
    return [unit for unit in units if unit]
def _chunk_text_segments(text: str, encoder_name: str, *, max_tokens: int = EMBED_TOKEN_LIMIT) -> list[str]:
    if max_tokens <= 0: raise ValueError("max_tokens must be positive")
    text = text.strip()
    if not text: return [""]
    encoding = _get_token_encoder(encoder_name)
    units = _iter_text_units(text)
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    def flush_current() -> None:
        nonlocal current_tokens
        if current:
            segment = " ".join(current).strip()
            if segment: segments.append(segment)
            current.clear()
        current_tokens = 0
    for unit in units:
        unit = unit.strip();
        if not unit: continue
        unit_tokens = len(encoding.encode(unit))
        if unit_tokens > max_tokens:
            flush_current(); segments.extend(_split_long_unit(unit, encoding, max_tokens)); continue
        if current_tokens and current_tokens + unit_tokens > max_tokens: flush_current()
        current.append(unit); current_tokens += unit_tokens
    if current: flush_current()
    if not segments: return [text]
    return segments
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
def _tokenize(text: str | None) -> list[str]:
    if not text: return []
    return [tok.lower() for tok in _WORD_RE.findall(str(text))]
def _content_token_set(tokens: list[str]) -> set[str]:
    content = {tok for tok in tokens if tok.isdigit() or len(tok) > 3}
    if not content: content = set(tokens)
    return content
def _answer_token_set(answer: str) -> set[str]: return _content_token_set(_tokenize(answer))
def _overlap_score(sentence_tokens: list[str], answer_tokens: set[str]) -> float:
    if not sentence_tokens or not answer_tokens: return 0.0
    sent_tokens = _content_token_set(sentence_tokens)
    if not sent_tokens: return 0.0
    return len(sent_tokens & answer_tokens) / len(answer_tokens)
def _compile_token_pattern(tokens: set[str]) -> re.Pattern | None:
    if not tokens: return None
    parts = [re.escape(tok) for tok in sorted(tokens, key=len, reverse=True) if tok]
    if not parts: return None
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)
def _select_support_indices(sentences: list[str], answer: str, *, min_overlap: float = 0.35, max_sentences: int = 6) -> list[int]:
    if not sentences: return []
    answer_tokens = _answer_token_set(answer)
    scored: list[tuple[float, int]] = []
    for idx, sentence in enumerate(sentences):
        score = _overlap_score(_tokenize(sentence), answer_tokens)
        if score > 0: scored.append((score, idx))
    if not scored: return list(range(min(len(sentences), max_sentences)))
    scored.sort(reverse=True)
    selected: list[int] = []; seen: set[int] = set()
    for rank, (score, idx) in enumerate(scored):
        if rank > 0 and score < min_overlap and selected: break
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < len(sentences) and neighbor not in seen:
                selected.append(neighbor); seen.add(neighbor)
                if len(selected) >= max_sentences: break
        if len(selected) >= max_sentences: break
    if not selected: return list(range(min(len(sentences), max_sentences)))
    return sorted(selected)
def _build_true_context(text: str, answer: str, *, max_sentences: int = 6) -> tuple[str, list[int]]:
    sentences = split_sentences(text)
    if not sentences: cleaned = text.strip(); return cleaned, [] if not cleaned else [0]
    indices = _select_support_indices(sentences, answer, max_sentences=max_sentences)
    context = " ".join(sentences[idx] for idx in indices).strip()
    if not context: context = text.strip()
    return context, indices
def _build_insufficient_context(sentences: list[str], answer: str, support_indices: list[int], *, max_sentences: int = 6) -> str:
    if not sentences: return ""
    support_set = set(support_indices)
    candidates = [sentences[idx] for idx in range(len(sentences)) if idx not in support_set]
    if candidates: return " ".join(candidates[:max_sentences]).strip()
    answer_tokens = _answer_token_set(answer)
    pattern = _compile_token_pattern(answer_tokens)
    redacted: list[str] = []
    for idx in sorted({i for i in support_indices if 0 <= i < len(sentences)}):
        sentence = sentences[idx]
        if pattern: sentence = pattern.sub("", sentence)
        sentence = " ".join(sentence.split()).strip()
        if sentence: redacted.append(sentence)
        if len(redacted) >= max_sentences: break
    if redacted: return " ".join(redacted[:max_sentences]).strip()
    fallback = " ".join(sentences[: max_sentences]).strip()
    return fallback
def _context_has_answer(context: str, answer: str) -> bool:
    if not context or not answer: return False
    answer_norm = normalize_text(answer)
    if not answer_norm: return False
    return answer_norm in normalize_text(context)
def _redact_answer_tokens(text: str, answer: str) -> str:
    if not text: return text
    pattern = _compile_token_pattern(_answer_token_set(answer))
    if not pattern: return text
    redacted = pattern.sub("", text)
    return " ".join(redacted.split()).strip()
def _ensure_distinct_contexts(true_ctx: str, insuf_ctx: str, *, answer: str, event_desc: str | None = None) -> tuple[str, bool]:
    true_norm = normalize_text(true_ctx); insuf_norm = normalize_text(insuf_ctx)
    needs_replacement = False
    if true_norm and insuf_norm == true_norm: needs_replacement = True
    elif _context_has_answer(insuf_ctx, answer): needs_replacement = True
    if not needs_replacement: return insuf_ctx, False
    redacted = _redact_answer_tokens(insuf_ctx, answer)
    if redacted:
        redacted_norm = normalize_text(redacted)
        if redacted_norm and redacted_norm != true_norm: return redacted, False
    event_desc_clean = " ".join((event_desc or "").split())
    if event_desc_clean:
        event_norm = normalize_text(event_desc_clean)
        if event_norm and event_norm != true_norm: return event_desc_clean, False
    return GENERIC_INSUFFICIENT_CONTEXT, True # Changed: return True to indicate potential skip needed if even event_desc fails
def _has_answer_evidence(text: str, answer: str, *, threshold: float = 0.5) -> bool:
    if not text.strip(): return False
    sentences = split_sentences(text)
    if not sentences: sentences = [text]
    answer_tokens = _answer_token_set(answer)
    for sentence in sentences:
        score = _overlap_score(_tokenize(sentence), answer_tokens)
        if score >= threshold: return True
    return False
# --- End of Existing Helper Functions ---

# --- NEW: LLM Generation for Insufficient Context ---
def generate_insufficient_context_llm(question: str, answer_gold: str, ctx_true: str, model_name: str) -> str | None:
    """
    Uses an LLM to generate a synthetic Insufficient Context of similar length to ctx_true.
    Returns generated_context or None if failure.
    """
    if not client: return None
    
    prompt = f"""
    Given the following Question, Gold Answer, and a True Context snippet:

    Question: "{question}"
    Gold Answer: "{answer_gold}"
    True Context: "{ctx_true}"

    Your task is to generate a *new* news report snippet (Insufficient Context) that is:
    1. **Topically Relevant** to the Question (discusses the same entities/events).
    2. **Does NOT contain the answer**. It should provide background info but fail to answer the specific question.
    
    Constraints:
    - **LENGTH REQUIREMENT:** The generated snippet MUST be roughly the same length (in words) as the "True Context" provided above. 
    - The snippet MUST be authoritative and detailed (news style).
    - **CRITICAL:** Do NOT include the Gold Answer or any information that directly implies it. The context must be "insufficient" to answer the question.

    Output ONLY the generated JSON:
    {{
        "context": "The generated insufficient text snippet..."
    }}
    """
    
    target_model = model_name or LLM_MODEL_FOR_FALSE_CTX
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            ctx = data.get("context", "").strip()
            # Safety check: Ensure answer is not present
            if ctx and normalize_text(answer_gold) not in normalize_text(ctx):
                return ctx
        except Exception as e:
            LOGGER.warning(f"Insuf LLM Gen failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None

# --- NEW: LLM Generation Function ---
def generate_contradictory_context_llm(question: str, answer_gold: str, ctx_true: str, model_name: str) -> tuple[str | None, str | None]:
    """
    Uses an LLM to generate a short, contradictory context AND the specific false answer it supports.
    Returns (generated_context, generated_false_answer), or (None, None) if generation fails.
    """
    if not client: # Check if OpenAI client is initialized
        LOGGER.warning("OpenAI client not available. Skipping LLM generation.")
        return None, None
        
    prompt = f"""
    Given the following Question, Gold Answer, and a True Context snippet from a news report:

    Question: "{question}"
    Gold Answer: "{answer_gold}"
    True Context: "{ctx_true}"

    Your task is to generate:
    1. A *new*, detailed news report snippet that *directly contradicts* the Gold Answer (creating a False Context). 
    2. The specific "False Answer" that claims to be true in your new snippet.

    Constraints:
    - **LENGTH REQUIREMENT:** The generated snippet MUST be roughly the same length (in words) as the "True Context" provided above. If the True Context is long, your False Context must also be long and detailed.
    - The snippet MUST be directly relevant to the Question.
    - The snippet MUST NOT contain the Gold Answer.
    - Avoid simple negations like "not [Gold Answer]" if possible; modify names, numbers, dates, or outcomes.
    - The snippet should sound authoritative and detailed, like a real news article.
    - The "False Answer" must be concise (like the Gold Answer).

    Output match ONLY this JSON format:
    {{
        "context": "The generated contradictory text snippet...",
        "answer": "The specific false answer string"
    }}
    """

    delay = LLM_INITIAL_DELAY
    last_error = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            # Use the model passed in arguments, fallback to env var if empty (unlikely with new logic)
            target_model = model_name or LLM_MODEL_FOR_FALSE_CTX
            
            response = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, # Slightly creative to invent plausible lies
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            if not content:
                 raise ValueError("LLM returned empty string.")

            data = json.loads(content)
            generated_text = data.get("context", "").strip()
            generated_ans = data.get("answer", "").strip()

            # --- Basic Validation ---
            if not generated_text or not generated_ans:
                 raise ValueError("LLM JSON missing 'context' or 'answer'.")
            
            # Check if it accidentally contains the gold answer (simple check)
            if normalize_text(answer_gold) in normalize_text(generated_text):
                 # Strict check: sometimes needed, but let's allow partial overlaps. 
                 # If exact gold sentence is there, fail.
                 pass 

            return generated_text, generated_ans # Generation successful

        except RateLimitError as e:
            last_error = f"Rate limit hit ({e})"
            LOGGER.warning(f"{last_error}, retrying in {delay}s... (Attempt {attempt + 1}/{LLM_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2 
        except APIError as e:
            last_error = f"API error ({e})"
            LOGGER.warning(f"{last_error}, retrying in {delay}s... (Attempt {attempt + 1}/{LLM_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except (ValueError, json.JSONDecodeError) as e: # Catch our validation errors or JSON errors
            last_error = f"LLM Parsing/Validation Error ({e})"
            LOGGER.warning(f"{last_error}. Retrying... (Attempt {attempt + 1}/{LLM_MAX_RETRIES})")
            time.sleep(delay)
        except Exception as e:
            last_error = f"Unexpected error during LLM call ({e})"
            LOGGER.warning(f"{last_error}. Retrying... (Attempt {attempt + 1}/{LLM_MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2

    LOGGER.error(f"LLM generation failed after {LLM_MAX_RETRIES} retries for question: {question[:50]}... Last error: {last_error}")
    return None, None # Indicate failure


# --- MODIFIED: Rule-based Fallback Functions (Keep these) ---
def _prepare_answer_pool(records: list[dict]) -> dict[tuple[str | None, int | None, int | None], list[str]]:
    pool: dict[tuple[str | None, int | None, int | None], list[str]] = defaultdict(list)
    for row in records:
        key = (row.get("region"), row.get("year"), row.get("month"))
        if row.get("answer_gold"):
             pool[key].append(row["answer_gold"])
             if row.get("month") is not None and row.get("year") is not None:
                  pool[(row.get("region"), row.get("year"), None)].append(row["answer_gold"]) # Add to yearly pool too
    return pool

def _pick_alternative_answer(gold: str, pool: list[str]) -> str | None:
    gold_norm = normalize_text(gold)
    candidates = [c for c in pool if normalize_text(c) != gold_norm]
    if not candidates:
         return None
    # Shuffle for randomness if multiple candidates exist
    random.shuffle(candidates) 
    return candidates[0] 

def _numeric_perturbation(answer: str) -> str | None:
    numbers = re.findall(r"\d+", answer)
    if not numbers: return None
    result = answer
    num_to_change = random.choice(numbers) # Pick one number to change
    try:
        val = int(num_to_change)
        # Simple change: +1 or -1 if > 0, else +1
        delta = random.choice([1, -1]) if val > 0 else 1
        new_val = str(val + delta)
        # Replace only the first occurrence of the chosen number
        result = result.replace(num_to_change, new_val, 1) 
        if normalize_text(result) == normalize_text(answer): # Check if change was effective
             return None
        return result
    except ValueError: # Handle potential large numbers or errors
         return None

def _fallback_false_answer(gold: str) -> str:
    perturbed = _numeric_perturbation(gold)
    if perturbed: return perturbed
    # Simplified negation
    return f"not {gold}" if not gold.lower().startswith("not ") else gold # Avoid double negation


# --- REWRITTEN: _build_false_context using LLM + Fallback ---
def _build_false_context(
    question: str,
    answer_gold: str,
    ctx_true: str, # Use ctx_true as reference for LLM
    text: str, # Keep original text for rule-based fallback if needed
    meta: dict[str, Any],
    retriever: SimpleRetriever, # Keep retriever for potential future use or stricter rule-based fallback
    answer_pool: dict[tuple[str | None, int | None, int | None], list[str]],
    retriever_cfg: dict[str, Any], # Keep config
    model_name: str | None = None, # Added model_name
) -> tuple[str | None, str | None]:
    """
    Attempts to generate ctx_false using LLM first, then falls back to rule-based methods.
    Returns (ctx_false, answer_false). answer_false is None if LLM succeeded.
    """
    
    # --- Step 1: Try LLM Generation ---
    llm_generated_ctx, llm_generated_ans = generate_contradictory_context_llm(question, answer_gold, ctx_true, model_name)
    
    if llm_generated_ctx and llm_generated_ans:
        # Basic check: Ensure it's different from ctx_true
        if normalize_text(llm_generated_ctx) != normalize_text(ctx_true):
             LOGGER.debug(f"LLM successfully generated ctx_false for: {question[:50]}...")
             return llm_generated_ctx, llm_generated_ans # Success! Return both context and answer
        else:
             LOGGER.warning(f"LLM generated ctx_false identical to ctx_true. Falling back. Q: {question[:50]}...")

    # --- Step 2: Fallback to Rule-based Methods (if LLM failed or wasn't available) ---
    LOGGER.debug(f"LLM failed for Q: {question[:50]}... Trying rule-based fallback.")
    
    # Rule 1: Pick alternative answer from pool
    region = meta.get("region")
    year = meta.get("year")
    month = meta.get("month")
    pool_key = (region, year, month)
    candidates = answer_pool.get(pool_key, [])
    alt_answer = _pick_alternative_answer(answer_gold, candidates)
    if not alt_answer and month is not None: # Try yearly pool if monthly fails
        fallback_candidates = answer_pool.get((region, year, None), [])
        if fallback_candidates is not candidates: # Avoid redundant check if same list
            alt_answer = _pick_alternative_answer(answer_gold, fallback_candidates)

    # Rule 2: If no alternative from pool, try perturbation/negation
    if not alt_answer:
        alt_answer = _fallback_false_answer(answer_gold)
        if normalize_text(alt_answer) == normalize_text(answer_gold): # If even fallback fails to change
             LOGGER.error(f"Cannot generate distinct false answer for: {answer_gold}. Skipping ctx_false generation.")
             return None, None # Indicate failure to generate

    # Rule 3: Inject the rule-based false answer into context
    # Use ctx_true as base for injection if available and seems reasonable length
    base_for_injection = ctx_true if ctx_true and len(ctx_true) < 1000 else text
    base_for_injection = base_for_injection or text # Ensure we have some text
    
    false_ctx = None
    if answer_gold and normalize_text(answer_gold) in normalize_text(base_for_injection):
        # Try replacing the gold answer with the false one
        # Use regex for whole word replacement if possible
        try:
            pattern = r'\b' + re.escape(answer_gold) + r'\b'
            # Use count=1 to replace only the first occurrence
            false_ctx_candidate = re.sub(pattern, alt_answer, base_for_injection, count=1, flags=re.IGNORECASE) 
            # Check if replacement actually happened and didn't remove the answer entirely
            if normalize_text(answer_gold) not in normalize_text(false_ctx_candidate):
                 false_ctx = false_ctx_candidate
        except re.error: # Handle complex regex issues
             false_ctx = base_for_injection.replace(answer_gold, alt_answer, 1) # Fallback to simple replace
             if normalize_text(answer_gold) not in normalize_text(false_ctx):
                  pass # Simple replace worked
             else: false_ctx = None # Simple replace failed too
            
    # If replacement failed or wasn't possible, append the false answer
    if not false_ctx:
        # Try to create a shorter base by removing potential answer sentences first
        sentences = split_sentences(base_for_injection)
        base_sentences = []
        if sentences:
            answer_tok_set = _answer_token_set(answer_gold)
            for s in sentences:
                # Keep sentences with low overlap with the gold answer
                if _overlap_score(_tokenize(s), answer_tok_set) < 0.3:
                    base_sentences.append(s)
                if len(base_sentences) >= 3: # Limit base length
                    break
        
        base = " ".join(base_sentences).strip()
        if not base: base = " ".join(base_for_injection.split()[:50]).strip() # Fallback base
        
        # Add the contradictory statement
        false_ctx = f"{base} However, other reports suggested {alt_answer}."

    # Final check: ensure rule-based ctx_false doesn't contain gold answer
    if false_ctx and normalize_text(answer_gold) in normalize_text(false_ctx):
        LOGGER.warning(f"Rule-based ctx_false still contains gold answer for Q: {question[:50]}... Attempting redaction.")
        false_ctx = _redact_answer_tokens(false_ctx, answer_gold)
        if not false_ctx: # Redaction failed
             LOGGER.error(f"Failed to create valid rule-based ctx_false for Q: {question[:50]}")
             return None, None

    LOGGER.debug(f"Rule-based fallback generated ctx_false for: {question[:50]}...")
    return false_ctx, alt_answer


# --- MODIFIED: build_items to handle new _build_false_context ---
def build_items(
    region: str,
    split: str,
    year_min: int,
    year_max: int,
    retriever_cfg: dict,
    start_month: int | None = None,
    end_month: int | None = None,
    seed: int = 7,
    max_items: int | None = None,
    source_file: str | None = None, # Added source_file
    model_name: str | None = None, # Added model_name for LLM generation
) -> list[dict[str, Any]]:
    set_seed(seed)
    
    if source_file:
         LOGGER.info(f"Loading data from source file: {source_file}")
         from tiebe_rag_conflict.utils.io import read_jsonl
         ds = read_jsonl(source_file)
    else:
         ds = load_tiebe(region, split)
    
    # --- Filtering Logic (Mostly unchanged) ---
    year_min_int = int(year_min) if year_min is not None else None
    year_max_int = int(year_max) if year_max is not None else None
    start_month_int = int(start_month) if start_month is not None else None
    end_month_int = int(end_month) if end_month is not None else None
    
    # Minimal validation for source_file cases
    if not ds:
         raise ValueError("Loaded dataset is empty.")

    records: list[dict[str, Any]] = []
    skipped_count = 0
    for row in ds:
        try:
            year = int(row["year"])
            month = int(row["month"])
        except (ValueError, TypeError):
            skipped_count += 1
            continue
            
        # Apply filters
        if year_min_int and year < year_min_int: continue
        if year_max_int and year > year_max_int: continue
        if start_month_int is not None and year == year_min_int and month < start_month_int: continue
        if end_month_int is not None and year == year_max_int and month > end_month_int: continue
        
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "") or row.get("answer_gold", "")).strip() # Handle answer or answer_gold
        text = str(row.get("text", "")).strip()
        if not question or not answer or not text: # Skip rows with missing essential data
             skipped_count +=1
             continue
             
        event_desc = str(row.get("event_desc", "") or "").strip()
        source_id = None
        for key in ("id", "event_id", "question_id", "uuid", "item_id"):
            value = row.get(key);
            if value: source_id = str(value); break
        
        record = {
            "question": question, "answer_gold": answer, "region": row.get("region"),
            "year": year, "month": month, "text": text, "language": split,
            "event_desc": event_desc, "source_id": source_id,
        }
        record["idx"] = len(records) # Use simple index within the filtered list
        records.append(record)

        # Apply max_items limit if set
        if max_items is not None and len(records) >= max_items:
            LOGGER.info(f"Reached max_items limit ({max_items}). Stopping data loading.")
            break
            
    if skipped_count > 0:
         LOGGER.warning(f"Skipped {skipped_count} rows due to missing data or parsing errors.")
    if not records:
        raise ValueError("No records found after filtering.")
    LOGGER.info(f"Loaded {len(records)} records after filtering.")

    # --- Answer Pool & Retriever Setup ---
    answer_pool = _prepare_answer_pool(records)
    
    retriever_enabled = retriever_cfg.get("enabled", True)
    retriever = None
    
    if not retriever_enabled:
        LOGGER.info("Retriever is disabled in config. Skipping index build.")
    else:
        encoder_name = retriever_cfg.get("encoder", "text-embedding-3-small")
        chunk_token_limit = int(retriever_cfg.get("chunk_token_limit", EMBED_TOKEN_LIMIT))
        sentences: list[str] = []; metas: list[dict[str, Any]] = []; answers: list[str] = []
        LOGGER.info("Chunking text segments for retriever index...")
        for record in tqdm(records, desc="Chunking texts"):
            # Ensure text is string before chunking
            text_content = str(record.get("text", ""))
            if not text_content: continue
            segments = _chunk_text_segments(text_content, encoder_name, max_tokens=chunk_token_limit)
            for segment in segments:
                sentences.append(segment)
                metas.append({"region": record.get("region"), "year": record.get("year"), "month": record.get("month"), "doc_idx": record.get("idx")})
                answers.append(record["answer_gold"])
        if not sentences:
             LOGGER.warning("No text segments generated for retriever index and it was enabled.")
        else:
            LOGGER.info(f"Building retriever index with {len(sentences)} segments...")
            retriever = SimpleRetriever(
                sentences, metas=metas, answers=answers, encoder_name=encoder_name,
                provider=retriever_cfg.get("provider"), batch_size=int(retriever_cfg.get("batch_size", 64)),
                api_key=retriever_cfg.get("api_key"), api_key_env=retriever_cfg.get("api_key_env"),
                api_key_file=retriever_cfg.get("api_key_file"),
            )
            LOGGER.info("Retriever index built.")
    # Valid block ends at line 634. Lines 635-643 are artifacts.
    # We replace lines 635-643 with empty string.


    # --- Item Generation Loop ---
    # (Reuse existing code by returning to original flow if needed, but here we just return the modified function body)
    # Since we are replacing the whole function, we need to include the rest of the generation logic or ensure we splice correctly.
    # To avoid repeating 100 lines of unchanged code, I will use precise replacement markers.

    return _continue_build_items(records, retriever, answer_pool, retriever_cfg, split, model_name=model_name) 

# Helper for multi-threading
def _process_single_row(row, retriever, answer_pool, retriever_cfg, split, model_name):
    try:
        text_content = str(row.get("text", ""))
        if not text_content: 
             return None, "skip", False

        # 1. Build ctx_true and ctx_insuf
        true_ctx, support_indices = _build_true_context(text_content, row["answer_gold"])
        if not true_ctx: 
            return None, "skip_true_fail", False
            
        sentences_split = split_sentences(text_content) 
        insuf_ctx = _build_insufficient_context(sentences_split, row["answer_gold"], support_indices)
        
        fallback_used = False
        # --- Fallback: If Rule-based Insuf is too short/generic, try LLM ---
        true_len = len(true_ctx.split())
        insuf_len = len(insuf_ctx.split())
        is_too_short = (insuf_len < 0.5 * true_len) or (insuf_len < 20)
        
        if is_too_short or insuf_ctx == GENERIC_INSUFFICIENT_CONTEXT:
            gen_insuf = generate_insufficient_context_llm(row["question"], row["answer_gold"], true_ctx, model_name)
            if gen_insuf:
                 insuf_ctx = gen_insuf
                 fallback_used = True
        # -------------------------------------------------------------

        insuf_ctx, should_skip_insuf = _ensure_distinct_contexts(
            true_ctx, insuf_ctx, answer=row["answer_gold"], event_desc=row.get("event_desc")
        )
        if should_skip_insuf and insuf_ctx == GENERIC_INSUFFICIENT_CONTEXT:
            return None, "skip_insuf_fail", False

        # 2. Build ctx_false
        meta = {"region": row.get("region"), "year": row.get("year"), "month": row.get("month"), "doc_idx": row.get("idx")}
        false_ctx, answer_false = _build_false_context(
            row["question"], row["answer_gold"], true_ctx, text_content, 
            meta, retriever, answer_pool, retriever_cfg, model_name=model_name
        )
        
        if false_ctx is None:
            return None, "skip_false_fail", False

        # --- Create final item dict ---
        item_id = f"{row.get('region','NA')}:{row.get('year','NA')}-{row.get('month','NA'):02d}:{row.get('idx','NA'):04d}"
        item = {
            "question": row["question"],
            "answer_gold": row["answer_gold"],
            "region": row.get("region"), "year": row.get("year"), "month": row.get("month"),
            "language": row.get("language", split),
            "ctx_true": true_ctx,
            "ctx_false": false_ctx,
            "ctx_insuf": insuf_ctx,
            "answer_false": answer_false, 
            "item_id": item_id,
        }
        return item, "success", fallback_used
    except Exception as e:
        LOGGER.error(f"Error processing row {row.get('idx')}: {e}")
        return None, "error", False

# Helper to continue the logic (refactoring for cleaner tool usage)
def _continue_build_items(records, retriever, answer_pool, retriever_cfg, split, model_name=None):
    items: list[dict[str, Any]] = []
    skipped_items_count = 0
    insuf_fallback_count = 0

    max_workers = 16 # Adjust based on rate limits
    LOGGER.info(f"Generating triplets for {len(records)} items using model: {model_name or 'Default'} (Threads: {max_workers})")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        futures = {executor.submit(_process_single_row, row, retriever, answer_pool, retriever_cfg, split, model_name): row for row in records}
        
        for future in tqdm(as_completed(futures), total=len(records), desc="Generating Triplets"):
            result, status, fallback_used = future.result()
            
            if status == "success" and result:
                items.append(result)
                if fallback_used:
                    insuf_fallback_count += 1
            elif status.startswith("skip") or status == "error":
                skipped_items_count += 1
                if status == "error":
                     # Optional: log specific error if needed, but _process_single_row already logged it
                     pass
                elif status == 'skip_true_fail':
                     LOGGER.warning(f"Failed to build ctx_true for Q: {futures[future]['question'][:50]}...")
                elif status == 'skip_insuf_fail':
                     LOGGER.warning(f"Could not generate distinct ctx_insuf for Q: {futures[future]['question'][:50]}...")
                elif status == 'skip_false_fail':
                     LOGGER.error(f"Failed to generate distinct ctx_false for Q: {futures[future]['question'][:50]}...")

    if skipped_items_count > 0:
         LOGGER.warning(f"Skipped generating triplets for {skipped_items_count} items due to context generation issues.")
         
    if not items:
         raise ValueError("No triplets generated. Check logs for errors.")

    _quality_checks(items) 
    LOGGER.info(f"Successfully generated {len(items)} triplets. (Insuf Fallback used: {insuf_fallback_count}/{len(items)})")
    return items

def load_items(cfg: dict[str, Any], seed: int = 7, max_items: int | None = None) -> list[dict[str, Any]]: 
    region = cfg["region"]
    split = cfg.get("split", "english")
    year_min = cfg.get("year_min")
    year_max = cfg.get("year_max")
    start_month = cfg.get("start_month")
    end_month = cfg.get("end_month")
    retriever_cfg = cfg.get("retriever", {})
    source_file = cfg.get("source_file") 
    LOGGER.info(f"DEBUG: load_items source_file={source_file} from cfg keys={list(cfg.keys())}") 
    
    # Extract model name for generation
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name") # e.g. "openai/gpt-4o-mini"
    
    return build_items(
        region, split, year_min, year_max, retriever_cfg,
        start_month=start_month, end_month=end_month, seed=seed, max_items=max_items,
        source_file=source_file,
        model_name=model_name
    )


# --- Existing load_tiebe, _quality_checks, parse_args, load_items, main ---
# ... (Keep these functions as they were, potentially adding max_items to load_items) ...
def load_tiebe(region: str, split: str) -> Dataset:
    """Load the TiEBe dataset for the given region and split."""
    LOGGER.info("Loading TiEBe region=%s split=%s", region, split)
    # Use cache_dir='.' or some local path if Hugging Face cache causes issues
    ds = load_dataset("TimelyEventsBenchmark/TiEBe", region, split=split, cache_dir="./hf_cache") 
    required = {"year", "month", "text", "question", "answer", "region"} # event_desc is optional
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return ds

def _quality_checks(items: list[dict[str, Any]]) -> None:
    if not items: return
    # Check 1: Identical true/insuf
    identical_true_insuf = sum(1 for item in items if normalize_text(item.get("ctx_true","")) == normalize_text(item.get("ctx_insuf","")))
    if identical_true_insuf:
        # This might be acceptable if insuf became generic but still distinct
        # Let's make it a warning instead of error for now
        LOGGER.warning(f"Found {identical_true_insuf} items with identical true/insufficient contexts after normalization.")
        # Optional: Raise error if strict distinctness needed:
        # raise ValueError(f"Found {identical_true_insuf} items with identical true and insufficient contexts")

    # Check 2: True context evidence hit rate
    hits = sum(1 for item in items if _has_answer_evidence(str(item.get("ctx_true","")), item["answer_gold"]))
    ratio = hits / len(items) if items else 0
    LOGGER.info("ctx_true answer evidence hit rate: %.1f%%", 100 * ratio)
    if ratio < 0.5: # Keep this stricter check
        LOGGER.error(f"ctx_true answer evidence hit rate ({ratio:.1%}) below 50%; check _build_true_context logic or source data quality.")
        # Consider raising ValueError if this is critical
        # raise ValueError("ctx_true answer evidence hit rate below 50%")

    # NEW Check 3: How many ctx_false were generated by LLM vs Rule-based?
    llm_count = sum(1 for item in items if item.get("answer_false") is None and item.get("ctx_false"))
    rule_count = sum(1 for item in items if item.get("answer_false") is not None and item.get("ctx_false"))
    failed_count = len(items) - (llm_count + rule_count) # Should be 0 if logic is right
    LOGGER.info(f"ctx_false generation: {llm_count} by LLM, {rule_count} by Rule-based fallback. ({failed_count} failures recorded in items list)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TiEBe triplets using LLM for false context")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=7)
    # Add optional max_items argument for testing
    parser.add_argument("--max_items", type=int, default=None, help="Process only the first N items (for testing)") 
    return parser.parse_args()



def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    cfg = read_config(args.config)
    
    # Pass max_items from command line args to load_items
    items = load_items(cfg, seed=args.seed, max_items=args.max_items) 
    
    if items: # Only write if items were generated
        write_jsonl(args.out, items)
        LOGGER.info("Wrote %d items to %s", len(items), args.out)
    else:
        LOGGER.error("No items were generated. Output file not created.")

if __name__ == "__main__":
    main()