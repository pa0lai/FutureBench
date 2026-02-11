from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import concurrent.futures
import datetime
import requests
from tqdm import tqdm
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tiebe_rag_conflict.eval.metrics import compute_csi
from tiebe_rag_conflict.eval.prompts import get_prompt
from tiebe_rag_conflict.utils.io import ensure_dir, read_config, read_jsonl, set_seed, write_jsonl

LOGGER = logging.getLogger(__name__)


DEFAULT_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


def call_openrouter(
    model: str,
    system_msg: str,
    user_msg: str,
    max_retries: int = 5,
    timeout: float = 60,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,

    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call the OpenRouter chat completions endpoint via the OpenAI SDK."""

    try:
        from openai import APIStatusError as _APIStatusError, OpenAI as _OpenAI, RateLimitError as _RateLimitError
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `openai` package is required for OpenRouter integration. Install it via `pip install openai`."
        ) from exc

    api_key = os.environ.get(DEFAULT_OPENROUTER_API_KEY_ENV)
    if not api_key:
        # Fallback to OPENAI_API_KEY
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"OpenRouter API key is missing. Set {DEFAULT_OPENROUTER_API_KEY_ENV} or OPENAI_API_KEY in the environment"
        )

    client = _OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    messages = (
        ([{"role": "system", "content": system_msg}] if system_msg else [])
        + [
            {"role": "user", "content": user_msg}
        ]
    )  # <-- 重要：一定要包含 user 訊息

    payload: dict[str, Any] = {
    "model": model,
    "messages": messages,
    "temperature": 0 if temperature is None else temperature,
    "max_tokens": 200 if max_tokens is None else max_tokens,
    }

    if response_format is not None:
        payload["response_format"] = response_format

# 不要把 extra_body merge 進去！


    backoff = 1.0
    last_err = "unknown"
    for _ in range(max_retries):
        try:

            completion = client.with_options(timeout=timeout).chat.completions.create(
                extra_headers=extra_headers,
                extra_body=extra_body,  
                **payload,
            )
            message = completion.choices[0].message
            content = message.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content is None:
                return ""
            return str(content).strip()
        except _RateLimitError as exc:  # pragma: no cover - network dependent
            last_err = str(exc)
        except _APIStatusError as exc:  # pragma: no cover - network dependent
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "response", None)
            body_text = ""
            if body is not None:
                text_attr = getattr(body, "text", None)
                if callable(text_attr):
                    body_text = str(text_attr())
                elif isinstance(text_attr, str):
                    body_text = text_attr
                else:
                    body_text = str(body)
            last_err = f"HTTP {status}: {body_text[:200]}" if status else str(exc)
        except Exception as exc:  # pragma: no cover - network dependent

            last_err = str(exc)
        time.sleep(backoff)
        backoff = min(backoff * 1.6, 15)
    raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


def _extract_openrouter_answer(raw: str, fields: list[str]) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(parsed, dict):
        if "error" in parsed:
            raise RuntimeError(f"OpenRouter error: {parsed['error']}")
        for field in fields:
            if field in parsed:
                value = parsed[field]
                if isinstance(value, str):
                    return value.strip()
                return json.dumps(value, ensure_ascii=False)
        for value in parsed.values():
            if isinstance(value, str):
                return value.strip()
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed).strip()


def ask_model(prompt: str, model_cfg: dict[str, Any]) -> str:
    if model_cfg.get("provider") == "openrouter":
        model_name = model_cfg.get("model") or model_cfg.get("name")
        if not model_name:
            raise RuntimeError("OpenRouter provider requires `model` (or `name`) in the config")
        system_msg = model_cfg.get("system_prompt") or model_cfg.get("system_msg") or ""
        response_field = model_cfg.get("response_field")
        if isinstance(response_field, str):
            response_fields = [response_field]
        elif isinstance(response_field, (list, tuple)):
            response_fields = [str(field) for field in response_field]
        else:
            response_fields = ["answer", "output", "text"]

        response_format_cfg = model_cfg.get("response_format")
        if isinstance(response_format_cfg, str):
            response_format_cfg = {"type": response_format_cfg}

        extra_headers_cfg = model_cfg.get("extra_headers")
        extra_headers = None
        if isinstance(extra_headers_cfg, dict):
            extra_headers = {str(key): str(value) for key, value in extra_headers_cfg.items()}

        extra_body_cfg = model_cfg.get("extra_body")
        extra_body = dict(extra_body_cfg) if isinstance(extra_body_cfg, dict) else {}

        # >>> ADD: inject reasoning block (only when supported & enabled)
        supports_reasoning = bool(model_cfg.get("supports_reasoning", True))
        r = model_cfg.get("reasoning") if isinstance(model_cfg.get("reasoning"), dict) else {}
        if supports_reasoning and r.get("enabled"):
            reasoning_obj: dict[str, Any] = {"enabled": True}
            # exclude（預設 True）
            reasoning_obj["exclude"] = bool(r.get("exclude", True))
            # 二擇一：effort 或 max_tokens
            if r.get("effort") is not None:
                reasoning_obj["effort"] = str(r["effort"])
            if r.get("max_tokens") is not None:
                reasoning_obj["max_tokens"] = int(r["max_tokens"])
            # 合併到 extra_body（OpenRouter 會忽略未知欄位；但對不支援的模型我們已用 supports_reasoning 防呆）
            extra_body["reasoning"] = reasoning_obj
        # <<< ADD

        if "api_key" in model_cfg:
            os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = str(model_cfg["api_key"])
        elif "api_key_file" in model_cfg:
            path = Path(str(model_cfg["api_key_file"]))
            if not path.exists():
                raise RuntimeError(f"OpenRouter API key file not found: {path}")
            key = path.read_text(encoding="utf-8").strip()
            if not key:
                raise RuntimeError(f"OpenRouter API key file {path} is empty")
            os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = key
        elif "api_key_env" in model_cfg:
            api_key = os.environ.get(str(model_cfg["api_key_env"]))
            if api_key:
                os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = api_key

        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "system_msg": system_msg,
            "user_msg": prompt,
            "max_retries": int(model_cfg.get("max_retries", 5)),
            "timeout": float(model_cfg.get("timeout", 60)),
            "temperature": model_cfg.get("temperature"),
            "max_tokens": model_cfg.get("max_tokens"),
            "extra_headers": extra_headers,
            "extra_body": extra_body,   # <<--- 已含 reasoning（若有）
        }
        if response_format_cfg is not None:
            call_kwargs["response_format"] = response_format_cfg

        raw = call_openrouter(**call_kwargs)
        return _extract_openrouter_answer(raw, response_fields)

    if model_cfg.get("provider") == "openrouter":
        model_name = model_cfg.get("model") or model_cfg.get("name")
        if not model_name:
            raise RuntimeError("OpenRouter provider requires `model` (or `name`) in the config")
        system_msg = model_cfg.get("system_prompt") or model_cfg.get("system_msg") or ""
        response_field = model_cfg.get("response_field")
        if isinstance(response_field, str):
            response_fields = [response_field]
        elif isinstance(response_field, (list, tuple)):
            response_fields = [str(field) for field in response_field]
        else:
            response_fields = ["answer", "output", "text"]
        response_format_cfg = model_cfg.get("response_format")
        if isinstance(response_format_cfg, str):
            response_format_cfg = {"type": response_format_cfg}

        extra_headers_cfg = model_cfg.get("extra_headers")
        extra_headers = None
        if isinstance(extra_headers_cfg, dict):
            extra_headers = {str(key): str(value) for key, value in extra_headers_cfg.items()}
        extra_body_cfg = model_cfg.get("extra_body")
        extra_body = None
        if isinstance(extra_body_cfg, dict):
            extra_body = dict(extra_body_cfg)
        if "api_key" in model_cfg:
            os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = str(model_cfg["api_key"])

        elif "api_key_file" in model_cfg:
            path = Path(str(model_cfg["api_key_file"]))
            if not path.exists():
                raise RuntimeError(f"OpenRouter API key file not found: {path}")
            key = path.read_text(encoding="utf-8").strip()
            if not key:
                raise RuntimeError(f"OpenRouter API key file {path} is empty")
            os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = key

        elif "api_key_env" in model_cfg:
            api_key = os.environ.get(str(model_cfg["api_key_env"]))
            if api_key:
                os.environ[DEFAULT_OPENROUTER_API_KEY_ENV] = api_key
        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "system_msg": system_msg,
            "user_msg": prompt,
            "max_retries": int(model_cfg.get("max_retries", 5)),
            "timeout": float(model_cfg.get("timeout", 60)),
            "temperature": model_cfg.get("temperature"),
            "max_tokens": model_cfg.get("max_tokens"),
            "extra_headers": extra_headers,
            "extra_body": extra_body,
        }
        if response_format_cfg is not None:
            call_kwargs["response_format"] = response_format_cfg

        raw = call_openrouter(**call_kwargs)
        return _extract_openrouter_answer(raw, response_fields)
    endpoint = model_cfg.get("endpoint")
    if not endpoint:
        raise RuntimeError("Model endpoint is not specified in the config")
    payload = {
        "prompt": prompt,
        "max_tokens": model_cfg.get("max_tokens", 128),
        "temperature": model_cfg.get("temperature", 0.0),
    }
    if "stop" in model_cfg:
        payload["stop"] = model_cfg["stop"]
    response = requests.post(endpoint, json=payload, timeout=model_cfg.get("timeout", 120))
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "output" in data:
            return str(data["output"]).strip()
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                if "text" in choice:
                    return str(choice["text"]).strip()
                message = choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"]).strip()
    return str(data).strip()


def run_eval(
    items: list[dict[str, Any]],
    mode: str,
    ctx_type: str,
    model_cfg: dict[str, Any],
    seed: int = 7,
    output_path: Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    prompt_template = get_prompt(mode)
    rows: list[dict[str, Any]] = []
    
    workers = int(model_cfg.get("workers", 5))

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        context = item.get(f"ctx_{ctx_type}", "")
        prompt = prompt_template.format(question=item["question"], context=context)
        try:
            if ctx_type == "false" and not context:
                pred = "NOT ENOUGH INFORMATION"
            else:
                print(f"[DEBUG] Asking model for Q: {item['question']}")
                pred = ask_model(prompt, model_cfg)
                print(f"[DEBUG] Raw Answer: {pred}")
        except Exception as exc:  # pragma: no cover - depends on external model
            LOGGER.error("Model call failed for question %s: %s", item["question"], exc)
            # Log full error to file
            with open("/tmp/FORCE_DEBUG.log", "a") as err_f:
                err_f.write(f"[{datetime.datetime.now()}] Mode: {mode} | Q: {item['question'][:20]}... | Error: {str(exc)}\n")
                err_f.flush()
            pred = "SYSTEM_ERROR"

        norm = (pred or "").strip()
        if norm.upper().rstrip(".") == "NOT ENOUGH INFORMATION":
            norm = ""
        pred = norm
        
        row = {
            "question": item["question"],
            "context": context,
            "pred": pred,
            "gold": item["answer_gold"],
            "false": item.get("answer_false"),
            "region": item.get("region"),
            "language": item.get("language"),
            "year": item.get("year"),
            "month": item.get("month"),
            "mode": mode,
            "ctx_type": ctx_type,
        }
        return row

    # For real-time saving & resume logic
    existing_questions = set()
    if output_path and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "question" in data:
                            existing_questions.add(data["question"])
            LOGGER.info("Found %d existing predictions in %s. Resuming...", len(existing_questions), output_path)
        except Exception as e:
            LOGGER.warning("Could not read existing file for resume: %s. Starting fresh.", e)
            output_path.write_text("", encoding="utf-8")
    elif output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    # Filter items that are already processed
    items_to_run = [item for item in items if item["question"] not in existing_questions]
    
    if not items_to_run:
        LOGGER.info("All items already processed for %s:%s. Skipping...", mode, ctx_type)
        # We still need to return the full rows for metrics
        all_rows = []
        if output_path and output_path.exists():
             with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_rows.append(json.loads(line))
        metrics = compute_csi(all_rows)
        metrics.update({"mode": mode, "ctx_type": ctx_type})
        return {"metrics": metrics, "rows": all_rows}

    LOGGER.info("Starting multi-threaded evaluation with %d workers (%d items to go)", workers, len(items_to_run))
    
    pbar = tqdm(total=len(items), desc=f"{mode}:{ctx_type}")
    pbar.update(len(existing_questions)) # 直接跳到已完成的進度
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in items_to_run}
        
        for future in concurrent.futures.as_completed(future_to_item):
            row = future.result()
            rows.append(row)
            pbar.update(1)
            
            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush() # 強迫存進硬碟，不留緩衝區
    pbar.close()

    # Combine existing and new results for final metrics
    final_rows = []
    if output_path and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_rows.append(json.loads(line))
    else:
        final_rows = rows

    metrics = compute_csi(final_rows)
    metrics.update({"mode": mode, "ctx_type": ctx_type})
    return {"metrics": metrics, "rows": final_rows}


def _normalize_to_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ValueError(f"{field_name} must be a string or a list of strings, got {type(value).__name__}")


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _resolve_eval_grid(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = cfg.get("eval")
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}

    if args.modes is not None:
        modes = [str(mode) for mode in args.modes]
    else:
        modes = _normalize_to_list(eval_cfg.get("modes"), "eval.modes")
    if not modes:
        raise ValueError("No evaluation modes specified. Provide --mode or set eval.modes in the config.")

    if args.ctx_types is not None:
        ctx_types = [str(ctx) for ctx in args.ctx_types]
    else:
        ctx_types = _normalize_to_list(eval_cfg.get("ctx_types"), "eval.ctx_types")
    if not ctx_types:
        raise ValueError("No context types specified. Provide --ctx or set eval.ctx_types in the config.")

    seed_value = args.seed if args.seed is not None else eval_cfg.get("seed")
    seed = int(seed_value) if seed_value is not None else 7

    limit_value = args.limit if args.limit is not None else eval_cfg.get("limit")
    limit = None if limit_value is None else int(limit_value)

    return {
        "modes": _deduplicate_preserve_order([str(mode) for mode in modes]),
        "ctx_types": _deduplicate_preserve_order([str(ctx) for ctx in ctx_types]),
        "seed": seed,
        "limit": limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TiEBe RAG conflict evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument(
        "--mode",
        dest="modes",
        nargs="+",
        choices=["openqa", "evidence_only", "baseline", "cot", "no_context"],
        help="One or more evaluation modes to run. Defaults to the config's eval.modes.",
    )
    parser.add_argument(
        "--ctx",
        dest="ctx_types",
        nargs="+",
        choices=["true", "false", "insuf"],
        help="One or more context types to run. Defaults to the config's eval.ctx_types.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config")

    # >>> ADD: reasoning controls
    parser.add_argument("--reasoning", choices=["on", "off"], default="off",
                        help="Enable model-side reasoning/CoT for supported OpenRouter models.")
    parser.add_argument("--reasoning-exclude", choices=["true", "false"], default="true",
                        help="If reasoning is on, whether to EXCLUDE reasoning text from the response (true = hidden CoT).")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        help="If reasoning is on, set effort level (mutually exclusive with --reasoning-max-tokens).")
    parser.add_argument("--reasoning-max-tokens", type=int, default=None,
                        help="If reasoning is on, set max reasoning tokens (mutually exclusive with --reasoning-effort).")
    # <<< ADD

    return parser.parse_args()



def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = read_config(args.config)
    plan = _resolve_eval_grid(args, cfg)
    model_cfg = cfg.get("model", {})
    items = read_jsonl(args.items)
    if plan["limit"] is not None:
        items = items[: plan["limit"]]
    
    # Priority: CLI argument > Config file > Default
    run_dir_str = args.output_dir if args.output_dir else cfg.get("output_dir", "runs/default")
    run_dir = Path(run_dir_str)
    ensure_dir(run_dir)

    # >>> ADD: merge CLI reasoning overrides into model_cfg
    # 標記哪些模型支援 reasoning（可在 config 內設 "supports_reasoning": true/false；預設 True）
    if "supports_reasoning" not in model_cfg:
        model_cfg["supports_reasoning"] = True  # 對 OpenRouter 上的 reasoning 模型預設 True；一般本地端/不支援者請在 YAML 設 False

    # 將 CLI 旗標寫入 model_cfg["reasoning"]
    rcfg = model_cfg.get("reasoning", {}) if isinstance(model_cfg.get("reasoning"), dict) else {}
    rcfg["enabled"] = (args.reasoning == "on")
    if args.reasoning_exclude is not None:
        rcfg["exclude"] = (args.reasoning_exclude == "true")
    if args.reasoning_effort and args.reasoning_max_tokens:
        raise ValueError("Use either --reasoning-effort OR --reasoning-max-tokens, not both.")
    if args.reasoning_effort:
        rcfg["effort"] = args.reasoning_effort
        rcfg.pop("max_tokens", None)
    if args.reasoning_max_tokens is not None:
        rcfg["max_tokens"] = int(args.reasoning_max_tokens)
        rcfg.pop("effort", None)
    model_cfg["reasoning"] = rcfg
    # <<< ADD

    for mode in plan["modes"]:
        for ctx_type in plan["ctx_types"]:
            LOGGER.info("Running evaluation for mode=%s ctx=%s seed=%s", mode, ctx_type, plan["seed"])
            preds_path = run_dir / f"preds_{mode}_{ctx_type}.jsonl"
            result = run_eval(items, mode, ctx_type, model_cfg, seed=plan["seed"], output_path=preds_path)
            
            # Save final metrics
            metrics_path = run_dir / f"metrics_{mode}_{ctx_type}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, ensure_ascii=False, indent=2)
            LOGGER.info("Saved final metrics to %s", metrics_path)
            LOGGER.info("Metrics: %s", result["metrics"])


if __name__ == "__main__":  # pragma: no cover
    main()

