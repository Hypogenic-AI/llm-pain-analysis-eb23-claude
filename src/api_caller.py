"""
API caller for multiple LLM providers.
Supports OpenAI direct API and OpenRouter for other models.
"""

import os
import json
import time
import asyncio
import httpx
from typing import Optional

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

# Model configurations: (provider, model_id, display_name)
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "name": "GPT-4.1",
    },
    "claude-sonnet-4-5": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
    },
    "gemini-2.5-pro": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-pro-preview",
        "name": "Gemini 2.5 Pro",
    },
    "llama-3.3-70b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
    },
    "deepseek-r1": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "name": "DeepSeek-R1",
    },
}


async def call_openai(
    client: httpx.AsyncClient,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """Call OpenAI API directly."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(5):
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  Rate limited (OpenAI), waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason", ""),
            }
        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            if attempt < 4:
                wait = 2 ** attempt * 2
                print(f"  Error (OpenAI): {e}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                return {"content": f"ERROR: {e}", "usage": {}, "finish_reason": "error"}
    return {"content": "ERROR: max retries exceeded", "usage": {}, "finish_reason": "error"}


async def call_openrouter(
    client: httpx.AsyncClient,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(5):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  Rate limited (OpenRouter), waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or len(data["choices"]) == 0:
                return {"content": f"ERROR: No choices in response: {json.dumps(data)[:200]}", "usage": {}, "finish_reason": "error"}
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason", ""),
            }
        except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
            if attempt < 4:
                wait = 2 ** attempt * 2
                print(f"  Error (OpenRouter {model_id}): {e}, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                return {"content": f"ERROR: {e}", "usage": {}, "finish_reason": "error"}
    return {"content": "ERROR: max retries exceeded", "usage": {}, "finish_reason": "error"}


async def call_model(
    client: httpx.AsyncClient,
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """Call any model by key."""
    config = MODELS[model_key]
    if config["provider"] == "openai":
        return await call_openai(client, config["model_id"], system_prompt, user_prompt, max_tokens, temperature)
    else:
        return await call_openrouter(client, config["model_id"], system_prompt, user_prompt, max_tokens, temperature)


async def call_model_batch(
    model_key: str,
    system_prompt: str,
    prompts: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    concurrency: int = 5,
) -> list[dict]:
    """Call a model on a batch of prompts with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async def _call(idx, prompt):
        async with semaphore:
            async with httpx.AsyncClient() as client:
                result = await call_model(client, model_key, system_prompt, prompt, max_tokens, temperature)
                result["prompt"] = prompt
                result["prompt_idx"] = idx
                results[idx] = result

    tasks = [_call(i, p) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    return results
