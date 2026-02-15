import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from ollama.backend.parsing import normalize_expense


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_telegram_api(token: str, method: str, params: dict | None = None, timeout: int = 35) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = None

    if params:
        data = urllib.parse.urlencode(params).encode("utf-8")

    request = urllib.request.Request(url, data=data, method="POST")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload.get("ok"):
        description = payload.get("description", "Unknown Telegram API error")
        raise RuntimeError(f"Telegram API error in {method}: {description}")

    return payload


def get_webhook_url(token: str) -> str | None:
    info = call_telegram_api(token, "getWebhookInfo")
    webhook_url = info.get("result", {}).get("url")
    if isinstance(webhook_url, str) and webhook_url:
        return webhook_url
    return None


def send_telegram_message(token: str, chat_id: int, text: str) -> None:
    call_telegram_api(
        token=token,
        method="sendMessage",
        params={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=30,
    )


def extract_sender_name(message: dict) -> str:
    sender = message.get("from", {})
    return (
        sender.get("username")
        or sender.get("first_name")
        or str(sender.get("id", "unknown"))
    )


def call_ollama_extract(base_url: str, model: str, text: str, timeout: int = 60) -> dict | None:
    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = (
        "Extract an expense from the user text (Spanish or English).\n"
        "Return ONLY valid JSON in this exact shape:\n"
        '{"category":"salida|obligacion|otro","amount":<number or null>}\n'
        "Rules:\n"
        "- Use only explicit numeric amounts found in the user text.\n"
        "- Do not invent numbers.\n"
        "- If there is no clear amount, return amount as null and category as 'otro'.\n"
        f"User text: {text}"
    )
    payload = post_json(
        url=url,
        payload={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )

    raw_response = payload.get("response")
    if not isinstance(raw_response, str):
        return None

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return None

    return normalize_expense(parsed.get("category"), parsed.get("amount"))


def call_ollama_embed(base_url: str, model: str, text: str, timeout: int = 60) -> list[float]:
    normalized_base_url = base_url.rstrip("/")
    embed_url = f"{normalized_base_url}/api/embed"
    try:
        payload = post_json(
            url=embed_url,
            payload={
                "model": model,
                "input": text,
            },
            timeout=timeout,
        )
        embeddings = payload.get("embeddings")
        if (
            isinstance(embeddings, list)
            and embeddings
            and isinstance(embeddings[0], list)
            and embeddings[0]
        ):
            return embeddings[0]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    legacy_payload = post_json(
        url=f"{normalized_base_url}/api/embeddings",
        payload={
            "model": model,
            "prompt": text,
        },
        timeout=timeout,
    )
    embedding = legacy_payload.get("embedding")
    if isinstance(embedding, list) and embedding:
        return embedding

    raise RuntimeError("Ollama embedding response does not contain a valid vector.")
