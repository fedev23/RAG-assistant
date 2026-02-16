import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from ollama.backend.parsing import collapse_spaces
from ollama.backend.parsing import extract_explicit_amount
from ollama.backend.parsing import normalize_category
from ollama.backend.parsing import normalize_expense
from ollama.backend.tuning import SIMILAR_EXAMPLES_LIMIT
from ollama.backend.tuning import SIMILAR_EXAMPLE_TEXT_MAX_CHARS
from ollama.backend.tuning import OLLAMA_EXTRACT_NUM_PREDICT


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


def _build_similarity_examples_block(similar_examples: list[dict[str, Any]] | None) -> str:
    if not similar_examples:
        return ""

    lines: list[str] = []
    for idx, example in enumerate(similar_examples[:SIMILAR_EXAMPLES_LIMIT], start=1):
        category = str(example.get("category") or "unclear")
        document = str(example.get("document") or "")
        raw_text = document.split("raw_text=", 1)[-1] if "raw_text=" in document else document
        raw_text = collapse_spaces(raw_text)[:SIMILAR_EXAMPLE_TEXT_MAX_CHARS]
        lines.append(
            f"{idx}. cat={category}; txt={raw_text}"
        )

    return "\n".join(lines)


def _parse_json_like_response(raw_response: str) -> dict[str, Any] | None:
    text = raw_response.strip()
    if not text:
        return None

    candidates: list[str] = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]

    return None


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def call_ollama_extract(
    base_url: str,
    model: str,
    text: str,
    similar_examples: list[dict[str, Any]] | None = None,
    timeout: int = 60,
) -> dict | None:
    url = f"{base_url.rstrip('/')}/api/generate"
    similar_examples_block = _build_similarity_examples_block(similar_examples)
    context_line = (
        f"Similar examples:\n{similar_examples_block}\n"
        if similar_examples_block
        else "Similar examples: none\n"
    )
    prompt = (
        "Extract an expense from the user text (Spanish or English).\n"
        "Return ONLY valid JSON in this exact shape:\n"
        '{"category":"salida|obligacion|unclear","amount":<number or null>}\n'
        "Rules:\n"
        "- Use category='salida' for discretionary/leisure spending.\n"
        "- Use category='obligacion' for essential/fixed/debt-related spending.\n"
        "- Use only explicit numeric amounts found in the user text.\n"
        "- Do not invent numbers.\n"
        "- If category is ambiguous, use 'unclear'.\n"
        "- If there is no clear amount, return amount as null and category as 'unclear'.\n"
        "- Use similar examples only as soft category hints, never for amount.\n"
        f"{context_line}"
        f"User text: {text}"
    )
    payload = post_json(
        url=url,
        payload={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_EXTRACT_NUM_PREDICT,
            },
        },
        timeout=timeout,
    )

    raw_response = payload.get("response")
    if not isinstance(raw_response, str):
        return None

    parsed = _parse_json_like_response(raw_response)
    if parsed is None:
        amount = extract_explicit_amount(text)
        if amount is None:
            return None
        return {
            "category": "unclear",
            "amount": amount,
        }

    raw_category = _first_present(parsed, ("category", "categoria", "tipo", "type"))
    raw_amount = _first_present(parsed, ("amount", "monto", "valor", "value"))

    normalized = normalize_expense(
        str(raw_category) if raw_category is not None else None,
        raw_amount,
    )
    if normalized is not None:
        return normalized

    amount = extract_explicit_amount(text)
    if amount is None:
        return None

    return {
        "category": normalize_category(str(raw_category)) if raw_category is not None else "unclear",
        "amount": amount,
    }


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
