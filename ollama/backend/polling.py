import json
import logging
import time
import urllib.error
from pathlib import Path

from ollama.backend.clients import call_ollama_extract
from ollama.backend.clients import call_telegram_api
from ollama.backend.clients import extract_sender_name
from ollama.backend.clients import send_telegram_message
from ollama.backend.formatting import format_money
from ollama.backend.parsing import normalize_category
from ollama.backend.query.parser import parse_query_intent
from ollama.backend.query.service import ExpenseQueryService
from ollama.backend.storage import ExpensePersistence
from ollama.backend.storage import load_offset
from ollama.backend.storage import save_offset
from ollama.backend.tuning import NEIGHBOR_PRIOR_MIN_CONSIDERED_OVERRIDE
from ollama.backend.tuning import NEIGHBOR_PRIOR_MIN_CONSIDERED_UNCLEAR
from ollama.backend.tuning import NEIGHBOR_PRIOR_RATIO_OVERRIDE
from ollama.backend.tuning import NEIGHBOR_PRIOR_RATIO_UNCLEAR
from ollama.backend.tuning import NEIGHBOR_PRIOR_TOP_K
from ollama.backend.tuning import SIMILAR_EXAMPLES_LIMIT
from ollama.backend.time_utils import get_now_iso_and_epoch


def build_expense_event(
    update_id: int,
    chat_id: int | None,
    sender: str,
    text: str,
    expense: dict | None,
    source: str,
    timestamp_iso: str,
    timestamp_epoch: int,
    similar_examples_count: int,
    extraction_error: str | None,
) -> dict:
    return {
        "update_id": update_id,
        "chat_id": chat_id,
        "sender": sender,
        "raw_text": text,
        "source": source,
        "detected_expense": expense,
        "timestamp_iso": timestamp_iso,
        "timestamp_epoch": timestamp_epoch,
        "similar_examples_count": similar_examples_count,
        "extraction_error": extraction_error,
    }


def _safe_reply(token: str, chat_id: int | None, text: str, logger: logging.Logger) -> None:
    if chat_id is None:
        logger.warning("Cannot reply because chat_id is missing.")
        return
    try:
        send_telegram_message(token=token, chat_id=chat_id, text=text)
    except Exception as exc:
        logger.exception("Failed to send Telegram reply: %s", exc)


def _build_expense_reply(expense: dict, persistence_result: dict) -> str:
    if persistence_result.get("status") != "stored":
        return "I could not save this expense."

    amount = float(expense["amount"])
    category = expense["category"]
    currency = persistence_result.get("currency", "ARS")
    month_key = persistence_result.get("month_key", "unknown-month")

    if persistence_result.get("inserted"):
        return (
            f"Expense saved: {category} {format_money(amount, currency)} "
            f"({month_key})."
        )
    return "This message was already processed before."


def _neighbor_majority(
    similar_examples: list[dict],
    top_k: int = NEIGHBOR_PRIOR_TOP_K,
) -> tuple[str | None, float, int]:
    votes: dict[str, int] = {}
    considered = 0

    for example in similar_examples[:top_k]:
        category = example.get("category")
        if not isinstance(category, str):
            continue
        normalized = normalize_category(category)
        votes[normalized] = votes.get(normalized, 0) + 1
        considered += 1

    if considered == 0:
        return None, 0.0, 0

    dominant_category, dominant_count = max(votes.items(), key=lambda item: item[1])
    return dominant_category, dominant_count / considered, considered


def _apply_neighbor_prior(
    expense: dict | None,
    similar_examples: list[dict],
    logger: logging.Logger,
) -> dict | None:
    if expense is None:
        return None

    current = expense.get("category")
    if not isinstance(current, str):
        return expense
    current_category = normalize_category(current)

    dominant_category, dominant_ratio, considered = _neighbor_majority(similar_examples)
    if dominant_category is None or dominant_category == "unclear":
        return expense

    should_adjust = False
    if (
        current_category == "unclear"
        and considered >= NEIGHBOR_PRIOR_MIN_CONSIDERED_UNCLEAR
        and dominant_ratio >= NEIGHBOR_PRIOR_RATIO_UNCLEAR
    ):
        should_adjust = True
    elif (
        current_category != dominant_category
        and considered >= NEIGHBOR_PRIOR_MIN_CONSIDERED_OVERRIDE
        and dominant_ratio >= NEIGHBOR_PRIOR_RATIO_OVERRIDE
    ):
        should_adjust = True

    if not should_adjust:
        return expense

    adjusted = dict(expense)
    adjusted["category"] = dominant_category
    logger.info(
        "Neighbor prior adjusted category %s -> %s (ratio=%.2f, considered=%s)",
        current_category,
        dominant_category,
        dominant_ratio,
        considered,
    )
    return adjusted


def _extract_expense_with_context(
    persistence: ExpensePersistence,
    ollama_base_url: str,
    ollama_model: str,
    chat_id: int | None,
    text: str,
    logger: logging.Logger,
) -> tuple[dict | None, list[dict], str | None]:
    similar_examples: list[dict] = []
    retrieval_error: str | None = None

    try:
        similar_examples = persistence.retrieve_similar_expenses(
            chat_id=chat_id,
            text=text,
            n_results=SIMILAR_EXAMPLES_LIMIT,
        )
    except Exception as exc:
        retrieval_error = f"vector_retrieval_error: {exc}"
        logger.warning("Vector retrieval failed, continuing without context: %s", exc)

    try:
        expense = call_ollama_extract(
            ollama_base_url,
            ollama_model,
            text,
            similar_examples=similar_examples,
        )
        expense = _apply_neighbor_prior(expense=expense, similar_examples=similar_examples, logger=logger)
    except Exception as exc:
        logger.exception("Ollama extraction failed: %s", exc)
        return None, similar_examples, f"llm_error: {exc}"

    if expense is None and retrieval_error is not None:
        return None, similar_examples, retrieval_error

    return expense, similar_examples, None


def run_long_polling(
    token: str,
    ollama_base_url: str,
    ollama_model: str,
    timezone_name: str,
    offset_file: Path,
    persistence: ExpensePersistence,
    query_service: ExpenseQueryService,
    logger: logging.Logger,
) -> None:
    offset = load_offset(offset_file, logger=logger)
    logger.info("Listening for Telegram messages (long polling)...")
    if offset is not None:
        logger.info("Recovered initial offset: %s", offset)

    while True:
        params: dict[str, str | int] = {"timeout": 30}
        if offset is not None:
            params["offset"] = offset

        try:
            updates = call_telegram_api(token, "getUpdates", params=params, timeout=40).get("result", [])
        except urllib.error.URLError as exc:
            logger.warning("Network error: %s. Retrying in 3 seconds...", exc)
            time.sleep(3)
            continue
        except Exception as exc:
            logger.exception("Telegram polling failed: %s. Retrying in 3 seconds...", exc)
            time.sleep(3)
            continue

        for update in updates:
            update_id = update.get("update_id")
            if not isinstance(update_id, int):
                logger.warning("Ignoring update with invalid update_id: %s", update)
                continue
            offset = update_id + 1

            message = update.get("message")
            if not message:
                continue

            chat_id = message.get("chat", {}).get("id")
            sender = extract_sender_name(message)
            text = message.get("text")

            if text is None:
                logger.info("[chat:%s] @%s sent a non-text message", chat_id, sender)
                save_offset(offset_file, offset)
                continue

            intent = parse_query_intent(text, timezone_name=timezone_name)
            if intent is not None:
                query_response = query_service.answer(chat_id=chat_id, intent=intent)
                if query_response.handled:
                    _safe_reply(token=token, chat_id=chat_id, text=query_response.text, logger=logger)

                logger.info(
                    "telegram_query_event=%s",
                    json.dumps(
                        {
                            "update_id": update_id,
                            "chat_id": chat_id,
                            "sender": sender,
                            "raw_text": text,
                            "query_intent": {
                                "operation": intent.operation,
                                "month": intent.month,
                                "year": intent.year,
                                "categories": intent.categories,
                            },
                            "query_response": {
                                "handled": query_response.handled,
                                "text": query_response.text,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
                save_offset(offset_file, offset)
                continue

            expense = None
            source = "llm"
            expense, similar_examples, extraction_error = _extract_expense_with_context(
                persistence=persistence,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
                chat_id=chat_id,
                text=text,
                logger=logger,
            )
            timestamp_iso, timestamp_epoch = get_now_iso_and_epoch(timezone_name, logger=logger)
            event = build_expense_event(
                update_id=update_id,
                chat_id=chat_id,
                sender=sender,
                text=text,
                expense=expense,
                source=source,
                timestamp_iso=timestamp_iso,
                timestamp_epoch=timestamp_epoch,
                similar_examples_count=len(similar_examples),
                extraction_error=extraction_error,
            )

            if expense is not None:
                persistence_result = persistence.store_expense(
                    update_id=update_id,
                    chat_id=chat_id,
                    category=expense["category"],
                    amount=expense["amount"],
                    occurred_at=timestamp_iso,
                    raw_text=text,
                    source=source,
                )
                event["persistence"] = persistence_result
                _safe_reply(
                    token=token,
                    chat_id=chat_id,
                    text=_build_expense_reply(expense=expense, persistence_result=persistence_result),
                    logger=logger,
                )
            else:
                skip_reason = "no_expense_detected" if extraction_error is None else "expense_extraction_error"
                event["persistence"] = {
                    "status": "skipped",
                    "reason": skip_reason,
                }
                _safe_reply(
                    token=token,
                    chat_id=chat_id,
                    text=(
                        "I could not detect an expense from that message."
                        if extraction_error is None
                        else "I had a temporary extraction error. Please try again."
                    ),
                    logger=logger,
                )

            logger.info("telegram_event=%s", json.dumps(event, ensure_ascii=False))
            save_offset(offset_file, offset)
