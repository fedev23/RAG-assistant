import json
import logging
import re
import time
import urllib.error
from pathlib import Path

from ollama.backend.clients import call_ollama_extract
from ollama.backend.clients import call_telegram_api
from ollama.backend.clients import extract_sender_name
from ollama.backend.clients import send_telegram_message
from ollama.backend.parsing import parse_expense_by_rule
from ollama.backend.query.parser import parse_query_intent
from ollama.backend.query.service import ExpenseQueryService
from ollama.backend.storage import ExpensePersistence
from ollama.backend.storage import load_offset
from ollama.backend.storage import save_offset
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
            f"Expense saved: {category} {amount:.2f} {currency} "
            f"({month_key})."
        )
    return "This message was already processed before."


def _looks_like_expense_message(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    has_number = bool(re.search(r"\d", normalized))
    has_expense_hint = bool(
        re.search(
            r"\b(gasto|gaste|gastado|salida|salidas|obligacion|obligaciones|pagu[eé]|pago|compr[eé])\b",
            normalized,
        )
    )
    return has_number and has_expense_hint


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

            expense = parse_expense_by_rule(text)
            source = "rule"
            if expense is None:
                if _looks_like_expense_message(text):
                    try:
                        expense = call_ollama_extract(ollama_base_url, ollama_model, text)
                        source = "llm"
                    except Exception as exc:
                        logger.exception("Ollama extraction failed: %s", exc)
                        source = "llm_error"
                else:
                    source = "not_expense_candidate"

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
                event["persistence"] = {
                    "status": "skipped",
                    "reason": "no_expense_detected",
                }
                _safe_reply(
                    token=token,
                    chat_id=chat_id,
                    text=(
                        "I could not detect an expense from that message. "
                        "You can register one like: 'Gaste 2700 en salidas' "
                        "or ask: 'cuanto gaste en salidas en febrero'."
                    ),
                    logger=logger,
                )

            logger.info("telegram_event=%s", json.dumps(event, ensure_ascii=False))
            save_offset(offset_file, offset)
