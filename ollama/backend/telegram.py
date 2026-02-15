#!/usr/bin/env python3
import logging

from ollama.backend.clients import get_webhook_url
from ollama.backend.config import load_config
from ollama.backend.polling import run_long_polling
from ollama.backend.query.service import ExpenseQueryService
from ollama.backend.storage import ExpensePersistence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        config = load_config()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    webhook_url = get_webhook_url(config.token)
    if webhook_url:
        logger.warning(
            "Active webhook detected. For long polling, run:\n"
            "curl -X POST \"https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook\"\n"
        )

    persistence = ExpensePersistence(
        db_path=config.db_path,
        chroma_path=config.chroma_path,
        chroma_collection_name=config.chroma_collection_name,
        ollama_base_url=config.ollama_base_url,
        ollama_embed_model=config.ollama_embed_model,
        default_currency=config.default_currency,
        logger=logger,
    )
    query_service = ExpenseQueryService(
        conn=persistence.conn,
        category_column=persistence.category_column,
        amount_column=persistence.amount_column,
        default_currency=config.default_currency,
    )

    try:
        run_long_polling(
            token=config.token,
            ollama_base_url=config.ollama_base_url,
            ollama_model=config.ollama_extract_model,
            timezone_name=config.timezone_name,
            offset_file=config.offset_file,
            persistence=persistence,
            query_service=query_service,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
        return 0
    finally:
        persistence.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
