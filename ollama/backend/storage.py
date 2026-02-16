from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Any

import chromadb

from ollama.backend.clients import call_ollama_embed
from ollama.backend.parsing import collapse_spaces
from ollama.backend.tuning import SIMILAR_EXAMPLES_LIMIT


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def load_offset(path: Path, logger: logging.Logger) -> int | None:
    if not path.exists():
        return None

    raw_value = path.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None

    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid offset value in %s. Restarting from scratch.", path)
        return None


def save_offset(path: Path, offset: int) -> None:
    ensure_parent_dir(path)
    path.write_text(str(offset), encoding="utf-8")


class ExpensePersistence:
    def __init__(
        self,
        db_path: Path,
        chroma_path: Path,
        chroma_collection_name: str,
        ollama_base_url: str,
        ollama_embed_model: str,
        default_currency: str,
        logger: logging.Logger,
    ) -> None:
        ensure_parent_dir(db_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        self.logger = logger
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(name=chroma_collection_name)

        self.ollama_base_url = ollama_base_url
        self.ollama_embed_model = ollama_embed_model
        self.default_currency = default_currency
        self._vector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector-upsert")

        self.logger.info("SQLite initialized at %s", db_path)
        self.logger.info("Chroma initialized at %s (collection=%s)", chroma_path, chroma_collection_name)

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_id INTEGER NOT NULL UNIQUE,
                chat_id INTEGER NOT NULL,
                categoria TEXT NOT NULL,
                monto REAL NOT NULL,
                currency TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                month_key TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(expenses)").fetchall()
        }
        if {"categoria", "monto"}.issubset(columns):
            self.category_column = "categoria"
            self.amount_column = "monto"
        elif {"category", "amount"}.issubset(columns):
            self.category_column = "category"
            self.amount_column = "amount"
            self.logger.warning(
                "Legacy SQL schema detected (category/amount). New rows will use legacy columns."
            )
        else:
            raise RuntimeError(f"Unsupported expenses schema. Columns found: {sorted(columns)}")

        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_expenses_chat_month_category
            ON expenses(chat_id, month_key, {self.category_column})
            """
        )
        self.conn.commit()

    def _insert_expense(
        self,
        update_id: int,
        chat_id: int,
        category: str,
        amount: float,
        currency: str,
        occurred_at: str,
        month_key: str,
        raw_text: str,
        source: str,
    ) -> tuple[int, bool]:
        created_at = datetime.utcnow().isoformat()
        try:
            cursor = self.conn.execute(
                f"""
                INSERT INTO expenses (
                    update_id, chat_id, {self.category_column}, {self.amount_column}, currency, occurred_at,
                    month_key, raw_text, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    update_id,
                    chat_id,
                    category,
                    amount,
                    currency,
                    occurred_at,
                    month_key,
                    raw_text,
                    source,
                    created_at,
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid), True
        except sqlite3.IntegrityError:
            row = self.conn.execute(
                "SELECT id FROM expenses WHERE update_id = ?",
                (update_id,),
            ).fetchone()
            if row is None:
                raise
            return int(row[0]), False

    def _upsert_vector(
        self,
        expense_id: int,
        chat_id: int,
        category: str,
        amount: float,
        currency: str,
        occurred_at: str,
        month_key: str,
        source: str,
        raw_text: str,
    ) -> None:
        occurred_dt = datetime.fromisoformat(occurred_at)
        metadata = {
            "expense_id": expense_id,
            "chat_id": chat_id,
            "categoria": category,
            "monto": amount,
            "currency": currency,
            "mes": occurred_dt.month,
            "year": occurred_dt.year,
            "month_key": month_key,
            "source": source,
        }
        normalized_text = collapse_spaces(raw_text)
        document = (
            f"expense_id={expense_id}; categoria={category}; monto={amount:.2f} {currency}; "
            f"occurred_at={occurred_at}; month_key={month_key}; raw_text={normalized_text}"
        )
        embedding = call_ollama_embed(
            base_url=self.ollama_base_url,
            model=self.ollama_embed_model,
            text=document,
        )
        self.collection.upsert(
            ids=[str(expense_id)],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def _upsert_vector_background(
        self,
        expense_id: int,
        chat_id: int,
        category: str,
        amount: float,
        currency: str,
        occurred_at: str,
        month_key: str,
        source: str,
        raw_text: str,
    ) -> None:
        try:
            self._upsert_vector(
                expense_id=expense_id,
                chat_id=chat_id,
                category=category,
                amount=amount,
                currency=currency,
                occurred_at=occurred_at,
                month_key=month_key,
                source=source,
                raw_text=raw_text,
            )
        except Exception as exc:
            self.logger.exception("Chroma upsert failed for expense_id=%s: %s", expense_id, exc)

    def retrieve_similar_expenses(
        self,
        chat_id: int | None,
        text: str,
        n_results: int = SIMILAR_EXAMPLES_LIMIT,
    ) -> list[dict[str, Any]]:
        if chat_id is None:
            return []

        normalized_text = collapse_spaces(text)
        if not normalized_text:
            return []

        query_embedding = call_ollama_embed(
            base_url=self.ollama_base_url,
            model=self.ollama_embed_model,
            text=normalized_text,
        )
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, n_results),
            where={"chat_id": chat_id},
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])
        metadatas = result.get("metadatas", [[]])
        distances = result.get("distances", [[]])
        if not documents or not isinstance(documents[0], list):
            return []

        doc_list = documents[0]
        metadata_list = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
        distance_list = distances[0] if distances and isinstance(distances[0], list) else []

        similar_items: list[dict[str, Any]] = []
        for idx, document in enumerate(doc_list):
            if not isinstance(document, str):
                continue

            metadata = metadata_list[idx] if idx < len(metadata_list) else {}
            if not isinstance(metadata, dict):
                metadata = {}

            distance = distance_list[idx] if idx < len(distance_list) else None
            if not isinstance(distance, (int, float)):
                distance = None

            similar_items.append(
                {
                    "document": document,
                    "category": metadata.get("categoria"),
                    "amount": metadata.get("monto"),
                    "distance": distance,
                    "month_key": metadata.get("month_key"),
                }
            )

        return similar_items

    def store_expense(
        self,
        update_id: int,
        chat_id: int | None,
        category: str,
        amount: float,
        occurred_at: str,
        raw_text: str,
        source: str,
    ) -> dict[str, Any]:
        if chat_id is None:
            return {
                "status": "skipped",
                "reason": "missing_chat_id",
            }

        occurred_dt = datetime.fromisoformat(occurred_at)
        month_key = occurred_dt.strftime("%Y-%m")
        expense_id, inserted = self._insert_expense(
            update_id=update_id,
            chat_id=chat_id,
            category=category,
            amount=amount,
            currency=self.default_currency,
            occurred_at=occurred_at,
            month_key=month_key,
            raw_text=raw_text,
            source=source,
        )

        vector_status = "queued"
        try:
            self._vector_executor.submit(
                self._upsert_vector_background,
                expense_id=expense_id,
                chat_id=chat_id,
                category=category,
                amount=amount,
                currency=self.default_currency,
                occurred_at=occurred_at,
                month_key=month_key,
                source=source,
                raw_text=raw_text,
            )
        except Exception as exc:
            vector_status = "error"
            self.logger.exception("Unable to queue Chroma upsert for expense_id=%s: %s", expense_id, exc)

        return {
            "status": "stored",
            "expense_id": expense_id,
            "inserted": inserted,
            "vector_status": vector_status,
            "month_key": month_key,
            "currency": self.default_currency,
        }

    def close(self) -> None:
        try:
            self._vector_executor.shutdown(wait=True)
        finally:
            self.conn.close()
