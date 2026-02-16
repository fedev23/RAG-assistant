#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import sys
import unicodedata
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from ollama.backend.clients import call_ollama_embed
from ollama.backend.parsing import collapse_spaces

DEFAULT_CSV_PATH = Path(__file__).with_name("rule.csv")
DEFAULT_CHROMA_PATH = Path(
    os.getenv(
        "CHROMA_PERSIST_DIR",
        str(Path(__file__).resolve().parents[1] / "backend" / "data" / "chroma"),
    )
)
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "expenses")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CURRENCY = os.getenv("APP_CURRENCY", "ARS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a labeled CSV into Chroma for better expense classification."
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help=f"CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--chat-id",
        type=int,
        required=True,
        help="Target chat_id used by bot retrieval.",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help=f"Chroma persist dir (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embed model (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate collection before ingesting.",
    )
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", collapse_spaces(value).lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_label(raw: str) -> str:
    value = _normalize_text(raw)
    if "oblig" in value:
        return "obligacion"
    if "ocio" in value or "salida" in value or "no oblig" in value:
        return "salida"
    return "unclear"


def _resolve_column(fieldnames: list[str], expected_normalized: str) -> str | None:
    for name in fieldnames:
        if _normalize_text(name) == expected_normalized:
            return name
    return None


def _build_id(csv_name: str, row_index: int, text: str, category: str) -> str:
    key = f"{csv_name}:{row_index}:{category}:{text}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()  # nosec: deterministic id for upsert
    return f"train:{csv_name}:{row_index}:{digest[:12]}"


def run() -> int:
    args = parse_args()

    try:
        import chromadb
    except ModuleNotFoundError:
        print("[error] Missing dependency 'chromadb'.", file=sys.stderr)
        return 1

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        label_column = _resolve_column(fieldnames, "categoria")
        text_column = _resolve_column(fieldnames, "descripcion")
        if label_column is None or text_column is None:
            print(
                "[error] CSV must include headers 'Categoría' and 'Descripción'.",
                file=sys.stderr,
            )
            return 1

        chroma_path = Path(args.chroma_path).expanduser().resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        if args.reset_collection:
            try:
                client.delete_collection(name=args.collection)
            except Exception:
                pass
        collection = client.get_or_create_collection(name=args.collection)

        inserted = 0
        skipped = 0
        for row_index, row in enumerate(reader, start=2):
            raw_text = str(row.get(text_column) or "")
            text = collapse_spaces(raw_text)
            if not text:
                skipped += 1
                continue

            raw_label = str(row.get(label_column) or "")
            category = _normalize_label(raw_label)
            doc_id = _build_id(csv_path.name, row_index, text, category)
            document = (
                f"expense_id=training-{row_index}; categoria={category}; monto=0.00 {DEFAULT_CURRENCY}; "
                f"occurred_at=training; month_key=training; source=training_csv:{csv_path.name}; raw_text={text}"
            )

            embedding = call_ollama_embed(
                base_url=args.base_url,
                model=args.embed_model,
                text=document,
            )
            metadata = {
                "expense_id": -row_index,
                "chat_id": args.chat_id,
                "categoria": category,
                "monto": 0.0,
                "currency": DEFAULT_CURRENCY,
                "mes": 0,
                "year": 0,
                "month_key": "training",
                "source": f"training_csv:{csv_path.name}",
                "csv_line": row_index,
            }
            collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            inserted += 1

    print(
        f"[done] inserted={inserted} skipped={skipped} "
        f"collection={args.collection} chat_id={args.chat_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
