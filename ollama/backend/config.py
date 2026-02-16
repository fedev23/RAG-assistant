from dataclasses import dataclass
import os
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
DEFAULT_DB_PATH = DATA_DIR / "expenses.db"
DEFAULT_CHROMA_PATH = DATA_DIR / "chroma"
DEFAULT_OFFSET_FILE = Path(__file__).with_name(".telegram_offset")


@dataclass(frozen=True)
class AppConfig:
    token: str
    ollama_base_url: str
    ollama_extract_model: str
    ollama_embed_model: str
    timezone_name: str
    default_currency: str
    db_path: Path
    chroma_path: Path
    chroma_collection_name: str
    offset_file: Path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_config() -> AppConfig:
    dotenv_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")

    return AppConfig(
        token=token,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_extract_model=os.getenv("OLLAMA_EXTRACT_MODEL", "llama3.2:1b"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        timezone_name=os.getenv("APP_TIMEZONE", "UTC"),
        default_currency=os.getenv("APP_CURRENCY", "ARS"),
        db_path=Path(os.getenv("EXPENSES_DB_PATH", str(DEFAULT_DB_PATH))),
        chroma_path=Path(os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_PATH))),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "expenses"),
        offset_file=Path(os.getenv("TELEGRAM_OFFSET_FILE", str(DEFAULT_OFFSET_FILE))),
    )
