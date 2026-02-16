from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

MONTH_MAP = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


@dataclass(frozen=True)
class QueryIntent:
    operation: str
    month: int | None
    year: int | None
    categories: list[str] | None


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_text(text: str) -> str:
    base = _strip_accents(text.lower())
    return " ".join(base.split())


def _extract_month(text: str) -> int | None:
    for month_name, month_number in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(month_name)}\b", text):
            return month_number
    return None


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", text)
    if not match:
        return None
    return int(match.group(1))


def _extract_categories(text: str) -> list[str] | None:
    categories: list[str] = []
    if re.search(r"\bsalida(s)?\b", text):
        categories.append("salida")
    if re.search(r"\bobligacion(es)?\b", text):
        categories.append("obligacion")
    if re.search(r"\b(unclear|otro(s)?)\b", text):
        categories.append("unclear")
    if not categories:
        return None
    return categories


def _is_query_candidate(text: str) -> bool:
    # Expense entry examples are handled separately and should not be considered queries.
    if "tipo de gasto" in text and "gasto" in text:
        return False

    analysis_words = (
        "cuanto",
        "cuanta",
        "cual",
        "cuando",
        "total",
        "maximo",
        "max",
        "suma",
        "sumar",
    )
    analysis_phrases = (
        "cada cosa",
        "por categoria",
        "por categorias",
        "desglose",
    )
    expense_context_words = (
        "gasto",
        "gaste",
        "gastado",
        "gastos",
        "salida",
        "salidas",
        "obligacion",
        "obligaciones",
        "unclear",
        "otro",
        "otros",
    )

    has_analysis_hint = any(word in text for word in analysis_words) or any(
        phrase in text for phrase in analysis_phrases
    )
    has_question_mark = text.endswith("?")
    has_relative_month_hint = "mes pasado" in text or "este mes" in text
    has_month_hint = has_relative_month_hint or _extract_month(text) is not None
    has_expense_context = any(word in text for word in expense_context_words)
    return (has_analysis_hint or has_question_mark) and (has_expense_context or has_month_hint)


def _current_month_year(timezone_name: str) -> tuple[int, int]:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz=tz)
    return now.month, now.year


def _previous_month_year(timezone_name: str) -> tuple[int, int]:
    month, year = _current_month_year(timezone_name)
    if month == 1:
        return 12, year - 1
    return month - 1, year


def parse_query_intent(text: str, timezone_name: str) -> QueryIntent | None:
    normalized = _normalize_text(text)
    if not _is_query_candidate(normalized):
        return None

    if "mes pasado" in normalized:
        month, year = _previous_month_year(timezone_name)
    elif "este mes" in normalized:
        month, year = _current_month_year(timezone_name)
    else:
        month = _extract_month(normalized)
        year = _extract_year(normalized)

    if "maximo" in normalized or re.search(r"\bmax\b", normalized):
        operation = "max"
    else:
        operation = "sum"

    categories = _extract_categories(normalized)
    return QueryIntent(
        operation=operation,
        month=month,
        year=year,
        categories=categories,
    )
