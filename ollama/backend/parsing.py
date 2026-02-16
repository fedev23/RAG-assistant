import re
import unicodedata


def collapse_spaces(text: str) -> str:
    return " ".join(text.split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_category(category: str) -> str:
    value = _strip_accents(collapse_spaces(category).lower())
    if value in {
        "salida",
        "salidas",
        "ocio",
        "entretenimiento",
        "recreacion",
        "recreativo",
        "hobby",
        "hobbies",
        "discrecional",
        "no esencial",
        "leisure",
    }:
        return "salida"
    if value in {
        "obligacion",
        "obligaciones",
        "obligatorio",
        "obligatoria",
        "esencial",
        "fijo",
        "fija",
        "deuda",
    }:
        return "obligacion"
    if value in {"unclear", "otro", "otros", "ambigua", "ambiguo", "indefinido"}:
        return "unclear"
    return "unclear"


def normalize_amount(amount: str | int | float) -> float | None:
    if isinstance(amount, str):
        raw_amount = _strip_accents(collapse_spaces(amount).lower())
        multiplier = 1.0

        if re.search(r"(?<=\d)\s*(mil|k)\b", raw_amount):
            multiplier = 1000.0
            raw_amount = re.sub(r"(?<=\d)\s*(mil|k)\b", "", raw_amount)

        raw_amount = re.sub(r"\b(ars|usd|peso|pesos|dolar|dolares)\b", "", raw_amount)
        amount = re.sub(r"[^\d,.\-]", "", raw_amount)
        if not amount or amount in {"-", ".", ","}:
            return None

        has_comma = "," in amount
        has_dot = "." in amount
        if has_comma and has_dot:
            if amount.rfind(",") > amount.rfind("."):
                amount = amount.replace(".", "").replace(",", ".")
            else:
                amount = amount.replace(",", "")
        elif has_comma:
            parts = amount.split(",")
            if len(parts) == 2:
                integer_part, fraction_part = parts
                if len(fraction_part) <= 2:
                    amount = amount.replace(",", ".")
                elif len(fraction_part) == 3 and len(integer_part) <= 3:
                    amount = amount.replace(",", "")
                else:
                    amount = amount.replace(",", ".")
            else:
                amount = amount.replace(",", "")
        elif has_dot:
            parts = amount.split(".")
            if len(parts) == 2:
                integer_part, fraction_part = parts
                if len(fraction_part) <= 2:
                    pass
                elif len(fraction_part) == 3 and len(integer_part) <= 3:
                    amount = amount.replace(".", "")
                else:
                    pass
            else:
                amount = amount.replace(".", "")

        try:
            amount = float(amount)
        except ValueError:
            return None
        amount *= multiplier
    elif isinstance(amount, (int, float)):
        amount = float(amount)
    else:
        return None

    if amount <= 0:
        return None

    return round(amount, 2)


def normalize_expense(category: str | None, amount: str | int | float | None) -> dict | None:
    if amount is None:
        return None

    normalized_amount = normalize_amount(amount)
    if normalized_amount is None:
        return None

    normalized_category = normalize_category(category or "unclear")

    return {
        "category": normalized_category,
        "amount": normalized_amount,
    }


def extract_explicit_amount(text: str) -> float | None:
    compact = _strip_accents(collapse_spaces(text).lower())

    mil_matches = re.findall(
        r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\s*(?:mil|k)\b|\d+(?:[.,]\d+)?\s*(?:mil|k)\b",
        compact,
    )
    if mil_matches:
        parsed_mil = normalize_amount(mil_matches[0])
        if parsed_mil is not None:
            return parsed_mil

    matches = re.findall(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?", compact)
    parsed_candidates = [value for value in (normalize_amount(token) for token in matches) if value is not None]
    if not parsed_candidates:
        return None
    return max(parsed_candidates)
