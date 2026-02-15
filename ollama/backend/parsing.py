import re

RULE_PATTERN = re.compile(
    r"^\s*tipo\s+de\s+gasto\s*:\s*(?P<category>[^,]+?)\s*,\s*gasto\s*:\s*(?P<amount>[0-9]+(?:[.,][0-9]+)?)\s*$",
    re.IGNORECASE,
)


def collapse_spaces(text: str) -> str:
    return " ".join(text.split())


def normalize_category(category: str) -> str:
    value = category.strip().lower()
    if value in {"salida", "salidas"}:
        return "salida"
    if value in {"obligacion", "obligaciones"}:
        return "obligacion"
    return "otro"


def normalize_amount(amount: str | int | float) -> float | None:
    if isinstance(amount, str):
        amount = amount.strip().replace(",", ".")
        try:
            amount = float(amount)
        except ValueError:
            return None
    elif isinstance(amount, (int, float)):
        amount = float(amount)
    else:
        return None

    if amount <= 0:
        return None

    return round(amount, 2)


def normalize_expense(category: str | None, amount: str | int | float | None) -> dict | None:
    if not category or amount is None:
        return None

    normalized_amount = normalize_amount(amount)
    if normalized_amount is None:
        return None

    return {
        "category": normalize_category(category),
        "amount": normalized_amount,
    }


def parse_expense_by_rule(text: str) -> dict | None:
    compact_text = collapse_spaces(text)
    match = RULE_PATTERN.match(compact_text)
    if not match:
        return None

    return normalize_expense(match.group("category"), match.group("amount"))
