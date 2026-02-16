def format_amount(amount: float) -> str:
    value = round(float(amount), 2)
    rendered = f"{value:,.2f}"
    rendered = rendered.replace(",", "_").replace(".", ",").replace("_", ".")
    if rendered.endswith(",00"):
        return rendered[:-3]
    return rendered


def format_money(amount: float, currency: str) -> str:
    return f"{format_amount(amount)} {currency}"
