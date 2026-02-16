from dataclasses import dataclass
import sqlite3

from ollama.backend.formatting import format_money
from ollama.backend.query.parser import MONTH_MAP
from ollama.backend.query.parser import QueryIntent

MONTH_NAME_BY_NUMBER = {value: key for key, value in MONTH_MAP.items() if key != "setiembre"}
CATEGORY_VALUES = {
    "salida": ("salida",),
    "obligacion": ("obligacion",),
    "unclear": ("unclear", "otro"),
}
CATEGORY_LABELS = {
    "salida": "salidas",
    "obligacion": "obligaciones",
    "unclear": "unclear",
}


@dataclass(frozen=True)
class QueryResponse:
    handled: bool
    text: str


class ExpenseQueryService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        category_column: str,
        amount_column: str,
        default_currency: str,
    ) -> None:
        self.conn = conn
        self.category_column = category_column
        self.amount_column = amount_column
        self.default_currency = default_currency

    def _resolve_year(self, chat_id: int, month: int, year: int | None) -> int | None:
        if year is not None:
            return year

        row = self.conn.execute(
            """
            SELECT MAX(CAST(substr(month_key, 1, 4) AS INTEGER))
            FROM expenses
            WHERE chat_id = ? AND substr(month_key, 6, 2) = ?
            """,
            (chat_id, f"{month:02d}"),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def _canonical_category(self, category: str) -> str:
        if category in {"otro", "unclear"}:
            return "unclear"
        return category

    def _expand_categories(self, categories: list[str] | None) -> list[str] | None:
        if not categories:
            return None

        expanded: list[str] = []
        for category in categories:
            canonical = self._canonical_category(category)
            values = CATEGORY_VALUES.get(canonical, (canonical,))
            for value in values:
                if value not in expanded:
                    expanded.append(value)

        return expanded

    def _build_filters(
        self,
        chat_id: int,
        month: int,
        year: int,
        categories: list[str] | None,
    ) -> tuple[str, list]:
        conditions = ["chat_id = ?", "month_key = ?"]
        params: list = [chat_id, f"{year:04d}-{month:02d}"]

        expanded_categories = self._expand_categories(categories)
        if expanded_categories:
            placeholders = ", ".join(["?"] * len(expanded_categories))
            conditions.append(f"{self.category_column} IN ({placeholders})")
            params.extend(expanded_categories)

        return " AND ".join(conditions), params

    def _render_scope(self, categories: list[str] | None) -> str:
        if not categories:
            return "all categories"

        labels: list[str] = []
        for category in categories:
            canonical = self._canonical_category(category)
            label = CATEGORY_LABELS.get(canonical, canonical)
            if label not in labels:
                labels.append(label)
        return ", ".join(labels)

    def _sum_for_where(self, where_clause: str, params: list) -> float:
        row = self.conn.execute(
            f"SELECT COALESCE(SUM({self.amount_column}), 0) FROM expenses WHERE {where_clause}",
            params,
        ).fetchone()
        return 0.0 if row is None or row[0] is None else float(row[0])

    def _sum_for_category(self, chat_id: int, month: int, year: int, category: str) -> float:
        where_clause, params = self._build_filters(
            chat_id=chat_id,
            month=month,
            year=year,
            categories=[category],
        )
        return self._sum_for_where(where_clause, params)

    def answer(self, chat_id: int | None, intent: QueryIntent) -> QueryResponse:
        if chat_id is None:
            return QueryResponse(
                handled=True,
                text="I could not identify the chat id for this query.",
            )

        if intent.month is None:
            return QueryResponse(
                handled=True,
                text=(
                    "I can answer that, but I need the month. "
                    "Example: 'How much did I spend in febrero 2026?'"
                ),
            )

        year = self._resolve_year(chat_id, intent.month, intent.year)
        month_name = MONTH_NAME_BY_NUMBER.get(intent.month, str(intent.month))

        if year is None:
            return QueryResponse(
                handled=True,
                text=f"I could not find expenses for {month_name}.",
            )

        where_clause, params = self._build_filters(
            chat_id=chat_id,
            month=intent.month,
            year=year,
            categories=intent.categories,
        )
        scope = self._render_scope(intent.categories)

        if intent.operation == "max":
            row = self.conn.execute(
                f"SELECT MAX({self.amount_column}) FROM expenses WHERE {where_clause}",
                params,
            ).fetchone()
            value = None if row is None else row[0]
            if value is None:
                return QueryResponse(
                    handled=True,
                    text=f"No expenses found for {scope} in {month_name} {year}.",
                )
            return QueryResponse(
                handled=True,
                text=(
                    f"Maximum expense for {scope} in {month_name} {year}: "
                    f"{format_money(float(value), self.default_currency)}."
                ),
            )

        if intent.categories is None:
            salida_total = self._sum_for_category(chat_id, intent.month, year, "salida")
            obligacion_total = self._sum_for_category(chat_id, intent.month, year, "obligacion")
            unclear_total = self._sum_for_category(chat_id, intent.month, year, "unclear")
            total = salida_total + obligacion_total + unclear_total
            return QueryResponse(
                handled=True,
                text=(
                    f"Total expense in {month_name} {year} "
                    f"(salidas + obligaciones + unclear): "
                    f"salidas {format_money(salida_total, self.default_currency)} + "
                    f"obligaciones {format_money(obligacion_total, self.default_currency)} + "
                    f"unclear {format_money(unclear_total, self.default_currency)} = "
                    f"{format_money(total, self.default_currency)}."
                ),
            )

        total = self._sum_for_where(where_clause, params)
        return QueryResponse(
            handled=True,
            text=(
                f"Total expense for {scope} in {month_name} {year}: "
                f"{format_money(total, self.default_currency)}."
            ),
        )
