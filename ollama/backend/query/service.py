from dataclasses import dataclass
import sqlite3

from ollama.backend.query.parser import MONTH_MAP
from ollama.backend.query.parser import QueryIntent

MONTH_NAME_BY_NUMBER = {value: key for key, value in MONTH_MAP.items() if key != "setiembre"}


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

    def _build_filters(
        self,
        chat_id: int,
        month: int,
        year: int,
        categories: list[str] | None,
    ) -> tuple[str, list]:
        conditions = ["chat_id = ?", "month_key = ?"]
        params: list = [chat_id, f"{year:04d}-{month:02d}"]

        if categories:
            placeholders = ", ".join(["?"] * len(categories))
            conditions.append(f"{self.category_column} IN ({placeholders})")
            params.extend(categories)

        return " AND ".join(conditions), params

    def _render_scope(self, categories: list[str] | None) -> str:
        if not categories:
            return "all categories"
        return ", ".join(categories)

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
                    f"{float(value):.2f} {self.default_currency}."
                ),
            )

        if intent.categories is None:
            salida_total = self._sum_for_category(chat_id, intent.month, year, "salida")
            obligacion_total = self._sum_for_category(chat_id, intent.month, year, "obligacion")
            total = salida_total + obligacion_total
            return QueryResponse(
                handled=True,
                text=(
                    f"Total expense in {month_name} {year}: "
                    f"salida {salida_total:.2f} {self.default_currency} + "
                    f"obligacion {obligacion_total:.2f} {self.default_currency} = "
                    f"{total:.2f} {self.default_currency}."
                ),
            )

        total = self._sum_for_where(where_clause, params)
        return QueryResponse(
            handled=True,
            text=(
                f"Total expense for {scope} in {month_name} {year}: "
                f"{total:.2f} {self.default_currency}."
            ),
        )
