# Expense Memory Bot

A conversational expense tracker that combines a Telegram bot, local LLM extraction, SQL accounting, and vector memory.

The goal is simple:
- let you write expenses in natural language,
- store them in a reliable ledger,
- and answer spending questions from chat.

## Project Idea

Most expense trackers force rigid forms. This project keeps the interface conversational:
- you send a normal message,
- the system extracts expense data,
- and your records become queryable over time.

It uses a hybrid memory design:
- **SQLite** for exact financial facts and deterministic calculations,
- **Chroma (vector DB)** for semantic memory (meaning-based similarity).

## How It Works (End-to-End)

1. A user sends a Telegram message.
2. The bot decides whether the message is:
- an expense query (example: "how much did I spend in February?"), or
- an expense entry (example: "I spent 2700 on going out").
3. If it is a query, the system parses intent (month, category, operation) and answers from SQLite.
4. If it is an expense entry:
- it first tries a strict rule-based parse,
- if that fails but still looks like an expense, it asks the LLM to extract `{category, amount}` in JSON.
5. Valid expenses are stored in SQLite as the source of truth.
6. The same expense is transformed into an embedding and upserted into Chroma.
7. The bot replies back in Telegram with a confirmation or the computed answer.

## Why Two Databases

### SQLite: Deterministic Ledger

SQLite stores structured records like:
- category,
- amount,
- currency,
- timestamp,
- chat/user context.

This is where exact totals and max values come from.

Use SQLite when you need:
- precision,
- reproducibility,
- clear auditability.

### Chroma: Semantic Memory

Chroma stores vector embeddings of expense records.

This makes it possible to retrieve records by **meaning**, not only by exact words.  
Example: a future query about "footwear" could match a stored record that said "sneakers."

Use Chroma when you need:
- fuzzy matching,
- natural-language recall,
- context retrieval for RAG-style answers.

## Current Query Behavior

Today, analytical answers are resolved from **SQLite** (sum/max by month/category).

Embeddings are already stored in Chroma, but vector retrieval is not yet the default path for answering user queries in the bot flow.

In other words:
- SQL is active for answers,
- vector memory is already being built for semantic features.

## What the AI Does

The AI is used mainly for extraction fallback:
- turn free text into a normalized expense object,
- avoid inventing numbers by enforcing strict extraction rules.

This keeps ingestion practical while preserving data quality.

## Reliability Concepts

- **Idempotency:** each Telegram `update_id` is unique, preventing duplicate inserts.
- **Offset tracking:** polling progress is persisted so restarts do not reprocess old updates.
- **Normalization:** categories and amounts are standardized before storage.

## Typical User Interactions

Expense entry:
- "I spent 1200 on obligations"
- "Tipo de gasto: salida, gasto: 2700"

Expense query:
- "How much did I spend in febrero 2026?"
- "What was my maximum salida in February?"

## Roadmap Direction

The natural next step is a hybrid answer pipeline:
1. keep SQL for exact math,
2. add Chroma retrieval for semantic context,
3. optionally use an LLM to compose richer natural-language answers grounded in retrieved records.

This gives both:
- accounting accuracy,
- and conversational flexibility.
