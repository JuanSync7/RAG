<!-- @summary
PostgreSQL tuning configuration and LiteLLM spend-log maintenance script shared by the `temporal-db` and `rag-postgres` containers.
@end-summary -->

# ops/postgres

This directory contains the PostgreSQL configuration and a maintenance helper used by the RAG stack. Both files are shared between the `temporal-db` and `rag-postgres` containers.

`postgresql.conf` is mounted read-only at `/etc/postgresql/postgresql.conf` and tunes connection limits, memory allocation, WAL durability, autovacuum aggressiveness, and query logging. Inline comments document recommended values and upgrade thresholds.

`litellm-maintenance.sh` is executed by a dedicated `pg-maintenance` container (Docker Compose `app` profile). It connects to the `litellm` database and deletes `litellmspendlogs` rows older than `LITELLM_LOG_RETENTION_DAYS` (default: 90) on a 24-hour loop — no cron daemon required.

## Contents

| Path | Purpose |
| --- | --- |
| `postgresql.conf` | PostgreSQL tuning: connections, memory, WAL, autovacuum, logging |
| `litellm-maintenance.sh` | Periodic deletion of aged LiteLLM spend-log rows |
