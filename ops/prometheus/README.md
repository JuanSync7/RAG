<!-- @summary
Prometheus scrape configuration and alerting rules for the RAG stack. Both files are mounted into the `prometheus` container.
@end-summary -->

# ops/prometheus

This directory contains the Prometheus configuration for the RAG monitoring stack. Files are mounted read-only into the `prometheus` container.

`prometheus.yml` sets a 15-second global scrape interval, configures the `rag-api` scrape target, and points alert rule evaluation at `alerts.yml` and notification routing at the `alertmanager` container on port 9093.

`alerts.yml` defines two warning-severity alerting rules: `RagApiHighErrorRate` (HTTP 500 rate exceeding 0.1 req/s over 5 minutes) and `RagApiLatencyP95High` (p95 latency above 5 seconds over 10 minutes).

## Contents

| Path | Purpose |
| --- | --- |
| `prometheus.yml` | Global scrape config, alertmanager target, and rule file references |
| `alerts.yml` | Alerting rules: error rate and p95 latency thresholds for `rag-api` |
