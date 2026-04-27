<!-- @summary
Grafana provisioning configuration and dashboard definitions for the RAG stack. Mounted into the `grafana` container to auto-configure the Prometheus datasource and load the RAG Overview dashboard.
@end-summary -->

# ops/grafana

This directory contains Grafana provisioning config and pre-built dashboard JSON for the RAG monitoring stack. Files are mounted into the `grafana` container so dashboards and datasources are available on first start without manual setup.

## Contents

| Path | Purpose |
| --- | --- |
| `provisioning/datasources/prometheus.yml` | Auto-provisions Prometheus as the default Grafana datasource |
| `provisioning/dashboards/dashboards.yml` | Tells Grafana to load dashboard JSON files from `/var/lib/grafana/dashboards` |
| `dashboards/rag-overview.json` | RAG Overview dashboard: request rate, p95 latency, per-stage latency, and rate-limit rejections |
