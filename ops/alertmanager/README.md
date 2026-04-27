<!-- @summary
Alertmanager configuration for the RAG stack. Defines alert routing rules and receivers, and is mounted into the `alertmanager` container.
@end-summary -->

# ops/alertmanager

This directory contains the Alertmanager configuration used by the RAG monitoring stack. The file is mounted read-only into the `alertmanager` container at `/etc/alertmanager/alertmanager.yml`.

The current configuration routes all alerts to a default (no-op) receiver, serving as a baseline to be extended with real notification channels (e.g. Slack, PagerDuty, email).

## Contents

| Path | Purpose |
| --- | --- |
| `alertmanager.yml` | Alert routing tree and receiver definitions |
