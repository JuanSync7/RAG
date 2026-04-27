<!-- @summary
nginx reverse proxy configuration for the RAG stack. Terminates TLS and forwards HTTPS traffic to the `rag-api` service, with SSE and WebSocket support.
@end-summary -->

# ops/nginx

This directory contains the nginx configuration used by the `rag-nginx` container (enabled via the `gateway` Docker Compose profile). The file is mounted read-only into the container at `/etc/nginx/conf.d/default.conf`.

The configuration redirects all HTTP traffic to HTTPS and proxies requests to the `rag-api` service on port 8000. It uses locally generated TLS certificates (see `scripts/generate-certs.sh`), disables proxy buffering for streaming/SSE responses, and sets generous timeouts suitable for LLM inference.

## Contents

| Path | Purpose |
| --- | --- |
| `nginx.conf` | HTTPS reverse proxy config: TLS termination, SSE/WebSocket headers, proxy rules |
