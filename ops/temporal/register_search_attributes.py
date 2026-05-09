#!/usr/bin/env python3
# @summary
# One-time admin script to register Deep Research custom search attributes
# (DRIterations, DRLLMCalls, DREarlyStopped, DRTopicCount, DREnabled) with
# the Temporal cluster. Idempotent — already-registered attributes are
# treated as success. Run once per cluster / namespace before deploying
# the worker, or after restoring a fresh dev cluster.
# Exports: register_dr_search_attributes (async), main
# Deps: temporalio, server.search_attributes, config.settings
# @end-summary
"""Register Deep Research custom search attributes on the Temporal cluster.

Usage::

    python -m ops.temporal.register_search_attributes
    # or with explicit overrides:
    RAG_TEMPORAL_TARGET_HOST=localhost:7233 \\
      RAG_TEMPORAL_NAMESPACE=default \\
      python -m ops.temporal.register_search_attributes

The Temporal Operator Service exposes ``AddSearchAttributes`` which is the
canonical way to declare custom attributes to the cluster. Without this step
``workflow.upsert_search_attributes`` will fail at runtime with an "unknown
search attribute" error.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from temporalio.api.operatorservice.v1 import AddSearchAttributesRequest
from temporalio.client import Client
from temporalio.common import SearchAttributeIndexedValueType as IVT

from server.search_attributes import ALL_DR_KEYS

logger = logging.getLogger("ops.temporal.register_search_attributes")


async def register_dr_search_attributes(
    target_host: str,
    namespace: str = "default",
) -> dict[str, str]:
    """Register the DR search-attribute keys. Returns a name → status map.

    ``status`` is one of ``"created"`` or ``"already_exists"``. Other errors
    bubble up.
    """
    client = await Client.connect(target_host, namespace=namespace)

    attrs = {key.name: IVT(key.indexed_value_type) for key in ALL_DR_KEYS}
    logger.info(
        "Registering %d DR search attributes on namespace=%s host=%s: %s",
        len(attrs), namespace, target_host, sorted(attrs.keys()),
    )

    request = AddSearchAttributesRequest(
        search_attributes={n: v.value for n, v in attrs.items()},
        namespace=namespace,
    )

    statuses: dict[str, str] = {}
    try:
        await client.operator_service.add_search_attributes(request)
        for name in attrs:
            statuses[name] = "created"
            logger.info("Registered: %s", name)
    except Exception as exc:  # noqa: BLE001
        # AlreadyExists comes back as an RPC error containing the substring
        # "already exists". Treat that as success — the script is meant to be
        # idempotent so deploys can re-run it freely.
        msg = str(exc).lower()
        if "already exists" in msg or "alreadyexists" in msg:
            for name in attrs:
                statuses[name] = "already_exists"
            logger.info("Search attributes already registered (idempotent no-op)")
        else:
            raise
    return statuses


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    target = os.environ.get("RAG_TEMPORAL_TARGET_HOST", "localhost:7233")
    namespace = os.environ.get("RAG_TEMPORAL_NAMESPACE", "default")
    statuses = asyncio.run(register_dr_search_attributes(target, namespace))
    for name, status in sorted(statuses.items()):
        print(f"{name}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
