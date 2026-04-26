# @summary
# Route module exports for query, admin, system, and document management API routers.
# Exports: create_query_router, run_query, create_admin_router, create_system_router,
#          build_health_response, create_documents_router
# Deps: server.routes.query, server.routes.admin, server.routes.system, server.routes.documents
# @end-summary
"""Server route package exports."""

from server.routes.admin import create_admin_router
from server.routes.documents import create_documents_router
from server.routes.ingest import create_ingest_router
from server.routes.query import create_query_router, run_query
from server.routes.system import build_health_response, create_system_router

__all__ = [
    "create_query_router",
    "run_query",
    "create_admin_router",
    "create_system_router",
    "build_health_response",
    "create_documents_router",
    "create_ingest_router",
]

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from server.routes.admin import (
    create_api_key_handler,
    list_api_keys_handler,
    list_quotas_handler,
)
