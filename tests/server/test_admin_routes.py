# @summary
# Offline route tests for server.routes.admin: API-key + quota admin endpoints.
# Patches src.platform.security backend fns (imported into the admin module
# namespace) and exercises the real require_role role gate for the 403 path.
# Deps: pytest, fastapi.testclient
# @end-summary
"""Tests for the admin API routes.

Strategy:
- Build a tiny FastAPI app mounting only ``create_admin_router()``.
- Override ``authenticate_request`` with a static admin Principal for the
  happy paths, and a non-admin Principal for the role-gate test (which then
  flows through the REAL ``require_role`` to produce a genuine 403).
- Patch the security backend functions (``list_api_keys``, ``create_api_key``,
  ``revoke_api_key``, ``list_quotas``, ``set_tenant_quota``,
  ``delete_tenant_quota``) on ``server.routes.admin`` because admin.py imports
  them into its own namespace.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes import admin as admin_mod
from server.routes.admin import create_admin_router, list_api_keys_handler
from server.schemas import CreateApiKeyRequest
from src.platform.security import Principal, authenticate_request


# ---------------------------------------------------------------------------
# Principals + app builders
# ---------------------------------------------------------------------------

ADMIN = Principal(
    subject="admin-1",
    tenant_id="t",
    roles=["admin"],
    auth_type="none",
    project_id="p",
)
NON_ADMIN = Principal(
    subject="user-1",
    tenant_id="t",
    roles=["query"],
    auth_type="none",
    project_id="p",
)


def _build_client(principal: Principal = ADMIN) -> TestClient:
    app = FastAPI()
    app.include_router(create_admin_router())
    app.dependency_overrides[authenticate_request] = lambda: principal
    return TestClient(app, raise_server_exceptions=False)


def _api_key_record(key_id: str = "k1", *, revoked: bool = False) -> dict:
    return {
        "key_id": key_id,
        "subject": "svc",
        "tenant_id": "t",
        "roles": ["query"],
        "description": "d",
        "created_at": 1000,
        "revoked_at": 9999 if revoked else None,
    }


# ---------------------------------------------------------------------------
# 1. GET /admin/api-keys
# ---------------------------------------------------------------------------


def test_list_api_keys_maps_records_and_default_include_revoked_false():
    with mock.patch.object(
        admin_mod, "list_api_keys", return_value=[_api_key_record("ka"), _api_key_record("kb")]
    ) as m:
        client = _build_client()
        resp = client.get("/admin/api-keys")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["key_id"] for r in body] == ["ka", "kb"]
    # default query param value forwarded
    m.assert_called_once_with(include_revoked=False)


def test_list_api_keys_forwards_include_revoked_true():
    with mock.patch.object(
        admin_mod, "list_api_keys", return_value=[_api_key_record("ka", revoked=True)]
    ) as m:
        client = _build_client()
        resp = client.get("/admin/api-keys", params={"include_revoked": True})
    assert resp.status_code == 200
    m.assert_called_once_with(include_revoked=True)
    assert resp.json()[0]["revoked_at"] == 9999


# ---------------------------------------------------------------------------
# 2. POST /admin/api-keys
# ---------------------------------------------------------------------------


def test_create_api_key_forwards_request_fields_and_returns_response():
    created = {**_api_key_record("new"), "api_key": "secret-token"}
    with mock.patch.object(admin_mod, "create_api_key", return_value=created) as m:
        client = _build_client()
        resp = client.post(
            "/admin/api-keys",
            json={
                "subject": "svc-acct",
                "tenant_id": "tenant-9",
                "roles": ["query", "admin"],
                "description": "ci key",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["api_key"] == "secret-token"
    assert resp.json()["key_id"] == "new"
    m.assert_called_once_with(
        subject="svc-acct",
        tenant_id="tenant-9",
        roles=["query", "admin"],
        description="ci key",
    )


# ---------------------------------------------------------------------------
# 3. DELETE /admin/api-keys/{key_id}
# ---------------------------------------------------------------------------


def test_revoke_api_key_found_returns_revoked_status():
    with mock.patch.object(admin_mod, "revoke_api_key", return_value=True) as m:
        client = _build_client()
        resp = client.delete("/admin/api-keys/key-7")
    assert resp.status_code == 200
    assert resp.json() == {"status": "revoked", "key_id": "key-7", "tenant_id": None}
    m.assert_called_once_with("key-7")


def test_revoke_api_key_missing_returns_404():
    with mock.patch.object(admin_mod, "revoke_api_key", return_value=False):
        client = _build_client()
        resp = client.delete("/admin/api-keys/nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. GET /admin/quotas
# ---------------------------------------------------------------------------


def test_list_quotas_returns_quotas_response():
    payload = {
        "defaults": {"requests_per_minute": 60},
        "tenants": {"t": {"requests_per_minute": 120}},
        "projects": {},
    }
    with mock.patch.object(admin_mod, "list_quotas", return_value=payload) as m:
        client = _build_client()
        resp = client.get("/admin/quotas")
    assert resp.status_code == 200
    assert resp.json() == payload
    m.assert_called_once_with()


# ---------------------------------------------------------------------------
# 5. PUT /admin/quotas/{tenant_id}
# ---------------------------------------------------------------------------


def test_set_tenant_quota_forwards_args_and_returns_response():
    ret = {"tenant_id": "tenant-3", "requests_per_minute": 250}
    with mock.patch.object(admin_mod, "set_tenant_quota", return_value=ret) as m:
        client = _build_client()
        resp = client.put("/admin/quotas/tenant-3", json={"requests_per_minute": 250})
    assert resp.status_code == 200
    assert resp.json() == ret
    m.assert_called_once_with("tenant-3", 250)


# ---------------------------------------------------------------------------
# 6. DELETE /admin/quotas/{tenant_id}
# ---------------------------------------------------------------------------


def test_delete_tenant_quota_existed_returns_deleted():
    with mock.patch.object(admin_mod, "delete_tenant_quota", return_value=True) as m:
        client = _build_client()
        resp = client.delete("/admin/quotas/tenant-5")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "key_id": None, "tenant_id": "tenant-5"}
    m.assert_called_once_with("tenant-5")


def test_delete_tenant_quota_missing_returns_noop():
    with mock.patch.object(admin_mod, "delete_tenant_quota", return_value=False):
        client = _build_client()
        resp = client.delete("/admin/quotas/ghost")
    assert resp.status_code == 200
    assert resp.json() == {"status": "noop", "key_id": None, "tenant_id": "ghost"}


# ---------------------------------------------------------------------------
# 7. Role gate via the REAL require_role (non-admin principal -> 403)
# ---------------------------------------------------------------------------


def test_non_admin_principal_is_rejected_403_on_list_api_keys():
    # No backend patch needed: require_role must raise before any delegation.
    with mock.patch.object(admin_mod, "list_api_keys", return_value=[]) as m:
        client = _build_client(principal=NON_ADMIN)
        resp = client.get("/admin/api-keys")
    assert resp.status_code == 403
    assert "Missing role: admin" in resp.json()["detail"]
    m.assert_not_called()


def test_non_admin_principal_is_rejected_403_on_delete_api_key():
    with mock.patch.object(admin_mod, "revoke_api_key", return_value=True) as m:
        client = _build_client(principal=NON_ADMIN)
        resp = client.delete("/admin/api-keys/key-7")
    assert resp.status_code == 403
    m.assert_not_called()


def test_non_admin_principal_is_rejected_403_on_delete_quota():
    with mock.patch.object(admin_mod, "delete_tenant_quota", return_value=True) as m:
        client = _build_client(principal=NON_ADMIN)
        resp = client.delete("/admin/quotas/t")
    assert resp.status_code == 403
    m.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Direct module-level handler coverage (no router)
# ---------------------------------------------------------------------------


def test_list_api_keys_handler_direct_admin():
    with mock.patch.object(
        admin_mod, "list_api_keys", return_value=[_api_key_record("kx")]
    ) as m:
        out = asyncio.run(list_api_keys_handler(True, ADMIN))
    assert [r.key_id for r in out] == ["kx"]
    m.assert_called_once_with(include_revoked=True)


def test_list_api_keys_handler_direct_non_admin_raises_403():
    from fastapi import HTTPException

    with mock.patch.object(admin_mod, "list_api_keys", return_value=[]):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(list_api_keys_handler(False, NON_ADMIN))
    assert ei.value.status_code == 403


def test_create_api_key_handler_direct_forwards_subject():
    created = {**_api_key_record("h"), "api_key": "tok"}
    req = CreateApiKeyRequest(
        subject="direct-subj", tenant_id="td", roles=["query"], description="x"
    )
    with mock.patch.object(admin_mod, "create_api_key", return_value=created) as m:
        out = asyncio.run(admin_mod.create_api_key_handler(req, ADMIN))
    assert out.api_key == "tok"
    m.assert_called_once_with(
        subject="direct-subj", tenant_id="td", roles=["query"], description="x"
    )
