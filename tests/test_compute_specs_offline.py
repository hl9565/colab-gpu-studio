from __future__ import annotations

import pytest
from app import main
from app.common import APIError, current_user, get_store
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.google import GoogleConfigurationError, GoogleError, GoogleOAuthError, GoogleService, GoogleTransientError
from app.gpu.repository import Repository
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = Store(f"sqlite:///{tmp_path / 's.sqlite'}")
    monkeypatch.setitem(main.app.dependency_overrides, get_store, lambda: store)
    monkeypatch.setitem(main.app.dependency_overrides, current_user, lambda: {"id": "u1", "role": "user"})
    test_client = TestClient(main.app, raise_server_exceptions=False)
    try:
        yield test_client
    finally:
        test_client.close()
        store.engine.dispose()


def test_colab_specs_offline_is_clear_409_not_500(monkeypatch, client):
    monkeypatch.setenv("STUDIO_OFFLINE_TEST", "1")
    r = client.get("/api/gpu/compute/specs?provider=colab")
    assert r.status_code == 409 and r.json()["error"]["code"] == "OFFLINE_MODE"


def test_colab_specs_real_disabled_is_clear_409(monkeypatch, client):
    monkeypatch.delenv("STUDIO_OFFLINE_TEST", raising=False)
    monkeypatch.setenv("GPU_REAL_ENABLED", "0")
    r = client.get("/api/gpu/compute/specs?provider=colab")
    assert r.status_code == 409 and r.json()["error"]["code"] == "REAL_GPU_DISABLED"


def test_mock_specs_still_work(client):
    assert client.get("/api/gpu/compute/specs?provider=mock").status_code == 200


@pytest.mark.parametrize(("error", "status", "code", "retryable", "message"), [
    (GoogleOAuthError("授权已失效"), 422, "GOOGLE_OAUTH_ERROR", False, "授权已失效"),
    (GoogleConfigurationError("配置不可用"), 422, "GOOGLE_CONFIG_ERROR", False, "配置不可用"),
    (GoogleTransientError("暂时不可用"), 502, "COLAB_CHECK_FAILED", True, "检查 Colab GPU 失败：暂时不可用"),
    (ColabError("响应异常"), 502, "COLAB_CHECK_FAILED", True, "检查 Colab GPU 失败：响应异常"),
    (APIError("LIMIT_REACHED", "已达到上限", 409), 409, "LIMIT_REACHED", False, "已达到上限"),
    (RuntimeError("private detail"), 500, "INTERNAL_ERROR", False, "后台操作失败，请查看已脱敏的任务事件"),
])
def test_colab_specs_returns_production_error_envelope(monkeypatch, client, error, status, code, retryable, message):
    monkeypatch.delenv("STUDIO_OFFLINE_TEST", raising=False)
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")

    def fail(_provider):
        raise error

    monkeypatch.setattr(GoogleColabProvider, "list_specs", fail)
    response = client.get("/api/gpu/compute/specs?provider=colab")
    assert response.status_code == status
    payload = response.json()
    assert payload["error"] == {"code": code, "retryable": retryable, "message": message}
    assert payload["data"] is None and payload["request_id"]
    assert payload["code"] == code and payload["message"] == message


@pytest.mark.parametrize(("error_type", "status", "code", "retryable"), [
    (GoogleOAuthError, 422, "GOOGLE_OAUTH_ERROR", False),
    (GoogleConfigurationError, 422, "GOOGLE_CONFIG_ERROR", False),
    (GoogleTransientError, 502, "GOOGLE_TRANSIENT_ERROR", True),
    (GoogleError, 422, "GOOGLE_REQUEST_FAILED", False),
])
def test_global_google_error_handler(monkeypatch, client, error_type, status, code, retryable):
    def fail(_provider):
        raise error_type("安全文案")

    monkeypatch.setattr(GoogleColabProvider, "subscription", fail)
    response = client.get("/api/gpu/compute/subscription")
    assert response.status_code == status
    assert response.json()["error"] == {"code": code, "retryable": retryable, "message": "安全文案"}


def test_account_oauth_start_returns_configuration_error(monkeypatch, client):
    repo = Repository(main.app.dependency_overrides[get_store]())
    account = repo.create("accounts", "u1", {"alias": "测试账号"})
    monkeypatch.setattr(main.app.state, "repo", repo, raising=False)

    def fail(_service, _purpose):
        raise GoogleConfigurationError("配置不可用")

    monkeypatch.setattr(GoogleService, "start", fail)
    response = client.post(
        f"/api/gpu/accounts/{account['id']}/auth/colab/start",
        headers={"Idempotency-Key": "config-error"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == {"code": "GOOGLE_CONFIG_ERROR", "retryable": False, "message": "配置不可用"}


def _batch_client(tmp_path):
    from app.batch_routes import build_batch_router
    from app.job_schemas import JobInput
    app = FastAPI()

    async def api_error(_request: Request, exc: APIError):
        return JSONResponse({"error": {"code": exc.code}}, status_code=exc.status)

    app.add_exception_handler(APIError, api_error)
    app.include_router(build_batch_router(JobInput, lambda *a, **k: {}, lambda j: j), prefix="/api/gpu")
    app.dependency_overrides[get_store] = lambda: Store(f"sqlite:///{tmp_path / 'b.sqlite'}")
    app.dependency_overrides[current_user] = lambda: {"id": "u1", "role": "user"}
    return TestClient(app)


def _body(mode):
    job = {"project_id": "p", "model_id": "m", "workflow_key": "w", "workflow_version": 1,
           "execution_policy": {"provider": "colab"}}
    return {"jobs": [job], "mode": mode}


def test_real_batch_blocked_when_offline_or_wrong_mode(monkeypatch, tmp_path):
    c = _batch_client(tmp_path)
    monkeypatch.setenv("STUDIO_OFFLINE_TEST", "1")
    r = c.post("/api/gpu/batches", json=_body("batch"), headers={"Idempotency-Key": "k"})
    assert r.status_code == 403
    monkeypatch.delenv("STUDIO_OFFLINE_TEST")
    monkeypatch.setenv("GPU_REAL_ENABLED", "1")
    r = c.post("/api/gpu/batches", json=_body("smart"), headers={"Idempotency-Key": "k"})
    assert r.status_code == 503
    assert r.json()['error']['code'] == 'REAL_LIFECYCLE_UNVERIFIED'
