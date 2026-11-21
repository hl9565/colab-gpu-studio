from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from app.db import Store
from app.gpu.google import GoogleConfigurationError, GoogleOAuthError, GoogleService, GoogleTransientError
from app.gpu.repository import Repository
from app.security import decrypt_secret, encrypt_secret


class MemoryRepository:
    def __init__(self):
        self.rows = {"credentials": [], "leases": []}

    def create(self, kind, owner_id, data, **kwargs):
        row = {"id": f"{kind}-{len(self.rows[kind]) + 1}", "owner_id": owner_id, **data}
        self.rows.setdefault(kind, []).append(row)
        return dict(row)

    def get(self, kind, record_id, owner_id=None):
        return next(
            (
                dict(row)
                for row in self.rows.get(kind, [])
                if row["id"] == record_id and (owner_id is None or row["owner_id"] == owner_id)
            ),
            None,
        )

    def list(self, kind, owner_id=None, **kwargs):
        return [
            dict(row) for row in self.rows.get(kind, []) if owner_id is None or row["owner_id"] == owner_id
        ]

    def update(self, kind, record_id, patch, owner_id=None):
        row = self.get(kind, record_id, owner_id)
        if row is None:
            raise KeyError(record_id)
        for current in self.rows[kind]:
            if current["id"] == record_id:
                current.update(patch)
                return dict(current)
        raise KeyError(record_id)

    def delete(self, kind, record_id, owner_id=None):
        old = len(self.rows.get(kind, []))
        self.rows[kind] = [
            r
            for r in self.rows[kind]
            if not (r["id"] == record_id and (owner_id is None or r["owner_id"] == owner_id))
        ]
        return len(self.rows[kind]) != old


def configured(monkeypatch):
    monkeypatch.setenv("GPU_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GPU_GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GPU_GOOGLE_REDIRECT_URI", "https://studio.example/api/gpu/auth/google/callback")


def test_oauth_start_uses_pkce_and_persists_owner_bound_state(monkeypatch):
    configured(monkeypatch)
    repo = MemoryRepository()
    result = GoogleService(repo, "owner-1").start("drive")
    params = parse_qs(urlparse(result["authorization_url"]).query)
    assert params["client_id"] == ["client-id"]
    assert params["scope"] == ["https://www.googleapis.com/auth/drive.file"]
    assert params["code_challenge_method"] == ["S256"]
    assert repo.rows["credentials"][0]["owner_id"] == "owner-1"
    assert repo.rows["credentials"][0]["purpose"] == "_oauth_state"
    assert repo.rows["credentials"][0]["oauth_purpose"] == "drive"


def test_callback_derives_owner_from_single_use_state_and_encrypts_tokens(monkeypatch):
    configured(monkeypatch)
    repo = MemoryRepository()

    def handler(request):
        form = parse_qs(request.content.decode())
        assert form["code"] == ["code-1"]
        assert "code_verifier" in form
        return httpx.Response(
            200, json={"access_token": "access-secret", "refresh_token": "refresh-secret", "expires_in": 3600}
        )

    service = GoogleService(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    started = service.start("colab")
    result = service.callback("code-1", started["state"])
    assert result == {"purpose": "colab", "connected": True, "owner_id": "owner-1"}
    credential = next(row for row in repo.rows["credentials"] if row.get("purpose") == "colab")
    assert decrypt_secret(credential["encrypted_access_token"]) == "access-secret"
    assert decrypt_secret(credential["encrypted_refresh_token"]) == "refresh-secret"
    assert "access-secret" not in str(result)
    with pytest.raises(GoogleOAuthError, match="state"):
        service.callback("code-1", started["state"])


@pytest.mark.parametrize(("next_client_id", "legacy", "reuse_allowed"), [
    ("client-b", False, False),
    ("client-id", True, False),
    ("client-id", False, True),
], ids=["different-web-client", "legacy-credential", "same-web-client"])
def test_callback_refresh_reuse_requires_same_issuing_web_client(monkeypatch, next_client_id, legacy, reuse_allowed):
    configured(monkeypatch)
    repo = MemoryRepository()
    payload = {"access_token": "first-access", "refresh_token": "first-refresh"}

    def handler(request):
        assert parse_qs(request.content.decode())["client_id"] == [expected_client_id]
        return httpx.Response(200, json=payload)

    expected_client_id = "client-id"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GoogleService(repo, "owner-1", client=client)
        if legacy:
            previous = refresh_credential(repo, account_id=None)
            assert "issued_client_id" not in previous
        else:
            assert service.callback("first-code", service.start("colab")["state"])["connected"] is True
            previous, = [row for row in repo.list("credentials") if row.get("purpose") == "colab"]
            assert previous["issued_client_id"] == "client-id"

        expected_client_id = next_client_id
        monkeypatch.setenv("GPU_GOOGLE_CLIENT_ID", next_client_id)
        payload = {"access_token": "second-access"}
        state = service.start("colab")["state"]
        if reuse_allowed:
            assert service.callback("second-code", state)["connected"] is True
            updated = repo.get("credentials", previous["id"], "owner-1")
            assert updated["issued_client_id"] == "client-id"
            assert decrypt_secret(updated["encrypted_refresh_token"]) == "first-refresh"
            assert decrypt_secret(updated["encrypted_access_token"]) == "second-access"
        else:
            with pytest.raises(GoogleOAuthError, match="Google OAuth未返回可持久化刷新凭证，请重新授权"):
                service.callback("second-code", state)
            assert repo.get("credentials", previous["id"], "owner-1") == previous


def test_refresh_preserves_existing_refresh_token(monkeypatch):
    configured(monkeypatch)
    from app.security import encrypt_secret

    repo = MemoryRepository()
    repo.create(
        "credentials",
        "owner-1",
        {
            "provider": "google",
            "purpose": "colab",
            "encrypted_access_token": encrypt_secret("old"),
            "encrypted_refresh_token": encrypt_secret("keep"),
            "expires_at": time.time() - 1,
        },
    )

    def handler(request):
        assert parse_qs(request.content.decode())["refresh_token"] == ["keep"]
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    service = GoogleService(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert service.access_token("colab") == "new"
    row = repo.rows["credentials"][0]
    assert decrypt_secret(row["encrypted_refresh_token"]) == "keep"
    assert row["reauth_required"] is False and row["reauth_message"] is None


def refresh_credential(repo, owner="owner-1", account="primary", purpose="colab", **patch):
    return repo.create("credentials", owner, {
        "provider": "google",
        "purpose": purpose,
        "account_id": account,
        "encrypted_access_token": encrypt_secret("old-access"),
        "encrypted_refresh_token": encrypt_secret("old-refresh"),
        "expires_at": 0,
        **patch,
    })


def test_repository_credential_patch_requires_owner_and_current_version(tmp_path):
    repo = Repository(Store(f"sqlite:///{tmp_path / 'versioned-credential.sqlite'}"))
    target = refresh_credential(repo)
    credential_id, version = target["id"], target["version"]
    patch = {"reauth_required": True, "reauth_message": "安全文案"}
    assert repo.update_credential_if_version("owner-2", credential_id, version, patch) is None
    assert repo.update_credential_if_version("owner-1", "missing", version, patch) is None
    updated = repo.update_credential_if_version("owner-1", credential_id, version, patch)
    assert updated == {**target, **patch, "version": version + 1, "updated_at": updated["updated_at"]}
    assert repo.update_credential_if_version("owner-1", credential_id, version, {"reauth_required": False}) is None
    assert repo.get("credentials", credential_id, "owner-1") == updated


@pytest.mark.parametrize(("status", "oauth_error", "message"), [
    (401, "invalid_client", "Google 授权已失效，请重新授权"),
    (400, "invalid_grant", "Google 授权已失效，请重新授权"),
    (400, "unauthorized_client", "Google OAuth客户端配置不可用，请检查配置并重新授权"),
    (400, "invalid_client", "Google OAuth客户端配置不可用，请检查配置并重新授权"),
])
def test_permanent_refresh_failure_marks_only_the_requested_credential(monkeypatch, status, oauth_error, message):
    configured(monkeypatch)
    repo = MemoryRepository()
    target = refresh_credential(repo)
    untouched = [
        refresh_credential(repo, purpose="drive"),
        refresh_credential(repo, account="secondary"),
        refresh_credential(repo, owner="owner-2"),
    ]

    def handler(request):
        return httpx.Response(status, json={"error": oauth_error, "error_description": "private upstream detail"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GoogleService(repo, "owner-1", account_id="primary", client=client)
        assert service.status()["colab"]["connected"] is True
        with pytest.raises(GoogleOAuthError) as caught:
            service.access_token("colab")
        assert str(caught.value) == message
        assert repo.get("credentials", target["id"], "owner-1") == {
            **target, "reauth_required": True, "reauth_message": message,
        }
        before_status = repo.list("credentials")
        status_by_purpose = service.status()
        assert status_by_purpose["colab"]["connected"] is False
        assert status_by_purpose["colab"]["message"] == message
        assert status_by_purpose["drive"]["connected"] is True
        assert repo.list("credentials") == before_status
        for row in untouched:
            assert repo.get("credentials", row["id"], row["owner_id"]) == row


@pytest.mark.parametrize(("status", "body", "message"), [
    (None, None, "Google OAuth刷新网络异常，请稍后重试"),
    (429, '{"error":"invalid_grant"}', "Google OAuth服务暂时不可用，请稍后重试"),
    (503, "private upstream detail", "Google OAuth服务暂时不可用，请稍后重试"),
    (400, '{"error":"invalid_request"}', "Google OAuth刷新响应异常，请稍后重试"),
    (400, "not JSON", "Google OAuth刷新响应异常，请稍后重试"),
    (403, '{"error":"invalid_grant"}', "Google OAuth刷新响应异常，请稍后重试"),
    (200, "not JSON", "Google OAuth刷新响应异常，请稍后重试"),
    (200, '{"access_token":"new","refresh_token":42}', "Google OAuth刷新响应异常，请稍后重试"),
])
def test_transient_refresh_failure_preserves_credential(monkeypatch, status, body, message):
    configured(monkeypatch)
    repo = MemoryRepository()
    target = refresh_credential(repo)

    def handler(request):
        if status is None:
            raise httpx.ReadTimeout("private upstream detail", request=request)
        return httpx.Response(status, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GoogleService(repo, "owner-1", account_id="primary", client=client)
        with pytest.raises(GoogleTransientError) as caught:
            service.access_token("colab")
        assert str(caught.value) == message
        assert repo.get("credentials", target["id"], "owner-1") == target
        assert service.status()["colab"]["connected"] is True


@pytest.mark.parametrize("field", ["encrypted_access_token", "encrypted_refresh_token"])
@pytest.mark.parametrize("ciphertext", [None, "", 123, "not-a-fernet-token"])
def test_unusable_ciphertext_marks_reauth_without_network(field, ciphertext):
    repo = MemoryRepository()
    target = refresh_credential(repo, expires_at=9999999999, **{field: ciphertext})
    service = GoogleService(repo, "owner-1", account_id="primary")
    message = "Google OAuth凭证不可用，请重新授权"
    with pytest.raises(GoogleOAuthError, match=message):
        service.access_token("colab")
    assert repo.get("credentials", target["id"], "owner-1") == {
        **target, "reauth_required": True, "reauth_message": message,
    }


@pytest.mark.parametrize("stored_message", [None, "请检查 OAuth 客户端并重新授权"])
def test_reauth_required_skips_cached_token_network_and_writes(monkeypatch, stored_message):
    repo = MemoryRepository()
    target = refresh_credential(repo, expires_at=9999999999, reauth_required=True, reauth_message=stored_message)
    service = GoogleService(repo, "owner-1", account_id="primary")
    message = stored_message or "Google 授权已失效，请重新授权"

    def unexpected(*args):
        pytest.fail("a flagged credential must not be decrypted or written again")

    monkeypatch.setattr(service, "_decrypt_tokens", unexpected)
    monkeypatch.setattr(repo, "update", unexpected)
    assert service.status()["colab"]["connected"] is False
    assert service.status()["colab"]["message"] == message
    with pytest.raises(GoogleOAuthError, match=message):
        service.access_token("colab")
    assert repo.get("credentials", target["id"], "owner-1") == target


def test_callback_requires_new_refresh_for_flagged_credential_and_clears_cli_binding(monkeypatch):
    configured(monkeypatch)
    repo = MemoryRepository()
    target = refresh_credential(repo, reauth_required=True, reauth_message="old message",
                                issued_client_id="client-id")
    payload = {"access_token": "authorized-access", "expires_in": 3600}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        service = GoogleService(repo, "owner-1", account_id="primary", client=client)
        with pytest.raises(GoogleOAuthError, match="Google OAuth未返回可持久化刷新凭证，请重新授权"):
            service.callback("code", service.start("colab")["state"])
        assert repo.get("credentials", target["id"], "owner-1") == target

        repo.update("credentials", target["id"], {
            "client_id": "old-cli", "encrypted_client_secret": encrypt_secret("old-cli-secret"),
        }, "owner-1")
        payload["refresh_token"] = "authorized-refresh"
        assert service.callback("code", service.start("colab")["state"])["connected"] is True
        updated = repo.get("credentials", target["id"], "owner-1")
        assert updated["reauth_required"] is False and updated["reauth_message"] is None
        assert updated["client_id"] is None and updated["encrypted_client_secret"] is None
        assert decrypt_secret(updated["encrypted_refresh_token"]) == "authorized-refresh"
        assert service.access_token("colab") == "authorized-access"


@pytest.mark.parametrize(("client_id", "client_secret"), [
    ("old-cli", "client-secret"),
    ("client-id", "old-cli-secret"),
])
def test_callback_rejects_refresh_reuse_for_different_cli_client(monkeypatch, client_id, client_secret):
    configured(monkeypatch)
    monkeypatch.delenv("GPU_GOOGLE_SECRET", raising=False)
    repo = MemoryRepository()
    target = refresh_credential(repo, issued_client_id="client-id", client_id=client_id,
                                encrypted_client_secret=encrypt_secret(client_secret))

    def handler(request):
        assert parse_qs(request.content.decode())["client_id"] == ["client-id"]
        return httpx.Response(200, json={"access_token": "web-access"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GoogleService(repo, "owner-1", account_id="primary", client=client)
        with pytest.raises(GoogleOAuthError, match="Google OAuth未返回可持久化刷新凭证，请重新授权"):
            service.callback("code", service.start("colab")["state"])
    assert repo.get("credentials", target["id"], "owner-1") == target


@pytest.mark.parametrize(("refresh_status", "oauth_error", "intervening_action", "expected_error"), [
    (401, None, "reauthorize", None),
    (400, "invalid_grant", "reauthorize", None),
    (400, "invalid_client", "reauthorize", None),
    (400, "unauthorized_client", "reauthorize", None),
    (401, None, "disconnect", "Google OAuth未连接，真实GPU能力已关闭"),
    (200, None, "disconnect", "Google OAuth未连接，真实GPU能力已关闭"),
    (200, None, "reauthorize", None),
])
def test_real_sqlite_stale_refresh_does_not_overwrite_new_credential(
    monkeypatch, tmp_path, refresh_status, oauth_error, intervening_action, expected_error,
):
    configured(monkeypatch)
    url = f"sqlite:///{tmp_path / 'refresh-race.sqlite'}"
    repo = Repository(Store(url))
    target = refresh_credential(repo)
    refresh_started, release_refresh = threading.Event(), threading.Event()

    def old_refresh(request):
        assert parse_qs(request.content.decode())["refresh_token"] == ["old-refresh"]
        refresh_started.set()
        assert release_refresh.wait(timeout=5)
        payload = {"access_token": "stale-access", "refresh_token": "stale-refresh"}
        return httpx.Response(refresh_status, json={"error": oauth_error} if oauth_error else payload)

    def new_authorization(request):
        return httpx.Response(200, json={"access_token": "fresh-access", "refresh_token": "fresh-refresh"})

    with (httpx.Client(transport=httpx.MockTransport(old_refresh)) as refresh_client,
          httpx.Client(transport=httpx.MockTransport(new_authorization)) as new_client,
          ThreadPoolExecutor(max_workers=1) as executor):
        old_service = GoogleService(Repository(Store(url)), "owner-1", account_id="primary", client=refresh_client)
        new_service = GoogleService(repo, "owner-1", account_id="primary", client=new_client)
        pending = executor.submit(old_service.access_token, "colab")
        try:
            assert refresh_started.wait(timeout=5)
            if intervening_action == "disconnect":
                assert new_service.disconnect("colab") is True
            else:
                assert new_service.callback("code", new_service.start("colab")["state"])["connected"] is True
            current = repo.get("credentials", target["id"], "owner-1")
            assert current["version"] == target["version"] + 1
        finally:
            release_refresh.set()
        if expected_error:
            with pytest.raises(GoogleOAuthError, match=expected_error):
                pending.result(timeout=5)
        else:
            assert pending.result(timeout=5) == "fresh-access"

        assert repo.get("credentials", target["id"], "owner-1") == current
        assert current["reauth_required"] is False and current["reauth_message"] is None
        if intervening_action == "disconnect":
            assert current["revoked"] is True
            assert current["encrypted_access_token"] is None and current["encrypted_refresh_token"] is None
            assert repo.update_credential_if_version("owner-1", target["id"], current["version"], {
                "reauth_required": True,
            }) is None
        else:
            assert decrypt_secret(current["encrypted_access_token"]) == "fresh-access"
            assert decrypt_secret(current["encrypted_refresh_token"]) == "fresh-refresh"


def test_unconfigured_google_fails_closed(monkeypatch):
    monkeypatch.delenv("GPU_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GPU_GOOGLE_CLIENT_SECRET", raising=False)
    with pytest.raises(GoogleConfigurationError):
        GoogleService(MemoryRepository(), "owner-1").start("drive")


def test_oauth_state_is_consumed_once_under_concurrent_callbacks(monkeypatch):
    configured(monkeypatch)
    repo = MemoryRepository()
    started = GoogleService(repo, "owner-1").start("drive")
    start_barrier = threading.Barrier(2)
    exchanges = 0
    exchange_lock = threading.Lock()

    def handler(request):
        nonlocal exchanges
        with exchange_lock:
            exchanges += 1
        return httpx.Response(
            200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}
        )

    clients = [httpx.Client(transport=httpx.MockTransport(handler)) for _ in range(2)]
    services = [GoogleService(repo, "owner-1", client=client) for client in clients]
    results = []

    def consume(service):
        try:
            start_barrier.wait(timeout=2)
            results.append(service.callback("code", started["state"]))
        except (GoogleOAuthError, KeyError):
            results.append(None)

    threads = [threading.Thread(target=consume, args=(service,)) for service in services]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert sum(result is not None for result in results) == 1
    assert exchanges == 1


def test_insufficient_scope_is_not_reported_as_connected(monkeypatch):
    configured(monkeypatch)
    repo = MemoryRepository()
    repo.create(
        "credentials",
        "owner-1",
        {
            "provider": "google",
            "purpose": "drive",
            "encrypted_access_token": "a",
            "encrypted_refresh_token": "r",
            "expires_at": 9999999999,
            "scope": "openid",
        },
    )
    status = GoogleService(repo, "owner-1").status()
    assert status["drive"]["connected"] is False


def test_real_sqlite_state_is_exchanged_once_concurrently(monkeypatch, tmp_path):
    configured(monkeypatch)
    database = tmp_path / "oauth.sqlite"
    repo = Repository(Store(f"sqlite:///{database}"))
    started = GoogleService(repo, "owner-1").start("drive")
    exchanges = 0
    exchange_lock = threading.Lock()

    def handler(request):
        nonlocal exchanges
        with exchange_lock:
            exchanges += 1
        return httpx.Response(
            200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}
        )

    services = [
        GoogleService(
            Repository(Store(f"sqlite:///{database}")),
            "owner-1",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        for _ in range(2)
    ]
    start_barrier = threading.Barrier(2)
    results = []

    def consume(service):
        try:
            start_barrier.wait(timeout=2)
            results.append(service.callback("code", started["state"]))
        except GoogleOAuthError:
            results.append(None)

    threads = [threading.Thread(target=consume, args=(service,)) for service in services]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sum(result is not None for result in results) == 1
    assert exchanges == 1


@pytest.mark.parametrize("with_confirmation", [False, True])
def test_disconnect_clears_local_tokens_when_remote_revoke_fails(monkeypatch, with_confirmation):
    configured(monkeypatch)
    repo = MemoryRepository()
    repo.create(
        "credentials",
        "owner-1",
        {
            "provider": "google",
            "purpose": "drive",
            "encrypted_access_token": encrypt_secret("access-secret"),
            "encrypted_refresh_token": encrypt_secret("refresh-secret"),
            "expires_at": 9999999999,
        },
    )

    def handler(request):
        assert parse_qs(request.content.decode())["token"] == ["refresh-secret"]
        return httpx.Response(503, text="upstream body must not escape")

    service = GoogleService(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    if with_confirmation:
        assert service.disconnect_with_confirmation("drive") == (True, False)
    else:
        assert service.disconnect("drive") is True
    row = repo.rows["credentials"][0]
    assert row["encrypted_access_token"] is None and row["encrypted_refresh_token"] is None
    assert row["disconnect_status"] == "remote_unconfirmed"
    assert service.status()["drive"]["connected"] is False
