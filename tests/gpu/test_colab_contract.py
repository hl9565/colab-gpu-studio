from __future__ import annotations

import httpx
import pytest
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.repository import Repository
from app.security import encrypt_secret


class MemoryRepository(Repository):
    def __init__(self):
        super().__init__(Store("sqlite:///:memory:"))
        self.create("credentials", "owner-1", {
            "provider": "google", "purpose": "colab", "encrypted_access_token": encrypt_secret("access"),
            "encrypted_refresh_token": encrypt_secret("refresh"), "expires_at": 9999999999,
        })


def make(handler):
    repo = MemoryRepository()
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "runtime_id": "atelier-runtime-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "owner_job_id": "job-1",
            "state": "creating",
        },
    )
    return GoogleColabProvider(repo, "owner-1", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_discovery_read_only_calls():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.url.path.endswith("runtimespecs"):
            return httpx.Response(200, json={"runtimeSpecs": [{"eligible": True}]})
        return httpx.Response(200, json={"name": "subscription", "tier": "SUBSCRIPTION_TIER_PRO"})

    provider = make(handler)
    assert provider.list_specs()[0]["eligible"] is True
    assert provider.subscription()["tier"] == "SUBSCRIPTION_TIER_PRO"
    assert paths == [("GET", "/v1beta/runtimespecs"), ("GET", "/v1beta/subscription")]


def test_create_matches_discovery_and_persists_operation():
    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.url.params["requestId"] == "11111111-1111-4111-8111-111111111111"
        assert request.url.params["runtimeId"] == "atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "operations/op-1",
                "done": True,
                "response": {"name": "runtimes/atelier-runtime-1"},
            },
        )

    provider = make(handler)
    result = provider.create(
        {
            "key": {"variant": "GPU", "accelerator": "T4", "shape": "STANDARD"},
            "eligible": True,
        },
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result == {
        "runtime_name": "runtimes/atelier-runtime-1",
        "operation_name": "operations/op-1",
        "ready": True,
        "runtime_id": "atelier-runtime-1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "owner_id": "owner-1",
    }


def test_captured_real_spec_uses_official_enums_and_requires_eligibility():
    captured_specs = [
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
            "eligible": True,
        },
        {
            "key": {"variant": "VARIANT_GPU", "accelerator": "H100", "shape": "SHAPE_STANDARD"},
        },
    ]

    assert GoogleColabProvider._spec(captured_specs[0]) == {
        "variant": "VARIANT_GPU",
        "accelerator": "T4",
        "shape": "SHAPE_STANDARD",
    }
    with pytest.raises(ColabError, match="eligible"):
        GoogleColabProvider._spec(captured_specs[1])


def test_captured_real_spec_create_payload_preserves_official_enums():
    captured_spec = {
        "key": {"variant": "VARIANT_GPU", "accelerator": "A100", "shape": "SHAPE_HIGHMEM"},
        "eligible": True,
    }

    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1beta/runtimes"
        assert request.content == (
            b'{"runtimeSpec":{"variant":"VARIANT_GPU","accelerator":"A100","shape":"SHAPE_HIGHMEM"}}'
        )
        return httpx.Response(200, json={"name": "operations/op-captured", "done": True})

    result = make(handler).create(
        captured_spec,
        "11111111-1111-4111-8111-111111111111",
        "job-1",
        "atelier-runtime-1",
    )
    assert result["operation_name"] == "operations/op-captured"


def test_inspect_without_operation_uses_runtime_get():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.url.path == "/v1beta/runtimes/atelier-runtime-1"
        return httpx.Response(
            200,
            json={
                "name": "runtimes/atelier-runtime-1",
                "connectionInfo": {"url": "https://runtime", "token": "secret"},
            },
        )

    result = make(handler).inspect(
        {
            "runtime_name": "runtimes/atelier-runtime-1",
            "runtime_id": "atelier-runtime-1",
            "owner_id": "owner-1",
        }
    )
    assert result["ready"] is True and "token" not in str(result)
    assert paths == ["/v1beta/runtimes/atelier-runtime-1"]


def test_delete_requires_owner_and_get_404_confirmation():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path))
        if request.method == "DELETE":
            assert request.url.params["allowMissing"] == "true"
            return httpx.Response(204)
        return httpx.Response(404)

    provider = make(handler)
    assert (
        provider.delete(
            {
                "runtime_name": "runtimes/atelier-runtime-1",
                "runtime_id": "atelier-runtime-1",
                "owner_id": "owner-1",
            }
        )
        is True
    )
    # An already-absent owned runtime is confirmed by GET, without an unnecessary DELETE.
    assert paths == [("GET", "/v1beta/runtimes/atelier-runtime-1")]
    with pytest.raises(Exception, match="归属"):
        provider.delete({"runtime_name": "runtimes/other", "runtime_id": "other", "owner_id": "other-owner"})


def test_3xx_redirect_rejected_fail_closed_in_rest_and_tunnel():
    def rest_handler(request):
        return httpx.Response(302, headers={"Location": "https://accounts.google.com/login"}, json={"runtimeSpecs": []})

    with pytest.raises(ColabError):
        make(rest_handler).list_specs()

    repo = MemoryRepository()
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "active",
            "owner_job_id": "job-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "runtime_id": "studio-ep-1",
            "tunnel_endpoint": "ep-1",
        },
    )

    def tunnel_handler(request):
        return httpx.Response(302, headers={"Location": "https://accounts.google.com/login"}, text=")]}'\n{\"assignments\": []}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        client=httpx.Client(transport=httpx.MockTransport(tunnel_handler)),
    )
    status = provider.inspect(
        {
            "owner_id": "owner-1",
            "owner_job_id": "job-1",
            "job_id": "job-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "runtime_id": "studio-ep-1",
            "runtime_name": "runtimes/studio-ep-1",
        }
    )
    assert status["deleted"] is False
    assert status["allocation_unknown"] is True
    assert status["remote_verified"] is False


def test_cli_tunnel_unassign_404_verifies_assignment_presence():
    assignments_state = {"items": [{"endpoint": "ep-1"}]}
    repo = MemoryRepository()
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "active",
            "owner_job_id": "job-1",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "runtime_id": "studio-ep-1",
            "tunnel_endpoint": "ep-1",
        },
    )

    def handler(request):
        if request.url.path == "/tun/m/unassign/ep-1" and request.method == "GET":
            return httpx.Response(200, text=")]}'\n{\"token\": \"xsrf-1\"}")
        if request.url.path == "/tun/m/unassign/ep-1" and request.method == "POST":
            return httpx.Response(404, text="{}")
        if request.url.path == "/tun/m/assignments":
            import json

            return httpx.Response(200, text=")]}'\n" + json.dumps({"assignments": assignments_state["items"]}))
        raise AssertionError(f"unexpected {request.method} {request.url}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    handle = {
        "owner_id": "owner-1",
        "owner_job_id": "job-1",
        "job_id": "job-1",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "runtime_id": "studio-ep-1",
        "runtime_name": "runtimes/studio-ep-1",
    }
    assert provider.delete(handle) is False
    assignments_state["items"] = []
    assert provider.delete(handle) is True
