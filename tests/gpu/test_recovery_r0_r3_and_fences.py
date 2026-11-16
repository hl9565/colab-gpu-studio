"""RED/GREEN regression tests for PR4 §5.2 Items 1-3:

- colab.py remote_verified HTTP evidence & CLI-tunnel fail-closed
- R0 -> R1 -> R1.5 -> R2 -> R3 recovery partition & alloc_generation pure-placeholder retirement
- Instance Allowed-Transition Matrix & cross-phase stop/delete/cleanup fences
- Linearized start x stop, _run MCPTransient + concurrent cancel across all exits, neighbor lease safety
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from app.config import settings
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.compute import GpuConfig, MockColabProvider, MockGPUExecutor
from app.gpu.instances import InstanceConflict, InstanceService
from app.gpu.mcp import MCPTransient
from app.gpu.repository import Repository, ensure_schema
from app.gpu.scheduler import Scheduler
from app.main import app
from app.security import sign_session
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from PIL import Image
from test_pr4_signoff_regressions import seed_execution_binding


def _valid_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_store(tmp_path=None, name: str = "recovery.db") -> tuple[Store, Repository]:
    url = "sqlite:///:memory:" if tmp_path is None else f"sqlite:///{tmp_path / name}"
    store = Store(url)
    repo = Repository(store)
    ensure_schema(store)
    return store, repo


def _setup_account(
    repo: Repository,
    owner: str = "owner-1",
    account_id: str = "acct-1",
    *,
    daily_limit: float = 100.0,
    monthly_limit: float = 500.0,
    provider: str = "mock",
) -> dict:
    return repo.create(
        "accounts",
        owner,
        {
            "account_id": account_id,
            "alias": f"Account {account_id}",
            "provider": provider,
            "connected": True,
            "auth_status": "connected",
            "enabled": True,
            "allow_allocation": True,
            "budget": {"daily": daily_limit, "monthly": monthly_limit},
            "daily_units_limit": daily_limit,
            "monthly_units_limit": monthly_limit,
        },
        record_id=account_id,
    )


def _colab_config(*, create_retries: int = 1, unit_rate: float = 1.0) -> GpuConfig:
    return GpuConfig(
        real_enabled=True,
        unit_rate=unit_rate,
        daily_units=100.0,
        monthly_units=500.0,
        max_job_units=50.0,
        max_runtime_minutes=30,
        create_retries=create_retries,
    )


def _tunnel_google(token: str = "oauth-tok-1") -> SimpleNamespace:
    return SimpleNamespace(
        access_token=lambda scope="colab": token,
        _credential=lambda service, account_id=None: {
            "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
        },
    )


def _rest_google(token: str = "oauth-tok-1") -> SimpleNamespace:
    return SimpleNamespace(
        access_token=lambda scope="colab": token,
        _credential=lambda service, account_id=None: {"client_id": "web-client-123.apps.googleusercontent.com"},
    )


def _tunnel_prefix(payload: dict) -> str:
    return ")]}'\n" + json.dumps(payload)


def _legacy_bare_job_body(
    project_id: str = "proj-1",
    *,
    provider: str = "mock",
    account_id: str = "acct-1",
    max_runtime_minutes: int = 10,
) -> dict:
    return {
        "project_id": project_id,
        "provider": provider,
        "execution_policy": {"provider": provider, "account_id": account_id, "lifecycle": "save"},
        "max_runtime_minutes": max_runtime_minutes,
        "snapshot": {"manifest": {"mock": provider == "mock", "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}}},
    }


def _service_budget_used(service: InstanceService, owner: str = "owner-1", account_id: str = "acct-1") -> float:
    with service._tx() as session:
        daily, _ = service._budget_totals(session, owner, account_id)
        return daily


# ---------------------------------------------------------------------------
# 1. colab.py remote_verified HTTP evidence & CLI-tunnel fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attempts,from_active", [(0, False), (0, True), (1, True)])
def test_cli_tunnel_missing_endpoint_zero_http_and_never_deleted(attempts, from_active):
    _, repo = _make_store()
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        raise AssertionError("HTTP must never be called when CLI tunnel endpoint is missing")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleColabProvider(
        repo,
        "owner-1",
        account_id="acct-1",
        client=client,
        google=_tunnel_google(),
    )
    runtime_id = "studio-missing-ep"
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "active" if from_active else "creating",
            "owner_job_id": "job-1",
            "request_id": "req-1",
            "runtime_id": runtime_id,
            "create_attempts": attempts,
            "operation_done": True,
            "from_active_missing_handle": from_active,
        },
    )
    handle = {
        "owner_id": "owner-1",
        "owner_job_id": "job-1",
        "job_id": "job-1",
        "request_id": "req-1",
        "runtime_id": runtime_id,
        "runtime_name": f"runtimes/{runtime_id}",
    }
    status = provider.inspect(handle)
    assert calls == []
    assert status["deleted"] is False
    assert status.get("identity_unverifiable") is True
    assert status.get("remote_verified") is False

    deleted = provider.delete(handle)
    assert calls == []
    assert deleted is False
    assert lease["id"]


def test_scheduler_quarantines_cli_tunnel_active_missing_handle_and_preserves_budget():
    _, repo = _make_store()
    _setup_account(repo, provider="colab")
    cfg = _colab_config(create_retries=1)
    service = InstanceService(repo, config=cfg, clock=lambda: 1000.0)
    http_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        http_calls.append(str(request.url))
        if request.url.path == "/tun/m/ccu-info":
            return httpx.Response(
                200,
                text=_tunnel_prefix({"eligibleGpus": ["T4"], "eligibleTpus": []}),
            )
        raise AssertionError(f"unexpected HTTP call: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleColabProvider(repo, "owner-1", account_id="acct-1", client=client, google=_tunnel_google())

    job = repo.create(
        "jobs",
        "owner-1",
        {
            **_legacy_bare_job_body(provider="colab"),
            "state": "allocating",
            "estimated_units": 10.0,
        },
    )
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "active",
            "owner_job_id": job["id"],
            "instance_id": inst["id"],
            "account_id": "acct-1",
            "request_id": "req-missing-ep",
            "runtime_id": "studio-missing-ep",
            "handle": None,
            "create_attempts": 1,
        },
    )
    job = repo.update("jobs", job["id"], {"lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")
    budget_before = _service_budget_used(service)
    assert budget_before > 0

    scheduler = Scheduler(
        repo,
        provider_factory=lambda j: provider,
        executor_factory=lambda j: MockGPUExecutor(),
        config=cfg,
        instance_service=service,
        clock=lambda: 1010.0,
    )
    result = scheduler.tick()
    assert http_calls == ["https://colab.research.google.com/tun/m/ccu-info?authuser=0"] or all(
        "/tun/m/ccu-info" in u for u in http_calls
    )
    saved_job = repo.get("jobs", job["id"], "owner-1")
    saved_lease = repo.get("leases", lease["id"], "owner-1")
    saved_inst = service.get("owner-1", inst["id"])
    assert saved_job["state"] == "failed"
    assert saved_job["error_code"] == "gpu_runtime_identity_missing"
    assert saved_lease["state"] == "quarantined"
    assert saved_lease["from_active_missing_handle"] is True
    assert saved_inst["state"] == "quarantined"
    assert _service_budget_used(service) == budget_before

    # Quarantined allocation cannot be freed or reused by another job
    with pytest.raises(Exception):
        service.allocate({**_legacy_bare_job_body(provider="colab"), "id": "job-reuse", "owner_id": "owner-1", "requested_instance_id": inst["id"]})
    assert result


def test_cli_tunnel_assignments_404_and_200_empty_dict_are_unknown_only_list_empty_is_absent():
    _, repo = _make_store()
    mode = {"case": "404"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/assignments":
            if mode["case"] == "404":
                return httpx.Response(404, text="{}")
            if mode["case"] == "missing_key":
                return httpx.Response(200, text=_tunnel_prefix({}))
            if mode["case"] == "non_list":
                return httpx.Response(200, text=_tunnel_prefix({"assignments": "bad"}))
            if mode["case"] == "empty_list":
                return httpx.Response(200, text=_tunnel_prefix({"assignments": []}))
        raise AssertionError(f"unexpected request {request.url}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        account_id="acct-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        google=_tunnel_google(),
    )
    runtime_id = "studio-ep-1"
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "active",
            "owner_job_id": "job-1",
            "request_id": "req-1",
            "runtime_id": runtime_id,
            "tunnel_endpoint": "ep-1",
        },
    )
    handle = {
        "owner_id": "owner-1",
        "owner_job_id": "job-1",
        "job_id": "job-1",
        "request_id": "req-1",
        "runtime_id": runtime_id,
        "runtime_name": f"runtimes/{runtime_id}",
    }

    for bad in ("404", "missing_key", "non_list"):
        mode["case"] = bad
        status = provider.inspect(handle)
        assert status["deleted"] is False
        assert status["allocation_unknown"] is True
        assert status["remote_verified"] is False
        with pytest.raises(ColabError):
            provider.delete(handle)

    mode["case"] = "empty_list"
    status = provider.inspect(handle)
    assert status["deleted"] is True
    assert status["remote_verified"] is True
    assert provider.delete(handle) is True


def test_rest_standalone_operation_404_is_unknown_only_verified_runtime_404_is_absent():
    _, repo = _make_store()
    case = {"op_status": 404, "op_body": {}, "rt_status": 404, "rt_body": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/operations/op-1":
            return httpx.Response(case["op_status"], json=case["op_body"])
        if request.url.path == "/v1beta/runtimes/studio-rest-1":
            return httpx.Response(case["rt_status"], json=case["rt_body"])
        raise AssertionError(f"unexpected request {request.url}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        account_id="acct-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        google=_rest_google(),
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "provider": "colab",
            "state": "creating",
            "owner_job_id": "job-1",
            "request_id": "req-1",
            "runtime_id": "studio-rest-1",
            "operation_name": "operations/op-1",
            "create_attempts": 1,
            "operation_done": False,
        },
    )
    handle = {
        "owner_id": "owner-1",
        "owner_job_id": "job-1",
        "job_id": "job-1",
        "request_id": "req-1",
        "runtime_id": "studio-rest-1",
        "runtime_name": "runtimes/studio-rest-1",
        "operation_name": "operations/op-1",
    }

    # Standalone operation 404 + runtime 404 -> unknown, remote_verified False
    st = provider.inspect(handle)
    assert st["deleted"] is False
    assert st["allocation_unknown"] is True
    assert st["remote_verified"] is False

    # Operation 404 + runtime 200 (alive) -> not deleted
    case["rt_status"] = 200
    case["rt_body"] = {"name": "runtimes/studio-rest-1", "connectionInfo": {"url": "http://rt", "token": "t"}}
    st2 = provider.inspect(handle)
    assert st2["deleted"] is False
    assert st2["ready"] is True

    # Operation done + runtime 404 -> verified absent
    case["op_status"] = 200
    case["op_body"] = {"name": "operations/op-1", "done": True}
    case["rt_status"] = 404
    case["rt_body"] = {}
    st3 = provider.inspect(handle)
    assert st3["deleted"] is True
    assert st3["remote_verified"] is True


# ---------------------------------------------------------------------------
# 2. Full R0 -> R1 -> R1.5 -> R2 -> R3 recovery matrix & alloc_generation
# ---------------------------------------------------------------------------


def test_r0_blocks_cleanup_pending_and_deleted_leases_with_handle():
    for lease_state in ("cleanup_pending", "deleted"):
        _, repo = _make_store()
        provider = MockColabProvider()
        lease = repo.create(
            "leases",
            "owner-1",
            {
                "provider": "mock",
                "state": lease_state,
                "owner_job_id": "job-1",
                "request_id": "req-1",
                "runtime_id": "rt-1",
                "handle": {"runtime_name": "mock-runtime-rt-1", "runtime_id": "rt-1", "request_id": "req-1", "job_id": "job-1", "owner_job_id": "job-1"},
                "create_attempts": 1,
            },
        )
        job = repo.create(
            "jobs",
            "owner-1",
            {"state": "allocating", "provider": "mock", "lease_id": lease["id"], "max_runtime_minutes": 5},
            record_id="job-1",
        )
        scheduler = Scheduler(repo, provider_factory=lambda j: provider, executor_factory=lambda j: MockGPUExecutor())
        res = scheduler._allocation(job)
        assert provider.create_calls == 0
        assert res["error_code"] == "gpu_allocation_aborted"


def test_r0_creating_missing_create_intent_id_attempts_0_vs_1():
    # attempts=0 missing create_intent_id -> stamps create_intent_id & from_active_missing_handle=False then creates
    _, repo0 = _make_store()
    lease0 = repo0.create(
        "leases",
        "owner-1",
        {"provider": "mock", "state": "creating", "owner_job_id": "job-0", "request_id": "req-0", "runtime_id": "rt-0", "create_attempts": 0},
    )
    job0 = repo0.create("jobs", "owner-1", {"state": "allocating", "provider": "mock", "lease_id": lease0["id"], "max_runtime_minutes": 5}, record_id="job-0")
    prov0 = MockColabProvider()
    res0 = Scheduler(repo0, provider_factory=lambda j: prov0)._allocation(job0)
    assert res0["state"] == "preparing"
    assert prov0.create_calls == 1
    saved0 = repo0.get("leases", lease0["id"], "owner-1")
    assert saved0["create_intent_id"]
    assert saved0["from_active_missing_handle"] is False

    # attempts=1 missing create_intent_id -> enters R1 probe first (never blind create before probe)
    _, repo1 = _make_store()
    lease1 = repo1.create(
        "leases",
        "owner-1",
        {"provider": "mock", "state": "creating", "owner_job_id": "job-1", "request_id": "req-1", "runtime_id": "rt-1", "create_attempts": 1},
    )
    job1 = repo1.create("jobs", "owner-1", {"state": "allocating", "provider": "mock", "lease_id": lease1["id"], "max_runtime_minutes": 5}, record_id="job-1")
    probe_calls: list[str] = []

    class UnknownProvider(MockColabProvider):
        def inspect(self, handle):
            probe_calls.append(handle["runtime_id"])
            return {"ready": False, "deleted": False, "allocation_unknown": True, "remote_verified": False}

    prov1 = UnknownProvider()
    res1 = Scheduler(repo1, provider_factory=lambda j: prov1, config=_colab_config(create_retries=1))._allocation(job1)
    assert probe_calls == ["rt-1"]
    assert prov1.create_calls == 0
    assert res1["state"] == "allocating"
    saved1 = repo1.get("leases", lease1["id"], "owner-1")
    assert saved1["state"] == "unknown"
    assert saved1["from_active_missing_handle"] is False


@pytest.mark.parametrize("legacy_setup", ["claim_existing_instance", "no_instance_attach_idle"])
@pytest.mark.parametrize("origin", ["creating_attempts_1", "active_missing_handle"])
def test_r0_to_r3_unknown_restart_absent_matrix(tmp_path, origin, legacy_setup):
    store, repo = _make_store(tmp_path, f"matrix_{origin}_{legacy_setup}.db")
    cfg = _colab_config(create_retries=1)
    use_service = legacy_setup == "claim_existing_instance"
    service = InstanceService(repo, config=cfg, clock=lambda: 1000.0) if use_service else None
    _setup_account(repo, provider="colab")

    probe_phase = {"mode": "unknown"}
    created: list[str] = []
    deleted_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/operations/op-1":
            if probe_phase["mode"] == "unknown":
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"name": "operations/op-1", "done": True})
        if request.url.path == "/v1beta/runtimes/studio-rec-1":
            if probe_phase["mode"] == "unknown":
                return httpx.Response(404, json={})
            return httpx.Response(404, json={})
        if request.url.path in {"/v1beta/runtimespecs", "/v1beta/runtimesspecs:list"} or "runtimeSpecs" in request.url.path:
            return httpx.Response(200, json={"runtimeSpecs": [{"eligible": True, "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}}]})
        if request.url.path.startswith("/v1beta/runtimes") and request.method == "POST":
            created.append(str(request.url))
            rt_id = (request.url.params or {}).get("runtimeId", "studio-new")
            return httpx.Response(200, json={"name": "operations/op-new", "done": True, "response": {"name": f"runtimes/{rt_id}"}})
        if request.url.path.startswith("/v1beta/runtimes/") and request.method == "DELETE":
            deleted_calls.append(request.url.path)
            return httpx.Response(200, json={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(handler)

    def make_provider(j):
        p = GoogleColabProvider(
            repo,
            "owner-1",
            account_id="acct-1",
            client=httpx.Client(transport=transport),
            google=_rest_google(),
        )
        # Stub list_specs to use valid key cleanly without hitting Google URL ambiguity
        p.list_specs = lambda: [{"eligible": True, "key": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}}]
        return p

    job = repo.create(
        "jobs",
        "owner-1",
        {
            **_legacy_bare_job_body(provider="colab"),
            "state": "allocating",
            "estimated_units": 10.0,
            **({"legacy_scheduler_only": True} if not use_service else {}),
        },
    )
    inst = None
    orig_request_id = str(uuid.uuid4())
    if use_service:
        # Simulate instance created & idle-claimed / legacy-reused
        prior = repo.create(
            "jobs",
            "owner-1",
            {**_legacy_bare_job_body(provider="colab"), "state": "queued"},
            record_id="prior-job",
        )
        idle_inst = service.allocate(prior)
        prior = repo.update("jobs", "prior-job", {"state": "completed"}, "owner-1")
        idle_lease = repo.create(
            "leases",
            "owner-1",
            {
                "provider": "colab",
                "state": "creating" if origin == "creating_attempts_1" else "active",
                "owner_job_id": job["id"],
                "instance_id": idle_inst["id"],
                "account_id": "acct-1",
                "request_id": orig_request_id,
                "runtime_id": "studio-rec-1",
                "operation_name": "operations/op-1",
                "create_attempts": 1 if origin == "creating_attempts_1" else 0,
                "operation_done": False if origin == "creating_attempts_1" else True,
                "handle": None,
            },
        )
        with service._tx() as session:
            row = service._find(session, "instances", idle_inst["id"], "owner-1")
            data = service._data(row)
            data.update(lease_id=idle_lease["id"], current_job_id=job["id"], hard_deadline_at=5000.0)
            service._set(row, data, state="allocating", session=session)
        job = repo.update("jobs", job["id"], {"lease_id": idle_lease["id"], "instance_id": idle_inst["id"]}, "owner-1")
        lease_id = idle_lease["id"]
        inst = idle_inst
        budget_unknown_before = _service_budget_used(service)
    else:
        # No-instance legacy setup (as if legacy attach_idle_lease / creating crashed)
        idle_lease = repo.create(
            "leases",
            "owner-1",
            {
                "provider": "colab",
                "state": "creating" if origin == "creating_attempts_1" else "active",
                "owner_job_id": job["id"],
                "request_id": orig_request_id,
                "runtime_id": "studio-rec-1",
                "operation_name": "operations/op-1",
                "create_attempts": 1 if origin == "creating_attempts_1" else 0,
                "operation_done": False if origin == "creating_attempts_1" else True,
                "handle": None,
            },
        )
        job = repo.update("jobs", job["id"], {"lease_id": idle_lease["id"]}, "owner-1")
        lease_id = idle_lease["id"]

    # Step 1: R1 probe returns "unknown"
    sched1 = Scheduler(repo, provider_factory=make_provider, config=cfg, instance_service=service, clock=lambda: 1005.0)
    res1 = sched1._allocation(repo.get("jobs", job["id"], "owner-1"))
    assert res1["state"] == "allocating"
    assert created == []
    assert deleted_calls == []
    lease_after_unknown = repo.get("leases", lease_id, "owner-1")
    assert lease_after_unknown["state"] == "unknown"
    expected_from_active = origin == "active_missing_handle"
    assert lease_after_unknown["from_active_missing_handle"] is expected_from_active
    if use_service:
        assert _service_budget_used(service) == budget_unknown_before
        assert service.get("owner-1", inst["id"])["state"] != "deleted"

    # Step 2: Simulate process restart & remote probe now verifies "absent"
    probe_phase["mode"] = "absent"
    sched2 = Scheduler(repo, provider_factory=make_provider, config=cfg, instance_service=service, clock=lambda: 1010.0)
    res2 = sched2._allocation(repo.get("jobs", job["id"], "owner-1"))

    if expected_from_active:
        # Active missing handle MUST NEVER call provider.create on absent
        assert created == []
        assert res2["state"] in {"cleaning_up", "failed"}
        final = sched2._cleanup(res2) if res2["state"] == "cleaning_up" else res2
        assert final["state"] == "failed"
        assert repo.get("leases", lease_id, "owner-1")["state"] == "deleted"
        if use_service:
            assert service.get("owner-1", inst["id"])["state"] == "deleted"
    else:
        # creating, attempts=1 performs bounded R2 retry keeping the SAME lease/instance/request/runtime identity
        assert len(created) == 1
        assert orig_request_id in created[0]
        assert deleted_calls == []
        assert res2["state"] == "preparing"
        assert res2["lease_id"] == lease_id
        rebound_lease = repo.get("leases", lease_id, "owner-1")
        assert rebound_lease["state"] == "active"
        assert rebound_lease["create_attempts"] == 2
        assert rebound_lease["request_id"] == orig_request_id
        assert rebound_lease["runtime_id"] == "studio-rec-1"
        if use_service:
            assert int(res2.get("alloc_generation") or 0) == 0
            assert service.get("owner-1", inst["id"])["state"] != "deleted"
            assert res2["instance_id"] == inst["id"]
            assert _service_budget_used(service) == pytest.approx(10.0)


@pytest.mark.parametrize("use_file_db", [False, True])
def test_r1_5_pure_placeholder_crash_between_r15_and_r3_then_resume_single_budget(tmp_path, monkeypatch, use_file_db):
    store, repo = _make_store(tmp_path if use_file_db else None, "r15_mid_crash.db")
    _setup_account(repo, provider="mock")
    cfg = GpuConfig(real_enabled=False, unit_rate=1.0, daily_units=100.0, monthly_units=500.0, create_retries=1)
    service = InstanceService(repo, config=cfg, clock=lambda: 1000.0)

    job = repo.create(
        "jobs",
        "owner-1",
        {**_legacy_bare_job_body(provider="mock"), "state": "allocating", "estimated_units": 10.0},
    )
    inst0 = service.allocate(job)
    # Give non-zero budget reservation to verify single charge invariant
    with service._tx() as session:
        row = service._find(session, "instances", inst0["id"], "owner-1")
        data = service._data(row)
        data["budget_reserved"] = 10.0
        service._set(row, data, session=session)
    assert _service_budget_used(service) == pytest.approx(10.0)

    lease0 = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "mock",
            "state": "creating",
            "owner_job_id": job["id"],
            "instance_id": inst0["id"],
            "account_id": "acct-1",
            "request_id": None,
            "runtime_id": None,
            "create_attempts": 0,
            "from_active_missing_handle": False,
        },
    )
    job = repo.update("jobs", job["id"], {"lease_id": lease0["id"], "instance_id": inst0["id"]}, "owner-1")

    class AbsentMockProvider(MockColabProvider):
        def inspect(self, handle):
            return {"ready": False, "deleted": True, "remote_verified": True}

    provider = AbsentMockProvider()
    real_allocate = service.allocate
    crash_once = {"armed": True}

    def crashing_allocate(j):
        if crash_once["armed"] and int(j.get("alloc_generation") or 0) >= 1:
            crash_once["armed"] = False
            raise RuntimeError("process died between R1.5 and R3 allocate")
        inst = real_allocate(j)
        with service._tx() as session:
            row = service._find(session, "instances", inst["id"], "owner-1")
            data = service._data(row)
            data["budget_reserved"] = 10.0
            service._set(row, data, session=session)
        return service.get("owner-1", inst["id"])

    monkeypatch.setattr(service, "allocate", crashing_allocate)
    sched1 = Scheduler(repo, provider_factory=lambda j: provider, config=cfg, instance_service=service, clock=lambda: 1001.0)
    with pytest.raises(RuntimeError, match="between R1.5 and R3"):
        sched1._allocation(repo.get("jobs", job["id"], "owner-1"))

    # Verify R1.5 durability after mid-crash: placeholder deleted, alloc_generation==1, zero leaked budget!
    mid_job = repo.get("jobs", job["id"], "owner-1")
    assert mid_job["alloc_generation"] == 1
    assert mid_job.get("lease_id") is None
    assert mid_job.get("instance_id") is None
    assert repo.get("leases", lease0["id"], "owner-1")["state"] == "deleted"
    assert service.get("owner-1", inst0["id"])["state"] == "deleted"
    assert _service_budget_used(service) == pytest.approx(0.0)

    # Restart scheduler -> R3 allocates generation 1 with single budget charge!
    sched2 = Scheduler(repo, provider_factory=lambda j: provider, config=cfg, instance_service=service, clock=lambda: 1002.0)
    resumed = sched2._allocation(repo.get("jobs", job["id"], "owner-1"))
    assert resumed["state"] == "preparing"
    assert resumed["alloc_generation"] == 1
    assert resumed["instance_id"] != inst0["id"]
    assert resumed["lease_id"] != lease0["id"]
    assert provider.create_calls == 1
    assert _service_budget_used(service) == pytest.approx(10.0)


def test_r1_5_dual_scheduler_race_creates_once_and_single_budget(tmp_path):
    _, repo = _make_store(tmp_path, "dual_race.db")
    _setup_account(repo, provider="mock")
    cfg = GpuConfig(real_enabled=False, unit_rate=1.0, daily_units=100.0, monthly_units=500.0, create_retries=1)
    service = InstanceService(repo, config=cfg, clock=lambda: 1000.0)
    job = repo.create(
        "jobs",
        "owner-1",
        {**_legacy_bare_job_body(provider="mock"), "state": "allocating", "estimated_units": 10.0},
    )
    inst0 = service.allocate(job)
    with service._tx() as session:
        row = service._find(session, "instances", inst0["id"], "owner-1")
        data = service._data(row)
        data["budget_reserved"] = 10.0
        service._set(row, data, session=session)
    lease0 = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "mock",
            "state": "creating",
            "owner_job_id": job["id"],
            "instance_id": inst0["id"],
            "account_id": "acct-1",
            "request_id": None,
            "runtime_id": None,
            "create_attempts": 0,
            "from_active_missing_handle": False,
        },
    )
    repo.update("jobs", job["id"], {"lease_id": lease0["id"], "instance_id": inst0["id"]}, "owner-1")

    create_lock = threading.Lock()
    create_count = {"n": 0}
    real_alloc = service.allocate

    def alloc_with_charge(j):
        inst = real_alloc(j)
        with service._tx() as session:
            row = service._find(session, "instances", inst["id"], "owner-1")
            data = service._data(row)
            if float(data.get("budget_reserved") or 0) == 0.0:
                data["budget_reserved"] = 10.0
                service._set(row, data, session=session)
        return service.get("owner-1", inst["id"])

    service.allocate = alloc_with_charge

    class SyncAbsentProvider(MockColabProvider):
        def inspect(self, handle):
            return {"ready": False, "deleted": True, "remote_verified": True}

        def create(self, spec, request_id, job_id, runtime_id):
            with create_lock:
                create_count["n"] += 1
            return super().create(spec, request_id, job_id, runtime_id)

    provider = SyncAbsentProvider()
    barrier = threading.Barrier(2)

    def run_worker():
        s = Scheduler(repo, provider_factory=lambda j: provider, config=cfg, instance_service=service, clock=lambda: 1005.0)
        barrier.wait()
        return s.tick()

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(run_worker)
        f2 = pool.submit(run_worker)
        f1.result()
        f2.result()

    assert create_count["n"] == 1
    assert _service_budget_used(service) == pytest.approx(10.0)
    final_job = repo.get("jobs", job["id"], "owner-1")
    assert final_job["state"] in {"preparing", "loading", "running", "uploading", "completed"}


def test_real_created_deleted_instance_never_resurrects():
    _, repo = _make_store()
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)
    job = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "id": "job-real", "state": "queued"}, record_id="job-real")
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {"provider": "mock", "state": "active", "owner_job_id": job["id"], "instance_id": inst["id"], "account_id": "acct-1", "handle": {"runtime_name": "rt-real"}},
    )
    inst = service.bind_lease(job, lease)
    inst = service.mark_ready(job)
    inst = service.job_busy(job)
    deleted = service.mark_deleted("owner-1", inst["id"], remote_deleted=True)
    assert deleted["state"] == "deleted"

    # Negative 1: cannot retire real created instance as pure placeholder
    with pytest.raises(InstanceConflict):
        service.retire_pure_placeholder("owner-1", inst["id"], lease_id=lease["id"])

    # Negative 2: allocate with same job/key or requested_instance_id NEVER resurrects deleted instance
    with pytest.raises(InstanceConflict):
        service.allocate(job)
    with pytest.raises(Exception):
        service.allocate({**_legacy_bare_job_body(), "id": "job-2", "owner_id": "owner-1", "requested_instance_id": inst["id"]})

    # Negative 3: complete/retention/bind_lease/mark_ready/job_busy NEVER resurrect deleted instance
    with pytest.raises(InstanceConflict):
        service.complete({**job, "retention_mode": "retain"}, success=True)
    with pytest.raises(InstanceConflict):
        service.retention("owner-1", inst["id"], "retain", 5, "ret-key-del")
    with pytest.raises(InstanceConflict):
        service.mark_ready(job)
    with pytest.raises(InstanceConflict):
        service.job_busy(job)
    assert service.get("owner-1", inst["id"])["state"] == "deleted"


# ---------------------------------------------------------------------------
# 3. Allowed-Transition Matrix & Cross-Phase Fencing
# ---------------------------------------------------------------------------


def test_instance_allowed_transition_matrix_and_legacy_cleanup_pending_never_idle():
    _, repo = _make_store()
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)

    # Cleanup states force retain=False on complete
    for cleanup_state in ("releasing", "cleanup_pending"):
        j = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "queued", "execution_policy": {"provider": "mock", "lifecycle": "retain"}})
        inst = service.allocate({**j, "retention_mode": "retain"})
        lease = repo.create("leases", "owner-1", {"provider": "mock", "state": "active", "owner_job_id": j["id"], "instance_id": inst["id"], "account_id": "acct-1", "handle": {"rt": 1}})
        service.bind_lease(j, lease)
        service.mark_ready(j)
        service.job_busy(j)
        if cleanup_state == "releasing":
            service.stop("owner-1", inst["id"], f"stop-{cleanup_state}")
        else:
            with service._tx() as session:
                row = service._find(session, "instances", inst["id"], "owner-1")
                data = service._data(row)
                service._set(row, data, state="cleanup_pending", session=session)
        # Re-attaching current_job_id must NOT retain or go backwards
        with pytest.raises(InstanceConflict):
            service.mark_ready(j)
        with pytest.raises(InstanceConflict):
            service.job_busy(j)
        # stop() on cleanup_pending must NOT move backwards to releasing
        if cleanup_state == "cleanup_pending":
            stopped = service.stop("owner-1", inst["id"], "stop-no-back")
            assert stopped["state"] == "cleanup_pending"
        done = service.complete(j, success=True)
        assert done["state"] == cleanup_state
        assert done["state"] != "retained_idle"

    # Legacy cleanup_pending lease NEVER transitions to idle/active
    _, repo2 = _make_store()
    _setup_account(repo2, provider="mock")
    old_job = repo2.create(
        "jobs",
        "owner-1",
        {**_legacy_bare_job_body(), "state": "cleaning_up", "batch_id": "b1", "cleanup_final_state": "completed", "compatibility_signature": "sig"},
    )
    next_job = repo2.create(
        "jobs",
        "owner-1",
        {**_legacy_bare_job_body(), "state": "waiting_for_capacity", "batch_id": "b1", "compatibility_signature": "sig"},
    )
    bad_lease = repo2.create(
        "leases",
        "owner-1",
        {
            "provider": "mock",
            "state": "cleanup_pending",
            "owner_job_id": old_job["id"],
            "deadline_epoch": 9999.0,
            "handle": {"runtime_name": "rt-bad", "job_id": old_job["id"], "owner_job_id": old_job["id"]},
        },
    )
    old_job = repo2.update("jobs", old_job["id"], {"lease_id": bad_lease["id"]}, "owner-1")
    assert repo2.attach_idle_lease("owner-1", next_job["id"], bad_lease["id"], 100.0) is False
    assert repo2.handoff_lease("owner-1", old_job["id"], next_job["id"], bad_lease["id"], 100.0) is False
    assert repo2.get("leases", bad_lease["id"], "owner-1")["state"] == "cleanup_pending"


@pytest.mark.parametrize("stage", ["preparing", "loading", "running", "uploading"])
@pytest.mark.parametrize("fence", ["deleted_instance", "cleanup_pending_lease", "manual_stop"])
def test_preparing_loading_running_uploading_fenced_before_advancing(stage, fence):
    _, repo = _make_store()
    cfg = GpuConfig(real_enabled=False, unit_rate=1.0, daily_units=100.0, monthly_units=500.0)
    service = InstanceService(repo, config=cfg, clock=lambda: 1000.0)
    _setup_account(repo, provider="mock")
    job = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating"})
    inst = service.allocate(job)
    provider = MockColabProvider()
    handle = provider.bind_owner(provider.create({}, "req-f", job["id"], "rt-f"), "owner-1")
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "mock",
            "state": "active",
            "owner_job_id": job["id"],
            "instance_id": inst["id"],
            "account_id": "acct-1",
            "request_id": "req-f",
            "runtime_id": "rt-f",
            "handle": handle,
        },
    )
    service.bind_lease(job, lease)
    if stage in {"loading", "running", "uploading"}:
        service.mark_ready(job)
    if stage in {"running", "uploading"}:
        service.job_busy(job)
        job = repo.update("jobs", job["id"], {"state": "loading", "lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")
        job = repo.prepare_execution(job["id"], "owner-1", lease["id"], 1000.0)
    job = repo.update(
        "jobs",
        job["id"],
        {
            "state": stage,
            "lease_id": lease["id"],
            "instance_id": inst["id"],
            **({"outputs": [{"filename": "a.png", "mime": "image/png", "content": "eA==", "content_encoding": "base64"}]} if stage == "uploading" else {}),
        },
        "owner-1",
    )

    if fence == "deleted_instance":
        service.mark_deleted("owner-1", inst["id"], remote_deleted=True)
    elif fence == "cleanup_pending_lease":
        repo.update("leases", lease["id"], {"state": "cleanup_pending"}, "owner-1")
    elif fence == "manual_stop":
        service.stop("owner-1", inst["id"], "manual-stop-k")

    started = []

    class ProbeExecutor(MockGPUExecutor):
        def start(self, h, j, eid):
            started.append(eid)
            return super().start(h, j, eid)

        def poll(self, h, j):
            return {"state": "completed", "execution_id": j["execution_id"], "outputs": [{"filename": "o.png", "mime": "image/png", "content": b"ok"}]}

    scheduler = Scheduler(
        repo,
        provider_factory=lambda j: provider,
        executor_factory=lambda j: ProbeExecutor(),
        config=cfg,
        instance_service=service,
        clock=lambda: 1005.0,
    )
    out = scheduler._advance(repo.get("jobs", job["id"], "owner-1"))
    assert started == []
    assert out["state"] not in {"loading", "running", "uploading", "completed"}
    assert out["state"] in {"cleaning_up", "cleanup_pending", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# 4. Linearized start x stop, _run MCPTransient + cancel, neighbor lease safety
# ---------------------------------------------------------------------------


def test_start_x_stop_barriers_start_intent_and_load_fences(tmp_path, monkeypatch):
    # Case A: Registered not started -> cancel/stop aborts start_intent -> _load refuses start & never sets running
    _, repo_a = _make_store()
    job_a = repo_a.create("jobs", "u", {"state": "loading", "provider": "mock"})
    prov_a = MockColabProvider()
    h_a = prov_a.bind_owner(prov_a.create({}, f"r1-{uuid.uuid4().hex[:6]}", job_a["id"], f"rt1-{uuid.uuid4().hex[:6]}"), "u")
    lease_a = repo_a.create("leases", "u", {"state": "active", "owner_job_id": job_a["id"], "handle": h_a})
    job_a = repo_a.update("jobs", job_a["id"], {"lease_id": lease_a["id"]}, "u")
    prepared_a = repo_a.prepare_execution(job_a["id"], "u", lease_a["id"], 1.0)
    assert (prepared_a.get("start_intent") or {}).get("state") == "starting"
    sched_a = Scheduler(repo_a, provider_factory=lambda j: prov_a, executor_factory=lambda j: MockGPUExecutor())
    sched_a.cancel("u", job_a["id"])
    refreshed_a = repo_a.get("jobs", job_a["id"], "u")
    assert refreshed_a["start_intent"]["state"] == "aborted"
    res_a = sched_a._load(refreshed_a)
    assert res_a["state"] != "running"

    # Case B: In-flight executor.start + concurrent POST /admin/compute/stop -> _commit_running_locked blocks running & job_busy
    monkeypatch.setattr(settings, "data_dir", tmp_path / "media")
    monkeypatch.setattr(settings, "session_secret", "test-session-private-signing-key")
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    st_b = Store("sqlite:///" + str(tmp_path / "start_stop.db"))
    repo_b = Repository(st_b)
    ensure_schema(st_b)
    user = st_b.create("users", "bootstrap", {"username": "gpu-admin", "role": "admin"})
    _setup_account(repo_b, owner=user["id"], provider="mock")
    service_b = InstanceService(repo_b, clock=lambda: 100.0)
    app.state.store = st_b
    app.state.instance_service = service_b
    token = sign_session({"uid": user["id"], "role": "admin", "iat": int(time.time()), "exp": time.time() + 3600})

    job_b = repo_b.create("jobs", user["id"], {**_legacy_bare_job_body(), "owner_id": user["id"], "state": "allocating"})
    inst_b = service_b.allocate(job_b)
    prov_b = MockColabProvider()
    h_b = prov_b.bind_owner(prov_b.create({}, f"req-b-{uuid.uuid4().hex[:6]}", job_b["id"], f"rt-b-{uuid.uuid4().hex[:6]}"), user["id"])
    lease_b = repo_b.create("leases", user["id"], {"provider": "mock", "state": "active", "owner_job_id": job_b["id"], "instance_id": inst_b["id"], "account_id": "acct-1", "handle": h_b})
    service_b.bind_lease(job_b, lease_b)
    service_b.mark_ready(job_b)
    job_b = repo_b.update("jobs", job_b["id"], {"state": "loading", "lease_id": lease_b["id"], "instance_id": inst_b["id"]}, user["id"])

    in_start = threading.Event()
    allow_start_return = threading.Event()

    class BarrierExecutor(MockGPUExecutor):
        def start(self, handle, job, execution_id):
            in_start.set()
            allow_start_return.wait(timeout=5.0)
            return super().start(handle, job, execution_id)

    sched_b = Scheduler(
        repo_b,
        provider_factory=lambda j: prov_b,
        executor_factory=lambda j: BarrierExecutor(),
        instance_service=service_b,
        clock=lambda: 101.0,
    )
    prev_store = getattr(app.state, "store", None)
    prev_svc = getattr(app.state, "instance_service", None)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(sched_b._load, repo_b.get("jobs", job_b["id"], user["id"]))
            assert in_start.wait(timeout=3.0)
            with TestClient(app) as client:
                client.headers["Authorization"] = "Bearer " + token
                resp = client.post("/api/gpu/admin/compute/stop", headers={"X-CSRF-Check": "1"})
                assert resp.status_code == 200
            allow_start_return.set()
            res_b = fut.result(timeout=5.0)
    finally:
        app.state.store = prev_store
        app.state.instance_service = prev_svc

    assert res_b["state"] != "running"
    assert res_b["requested_outcome"] == "cancelled"
    assert service_b.get(user["id"], inst_b["id"])["state"] != "busy"

    # Case C: job_busy refusal NOT swallowed
    _, repo_c = _make_store()
    _setup_account(repo_c, provider="mock")
    service_c = InstanceService(repo_c, clock=lambda: 100.0)
    job_c = repo_c.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating"})
    inst_c = service_c.allocate(job_c)
    prov_c = MockColabProvider()
    h_c = prov_c.bind_owner(prov_c.create({}, "req-c", job_c["id"], "rt-c"), "owner-1")
    lease_c = repo_c.create("leases", "owner-1", {"provider": "mock", "state": "active", "owner_job_id": job_c["id"], "instance_id": inst_c["id"], "account_id": "acct-1", "handle": h_c})
    service_c.bind_lease(job_c, lease_c)
    service_c.mark_ready(job_c)
    job_c = repo_c.update("jobs", job_c["id"], {"state": "loading", "lease_id": lease_c["id"], "instance_id": inst_c["id"]}, "owner-1")

    def stop_during_start(handle, j, eid):
        service_c.stop("owner-1", inst_c["id"], "conflict-key")
        return {"started": True, "execution_id": eid}

    res_c = Scheduler(
        repo_c,
        provider_factory=lambda j: prov_c,
        executor_factory=lambda j: SimpleNamespace(start=stop_during_start),
        instance_service=service_c,
        clock=lambda: 101.0,
    )._load(job_c)
    assert res_c["state"] != "running"

    # Case D: _load early-return when execution missing or failed/aborted -> NEVER sets running
    for exec_state in (None, "failed", "aborted", "cancelled"):
        _, repo_d = _make_store()
        exec_id = "missing-exec"
        if exec_state is not None:
            ex = repo_d.create("executions", "u", {"state": exec_state, "job_id": "job-d"})
            exec_id = ex["id"]
        lease_d = repo_d.create("leases", "u", {"state": "active", "owner_job_id": "job-d", "handle": {"rt": 1}})
        job_d = repo_d.create("jobs", "u", {"state": "loading", "provider": "mock", "execution_id": exec_id, "lease_id": lease_d["id"]}, record_id="job-d")
        res_d = Scheduler(repo_d)._load(job_d)
        assert res_d["state"] != "running"

    # Case E: an explicitly bound mock execution starts through _commit_running_locked cleanly
    _, repo_e = _make_store()
    prov_e = MockColabProvider()
    job_e = repo_e.create("jobs", "u", {"state": "loading", "provider": "mock"}, record_id="job-e")
    h_e = prov_e.bind_owner(prov_e.create({}, f"req-e-{uuid.uuid4().hex[:6]}", "job-e", f"rt-e-{uuid.uuid4().hex[:6]}"), "u")
    lease_e = repo_e.create("leases", "u", {"state": "active", "owner_job_id": "job-e", "handle": h_e})
    job_e = repo_e.update("jobs", "job-e", {"lease_id": lease_e["id"]}, "u")
    job_e = repo_e.prepare_execution(job_e["id"], "u", lease_e["id"], 1.0)
    job_e = seed_execution_binding(repo_e, job_e, lease_e)
    sched_e = Scheduler(repo_e, provider_factory=lambda j: prov_e, executor_factory=lambda j: MockGPUExecutor())
    called_commit = {"n": 0}
    orig_commit = sched_e._commit_running_locked

    def wrapped_commit(*a, **k):
        called_commit["n"] += 1
        return orig_commit(*a, **k)

    sched_e._commit_running_locked = wrapped_commit
    res_e = sched_e._load(job_e)
    assert res_e["state"] == "running"
    assert called_commit["n"] == 1


def test_running_mcp_transient_resume_preserves_running_no_re_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "media")
    _, repo = _make_store()
    _setup_account(repo, provider="mock")
    now = [1000.0]
    service = InstanceService(repo, clock=lambda: now[0])
    job = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating"})
    inst = service.allocate(job)
    prov = MockColabProvider()
    h = prov.bind_owner(prov.create({}, f"req-r-{uuid.uuid4().hex[:6]}", job["id"], f"rt-r-{uuid.uuid4().hex[:6]}"), "owner-1")
    lease = repo.create("leases", "owner-1", {"provider": "mock", "state": "active", "owner_job_id": job["id"], "instance_id": inst["id"], "account_id": "acct-1", "handle": h})
    service.bind_lease(job, lease)
    service.mark_ready(job)
    job = repo.update("jobs", job["id"], {"state": "loading", "lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")

    busy_calls = {"n": 0}
    orig_busy = service.job_busy

    def track_busy(j):
        busy_calls["n"] += 1
        return orig_busy(j)

    service.job_busy = track_busy

    poll_step = {"n": 0}

    class FlakyExecutor(MockGPUExecutor):
        def poll(self, handle, j):
            poll_step["n"] += 1
            if poll_step["n"] == 1:
                raise MCPTransient("temporary disconnect")
            return {"state": "completed", "execution_id": j["execution_id"], "outputs": [{"filename": "out.png", "mime": "image/png", "content": _valid_png_bytes()}]}

    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: FlakyExecutor(), instance_service=service, clock=lambda: now[0])
    running = sched._load(job)
    assert running["state"] == "running"
    assert service.get("owner-1", inst["id"])["state"] == "busy"
    assert (
        sum(
            1
            for ev in repo.list("events", "owner-1")
            if ev.get("instance_id") == inst["id"] and ev.get("state") == "busy"
        )
        == 1
    )

    transient = sched._run(running)
    assert transient["state"] == "running"
    assert transient["poll_retry_count"] == 1
    assert repo.get("executions", running["execution_id"], "owner-1")["state"] == "running"

    now[0] = float(transient["poll_retry_at"]) + 1.0
    uploaded = sched._run(repo.get("jobs", job["id"], "owner-1"))
    assert uploaded["state"] == "uploading"
    assert service.get("owner-1", inst["id"])["state"] == "busy"
    assert (
        sum(
            1
            for ev in repo.list("events", "owner-1")
            if ev.get("instance_id") == inst["id"] and ev.get("state") == "busy"
        )
        == 1
    )
    final = sched._cleanup(sched._upload(uploaded))
    assert final["state"] == "completed"


@pytest.mark.parametrize(
    "exit_case",
    [
        "mcp_transient_retry",
        "mcp_transient_exhausted",
        "backoff_clear_in_progress",
        "execution_id_mismatch",
        "gpu_diagnostics_write",
        "stage_timings_progress_write",
        "completed_outputs",
        "completed_artifacts",
        "failed_result",
        "completed_empty_outputs",
    ],
)
def test_concurrent_cancel_racing_late_poll_blocked_across_all_run_exits(tmp_path, exit_case):
    _, repo = _make_store(tmp_path, f"race_{exit_case}.db")
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)
    job = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating"})
    inst = service.allocate(job)
    prov = MockColabProvider()
    h = prov.bind_owner(prov.create({}, f"req-p-{exit_case}-{uuid.uuid4().hex[:6]}", job["id"], f"rt-p-{exit_case}-{uuid.uuid4().hex[:6]}"), "owner-1")
    lease = repo.create("leases", "owner-1", {"provider": "mock", "state": "active", "owner_job_id": job["id"], "instance_id": inst["id"], "account_id": "acct-1", "handle": h})
    service.bind_lease(job, lease)
    service.mark_ready(job)
    job = repo.update("jobs", job["id"], {"state": "loading", "lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")
    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockGPUExecutor(), instance_service=service, clock=lambda: 1000.0)
    running = sched._load(job)
    assert running["state"] == "running"
    exec_id = running["execution_id"]

    if exit_case == "mcp_transient_exhausted":
        running = repo.update("jobs", job["id"], {"poll_retry_count": 5, "poll_retry_at": 0}, "owner-1")
    elif exit_case == "backoff_clear_in_progress":
        running = repo.update("jobs", job["id"], {"poll_retry_count": 2, "poll_retry_at": 0}, "owner-1")

    in_poll = threading.Event()
    allow_poll_return = threading.Event()

    class RacingExecutor:
        def poll(self, handle, j):
            in_poll.set()
            allow_poll_return.wait(timeout=5.0)
            if exit_case in {"mcp_transient_retry", "mcp_transient_exhausted"}:
                raise MCPTransient("lost link")
            if exit_case in {"backoff_clear_in_progress", "stage_timings_progress_write"}:
                return {"state": "running", "execution_id": exec_id, "stage": "sampling", "progress": 0.5, "timings": {"sampling": 3.2}}
            if exit_case == "execution_id_mismatch":
                return {"state": "completed", "execution_id": "wrong-exec", "outputs": [{"filename": "a.png", "mime": "image/png", "content": b"x"}]}
            if exit_case == "gpu_diagnostics_write":
                return {"state": "running", "execution_id": exec_id, "gpu_diagnostics": {"devices": [{"name": "T4"}]}}
            if exit_case == "completed_outputs":
                return {
                    "state": "completed",
                    "execution_id": exec_id,
                    "outputs": [{"filename": "a.png", "mime": "image/png", "content": b"x"}],
                    "warm_saved": True,
                    "worker_reported_execution_id": exec_id,
                    "warm_epoch_id": "ep1",
                    "warm_fingerprint": "fp1",
                    "runtime_meta": {"python_mm": "3.12"},
                }
            if exit_case == "completed_artifacts":
                return {
                    "state": "completed",
                    "execution_id": exec_id,
                    "artifacts": {"files": []},
                    "warm_saved": True,
                    "worker_reported_execution_id": exec_id,
                    "warm_epoch_id": "ep1",
                    "warm_fingerprint": "fp1",
                    "runtime_meta": {"python_mm": "3.12"},
                }
            if exit_case == "failed_result":
                return {"state": "failed", "execution_id": exec_id, "error_code": "oom", "error_message": "OOM"}
            if exit_case == "completed_empty_outputs":
                return {"state": "completed", "execution_id": exec_id, "outputs": []}
            raise AssertionError(exit_case)

    race_sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: RacingExecutor(), instance_service=service, clock=lambda: 1001.0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(race_sched._run, running)
        assert in_poll.wait(timeout=3.0)
        race_sched.cancel("owner-1", job["id"])
        allow_poll_return.set()
        after = fut.result(timeout=5.0)

    assert after["requested_outcome"] == "cancelled"
    assert after["state"] != "uploading"
    assert after.get("outputs") in (None, [])
    assert after.get("gpu_reported") is None
    assert after.get("gpu_diagnostics") is None
    assert after.get("stage") != "sampling"
    if exit_case == "mcp_transient_retry":
        assert not after.get("poll_retry_count")
    assert repo.get("leases", lease["id"], "owner-1").get("warm_proof") is None


def test_neighbor_job_a_lease_with_stolen_current_job_id_never_zeroes_budget_or_deletes():
    _, repo = _make_store()
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)

    job_a = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating"}, record_id="job-a")
    inst_a = service.allocate(job_a)
    with service._tx() as session:
        row = service._find(session, "instances", inst_a["id"], "owner-1")
        data = service._data(row)
        data["budget_reserved"] = 10.0
        service._set(row, data, session=session)

    prov = MockColabProvider()
    h_a = prov.bind_owner(prov.create({}, "req-a", "job-a", "rt-a"), "owner-1")
    lease_a = repo.create(
        "leases",
        "owner-1",
        {
            "provider": "mock",
            "state": "active",
            "owner_job_id": "job-a",
            "current_job_id": "job-b",  # stolen by neighbor job B
            "instance_id": inst_a["id"],
            "account_id": "acct-1",
            "request_id": "req-a",
            "runtime_id": "rt-a",
            "handle": h_a,
        },
    )
    job_a = repo.update("jobs", "job-a", {"state": "running", "lease_id": lease_a["id"], "instance_id": inst_a["id"]}, "owner-1")
    job_b = repo.create("jobs", "owner-1", {**_legacy_bare_job_body(), "state": "allocating", "lease_id": lease_a["id"], "instance_id": inst_a["id"]}, record_id="job-b")

    sched = Scheduler(repo, provider_factory=lambda j: prov, executor_factory=lambda j: MockGPUExecutor(), instance_service=service, clock=lambda: 1005.0)
    # Job B failing must NEVER delete Job A's runtime or zero Job A's instance budget
    res_b = sched._begin_failure(job_b, "gpu_test_fail", "b failed")
    if res_b["state"] == "cleaning_up":
        sched._cleanup(res_b)
    # Job A failing while B holds current_job_id must also NOT zero/delete torn lease runtime or mark_deleted
    res_a = sched._begin_failure(job_a, "gpu_test_fail", "a failed")
    if res_a["state"] == "cleaning_up":
        sched._cleanup(res_a)
    assert prov.deleted == []
    assert _service_budget_used(service) == pytest.approx(10.0)
    assert service.get("owner-1", inst_a["id"])["state"] != "deleted"
