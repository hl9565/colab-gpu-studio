"""Regression tests covering all PR4 Astra Stage-A sign-off review findings (P0-1..P0-3, P1-1..P1-8, P2)."""

from __future__ import annotations

import contextlib
import json
import os
import threading
from types import SimpleNamespace

import httpx
import pytest
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.compute import ComputeError, GpuConfig, MockColabProvider, MockGPUExecutor
from app.gpu.estimator import assess_job_reuse_window, check_job_reuse_guard, is_warm_verified_for_job
from app.gpu.instances import AllocationError, InstanceConflict, InstanceService
from app.gpu.mcp import MCPGPUExecutor, MCPTransient, build_poll_code
from app.gpu.repository import Repository, ensure_schema, is_valid_runtime_handle, normalize_tunnel_endpoint
from app.gpu.runtime_worker import compute_manifest_warm_fingerprint
from app.gpu.scheduler import Scheduler

MANIFEST_MODEL = {
    "state": "published",
    "reviewed": True,
    "comfy_commit": "a" * 40,
    "node_allowlist": ["KSampler"],
    "compute_spec": {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"},
    "weight_files": [
        {
            "source": "huggingface",
            "repo_id": "org/model",
            "filename": "model.safetensors",
            "revision": "a" * 40,
            "target": "checkpoints/model.safetensors",
            "sha256": "b" * 64,
            "size": 8,
        }
    ],
}


def _make_repo(tmp_path=None, name: str = "pr4_signoff.db") -> tuple[Store, Repository]:
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
    provider: str = "mock",
    daily_limit: float = 100.0,
    monthly_limit: float = 500.0,
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


def _tunnel_body(payload: dict) -> str:
    return ")]}'\n" + json.dumps(payload)


def _calibrated_colab_job(
    *,
    job_id: str = "job-1",
    owner_id: str = "owner-1",
    state: str = "queued",
    lifecycle: str = "retain",
) -> dict:
    return {
        "id": job_id,
        "owner_id": owner_id,
        "project_id": "proj-1",
        "workflow_key": "minimax-h3-ref2v",
        "workflow_version": "1.4.0",
        "gpu_tier": "L4",
        "state": state,
        "provider": "colab",
        "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": lifecycle},
        "lifecycle": lifecycle,
        "compatibility_signature": "sig-ref2v",
        "max_runtime_minutes": 30,
        "estimated_units": 2.0,
        "snapshot": {
            "manifest": {**MANIFEST_MODEL, "workflow_key": "minimax-h3-ref2v", "version": "1.4.0", "modality": "video"},
            "resolved_params": {"width": 544, "height": 960, "length": 124, "mode": "fast8"},
            "bound_graph": {},
        },
    }


def _budget_totals(service: InstanceService, owner: str = "owner-1", account_id: str = "acct-1") -> tuple[float, float]:
    with service._tx() as session:
        return service._budget_totals(session, owner, account_id)


# ---------------------------------------------------------------------------
# P0-1: Rejected lease retirement must abort create and enter reconciliation
# ---------------------------------------------------------------------------


def test_p0_1_rejected_retirement_with_active_lease_a_never_creates_lease_b_or_rpc():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())
    provider = MockColabProvider()
    executor = MockGPUExecutor()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: executor,
        clock=lambda: 1000.0,
        config=_colab_config(),
        instance_service=service,
    )

    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": "job-b",
            "lease_id": "lease-a",
            "budget_reserved": 2.0,
            "generation": 1,
        },
        record_id="inst-shared",
    )
    lease_a = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-a",
            "current_job_id": "job-a",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "runtime_id": "rt-a",
            "request_id": "req-a",
            "handle": {"runtime_id": "rt-a", "request_id": "req-a", "job_id": "job-a", "owner_job_id": "job-a"},
        },
        record_id="lease-a",
    )
    job_b = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acct-1",
            "project_id": "proj-1",
            "instance_id": inst["id"],
            "lease_id": "lease-missing",
            "alloc_generation": 0,
            "max_runtime_minutes": 10,
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
            "snapshot": {"manifest": {"compute_spec": MANIFEST_MODEL["compute_spec"]}},
        },
        record_id="job-b",
    )

    updated_b = scheduler._allocation(job_b)

    assert updated_b["state"] == "cleanup_pending"
    assert updated_b["error_code"] == "gpu_instance_conflict"
    assert provider.create_calls == 0
    assert [row["id"] for row in repo.list("leases", "owner-1")] == [lease_a["id"]]
    assert repo.get("leases", lease_a["id"], "owner-1")["state"] == "active"
    assert repo.get("instances", inst["id"], "owner-1")["lease_id"] == lease_a["id"]


def test_p0_1_bind_lease_rejects_second_non_deleted_lease_on_instance_or_runtime():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())

    repo.create(
        "instances",
        "owner-1",
        {"state": "allocating", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-b", "lease_id": "lease-a"},
        record_id="inst-1",
    )
    repo.create(
        "leases",
        "owner-1",
        {"state": "active", "owner_job_id": "job-a", "instance_id": "inst-1", "runtime_id": "rt-1", "account_id": "acct-1"},
        record_id="lease-a",
    )
    lease_b = repo.create(
        "leases",
        "owner-1",
        {"state": "creating", "owner_job_id": "job-b", "runtime_id": "rt-1", "account_id": "acct-1"},
        record_id="lease-b",
    )
    job_b = repo.create("jobs", "owner-1", {"state": "allocating", "instance_id": "inst-1", "account_id": "acct-1"}, record_id="job-b")

    with pytest.raises(InstanceConflict):
        service.bind_lease(job_b, lease_b)


# ---------------------------------------------------------------------------
# P0-2 + P2: Fail-closed tunnel assignments, verified delete proof, Mock tombstone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_entry", [None, {}, {"proxy": "https://x"}, {"endpoint": "   "}])
def test_p0_2_colab_tunnel_invalid_assignment_entry_never_confirms_deleted(bad_entry):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/assignments":
            return httpx.Response(200, text=_tunnel_body({"assignments": [bad_entry]}))
        raise AssertionError(f"unexpected {request.method} {request.url}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        account_id="acct-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        google=_tunnel_google(),
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-1",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-1",
            "request_id": "req-1",
            "tunnel_endpoint": "ep-1",
        },
        record_id="lease-1",
    )
    handle = {
        "runtime_name": "runtimes/rt-1",
        "runtime_id": "rt-1",
        "request_id": "req-1",
        "owner_id": "owner-1",
        "job_id": "job-1",
        "owner_job_id": "job-1",
    }

    inspected = provider.inspect(handle)
    assert inspected["deleted"] is False
    assert inspected["allocation_unknown"] is True
    assert inspected["remote_verified"] is False

    with pytest.raises(ColabError):
        provider.delete(handle)


def test_p0_2_unverified_delete_retains_budget_and_retries_until_verified():
    class UnverifiedThenVerifiedProvider:
        def __init__(self):
            self.inspect_calls = 0
            self.delete_calls = 0
            self.verified = False

        def inspect(self, handle):
            self.inspect_calls += 1
            return {"ready": False, "deleted": True, "remote_verified": False}

        def delete(self, handle):
            self.delete_calls += 1
            if self.verified:
                return {"deleted": True, "remote_verified": True}
            return {"deleted": True, "remote_verified": False}

    _, repo = _make_repo()
    _setup_account(repo, provider="colab", daily_limit=20.0, monthly_limit=50.0)
    now = [1000.0]
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: now[0], config=cfg)
    provider = UnverifiedThenVerifiedProvider()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: now[0],
        config=cfg,
        instance_service=service,
    )

    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "releasing",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": "job-1",
            "lease_id": "lease-1",
            "budget_reserved": 4.0,
            "runtime_created": True,
        },
        record_id="inst-1",
    )
    handle = {"runtime_name": "rt-1", "runtime_id": "rt-1", "request_id": "req-1", "owner_id": "owner-1", "job_id": "job-1", "owner_job_id": "job-1"}
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-1",
            "current_job_id": "job-1",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "runtime_id": "rt-1",
            "request_id": "req-1",
            "handle": handle,
        },
        record_id="lease-1",
    )
    repo.create(
        "jobs",
        "owner-1",
        {
            "state": "cleaning_up",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "lease_id": "lease-1",
            "requested_outcome": "failed",
            "cleanup_final_state": "failed",
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
        },
        record_id="job-1",
    )

    scheduler.tick()
    assert repo.get("jobs", "job-1", "owner-1")["state"] == "cleanup_pending"
    assert repo.get("leases", "lease-1", "owner-1")["state"] == "cleanup_pending"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(service) == (4.0, 4.0)

    provider.verified = True
    now[0] += 100.0
    scheduler.tick()
    assert repo.get("jobs", "job-1", "owner-1")["state"] == "failed"
    assert repo.get("leases", "lease-1", "owner-1")["state"] == "deleted"
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "deleted"
    assert _budget_totals(service) == (0.0, 0.0)


def test_p0_2_orphan_watchdog_unverified_delete_retains_budget_and_retries():
    class UnverifiedOrphanProvider:
        def __init__(self):
            self.verified = False

        def inspect(self, handle):
            return {"ready": False, "deleted": True, "remote_verified": False}

        def delete(self, handle):
            return {"deleted": True, "remote_verified": self.verified}

    _, repo = _make_repo()
    _setup_account(repo, provider="colab", daily_limit=20.0, monthly_limit=50.0)
    now = [1000.0]
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: now[0], config=cfg)
    provider = UnverifiedOrphanProvider()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: now[0],
        config=cfg,
        instance_service=service,
    )
    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "releasing",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": None,
            "lease_id": "lease-orp",
            "budget_reserved": 3.0,
            "runtime_created": True,
        },
        record_id="inst-orp",
    )
    repo.create(
        "jobs",
        "owner-1",
        {"state": "failed", "provider": "colab", "account_id": "acct-1", "instance_id": inst["id"], "lease_id": "lease-orp"},
        record_id="job-orp",
    )
    handle = {"runtime_name": "rt-orp", "runtime_id": "rt-orp", "request_id": "req-orp", "owner_id": "owner-1", "job_id": "job-orp", "owner_job_id": "job-orp"}
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-orp",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "runtime_id": "rt-orp",
            "request_id": "req-orp",
            "handle": handle,
            "deadline_epoch": 500.0,
        },
        record_id="lease-orp",
    )

    scheduler.watchdog()
    assert repo.get("leases", "lease-orp", "owner-1")["state"] == "cleanup_pending"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(service) == (3.0, 3.0)

    provider.verified = True
    now[0] += 100.0
    scheduler.watchdog()
    assert repo.get("leases", "lease-orp", "owner-1")["state"] == "deleted"
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "deleted"
    assert _budget_totals(service) == (0.0, 0.0)


def test_p2_mock_provider_requires_verified_tombstone_and_deletion_record():
    provider = MockColabProvider()
    handle = provider.create({"provider": "mock"}, "req-1", "job-1", "rt-1")
    handle = provider.bind_owner(handle, "owner-1")
    live = provider.inspect(handle)
    assert live["deleted"] is False
    assert live["remote_verified"] is True

    fake_uncreated = {
        "runtime_name": "mock-runtime-unknown",
        "runtime_id": "unknown",
        "request_id": "req-missing",
        "owner_id": "owner-1",
        "mock": True,
    }
    with pytest.raises(ComputeError):
        provider.inspect(fake_uncreated)
    with pytest.raises(ComputeError):
        provider.delete(fake_uncreated)

    assert provider.delete(handle) is True
    gone = provider.inspect(handle)
    assert gone["deleted"] is True
    assert gone["remote_verified"] is True

    # Removing tombstone without deletion confirmation must fail closed
    provider._tombstones.pop(handle["runtime_name"], None)
    with pytest.raises(ComputeError):
        provider.inspect(handle)


# ---------------------------------------------------------------------------
# P0-3: Watchdog must NOT locally delete non-pure creating+handle=None+attempts=0 leases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_fields,expected_lease_state",
    [
        ({"was_active_missing_handle": True, "request_id": None, "runtime_id": None}, "quarantined"),
        ({"was_active_missing_handle": True}, "quarantined"),
        ({"create_intent": {"state": "claimed"}, "create_intent_id": "ci-1", "request_id": None, "runtime_id": None}, "quarantined"),
        ({"tunnel_endpoint": "ep-live"}, "cleanup_pending"),
    ],
)
def test_p0_3_watchdog_never_locally_deletes_active_origin_intent_or_endpoint_lease(
    extra_fields, expected_lease_state
):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class ReconcileOrFailProvider:
        def inspect(self, handle):
            if handle.get("tunnel_endpoint"):
                return {"ready": True, "deleted": False, "remote_verified": True}
            raise ComputeError("runtime handle identity mismatch")

        def delete(self, handle):
            return {"deleted": False, "remote_verified": False}

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: ReconcileOrFailProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )

    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "releasing",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": None,
            "lease_id": "lease-suspect",
            "budget_reserved": 5.0,
        },
        record_id="inst-suspect",
    )
    repo.create(
        "jobs",
        "owner-1",
        {"state": "failed", "provider": "colab", "account_id": "acct-1", "instance_id": inst["id"], "lease_id": "lease-suspect"},
        record_id="job-dead",
    )
    lease_payload = {
        "state": "creating",
        "owner_job_id": "job-dead",
        "current_job_id": "job-dead",
        "provider": "colab",
        "account_id": "acct-1",
        "instance_id": inst["id"],
        "request_id": "req-s",
        "runtime_id": "rt-s",
        "create_attempts": 0,
        "handle": None,
        "deadline_epoch": 500.0,
        **extra_fields,
    }
    repo.create("leases", "owner-1", lease_payload, record_id="lease-suspect")

    scheduler.watchdog()

    lease = repo.get("leases", "lease-suspect", "owner-1")
    updated_inst = repo.get("instances", inst["id"], "owner-1")
    assert lease["state"] == expected_lease_state
    assert lease["state"] != "deleted"
    assert updated_inst["state"] != "deleted"
    assert _budget_totals(service) == (5.0, 5.0)


def test_p0_3_watchdog_quarantines_creating_zero_attempt_lease_when_execution_exists():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    scheduler = Scheduler(repo, clock=lambda: 1000.0, config=cfg, instance_service=service)

    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "releasing",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": None,
            "lease_id": "lease-exec",
            "budget_reserved": 6.0,
        },
        record_id="inst-exec",
    )
    repo.create(
        "jobs",
        "owner-1",
        {"state": "failed", "provider": "colab", "account_id": "acct-1", "instance_id": inst["id"], "lease_id": "lease-exec"},
        record_id="job-exec",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "owner_job_id": "job-exec",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "create_attempts": 0,
            "handle": None,
            "deadline_epoch": 500.0,
        },
        record_id="lease-exec",
    )
    repo.create(
        "executions",
        "owner-1",
        {"state": "failed", "job_id": "job-exec", "lease_id": "lease-exec", "instance_id": inst["id"]},
        record_id="exec-1",
    )

    scheduler.watchdog()
    assert repo.get("leases", "lease-exec", "owner-1")["state"] == "quarantined"
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "quarantined"
    assert _budget_totals(service) == (6.0, 6.0)


# ---------------------------------------------------------------------------
# P1-1: Allocation must re-read latest job & commit atomically inside admission_lock
# ---------------------------------------------------------------------------


def test_p1_1_atomic_allocation_aborts_with_zero_writes_if_job_cancelled():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    job = repo.create_job("owner-1", _calibrated_colab_job(state="waiting_for_capacity"), "k-p1-1")[0]

    repo.request_job_cancel("owner-1", job["id"], now=1001.0)
    stale_snapshot = {**job, "state": "waiting_for_capacity", "cancel_requested": False}

    with pytest.raises(InstanceConflict):
        service.allocate(stale_snapshot, atomic_lease=True)

    assert repo.list("instances", "owner-1") == []
    assert repo.list("leases", "owner-1") == []
    assert _budget_totals(service) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# P1-2: Admission lock timeout must never overwrite job with None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recover_mode", ["claim", "retry_absent"])
def test_p1_2_admission_lock_timeout_preserves_job_without_attribute_error(recover_mode):
    store, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class AbsentProvider:
        def list_specs(self):
            return [{"key": MANIFEST_MODEL["compute_spec"], "eligible": True}]

        def inspect(self, handle):
            return {"ready": False, "deleted": True, "remote_verified": True}

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: AbsentProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "allocating", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-lock", "lease_id": "lease-lock", "budget_reserved": 2.0},
        record_id="inst-lock",
    )
    lease_payload = {
        "state": "creating" if recover_mode == "claim" else "unknown",
        "owner_job_id": "job-lock",
        "current_job_id": "job-lock",
        "provider": "colab",
        "account_id": "acct-1",
        "instance_id": inst["id"],
        "request_id": "req-lock",
        "runtime_id": "rt-lock",
        "create_attempts": 0 if recover_mode == "claim" else 1,
        "handle": None,
        "deadline_epoch": 2000.0,
    }
    repo.create("leases", "owner-1", lease_payload, record_id="lease-lock")
    repo.create(
        "jobs",
        "owner-1",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acct-1",
            "project_id": "proj-1",
            "instance_id": inst["id"],
            "lease_id": "lease-lock",
            "max_runtime_minutes": 10,
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
            "snapshot": {"manifest": {"compute_spec": MANIFEST_MODEL["compute_spec"]}},
        },
        record_id="job-lock",
    )

    calls = [0]

    @contextlib.contextmanager
    def lock_after_bind(*_args, **_kwargs):
        calls[0] += 1
        # Allow bind_lease on call 1, then simulate contention on claim/retry lock
        yield calls[0] == 1

    store.admission_lock = lock_after_bind
    scheduler.tick()
    assert repo.get("jobs", "job-lock", "owner-1")["state"] == "allocating"


# ---------------------------------------------------------------------------
# P1-3: R2 must honor was_active_missing_handle and create_intent_origin across retries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin_patch",
    [
        {"was_active_missing_handle": True, "from_active_missing_handle": False},
        {"create_intent_origin": "once_active", "from_active_missing_handle": False},
    ],
)
def test_p1_3_once_active_lease_never_performs_fresh_create(origin_patch):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class AbsentInspectProvider:
        def __init__(self):
            self.create_calls = 0
            self.inspect_calls = 0
            self.confirmed_absent = False

        def list_specs(self):
            return [{"key": MANIFEST_MODEL["compute_spec"], "eligible": True}]

        def create(self, *args, **kwargs):
            self.create_calls += 1
            raise AssertionError("once-active lease must NEVER invoke provider.create")

        def inspect(self, handle):
            self.inspect_calls += 1
            if not self.confirmed_absent:
                return {"ready": False, "deleted": False, "allocation_unknown": True, "remote_verified": False}
            return {"ready": False, "deleted": True, "remote_verified": True}

    provider = AbsentInspectProvider()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "allocating", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-once", "lease_id": "lease-once", "budget_reserved": 2.0},
        record_id="inst-once",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "owner_job_id": "job-once",
            "current_job_id": "job-once",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "request_id": "req-once",
            "runtime_id": "rt-once",
            "create_attempts": 0,
            "handle": None,
            "deadline_epoch": 2000.0,
            **origin_patch,
        },
        record_id="lease-once",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "allocating",
            "provider": "colab",
            "account_id": "acct-1",
            "project_id": "proj-1",
            "instance_id": inst["id"],
            "lease_id": "lease-once",
            "max_runtime_minutes": 10,
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
            "snapshot": {"manifest": {"compute_spec": MANIFEST_MODEL["compute_spec"]}},
        },
        record_id="job-once",
    )

    # Round 1: unknown probe persists once-active marker across restart/retry
    scheduler._allocation(job)
    assert repo.get("leases", "lease-once", "owner-1")["state"] == "unknown"
    assert repo.get("leases", "lease-once", "owner-1")["was_active_missing_handle"] is True

    # Round 2: absent remote proof closes cleanly without calling provider.create()
    provider.confirmed_absent = True
    scheduler._allocation(repo.get("jobs", "job-once", "owner-1"))
    assert provider.create_calls == 0
    assert repo.get("leases", "lease-once", "owner-1")["state"] == "deleted"


# ---------------------------------------------------------------------------
# P1-4: Cancel during slow prep aborts before dispatch claim with zero RPC
# ---------------------------------------------------------------------------


def test_p1_4_cancel_during_slow_model_cache_aborts_with_zero_executor_start():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    executor = MockGPUExecutor()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: executor,
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )

    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "ready", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-p14", "lease_id": "lease-p14", "budget_reserved": 2.0},
        record_id="inst-p14",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-p14",
            "current_job_id": "job-p14",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "request_id": "req-p14",
            "runtime_id": "rt-p14",
            "handle": {"runtime_name": "rt-p14", "runtime_id": "rt-p14", "request_id": "req-p14", "job_id": "job-p14", "owner_job_id": "job-p14"},
        },
        record_id="lease-p14",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "loading",
            "provider": "colab",
            "account_id": "acct-1",
            "project_id": "proj-1",
            "instance_id": inst["id"],
            "lease_id": "lease-p14",
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
            "snapshot": {"manifest": {"compute_spec": MANIFEST_MODEL["compute_spec"]}},
        },
        record_id="job-p14",
    )

    def slow_cache(current_job):
        scheduler.cancel("owner-1", current_job["id"])
        return None

    scheduler._model_cache = slow_cache
    result = scheduler._load(job)
    assert executor.started == []
    assert result["state"] in {"cleaning_up", "cancelled"}


def test_p1_4_in_flight_cancel_after_dispatch_claim_fences_running_and_cleans_up():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class RaceExecutor(MockGPUExecutor):
        def start(self, handle, job, execution_id):
            scheduler.cancel("owner-1", job["id"])
            return super().start(handle, job, execution_id)

    executor = RaceExecutor()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: executor,
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "ready", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-race", "lease_id": "lease-race", "budget_reserved": 2.0},
        record_id="inst-race",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-race",
            "current_job_id": "job-race",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "request_id": "req-race",
            "runtime_id": "rt-race",
            "handle": {"runtime_name": "runtimes/rt-race", "runtime_id": "rt-race", "request_id": "req-race", "job_id": "job-race", "owner_job_id": "job-race"},
        },
        record_id="lease-race",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "loading",
            "provider": "colab",
            "account_id": "acct-1",
            "project_id": "proj-1",
            "instance_id": inst["id"],
            "lease_id": "lease-race",
            "execution_policy": {"provider": "colab", "account_id": "acct-1", "lifecycle": "save"},
            "snapshot": {"manifest": {"compute_spec": MANIFEST_MODEL["compute_spec"]}},
        },
        record_id="job-race",
    )

    result = scheduler._load(job)
    assert len(executor.started) == 1
    assert result["state"] == "cleaning_up"
    assert result["requested_outcome"] == "cancelled"
    assert result["start_intent"]["state"] == "aborted"


# ---------------------------------------------------------------------------
# P1-5: No-instance legacy recovery validates active lease + handle + non-stopped
# ---------------------------------------------------------------------------


def test_p1_5_no_instance_legacy_recovery_rejects_creating_without_handle():
    _, repo = _make_repo()
    lease = repo.create(
        "leases",
        "owner-1",
        {"state": "creating", "owner_job_id": "job-legacy", "provider": "mock", "handle": None},
        record_id="lease-legacy",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {"state": "loading", "provider": "mock", "lease_id": lease["id"], "legacy_scheduler_only": True},
        record_id="job-legacy",
    )

    with pytest.raises(ValueError, match="lease is not active"):
        repo.prepare_execution(job["id"], "owner-1", lease["id"], 1000.0)


def test_p1_5_no_instance_legacy_start_fence_blocks_creating_or_stopped_and_allows_active():
    _, repo = _make_repo()
    lease = repo.create(
        "leases",
        "owner-1",
        {"generation": 1, "state": "creating", "owner_job_id": "job-leg", "provider": "mock", "handle": None},
        record_id="lease-leg",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "loading",
            "provider": "mock",
            "lease_id": lease["id"],
            "execution_id": "exec-leg",
            "start_intent": {"intent_id": "i-1", "execution_id": "exec-leg", "state": "starting"},
        },
        record_id="job-leg",
    )
    repo.create(
        "executions",
        "owner-1",
        {
            "generation": 1, "state": "starting",
            "job_id": job["id"],
            "lease_id": lease["id"],
            "start_intent": {"intent_id": "i-1", "execution_id": "exec-leg", "state": "starting"},
        },
        record_id="exec-leg",
    )

    before = _v5_rows(repo)
    assert repo.claim_execution_dispatch(job["id"], "owner-1", lease_id=lease["id"],
        execution_id="exec-leg", now=999.0)[0] == "stale"
    assert _v5_rows(repo) == before
    job = seed_execution_binding(repo, job, lease)
    status, fenced_job, _ = repo.claim_execution_dispatch(
        job["id"], "owner-1", lease_id=lease["id"], execution_id="exec-leg", now=1000.0
    )
    assert status == "fenced"
    assert fenced_job["state"] == "cleaning_up"

    repo.update(
        "leases",
        lease["id"],
        {"state": "active", "handle": {"runtime_name": "rt-leg", "runtime_id": "rt-leg", "job_id": "job-leg", "owner_job_id": "job-leg"}},
        "owner-1",
    )
    repo.update(
        "jobs",
        "job-leg",
        {"state": "loading", "start_intent": {"intent_id": "i-2", "execution_id": "exec-leg-2", "state": "starting"}, "execution_id": "exec-leg-2"},
        "owner-1",
    )
    repo.create(
        "executions",
        "owner-1",
        {"generation": 1, "state": "starting", "job_id": "job-leg", "lease_id": lease["id"], "start_intent": {"intent_id": "i-2", "execution_id": "exec-leg-2", "state": "starting"}},
        record_id="exec-leg-2",
    )
    seed_execution_binding(repo, repo.get("jobs", "job-leg", "owner-1"), repo.get("leases", lease["id"], "owner-1"))
    ok_status, ok_job, _ = repo.claim_execution_dispatch(
        "job-leg", "owner-1", lease_id=lease["id"], execution_id="exec-leg-2", now=1005.0
    )
    assert ok_status == "ok"
    assert ok_job["start_intent"]["state"] == "dispatching"


# ---------------------------------------------------------------------------
# P1-6: Poll writers uniformly fence on job/lease/instance stop or cleanup flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_collection,patch",
    [
        ("instances", {"manual_stop_requested": True}),
        ("instances", {"cleanup_claim": "tok-1"}),
        ("instances", {"state": "deleting"}),
        ("leases", {"cleanup_claim_token": "tok-2"}),
        ("leases", {"state": "deleting"}),
        ("jobs", {"manual_stop_requested": True}),
    ],
)
def test_p1_6_poll_and_diagnostic_writers_fenced_with_zero_writes(target_collection, patch):
    _, repo = _make_repo()
    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "provider": "colab",
            "account_id": "acct-1",
            "current_job_id": "job-run",
            "lease_id": "lease-run",
            "active_execution": {"active_execution_id": "exec-run", "lease_id": "lease-run", "instance_id": "inst-run", "generation": 1},
        },
        record_id="inst-run",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-run",
            "current_job_id": "job-run",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "active_execution": {"active_execution_id": "exec-run", "lease_id": "lease-run", "instance_id": "inst-run", "generation": 1},
        },
        record_id="lease-run",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            "state": "running",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "lease_id": lease["id"],
            "execution_id": "exec-run",
        },
        record_id="job-run",
    )
    repo.create(
        "executions",
        "owner-1",
        {"state": "running", "job_id": job["id"], "lease_id": lease["id"], "instance_id": inst["id"], "generation": 1},
        record_id="exec-run",
    )
    target_id = {"instances": inst["id"], "leases": lease["id"], "jobs": job["id"]}[target_collection]
    repo.update(target_collection, target_id, patch, "owner-1")
    before_version = repo.get("jobs", job["id"], "owner-1").get("version", 0)

    status, returned = repo.commit_run_progress(
        job["id"],
        "owner-1",
        execution_id="exec-run",
        now=1010.0,
        patch={"poll_step": 42},
        expected_state="running",
        expected_generation=1,
        require_instance=True,
    )
    assert status == "fenced"
    assert returned.get("version", 0) == before_version
    assert "poll_step" not in repo.get("jobs", job["id"], "owner-1")

    diag_res = repo.record_gpu_diagnostics(
        job,
        {"source": "runtime_nvidia_smi", "captured_at": 1005.0, "devices": [{"name": "NVIDIA L4", "memory_mib": 24576}]},
    )
    assert diag_res.get("version", 0) == before_version
    assert "gpu_reported" not in repo.get("jobs", job["id"], "owner-1")

    term_res = repo.commit_execution_terminal(
        job["id"],
        "owner-1",
        execution_id="exec-run",
        terminal_state="uploading",
        message="should be fenced",
        patch={"outputs": [{"filename": "late.png"}]},
        now=1011.0,
        proof={"warm_saved": False},
        expected_state="running",
        expected_generation=1,
        require_instance=True,
    )
    assert term_res.get("version", 0) == before_version
    assert repo.get("jobs", job["id"], "owner-1")["state"] == "running"


# ---------------------------------------------------------------------------
# P1-7: Retained cleanup finalization checks stop/cleanup claim under admission_lock
# ---------------------------------------------------------------------------


def test_p1_7_finalize_retained_cleanup_aborts_if_cleanup_or_stop_claimed_instance():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "retained_idle", "provider": "colab", "account_id": "acct-1", "current_job_id": None, "lease_id": "lease-ret", "idle_deadline_at": 1900.0, "hard_deadline_at": 2800.0},
        record_id="inst-ret",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-ret",
            "current_job_id": "job-ret",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "handle": {"runtime_name": "runtimes/rt-ret", "runtime_id": "rt-ret", "job_id": "job-ret", "owner_job_id": "job-ret"},
        },
        record_id="lease-ret",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {"state": "cleaning_up", "provider": "colab", "account_id": "acct-1", "instance_id": inst["id"], "lease_id": lease["id"], "execution_id": "exec-ret"},
        record_id="job-ret",
    )
    repo.create("executions", "owner-1", {"state": "running", "job_id": job["id"], "lease_id": lease["id"], "instance_id": inst["id"], "generation": 1}, record_id="exec-ret")
    for kind, row in (("jobs", job), ("leases", lease), ("instances", inst)):
        repo.update(kind, row["id"], {"generation": 1}, "owner-1")

    repo.update("leases", lease["id"], {"cleanup_claim_token": "claim-race"}, "owner-1")
    status, ret_job, ret_lease = repo.finalize_retained_cleanup(
        job["id"], "owner-1", lease_id=lease["id"], now=1000.0, instance_id=inst["id"], expected_instance_state="retained_idle"
    )
    assert status == "aborted"
    assert ret_lease["state"] == "active"
    assert ret_job["state"] == "cleaning_up"

    repo.update("leases", lease["id"], {"cleanup_claim_token": None}, "owner-1")
    ok_status, ok_job, ok_lease = repo.finalize_retained_cleanup(
        job["id"], "owner-1", lease_id=lease["id"], now=1000.0, instance_id=inst["id"], expected_instance_state="retained_idle"
    )
    assert ok_status == "ok"
    assert ok_lease["state"] == "idle"
    assert ok_job["state"] == "completed"
    assert repo.get("executions", "exec-ret", "owner-1")["state"] == "completed"


# ---------------------------------------------------------------------------
# P1-8: Warm proof validation verifies generation, epoch, execution ownership, fingerprint
# ---------------------------------------------------------------------------


def test_p1_8_stale_previous_generation_warm_proof_falls_back_to_cold_start():
    job = _calibrated_colab_job()
    fp = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    stale_inst_proof = {
        "warm_saved": True,
        "warm_execution_id": "exec-old", "worker_reported_execution_id": "exec-old",
        "warm_epoch_id": "epoch-old",
        "warm_fingerprint": fp,
        "runtime_meta": {"python_mm": "3.12"},
    }
    inst = {
        "id": "inst-1",
        "lease_id": "lease-1",
        "generation": 1,
        "warm_proof": stale_inst_proof,
        "compatibility_signature": job["compatibility_signature"],
    }
    lease = {
        "id": "lease-1",
        "instance_id": "inst-1",
        "generation": 1,
        "warm_proof": None,
        "compatibility_signature": job["compatibility_signature"],
    }
    assert is_warm_verified_for_job(job, instance_data=inst, lease_data=lease) is False

    lease_stale = {**lease, "warm_proof": stale_inst_proof}
    assert is_warm_verified_for_job(job, instance_data=inst, lease_data=lease_stale) is False

    valid_proof = {**stale_inst_proof, "generation": 1}
    executions = {
        "exec-old": {
            "id": "exec-old",
            "state": "completed",
            "instance_id": "inst-1",
            "lease_id": "lease-1",
            "generation": 1,
            "warm_epoch_id": "epoch-old",
            "warm_fingerprint": fp,
        },
        "exec-leg": {
            "id": "exec-leg",
            "state": "completed",
            "lease_id": "lease-leg",
            "generation": 1,
            "warm_epoch_id": "epoch-old",
            "warm_fingerprint": fp,
        },
    }
    assert (
        is_warm_verified_for_job(
            job,
            instance_data={**inst, "warm_proof": valid_proof},
            lease_data={**lease, "warm_proof": valid_proof},
            executions_by_id=executions,
        )
        is True
    )

    bad_exec = {
        "exec-old": {"id": "exec-old", "state": "completed", "instance_id": "inst-1", "lease_id": "lease-other", "generation": 1}
    }
    assert (
        is_warm_verified_for_job(
            job,
            instance_data={**inst, "warm_proof": valid_proof},
            lease_data={**lease, "warm_proof": valid_proof},
            executions_by_id=bad_exec,
        )
        is False
    )

    legacy_proof = {**valid_proof, "warm_execution_id": "exec-leg", "worker_reported_execution_id": "exec-leg"}
    assert (
        is_warm_verified_for_job(
            job,
            instance_data=None,
            lease_data={"id": "lease-leg", "warm_proof": legacy_proof, "generation": 1},
            executions_by_id=executions,
        )
        is False
    )


# ===========================================================================
# Stage-A V2 Sign-Off Adversarial Regressions (ALL 12 items: 4 P0 + 8 P1)
# ===========================================================================


def test_v2_p0_1_repro_attach_idle_lease_rejects_when_active_lease_on_instance_or_runtime():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "ready",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-shared",
            "current_job_id": "job-a",
            "lease_id": "lease-a",
            "budget_reserved": 3.0,
            "hard_deadline_at": 2000.0,
        },
        record_id="inst-shared",
    )
    job_a = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-a", state="running"), record_id="job-a")
    lease_a = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "runtime_id": "rt-shared",
            "owner_job_id": job_a["id"],
            "current_job_id": job_a["id"],
            "deadline_epoch": 2000.0,
            "handle": {"runtime_name": "runtimes/rt-shared", "runtime_id": "rt-shared", "job_id": job_a["id"]},
        },
        record_id="lease-a",
    )
    job_b = repo.create(
        "jobs",
        "owner-1",
        _calibrated_colab_job(job_id="job-b", state="waiting_for_capacity"),
        record_id="job-b",
    )
    fp = compute_manifest_warm_fingerprint(job_b["snapshot"]["manifest"], python_mm="3.12")
    warm_proof = {
        "warm_saved": True,
        "warm_execution_id": "exec-prev", "worker_reported_execution_id": "exec-prev",
        "warm_epoch_id": "epoch-prev",
        "warm_fingerprint": fp,
        "generation": 1,
        "runtime_meta": {"python_mm": "3.12"},
    }
    lease_b = repo.create(
        "leases",
        "owner-1",
        {
            "state": "idle",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "runtime_id": "rt-shared",
            "owner_job_id": "prev",
            "current_job_id": None,
            "compatibility_signature": "sig-ref2v",
            "deadline_epoch": 2000.0,
            "idle_deadline_at": 1800.0,
            "generation": 1,
            "warm_proof": warm_proof,
            "handle": {"runtime_name": "runtimes/rt-shared", "runtime_id": "rt-shared", "job_id": "prev"},
        },
        record_id="lease-b",
    )
    repo.create(
        "executions",
        "owner-1",
        {"state": "completed", "job_id": "prev", "instance_id": inst["id"], "lease_id": lease_b["id"], "generation": 1, "warm_epoch_id": "epoch-prev", "warm_fingerprint": fp},
        record_id="exec-prev",
    )

    assert repo.attach_idle_lease("owner-1", job_b["id"], lease_b["id"], 1000.0, guard_mode="enforce") is False
    assert repo.get("leases", "lease-a", "owner-1")["state"] == "active"
    assert repo.get("leases", "lease-b", "owner-1")["state"] == "idle"
    assert repo.get("instances", inst["id"], "owner-1")["lease_id"] == lease_a["id"]
    assert repo.get("jobs", job_b["id"], "owner-1")["state"] == "waiting_for_capacity"


def test_v2_p0_1_repro_two_instances_same_runtime_cannot_enter_retained_idle():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    repo.create(
        "instances",
        "owner-1",
        {"state": "busy", "provider": "colab", "account_id": "acct-1", "runtime_id": "rt-dup", "current_job_id": "job-a", "lease_id": "lease-a", "budget_reserved": 1.0},
        record_id="inst-a",
    )
    repo.create(
        "leases",
        "owner-1",
        {"state": "active", "provider": "colab", "account_id": "acct-1", "instance_id": "inst-a", "runtime_id": "rt-dup", "owner_job_id": "job-a", "current_job_id": "job-a"},
        record_id="lease-a",
    )
    job_b = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-b", state="cleaning_up", lifecycle="retain"), record_id="job-b")
    repo.create(
        "instances",
        "owner-1",
        {"state": "busy", "provider": "colab", "account_id": "acct-1", "runtime_id": "rt-dup", "current_job_id": "job-b", "lease_id": "lease-b", "retention_mode": "retain", "budget_reserved": 1.0},
        record_id="inst-b",
    )
    repo.create(
        "leases",
        "owner-1",
        {"state": "active", "provider": "colab", "account_id": "acct-1", "instance_id": "inst-b", "runtime_id": "rt-dup", "owner_job_id": "job-b", "current_job_id": "job-b"},
        record_id="lease-b",
    )

    with pytest.raises(InstanceConflict):
        service.complete({**job_b, "instance_id": "inst-b", "lease_id": "lease-b"}, success=True)
    assert repo.get("instances", "inst-b", "owner-1")["state"] != "retained_idle"

    with pytest.raises(InstanceConflict):
        service.retention("owner-1", "inst-b", "retain", 10, "retain-dup-rt")
    assert repo.get("instances", "inst-b", "owner-1")["state"] != "retained_idle"


def test_v2_p0_2a_partial_206_content_range_and_whitespace_endpoint_normalization():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    calls: list[tuple[str, str]] = []
    mode = ["partial_206"]

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path == "/tun/m/assignments":
            if mode[0] == "partial_206":
                return httpx.Response(
                    206,
                    headers={"Content-Range": "items 0-0/2"},
                    text=_tunnel_body({"assignments": [{"endpoint": "ep-other"}]}),
                )
            return httpx.Response(
                200,
                text=_tunnel_body({"assignments": [{"endpoint": " ep-1 "}]}),
            )
        raise AssertionError(f"unexpected unassign/delete request in mode={mode[0]}: {request.method} {request.url}")

    provider = GoogleColabProvider(
        repo,
        "owner-1",
        account_id="acct-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        google=_tunnel_google(),
    )
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "releasing", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-206", "lease_id": "lease-206", "budget_reserved": 3.0},
        record_id="inst-206",
    )
    handle = {
        "runtime_name": "runtimes/rt-206",
        "runtime_id": "rt-206",
        "request_id": "req-206",
        "job_id": "job-206",
        "owner_job_id": "job-206",
        "owner_id": "owner-1",
    }
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "cleaning_up",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-206",
            "current_job_id": "job-206",
            "runtime_id": "rt-206",
            "request_id": "req-206",
            "tunnel_endpoint": "ep-1",
            "handle": handle,
        },
        record_id="lease-206",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-206", state="cleaning_up"), "instance_id": inst["id"], "lease_id": lease["id"], "cleanup_final_state": "failed"},
        record_id="job-206",
    )

    inspected_206 = provider.inspect(handle)
    assert inspected_206["deleted"] is False
    assert inspected_206["remote_verified"] is False
    assert inspected_206["allocation_unknown"] is True

    calls.clear()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    after_cleanup = scheduler._cleanup(job)
    assert after_cleanup["state"] == "cleanup_pending"
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "deleted"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(service) == (3.0, 3.0)
    assert len([c for c in calls if "unassign" in c[1]]) == 0

    mode[0] = "padded_present"
    inspected_present = provider.inspect(handle)
    assert inspected_present["ready"] is True
    assert inspected_present["deleted"] is False


def test_v2_p0_2b_stale_absent_probe_never_deletes_attempt2_and_preserves_late_handle():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config(create_retries=1)
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "allocating", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-race", "lease_id": "lease-race", "budget_reserved": 3.0},
        record_id="inst-race",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-race",
            "current_job_id": "job-race",
            "request_id": "req-race",
            "runtime_id": "rt-race",
            "create_attempts": 1,
            "create_intent_id": "ci-race",
            "deadline_epoch": 3000.0,
        },
        record_id="lease-race",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-race", state="allocating"), "instance_id": inst["id"], "lease_id": lease["id"]},
        record_id="job-race",
    )

    spec = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
    probe_version = lease.get("version", 0)
    probe_attempts = lease["create_attempts"]

    # B retries attempt 1 -> 2 and invokes remote create; meanwhile job gets cancelled remotely while B's RPC is in flight
    retry_status, _, lease_v2 = repo.retry_absent_lease_create(
        job["id"],
        "owner-1",
        lease["id"],
        spec=spec,
        max_retries=1,
        now=1001.0,
        expected_lease_version=probe_version,
        expected_create_attempts=probe_attempts,
    )
    assert retry_status == "create"
    assert lease_v2["create_attempts"] == 2

    # Late worker A with stale (probe_version, probe_attempts=1) MUST be rejected as stale (never exhausted/deleted)
    late_retry_status, _, _ = repo.retry_absent_lease_create(
        job["id"],
        "owner-1",
        lease["id"],
        spec=spec,
        max_retries=1,
        now=1002.0,
        expected_lease_version=probe_version,
        expected_create_attempts=probe_attempts,
    )
    assert late_retry_status == "stale"
    late_retire_status, _, _ = repo.retire_absent_lease_cas(
        job["id"],
        "owner-1",
        lease["id"],
        expected_lease_version=probe_version,
        expected_create_attempts=probe_attempts,
        from_active=False,
        now=1002.0,
    )
    assert late_retire_status == "stale"
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(service) == (3.0, 3.0)

    # Even if job was cancelled while B's RPC was in-flight, B's live returned handle is preserved for cleanup
    repo.request_job_cancel("owner-1", job["id"], now=1003.0)
    live_handle = {"runtime_name": "runtimes/rt-race", "ready": True, "runtime_id": "rt-race", "request_id": "req-race"}

    class LateCreateProvider:
        def create(self, *_args, **_kwargs):
            return live_handle

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: LateCreateProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1004.0,
        config=cfg,
        instance_service=service,
    )
    scheduler._create_runtime_on_lease(
        repo.get("jobs", job["id"], "owner-1"),
        repo.get("leases", lease["id"], "owner-1"),
        LateCreateProvider(),
        spec,
        allow_recovery=False,
    )
    saved_lease = repo.get("leases", lease["id"], "owner-1")
    assert saved_lease["state"] == "cleanup_pending"
    assert saved_lease["remote_deleted"] is False
    assert isinstance(saved_lease.get("handle"), dict) and saved_lease["handle"].get("runtime_name") == "runtimes/rt-race"
    assert repo.get("instances", inst["id"], "owner-1")["runtime_created"] is True
    assert _budget_totals(service) == (3.0, 3.0)


@pytest.mark.parametrize(
    "extra_fields",
    [
        {"warm_proof": {"warm_saved": True, "warm_execution_id": "e1", "worker_reported_execution_id": "e1", "warm_epoch_id": "ep1", "warm_fingerprint": "fp1", "generation": 1}},
        {"tunnel_nbh": {"nbh": "secret-token"}},
        {"create_intent": {"intent_id": "ci-pending", "origin": "new", "from_active": False, "state": "pending"}},
    ],
)
def test_v2_p0_3_watchdog_never_locally_deletes_history_or_pending_intent(extra_fields):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class CountingProvider:
        def __init__(self):
            self.inspect_calls = 0
            self.delete_calls = 0

        def inspect(self, _handle):
            self.inspect_calls += 1
            return {"deleted": False, "allocation_unknown": True}

        def delete(self, _handle):
            self.delete_calls += 1
            return False

    provider = CountingProvider()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 2000.0,
        config=cfg,
        instance_service=service,
    )
    # Expired legacy/no-instance orphan job whose lease is still creating with persisted history
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "orphan-job",
            "current_job_id": None,
            "deadline_epoch": 1500.0,
            **extra_fields,
        },
        record_id="lease-orphan-hist",
    )
    scheduler.watchdog()
    saved = repo.get("leases", lease["id"], "owner-1")
    assert saved["state"] == "quarantined"
    assert provider.inspect_calls == 0
    assert provider.delete_calls == 0


def test_v2_p0_3_watchdog_transient_execution_query_failure_fails_closed():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    class CountingProvider:
        inspect_calls = 0
        delete_calls = 0

        def inspect(self, _handle):
            self.inspect_calls += 1
            return {"deleted": False}

        def delete(self, _handle):
            self.delete_calls += 1
            return False

    provider = CountingProvider()
    lease = repo.create(
        "leases",
        "owner-1",
        {"state": "creating", "provider": "colab", "account_id": "acct-1", "owner_job_id": "job-qe", "deadline_epoch": 1500.0},
        record_id="lease-qe",
    )
    repo.create("executions", "owner-1", {"state": "running", "job_id": "job-qe", "lease_id": lease["id"]}, record_id="exec-qe")

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 2000.0,
        config=cfg,
        instance_service=service,
    )
    orig_list = repo.list

    def flaky_list(kind, owner_id=None, project_id=None):
        if kind == "executions":
            raise RuntimeError("transient sqlite busy")
        return orig_list(kind, owner_id, project_id)

    repo.list = flaky_list  # type: ignore[method-assign]
    try:
        scheduler.watchdog()
    finally:
        repo.list = orig_list  # type: ignore[method-assign]

    saved = repo.get("leases", lease["id"], "owner-1")
    assert saved["state"] == "quarantined"
    assert provider.inspect_calls == 0
    assert provider.delete_calls == 0


def test_v2_p1_1_missing_or_stopped_job_rejected_and_stop_during_list_specs_blocks_create():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    # 1. Missing GpuJobRow -> reject with zero writes
    with pytest.raises(AllocationError):
        service.allocate(_calibrated_colab_job(job_id="ghost-job"))
    assert repo.list("instances", "owner-1") == []
    assert repo.list("leases", "owner-1") == []
    assert _budget_totals(service) == (0.0, 0.0)

    # 2. manual_stop_requested=True on job -> reject with zero writes
    stopped_job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-mstop", state="waiting_for_capacity"), "manual_stop_requested": True},
        record_id="job-mstop",
    )
    with pytest.raises(InstanceConflict):
        service.allocate(stopped_job, atomic_lease=True)
    assert repo.list("instances", "owner-1") == []
    assert _budget_totals(service) == (0.0, 0.0)

    # 3. Real stop() called during list_specs -> claim_create_intent sees releasing instance -> zero provider.create
    job = repo.create(
        "jobs",
        "owner-1",
        _calibrated_colab_job(job_id="job-stop-spec", state="waiting_for_capacity"),
        record_id="job-stop-spec",
    )
    inst = service.allocate(job, atomic_lease=True)
    job = repo.update("jobs", job["id"], {"state": "allocating"}, "owner-1")

    class StopDuringSpecsProvider:
        def __init__(self):
            self.create_calls = 0

        def list_specs(self):
            service.stop("owner-1", inst["id"], "stop-during-specs")
            return [{"key": MANIFEST_MODEL["compute_spec"], "eligible": True}]

        def create(self, *_args, **_kwargs):
            self.create_calls += 1
            raise AssertionError("provider.create must not run after stop() during list_specs")

    provider = StopDuringSpecsProvider()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    after = scheduler._allocation(repo.get("jobs", job["id"], "owner-1"))
    assert provider.create_calls == 0
    assert after["state"] in {"cleaning_up", "cleanup_pending", "failed"}
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "releasing"


def test_v2_p1_3_stale_unknown_write_cannot_downgrade_recovered_active_lease():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
    )
    lease_stale_snapshot = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "owner_job_id": "job-rec",
            "current_job_id": "job-rec",
            "from_active_missing_handle": True,
            "was_active_missing_handle": True,
            "create_intent_origin": "once_active",
            "create_attempts": 1,
        },
        record_id="lease-rec",
    )
    job = repo.create("jobs", "owner-1", {"state": "allocating", "provider": "colab", "lease_id": "lease-rec"}, record_id="job-rec")

    # Worker B recovers the lease active with live handle & puts job into preparing
    repo.update(
        "leases",
        "lease-rec",
        {
            "state": "active",
            "handle": {"runtime_name": "runtimes/rt-rec", "runtime_id": "rt-rec", "job_id": "job-rec"},
            "from_active_missing_handle": True,
            "was_active_missing_handle": True,
            "create_intent_origin": "once_active",
        },
        "owner-1",
    )
    repo.update("jobs", "job-rec", {"state": "preparing"}, "owner-1")

    # Worker A's stale unknown write (from pre-recovery version 0) MUST perform zero writes
    scheduler._mark_lease_unknown(
        lease_stale_snapshot,
        job,
        from_active=False,
        error="late unknown",
        expected_version=lease_stale_snapshot.get("version", 0),
        expected_create_attempts=1,
    )
    final_lease = repo.get("leases", "lease-rec", "owner-1")
    assert final_lease["state"] == "active"
    assert final_lease["create_intent_origin"] == "once_active"
    assert final_lease["was_active_missing_handle"] is True
    assert final_lease["handle"]["runtime_name"] == "runtimes/rt-rec"

    rebuilt = Scheduler(
        Repository(repo.store),
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1001.0,
        config=cfg,
    )
    assert rebuilt._lease(repo.get("jobs", "job-rec", "owner-1"))["state"] == "active"


def test_v2_p1_4_concurrent_claim_execution_dispatch_returns_in_flight_without_aborting():
    _, repo = _make_repo()
    job = repo.create(
        "jobs",
        "owner-1",
        {"state": "loading", "provider": "mock", "snapshot": {"manifest": {"mock": True}}},
        record_id="job-disp",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": job["id"],
            "current_job_id": job["id"],
            "handle": {"runtime_name": "mock-runtime-rt-disp", "runtime_id": "rt-disp", "job_id": job["id"], "owner_id": "owner-1"},
        },
        record_id="lease-disp",
    )
    job = repo.update("jobs", job["id"], {"lease_id": lease["id"]}, "owner-1")
    job = repo.prepare_execution(job["id"], "owner-1", lease["id"], now=1000.0)
    exec_id = job["execution_id"]

    job = seed_execution_binding(repo, job, lease)
    s1, j1, _ = repo.claim_execution_dispatch(job["id"], "owner-1", lease_id=lease["id"], execution_id=exec_id, now=1001.0)
    assert s1 == "ok"
    assert j1["start_intent"]["state"] == "dispatching"
    version_after_first = j1["version"]

    # Second concurrent caller without cancel/stop MUST get "in_flight" with ZERO writes
    s2, j2, _ = repo.claim_execution_dispatch(job["id"], "owner-1", lease_id=lease["id"], execution_id=exec_id, now=1002.0)
    assert s2 == "in_flight"
    assert j2["state"] == "loading"
    assert j2["version"] == version_after_first
    assert j2["start_intent"]["state"] == "dispatching"
    assert repo.get("executions", exec_id, "owner-1")["start_intent"]["state"] == "dispatching"


def test_v2_p1_5_invalid_handles_fail_before_prepare_and_legacy_load_resume():
    _, repo = _make_repo()
    lease_none = repo.create("leases", "owner-1", {"state": "active", "owner_job_id": "j-none", "handle": None}, record_id="l-none")
    job_none = repo.create("jobs", "owner-1", {"state": "loading", "lease_id": lease_none["id"]}, record_id="j-none")
    with pytest.raises(ValueError):
        repo.prepare_execution(job_none["id"], "owner-1", lease_none["id"], 1000.0)
    assert repo.list("executions", "owner-1") == []
    assert repo.get("leases", lease_none["id"], "owner-1").get("generation") is None

    lease_bad = repo.create(
        "leases",
        "owner-1",
        {"state": "active", "owner_job_id": "j-bad", "handle": {"unrelated": 1}},
        record_id="l-bad",
    )
    job_bad = repo.create("jobs", "owner-1", {"state": "loading", "lease_id": lease_bad["id"]}, record_id="j-bad")
    with pytest.raises(ValueError):
        repo.prepare_execution(job_bad["id"], "owner-1", lease_bad["id"], 1000.0)
    assert repo.list("executions", "owner-1") == []
    assert "start_intent" not in repo.get("jobs", job_bad["id"], "owner-1")

    # Existing execution + {"unrelated": 1} handle -> Scheduler._load() MUST NOT put job into running
    exec_bad = repo.create(
        "executions",
        "owner-1",
        {"state": "starting", "job_id": job_bad["id"], "lease_id": lease_bad["id"], "start_intent": {"state": "starting", "execution_id": "exec-unrel"}},
        record_id="exec-unrel",
    )
    job_bad = repo.update(
        "jobs",
        job_bad["id"],
        {"execution_id": exec_bad["id"], "start_intent": {"state": "starting", "execution_id": exec_bad["id"]}},
        "owner-1",
    )
    scheduler = Scheduler(repo, provider_factory=lambda _: MockColabProvider(), executor_factory=lambda _: MockGPUExecutor())
    after_load = scheduler._load(job_bad)
    assert after_load["state"] != "running"


@pytest.mark.parametrize("bad_entity,patch", [
    ("job", {"runtime_deleted": True}),
    ("job", {"remote_deleted": True}),
    ("lease", {"state": "deleted"}),
])
def test_v2_p1_6_runtime_remote_deleted_and_deleted_lease_block_diagnostics_progress_terminal(bad_entity, patch):
    _, repo = _make_repo()
    active_exec = {"active_execution_id": "e-p16", "lease_id": "l-p16", "instance_id": "i-p16", "generation": 1}
    inst = repo.create(
        "instances",
        "owner-1",
        {"state": "busy", "current_job_id": "j-p16", "lease_id": "l-p16", "generation": 1, "active_execution": active_exec},
        record_id="i-p16",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "j-p16",
            "current_job_id": "j-p16",
            "instance_id": inst["id"],
            "generation": 1,
            "active_execution": active_exec,
            "handle": {"runtime_name": "mock-runtime-rt", "runtime_id": "rt"},
        },
        record_id="l-p16",
    )
    exec_row = repo.create("executions", "owner-1", {"state": "running", "job_id": "j-p16", "lease_id": lease["id"], "instance_id": inst["id"], "generation": 1}, record_id="e-p16")
    job = repo.create(
        "jobs",
        "owner-1",
        {"state": "running", "lease_id": lease["id"], "instance_id": inst["id"], "execution_id": exec_row["id"], "generation": 1},
        record_id="j-p16",
    )
    if bad_entity == "job":
        job = repo.update("jobs", job["id"], patch, "owner-1")
    else:
        repo.update("leases", lease["id"], patch, "owner-1")

    before_ver = job.get("version", 0)
    diag_out = repo.record_gpu_diagnostics(
        job, {"source": "runtime_nvidia_smi", "captured_at": 1005.0, "devices": [{"name": "NVIDIA L4", "memory_mib": 24576}]}
    )
    assert diag_out.get("version", 0) == before_ver
    assert "gpu_reported" not in repo.get("jobs", job["id"], "owner-1")

    p_status, p_job = repo.commit_run_progress(
        job["id"], "owner-1", execution_id=exec_row["id"], now=1006.0, patch={"stage": "late"}, expected_state="running", expected_generation=1
    )
    assert p_status == "fenced"
    assert p_job.get("version", 0) == before_ver
    assert "stage" not in repo.get("jobs", job["id"], "owner-1")

    t_job = repo.commit_execution_terminal(
        job["id"], "owner-1", execution_id=exec_row["id"], terminal_state="uploading", message="late", patch={}, now=1007.0, expected_state="running", expected_generation=1
    )
    assert t_job["state"] == "running"
    assert t_job.get("version", 0) == before_ver


def test_v2_p1_7_finalize_retained_cleanup_authoritative_db_instance_runtime_deleted_and_tightened_deadline():
    # Case 1: missing instance row in DB -> aborted
    _, repo1 = _make_repo()
    l1 = repo1.create("leases", "owner-1", {"state": "active", "owner_job_id": "j1", "instance_id": "missing-inst", "handle": {"runtime_name": "rt-1", "runtime_id": "rt-1"}}, record_id="l1")
    j1 = repo1.create("jobs", "owner-1", {"state": "cleaning_up", "lease_id": l1["id"], "instance_id": "missing-inst"}, record_id="j1")
    s1, rj1, rl1 = repo1.finalize_retained_cleanup(j1["id"], "owner-1", lease_id=l1["id"], now=1000.0, instance_id="missing-inst", expected_instance_state="retained_idle", idle_deadline_at=1900.0)
    assert s1 == "aborted"
    assert rj1["state"] == "cleaning_up"
    assert rl1["state"] == "active"

    # Case 2: job.runtime_deleted=True -> aborted and runtime_deleted preserved True
    inst2 = repo1.create("instances", "owner-1", {"state": "retained_idle", "lease_id": "l2", "current_job_id": None, "retain_until": 1060.0}, record_id="inst2")
    l2 = repo1.create("leases", "owner-1", {"state": "active", "owner_job_id": "j2", "instance_id": inst2["id"], "handle": {"runtime_name": "rt-2", "runtime_id": "rt-2"}}, record_id="l2")
    j2 = repo1.create("jobs", "owner-1", {"state": "cleaning_up", "lease_id": l2["id"], "instance_id": inst2["id"], "runtime_deleted": True}, record_id="j2")
    s2, rj2, _ = repo1.finalize_retained_cleanup(j2["id"], "owner-1", lease_id=l2["id"], now=1000.0, instance_id=inst2["id"], expected_instance_state="retained_idle", idle_deadline_at=1900.0)
    assert s2 == "aborted"
    assert rj2["runtime_deleted"] is True

    # Case 3: DB retain_until tightened to 1060.0 -> stale caller param 1900.0 CANNOT extend past 1060.0
    inst3 = repo1.create("instances", "owner-1", {"state": "retained_idle", "lease_id": "l3", "current_job_id": None, "retain_until": 1060.0, "idle_deadline_at": 1060.0}, record_id="inst3")
    l3 = repo1.create("leases", "owner-1", {"state": "active", "owner_job_id": "j3", "instance_id": inst3["id"], "handle": {"runtime_name": "rt-3", "runtime_id": "rt-3"}}, record_id="l3")
    j3 = repo1.create("jobs", "owner-1", {"state": "cleaning_up", "lease_id": l3["id"], "instance_id": inst3["id"]}, record_id="j3")
    j3 = repo1.update("jobs", j3["id"], {"execution_id": "exec-j3"}, "owner-1")
    j3 = seed_execution_binding(repo1, j3, l3, instance=inst3)
    s3_exp, _, _ = repo1.finalize_retained_cleanup(j3["id"], "owner-1", lease_id=l3["id"], now=1061.0, instance_id=inst3["id"], expected_instance_state="retained_idle", idle_deadline_at=1900.0)
    assert s3_exp == "expired"

    s3_ok, _, ok_l3 = repo1.finalize_retained_cleanup(j3["id"], "owner-1", lease_id=l3["id"], now=1050.0, instance_id=inst3["id"], expected_instance_state="retained_idle", idle_deadline_at=1900.0)
    assert s3_ok == "ok"
    assert ok_l3["idle_deadline_at"] == 1060.0
    assert ok_l3["retain_until"] == 1060.0


@pytest.mark.parametrize("proof_gen", [None, 2])
def test_v2_p1_8_missing_execution_row_forces_cold_start_and_rejects_450s_remaining(proof_gen):
    _, repo = _make_repo()
    now = 1000.0
    job = repo.create(
        "jobs",
        "owner-1",
        _calibrated_colab_job(job_id="job-p18", state="waiting_for_capacity"),
        record_id="job-p18",
    )
    fp = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    proof = {
        "warm_saved": True,
        "warm_execution_id": "exec-missing", "worker_reported_execution_id": "exec-missing",
        "warm_epoch_id": "ep-1",
        "warm_fingerprint": fp,
        "runtime_meta": {"python_mm": "3.12"},
        **({"generation": proof_gen} if proof_gen is not None else {}),
    }
    inst = {
        "id": "inst-p18",
        "lease_id": "lease-p18",
        "hard_deadline_at": now + 450.0,
        "warm_proof": proof,
        "compatibility_signature": "sig-ref2v",
        **({"generation": proof_gen} if proof_gen is not None else {}),
    }
    lease = {
        "id": "lease-p18",
        "instance_id": "inst-p18",
        "deadline_epoch": now + 450.0,
        "warm_proof": proof,
        "compatibility_signature": "sig-ref2v",
        **({"generation": proof_gen} if proof_gen is not None else {}),
    }
    assert is_warm_verified_for_job(job, instance_data=inst, lease_data=lease, executions_by_id={}) is False
    allowed, assessment = check_job_reuse_guard(
        job,
        guard_mode="enforce",
        now=now,
        instance_data=inst,
        lease_data=lease,
        repo=repo,
        owner_id="owner-1",
    )
    assert allowed is False
    assert assessment is not None and assessment.warm_verified is False and assessment.need_seconds == pytest.approx(780.0)


def test_v2_new_p1_backed_off_cleanup_does_not_starve_running_job():
    _, repo = _make_repo()

    class CountingExecutor:
        def __init__(self):
            self.poll_calls = 0

        def poll(self, _handle, job):
            self.poll_calls += 1
            return {"state": "running", "execution_id": job["execution_id"]}

    executor = CountingExecutor()
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: executor,
        clock=lambda: 1000.0,
    )
    lease_a = repo.create(
        "leases",
        "owner-1",
        {"state": "active", "owner_job_id": "job-run-a", "current_job_id": "job-run-a", "handle": {"runtime_name": "mock-runtime-a", "runtime_id": "a"}},
        record_id="lease-a-starve",
    )
    repo.create(
        "executions",
        "owner-1",
        {"state": "running", "job_id": "job-run-a", "lease_id": lease_a["id"], "start_intent": {"state": "running", "execution_id": "exec-run-a"}},
        record_id="exec-run-a",
    )
    repo.create(
        "jobs",
        "owner-1",
        {"state": "running", "provider": "mock", "lease_id": lease_a["id"], "execution_id": "exec-run-a", "created_epoch": 200.0},
        record_id="job-run-a",
    )
    # Earlier conflict cleanup job B backed off until 1005.0 while clock=1000.0
    repo.create(
        "jobs",
        "owner-1",
        {"state": "cleanup_pending", "provider": "mock", "cleanup_next_epoch": 1005.0, "created_epoch": 100.0},
        record_id="job-clean-b",
    )

    for _ in range(5):
        res = scheduler.tick()
        assert res["job_id"] == "job-run-a"
    assert executor.poll_calls == 5


# ===========================================================================
# Stage-A V3 Final Sign-Off Adversarial Regressions (5 P0 + 6 P1)
# ===========================================================================


def test_v3_p0_1a_complete_retain_blocked_when_other_instance_shares_runtime_missing_lease_runtime_id():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())

    inst_a = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-dup",
            "current_job_id": "job-a",
            "lease_id": "lease-a",
            "budget_reserved": 2.0,
            "hard_deadline_at": 2500.0,
        },
        record_id="inst-a",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_a["id"],
            "owner_job_id": "job-a",
            "current_job_id": "job-a",
            # Lease A omits runtime_id; authoritative resolver must pull rt-dup from inst_a!
            "runtime_id": None,
            "handle": {"runtime_name": "projects/p/locations/l/runtimes/rt-dup", "job_id": "job-a"},
        },
        record_id="lease-a",
    )
    inst_b = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-dup",
            "current_job_id": "job-b",
            "lease_id": "lease-b",
            "budget_reserved": 2.0,
            "hard_deadline_at": 2500.0,
        },
        record_id="inst-b",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_b["id"],
            "owner_job_id": "job-b",
            "current_job_id": "job-b",
            "runtime_id": None,
            "handle": {"tunnel_endpoint": "https://dup.trycloudflare.com/"},
        },
        record_id="lease-b",
    )
    job_b = repo.create(
        "jobs",
        "owner-1",
        _calibrated_colab_job(job_id="job-b", state="uploading", lifecycle="retain"),
        record_id="job-b",
    )
    job_b = repo.update("jobs", "job-b", {"instance_id": inst_b["id"], "lease_id": "lease-b"}, "owner-1")

    with pytest.raises(InstanceConflict):
        service.complete({**job_b, "retention_mode": "retain"}, success=True)
    assert service.get("owner-1", inst_b["id"])["state"] != "retained_idle"


def test_v3_p0_1b_scheduler_orphan_cleanup_never_unassigns_shared_runtime_or_rebinds_while_claimed():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if "/tun/m/assignments" in str(request.url) and request.method == "GET":
            return httpx.Response(
                200,
                text=_tunnel_body(
                    {
                        "assignments": [
                            {
                                "endpoint": "https://dup-ep.trycloudflare.com",
                                "runtimeName": "projects/p/locations/l/runtimes/rt-dup",
                            }
                        ]
                    }
                ),
            )
        if "/tun/m/unassign" in str(request.url):
            raise AssertionError("orphan B MUST NOT unassign shared runtime rt-dup owned by active A")
        return httpx.Response(200, json={"done": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleColabProvider(repo, "owner-1", account_id="acct-1", client=client, google=_tunnel_google())
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: provider,
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )

    inst_a = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-dup",
            "tunnel_endpoint": "https://dup-ep.trycloudflare.com:443/",
            "current_job_id": "job-a",
            "lease_id": "lease-a",
            "budget_reserved": 2.0,
            "hard_deadline_at": 2500.0,
        },
        record_id="inst-a",
    )
    repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-a", state="running"), "instance_id": inst_a["id"], "lease_id": "lease-a"},
        record_id="job-a",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_a["id"],
            "owner_job_id": "job-a",
            "current_job_id": "job-a",
            # Lease A omits runtime_id; resolved from inst_a / handle.
            "runtime_id": None,
            "request_id": "req-a",
            "deadline_epoch": 2500.0,
            "handle": {
                "runtime_name": "projects/p/locations/l/runtimes/rt-dup",
                "tunnel_endpoint": "https://dup-ep.trycloudflare.com",
                "job_id": "job-a",
                "owner_job_id": "job-a",
                "owner_id": "owner-1",
            },
        },
        record_id="lease-a",
    )

    # Expired orphan B on a different instance also referencing rt-dup + normalized endpoint
    inst_b = repo.create(
        "instances",
        "owner-1",
        {
            "state": "releasing",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-dup",
            "tunnel_endpoint": "https://DUP-EP.trycloudflare.com/",
            "current_job_id": None,
            "lease_id": "lease-b",
            "budget_reserved": 2.0,
            "hard_deadline_at": 900.0,
        },
        record_id="inst-b",
    )
    repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-b", state="completed"), "instance_id": inst_b["id"], "lease_id": "lease-b"},
        record_id="job-b",
    )
    repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_b["id"],
            "owner_job_id": "job-b",
            "current_job_id": None,
            "runtime_id": None,
            "request_id": "req-b",
            "deadline_epoch": 900.0,
            "handle": {
                "runtime_name": "projects/p/locations/l/runtimes/rt-dup",
                "tunnel_endpoint": "https://dup-ep.trycloudflare.com",
                "request_id": "req-b",
                "job_id": "job-b",
                "owner_job_id": "job-b",
                "owner_id": "owner-1",
            },
        },
        record_id="lease-b",
    )

    res = scheduler.watchdog()
    assert res["recovered"] == 0
    assert not any("/tun/m/unassign" in url or method == "DELETE" for method, url in calls)

    # Part 2: while orphan cleanup claim is active on rt-dup, nobody can rebind rt-dup
    repo.update("leases", "lease-a", {"state": "deleted"}, "owner-1")
    repo.update("instances", inst_a["id"], {"state": "deleted", "lease_id": None}, "owner-1")
    token = "clean-claim-1"
    claimed = repo.claim_orphan_lease("owner-1", "lease-b", inst_b["id"], token, 1000.0)
    assert claimed is not None
    assert repo.get("leases", "lease-b", "owner-1")["state"] == "deleting"

    inst_c = repo.create(
        "instances",
        "owner-1",
        {"state": "allocating", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-c", "budget_reserved": 2.0},
        record_id="inst-c",
    )
    job_c = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-c", state="allocating"), "instance_id": inst_c["id"]},
        record_id="job-c",
    )
    with pytest.raises(InstanceConflict):
        service.bind_lease(job_c, {"id": "lease-new", "runtime_id": "rt-dup", "handle": {"runtime_id": "rt-dup"}})


@pytest.mark.parametrize(
    "case",
    ["link_next_header", "nested_page_info_has_next", "next_page_token", "x_has_more_header"],
)
def test_v3_p0_2a_colab_provider_rejects_paginated_or_truncated_tunnel_assignments(case):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if "/tun/m/assignments" in str(request.url):
            base_body = {"assignments": [{"endpoint": "https://other.trycloudflare.com", "runtimeName": "runtimes/rt-other"}]}
            if case == "link_next_header":
                return httpx.Response(
                    200,
                    headers={"Link": '<https://colab.research.google.com/tun/m/assignments?page=2>; rel="next"'},
                    text=_tunnel_body(base_body),
                )
            if case == "nested_page_info_has_next":
                return httpx.Response(
                    200,
                    text=_tunnel_body({**base_body, "pageInfo": {"hasNextPage": True}}),
                )
            if case == "next_page_token":
                return httpx.Response(
                    200,
                    text=_tunnel_body({"items": base_body["assignments"], "nextPageToken": "page-2-tok"}),
                )
            return httpx.Response(200, headers={"X-Has-More": "true"}, text=_tunnel_body(base_body))
        if "/tun/m/unassign" in str(request.url):
            raise AssertionError("paginated listing MUST NOT trigger unassign")
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleColabProvider(repo, "owner-1", account_id="acct-1", client=client, google=_tunnel_google())

    handle = {
        "runtime_id": "rt-target",
        "runtime_name": "runtimes/rt-target",
        "tunnel_endpoint": "https://target.trycloudflare.com",
        "request_id": "req-1",
        "job_id": "job-1",
        "owner_job_id": "job-1",
        "owner_id": "owner-1",
    }
    job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-1", state="allocating"), "estimated_units": 3.0},
        record_id="job-1",
    )
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "unknown",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-1",
            "current_job_id": "job-1",
            "runtime_id": "rt-target",
            "request_id": "req-1",
            "tunnel_endpoint": "https://target.trycloudflare.com",
            "create_attempts": 1,
        },
        record_id="lease-1",
    )
    job = repo.update("jobs", "job-1", {"lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")

    status = provider.inspect(handle)
    assert status["allocation_unknown"] is True
    assert status["deleted"] is False
    assert status["remote_verified"] is False

    scheduler = Scheduler(repo, provider_factory=lambda _: provider, executor_factory=lambda _: MockGPUExecutor(), clock=lambda: 1000.0, config=cfg, instance_service=service)
    res = scheduler._allocation(job)
    assert res["state"] == "allocating"
    assert repo.get("leases", lease["id"], "owner-1")["state"] == "unknown"
    assert _budget_totals(service)[0] == pytest.approx(3.0)
    assert not any("/tun/m/unassign" in url or method == "DELETE" for method, url in calls)


def test_v3_p0_2b_a_version_0_serialization_and_cas_rejects_stale_absent_same_attempt():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config(create_retries=2)
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    job = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-v0", state="allocating"), record_id="job-v0")
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-v0",
            "current_job_id": "job-v0",
            "runtime_id": "rt-v0",
            "request_id": "req-v0",
            "create_attempts": 2,
        },
        record_id="lease-v0",
    )
    assert lease.get("version") == 0

    job = repo.update("jobs", "job-v0", {"lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")

    class MutatingInspectProvider:
        def inspect(self, _handle):
            # Concurrently register newly discovered endpoint on SAME create_attempts=2 (version 0 -> 1)
            repo.update(
                "leases",
                lease["id"],
                {
                    "state": "active",
                    "tunnel_endpoint": "https://new-ep.trycloudflare.com",
                    "handle": {"runtime_id": "rt-v0", "tunnel_endpoint": "https://new-ep.trycloudflare.com"},
                    "create_attempts": 2,
                },
                "owner-1",
            )
            return {"deleted": True, "remote_verified": True}

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MutatingInspectProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    scheduler._handle_recovery(job, lease, MutatingInspectProvider())
    after = repo.get("leases", lease["id"], "owner-1")
    assert after["state"] == "active"
    assert after["version"] == 1
    assert after["tunnel_endpoint"] == "https://new-ep.trycloudflare.com"


def test_v3_p0_2b_b_dual_scheduler_inflight_create_cannot_retire_and_late_handle_recovers_quota(tmp_path):
    _, repo = _make_repo(tmp_path, "dual_inflight.db")
    _setup_account(repo, provider="colab")
    cfg = _colab_config(create_retries=2, unit_rate=3.0)
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-inf", state="allocating"), "estimated_units": 3.0},
        record_id="job-inf",
    )
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "unknown",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-inf",
            "current_job_id": "job-inf",
            "runtime_id": "rt-inf",
            "request_id": "req-inf",
            "create_attempts": 1,
            "create_intent_id": "ci-inf",
            "create_intent_origin": "new",
        },
        record_id="lease-inf",
    )
    job = repo.update("jobs", "job-inf", {"lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")

    in_create = threading.Event()
    allow_create_return = threading.Event()

    class BlockingAttempt2Provider:
        def list_specs(self):
            return [{"key": MANIFEST_MODEL["compute_spec"], "eligible": True}]

        def inspect(self, _handle):
            return {"deleted": True, "remote_verified": True}

        def create(self, _spec, _req_id, _job_id, rt_id):
            in_create.set()
            assert allow_create_return.wait(timeout=5.0)
            return {
                "runtime_name": f"runtimes/{rt_id}",
                "runtime_id": rt_id,
                "tunnel_endpoint": "https://late-recovered.trycloudflare.com",
            }

    class EmptyInspectProvider:
        def list_specs(self):
            return [{"key": MANIFEST_MODEL["compute_spec"], "eligible": True}]

        def inspect(self, _handle):
            return {"deleted": True, "remote_verified": True}

    sched1 = Scheduler(repo, provider_factory=lambda _: BlockingAttempt2Provider(), executor_factory=lambda _: MockGPUExecutor(), clock=lambda: 1005.0, config=cfg, instance_service=service)
    sched2 = Scheduler(repo, provider_factory=lambda _: EmptyInspectProvider(), executor_factory=lambda _: MockGPUExecutor(), clock=lambda: 1006.0, config=cfg, instance_service=service)

    t1_out: list[dict] = []

    def run_t1():
        t1_out.append(sched1._allocation(repo.get("jobs", "job-inf", "owner-1")))

    t1 = threading.Thread(target=run_t1)
    t1.start()
    assert in_create.wait(timeout=5.0)

    mid_lease = repo.get("leases", lease["id"], "owner-1")
    assert mid_lease["create_attempts"] == 2
    assert mid_lease["create_rpc_in_flight"] is True

    # Other scheduler sees empty listing while attempt 2 POST is in flight -> MUST NOT retire!
    sched2._handle_recovery(repo.get("jobs", "job-inf", "owner-1"), mid_lease, EmptyInspectProvider())
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "deleted"
    assert repo.get("jobs", "job-inf", "owner-1")["state"] == "allocating"

    # Simulate external cancel/delete occurring before late POST resolves
    repo.update("leases", lease["id"], {"state": "deleted", "remote_deleted": True}, "owner-1")
    repo.update("instances", inst["id"], {"state": "deleted", "runtime_deleted": True, "budget_reserved": 0.0}, "owner-1")
    repo.update("jobs", "job-inf", {"cancel_requested": True, "requested_outcome": "cancelled"}, "owner-1")

    allow_create_return.set()
    t1.join(timeout=5.0)

    final_lease = repo.get("leases", lease["id"], "owner-1")
    final_inst = repo.get("instances", inst["id"], "owner-1")
    assert final_lease["state"] == "cleanup_pending"
    assert final_lease["remote_deleted"] is False
    assert isinstance(final_lease.get("handle"), dict) and final_lease["handle"].get("runtime_id") == "rt-inf"
    assert final_inst["state"] == "cleanup_pending"
    assert _budget_totals(service)[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "extra",
    [
        {"create_intent_origin": "new", "origin": "active"},
        {"create_intent_origin": "new", "create_intent": {"intent_id": "ci-1", "origin": "active", "state": "claimed"}},
    ],
)
def test_v3_p0_3a_origin_active_not_shadowed_by_create_intent_origin_new(extra):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    inspect_calls = {"n": 0}

    class NonDeletingProvider:
        def inspect(self, _handle):
            inspect_calls["n"] += 1
            return {"deleted": False, "allocation_unknown": True, "identity_unverifiable": True}

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: NonDeletingProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=_colab_config(),
    )
    repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-shadow", state="completed"), record_id="job-shadow")
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "owner_job_id": "job-shadow",
            "account_id": "acct-1",
            "request_id": None,
            "runtime_id": None,
            "create_attempts": 0,
            "from_active_missing_handle": False,
            **extra,
        },
        record_id="lease-shadow",
    )

    retire_status, _ = repo.retire_pure_orphan_placeholder_cas(
        "owner-1", lease["id"], expected_lease_version=lease["version"], now=1000.0
    )
    assert retire_status != "retired"
    assert repo.get("leases", lease["id"], "owner-1")["state"] == "creating"

    res = scheduler.watchdog()
    assert res["recovered"] == 0
    assert inspect_calls["n"] == 0
    assert repo.get("leases", lease["id"], "owner-1")["state"] == "quarantined"


def test_v3_p0_3b_single_tx_watchdog_aborts_cleanly_on_concurrent_activation(tmp_path):
    _, repo = _make_repo(tmp_path, "watchdog_race.db")
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    job = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-w", state="queued"), record_id="job-w")
    inst = service.allocate(job)
    repo.update("jobs", "job-w", {"state": "completed", "instance_id": inst["id"]}, "owner-1")
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "owner_job_id": "job-w",
            "current_job_id": "job-w",
            "instance_id": inst["id"],
            "account_id": "acct-1",
            "request_id": "req-w",
            "runtime_id": "studio-placeholder-w",
            "create_attempts": 0,
            "deadline_epoch": 500.0,
        },
        record_id="lease-w",
    )
    repo.update("jobs", "job-w", {"lease_id": lease["id"]}, "owner-1")

    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )

    entered = threading.Event()
    allow_proceed = threading.Event()
    orig_retire = repo.retire_pure_orphan_placeholder_cas

    def hooked_retire(*args, **kwargs):
        entered.set()
        assert allow_proceed.wait(timeout=5.0)
        return orig_retire(*args, **kwargs)

    repo.retire_pure_orphan_placeholder_cas = hooked_retire

    wd_result: list[dict] = []

    def run_watchdog():
        wd_result.append(scheduler.watchdog())

    t = threading.Thread(target=run_watchdog)
    t.start()
    assert entered.wait(timeout=5.0)

    # Thread 2 concurrently activates lease + marks instance busy + starts execution!
    repo.update(
        "leases",
        lease["id"],
        {"state": "active", "handle": {"runtime_name": "runtimes/rt-live", "runtime_id": "rt-live"}},
        "owner-1",
    )
    repo.update("instances", inst["id"], {"state": "busy", "lease_id": lease["id"]}, "owner-1")
    repo.create(
        "executions",
        "owner-1",
        {"state": "running", "job_id": "job-w", "lease_id": lease["id"], "instance_id": inst["id"], "generation": 1},
        record_id="exec-w",
    )

    allow_proceed.set()
    t.join(timeout=5.0)

    assert wd_result[0]["recovered"] == 0
    assert repo.get("leases", lease["id"], "owner-1")["state"] == "active"
    assert repo.get("instances", inst["id"], "owner-1")["state"] == "busy"
    assert repo.get("executions", "exec-w", "owner-1")["state"] == "running"


def test_v3_p1_3_cleanup_ownership_cas_quarantine_cas_and_no_handle_active_retire():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    # (a) Scheduler A checks old failed op while B restores same runtime to preparing/active -> ZERO DELETE
    job_a = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-c1", state="allocating"), record_id="job-c1")
    inst_a = service.allocate(job_a)
    lease_a = repo.create(
        "leases",
        "owner-1",
        {
            "state": "unknown",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_a["id"],
            "owner_job_id": "job-c1",
            "current_job_id": "job-c1",
            "runtime_id": "rt-c1",
            "request_id": "req-c1",
            "operation_name": "operations/op-old",
            "create_attempts": 1,
            "handle": {"runtime_name": "runtimes/rt-c1", "runtime_id": "rt-c1", "operation_name": "operations/op-old"},
        },
        record_id="lease-c1",
    )
    job_a = repo.update("jobs", "job-c1", {"lease_id": lease_a["id"], "instance_id": inst_a["id"]}, "owner-1")

    delete_calls = {"n": 0}

    class StaleInspectProviderA:
        def inspect(self, _handle):
            # B recovers same runtime to active/preparing via op-new concurrently
            repo.update(
                "leases",
                lease_a["id"],
                {
                    "state": "active",
                    "operation_name": "operations/op-new",
                    "handle": {"runtime_name": "runtimes/rt-c1", "runtime_id": "rt-c1", "operation_name": "operations/op-new"},
                },
                "owner-1",
            )
            repo.update("jobs", "job-c1", {"state": "preparing"}, "owner-1")
            return {"allocation_failed": True, "error": "old op failed"}

        def delete(self, _handle):
            delete_calls["n"] += 1
            return True

    sched = Scheduler(
        repo,
        provider_factory=lambda _: StaleInspectProviderA(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    sched._handle_recovery(job_a, lease_a, StaleInspectProviderA())
    assert delete_calls["n"] == 0
    assert repo.get("leases", lease_a["id"], "owner-1")["state"] == "active"

    # (b) Stale identity_unverifiable cannot quarantine B's recovered active lease
    job_b = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-c2", state="allocating"), record_id="job-c2")
    inst_b = service.allocate(job_b)
    lease_b = repo.create(
        "leases",
        "owner-1",
        {
            "state": "unknown",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_b["id"],
            "owner_job_id": "job-c2",
            "current_job_id": "job-c2",
            "runtime_id": "rt-c2",
            "request_id": "req-c2",
            "create_attempts": 1,
            "handle": {"runtime_name": "runtimes/rt-c2", "runtime_id": "rt-c2"},
        },
        record_id="lease-c2",
    )
    job_b = repo.update("jobs", "job-c2", {"lease_id": lease_b["id"], "instance_id": inst_b["id"]}, "owner-1")

    class UnverifiableRacedProvider:
        def inspect(self, _handle):
            repo.update(
                "leases",
                lease_b["id"],
                {"state": "active", "handle": {"runtime_name": "runtimes/rt-c2", "runtime_id": "rt-c2"}},
                "owner-1",
            )
            repo.update("jobs", "job-c2", {"state": "preparing"}, "owner-1")
            return {"identity_unverifiable": True, "deleted": False, "remote_verified": False}

    sched._handle_recovery(job_b, lease_b, UnverifiableRacedProvider())
    assert repo.get("leases", lease_b["id"], "owner-1")["state"] == "active"
    assert repo.get("jobs", "job-c2", "owner-1")["state"] == "preparing"

    # (c) No-handle historical active lease with confirmed remote_verified=True absence retires in 1 tick
    job_c = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-c3", state="allocating"), record_id="job-c3")
    inst_c = service.allocate(job_c)
    lease_c = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_c["id"],
            "owner_job_id": "job-c3",
            "current_job_id": "job-c3",
            "runtime_id": "rt-c3",
            "request_id": "req-c3",
            "create_attempts": 1,
            "create_intent_origin": "once_active",
            "from_active_missing_handle": True,
            "handle": None,
        },
        record_id="lease-c3",
    )
    job_c = repo.update("jobs", "job-c3", {"lease_id": lease_c["id"], "instance_id": inst_c["id"]}, "owner-1")

    class AbsentConfirmProvider:
        def inspect(self, _handle):
            return {"deleted": True, "remote_verified": True}

    sched_c = Scheduler(
        repo,
        provider_factory=lambda _: AbsentConfirmProvider(),
        executor_factory=lambda _: MockGPUExecutor(),
        clock=lambda: 1000.0,
        config=cfg,
        instance_service=service,
    )
    out = sched_c._allocation(job_c)
    assert out["error_code"] == "gpu_runtime_missing"
    assert repo.get("leases", lease_c["id"], "owner-1")["state"] == "deleted"
    assert repo.get("instances", inst_c["id"], "owner-1")["state"] == "deleted"
    final_c = sched_c._cleanup(out)
    assert final_c["state"] == "failed"


@pytest.mark.parametrize("bad_state", ["running", "uploading", "completed", "cancelled", "failed", "preparing", "loading"])
def test_v3_p1_1_instance_service_allocate_whitelist_rejects_non_allocatable_states(bad_state):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())

    job = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id=f"job-{bad_state}", state=bad_state))
    before_instances = len(repo.list("instances", "owner-1"))
    before_leases = len(repo.list("leases", "owner-1"))
    before_daily = _budget_totals(service)[0]

    with pytest.raises(InstanceConflict):
        service.allocate(job, atomic_lease=True)

    assert len(repo.list("instances", "owner-1")) == before_instances
    assert len(repo.list("leases", "owner-1")) == before_leases
    assert _budget_totals(service)[0] == pytest.approx(before_daily)


def test_v3_p1_4_dispatching_reconciles_remote_poll_and_never_starves_queued_jobs():
    # Part (a): remotely accepted dispatching job transitions to running on 1 tick without 2nd start
    _, repo = _make_repo()
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)
    job_a = repo.create("jobs", "owner-1", {**_calibrated_colab_job(job_id="job-d1", state="allocating"), "provider": "mock", "execution_policy": {"provider": "mock", "lifecycle": "save"}}, record_id="job-d1")
    inst_a = service.allocate(job_a)
    handle_a = {"runtime_name": "runtimes/rt-d1", "runtime_id": "rt-d1", "job_id": "job-d1", "owner_job_id": "job-d1", "owner_id": "owner-1"}
    lease_a = repo.create("leases", "owner-1", {"state": "active", "provider": "mock", "account_id": "acct-1", "instance_id": inst_a["id"], "owner_job_id": "job-d1", "handle": handle_a}, record_id="lease-d1")
    service.bind_lease(job_a, lease_a)
    service.mark_ready(job_a)
    job_a = repo.update("jobs", "job-d1", {"state": "loading", "lease_id": lease_a["id"], "instance_id": inst_a["id"], "created_epoch": 100.0}, "owner-1")
    job_a = repo.prepare_execution("job-d1", "owner-1", lease_a["id"], 1000.0)
    status, job_a, _ = repo.claim_execution_dispatch("job-d1", "owner-1", lease_id=lease_a["id"], execution_id=job_a["execution_id"], now=1000.0)
    assert status == "ok"

    start_calls = {"n": 0}
    poll_calls = {"n": 0}

    class AcceptedRemoteExecutor:
        def start(self, *_args):
            start_calls["n"] += 1
            return {}

        def poll(self, _handle, job):
            poll_calls["n"] += 1
            return {"execution_id": job["execution_id"], "worker_reported_execution_id": job["execution_id"], "state": "running"}

    sched_a = Scheduler(repo, provider_factory=lambda _: MockColabProvider(), executor_factory=lambda _: AcceptedRemoteExecutor(), instance_service=service, clock=lambda: 1001.0)
    res_a = sched_a.tick()
    assert res_a["job_id"] == "job-d1"
    assert res_a["state"] == "running"
    assert start_calls["n"] == 0
    assert poll_calls["n"] == 1

    # Part (b): pending/transient dispatching job sets backoff and NEVER starves queued job B for 4 ticks
    _, repo_b = _make_repo()
    _setup_account(repo_b, provider="mock")
    service_b = InstanceService(repo_b, clock=lambda: 1000.0)
    stuck_a = repo_b.create("jobs", "owner-1", {**_calibrated_colab_job(job_id="job-stuck", state="allocating"), "provider": "mock", "execution_policy": {"provider": "mock", "lifecycle": "save"}}, record_id="job-stuck")
    inst_s = service_b.allocate(stuck_a)
    handle_s = {"runtime_name": "runtimes/rt-s1", "runtime_id": "rt-s1", "job_id": "job-stuck", "owner_job_id": "job-stuck", "owner_id": "owner-1"}
    lease_s = repo_b.create("leases", "owner-1", {"state": "active", "provider": "mock", "account_id": "acct-1", "instance_id": inst_s["id"], "owner_job_id": "job-stuck", "handle": handle_s}, record_id="lease-s1")
    service_b.bind_lease(stuck_a, lease_s)
    service_b.mark_ready(stuck_a)
    stuck_a = repo_b.update("jobs", "job-stuck", {"state": "loading", "lease_id": lease_s["id"], "instance_id": inst_s["id"], "created_epoch": 100.0}, "owner-1")
    stuck_a = repo_b.prepare_execution("job-stuck", "owner-1", lease_s["id"], 1000.0)
    repo_b.claim_execution_dispatch("job-stuck", "owner-1", lease_id=lease_s["id"], execution_id=stuck_a["execution_id"], now=1000.0)

    queued_b = repo_b.create("jobs", "owner-1", {**_calibrated_colab_job(job_id="job-q2", state="queued"), "provider": "mock", "execution_policy": {"provider": "mock", "lifecycle": "save"}, "created_epoch": 200.0}, record_id="job-q2")

    class TransientPollExecutor(MockGPUExecutor):
        def poll(self, handle, job):
            if job["id"] == "job-stuck":
                raise MCPTransient("worker still starting")
            return super().poll(handle, job)

    sched_b = Scheduler(repo_b, provider_factory=lambda _: MockColabProvider(), executor_factory=lambda _: TransientPollExecutor(), instance_service=service_b, clock=lambda: 1001.0)
    picked_jobs: list[str] = []
    for _ in range(4):
        picked_jobs.append(sched_b.tick()["job_id"])
    assert queued_b["id"] in picked_jobs
    assert repo_b.get("jobs", queued_b["id"], "owner-1")["state"] != "queued"


def test_v3_p1_5_runtime_name_vs_runtime_id_mismatch_rejected_everywhere():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    bad_handle = {
        "runtime_name": "projects/p/locations/l/runtimes/rt-b",
        "runtime_id": "rt-a",
        "tunnel_endpoint": "https://x.trycloudflare.com",
        "owner_id": "owner-1",
    }
    assert is_valid_runtime_handle(bad_handle) is False

    job = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-rt-mm", state="allocating"), record_id="job-rt-mm")
    inst = service.allocate(job)
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst["id"],
            "owner_job_id": "job-rt-mm",
            "runtime_id": "rt-a",
            "handle": {**bad_handle, "job_id": "job-rt-mm", "owner_job_id": "job-rt-mm"},
        },
        record_id="lease-rt-mm",
    )
    service.bind_lease(job, {**lease, "runtime_id": "rt-a"})
    service.mark_ready(job)
    job = repo.update("jobs", "job-rt-mm", {"state": "loading", "lease_id": lease["id"], "instance_id": inst["id"]}, "owner-1")

    with pytest.raises(ValueError):
        repo.prepare_execution("job-rt-mm", "owner-1", lease["id"], 1000.0)

    # Also rejected in Scheduler._load even if reached directly
    sched = Scheduler(repo, provider_factory=lambda _: MockColabProvider(), executor_factory=lambda _: MockGPUExecutor(), instance_service=service, clock=lambda: 1000.0, config=cfg)
    failed = sched._load(job)
    assert failed["state"] == "cleaning_up"
    assert failed["error_code"] == "gpu_lease_invalid"
    assert repo.list("executions", "owner-1") == []
    assert sched.tick()["state"] == "cleanup_pending"


def test_v3_p1_6_execution_lease_id_mismatch_blocks_diagnostics_progress_and_terminal():
    _, repo = _make_repo()
    _setup_account(repo, provider="mock")
    service = InstanceService(repo, clock=lambda: 1000.0)

    job = repo.create("jobs", "owner-1", {**_calibrated_colab_job(job_id="job-el", state="allocating"), "provider": "mock", "execution_policy": {"provider": "mock", "lifecycle": "save"}}, record_id="job-el")
    inst = service.allocate(job)
    lease = repo.create("leases", "owner-1", {"state": "active", "provider": "mock", "account_id": "acct-1", "instance_id": inst["id"], "owner_job_id": "job-el", "handle": {"runtime_name": "rt-el"}}, record_id="lease-real")
    service.bind_lease(job, lease)
    service.mark_ready(job)
    service.job_busy(job)
    # Seed execution bound to lease-other while job references lease-real
    repo.create(
        "executions",
        "owner-1",
        {
            "state": "running",
            "job_id": "job-el",
            "instance_id": inst["id"],
            "lease_id": "lease-other",
            "generation": 1,
            "start_intent": {"state": "running", "execution_id": "exec-el", "lease_id": "lease-other", "generation": 1},
        },
        record_id="exec-el",
    )
    repo.update(
        "jobs",
        "job-el",
        {
            "state": "running",
            "lease_id": lease["id"],
            "instance_id": inst["id"],
            "execution_id": "exec-el",
            "active_execution": {"execution_id": "exec-el", "lease_id": lease["id"], "generation": 1},
            "start_intent": {"state": "running", "execution_id": "exec-el", "lease_id": lease["id"], "generation": 1},
        },
        "owner-1",
    )

    repo.record_gpu_diagnostics(
        repo.get("jobs", "job-el", "owner-1"),
        {"source": "runtime_nvidia_smi", "captured_at": 1001.0, "devices": [{"name": "L4", "memory_mib": 24000}]},
    )
    assert repo.get("jobs", "job-el", "owner-1").get("gpu_diagnostics") is None

    prog_status, _ = repo.commit_run_progress(
        "job-el",
        "owner-1",
        execution_id="exec-el",
        expected_generation=1,
        patch={"stage": "sampling"},
        message="sampling",
        now=1002.0,
    )
    assert prog_status in {"stale", "fenced"}
    assert repo.get("jobs", "job-el", "owner-1").get("stage") is None

    term_job = repo.commit_execution_terminal(
        "job-el",
        "owner-1",
        execution_id="exec-el",
        terminal_state="completed",
        message="done",
        patch=None,
        expected_generation=1,
        now=1003.0,
    )
    assert term_job["state"] == "running"
    assert repo.get("executions", "exec-el", "owner-1")["state"] == "running"


def test_v3_p1_7_finalize_retained_cleanup_validates_all_three_pointers():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")

    inst_i = repo.create("instances", "owner-1", {"state": "releasing", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-ptr", "lease_id": "lease-ptr", "runtime_id": "rt-i", "budget_reserved": 2.0}, record_id="inst-i")
    inst_j = repo.create("instances", "owner-1", {"state": "busy", "provider": "colab", "account_id": "acct-1", "current_job_id": "job-other", "lease_id": "lease-ptr", "runtime_id": "rt-j", "budget_reserved": 2.0}, record_id="inst-j")
    lease = repo.create("leases", "owner-1", {"state": "cleanup_pending", "provider": "colab", "account_id": "acct-1", "instance_id": inst_j["id"], "owner_job_id": "job-ptr", "runtime_id": "rt-j"}, record_id="lease-ptr")
    repo.create("jobs", "owner-1", {**_calibrated_colab_job(job_id="job-ptr", state="cleaning_up"), "instance_id": inst_i["id"], "lease_id": lease["id"]}, record_id="job-ptr")

    status, _, _ = repo.finalize_retained_cleanup(
        "job-ptr",
        "owner-1",
        lease_id=lease["id"],
        instance_id=inst_i["id"],
        idle_deadline_at=1500.0,
        now=1000.0,
    )
    assert status == "aborted"
    assert repo.get("instances", inst_i["id"], "owner-1")["state"] == "releasing"
    assert repo.get("instances", inst_j["id"], "owner-1")["state"] == "busy"
    assert repo.get("jobs", "job-ptr", "owner-1")["state"] == "cleaning_up"


def test_v3_p1_8_estimator_requires_non_empty_execution_metadata_and_producer_persists_it():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    cfg = _colab_config()
    service = InstanceService(repo, clock=lambda: 1000.0, config=cfg)

    job = _calibrated_colab_job(job_id="job-warm-v3", state="queued")
    fp = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    assert fp is not None
    proof = {
        "warm_saved": True,
        "warm_execution_id": "exec-w3", "worker_reported_execution_id": "exec-w3",
        "warm_epoch_id": "ep-w3",
        "warm_fingerprint": fp,
        "generation": 1,
        "runtime_meta": {"python_mm": "3.12"},
    }
    inst = {"id": "inst-w3", "lease_id": "lease-w3", "generation": 1, "warm_proof": proof, "compatibility_signature": job["compatibility_signature"], "hard_deadline_at": 1450.0, "gpu_tier": "L4"}
    lease = {"id": "lease-w3", "instance_id": "inst-w3", "generation": 1, "warm_proof": proof, "compatibility_signature": job["compatibility_signature"]}

    for bad_fields in [
        {"warm_epoch_id": None},
        {"instance_id": None},
        {"lease_id": None},
        {"warm_fingerprint": None},
        {"warm_fingerprint": "deadbeef" * 8},
    ]:
        exec_row = {
            "id": "exec-w3",
            "state": "completed",
            "instance_id": "inst-w3",
            "lease_id": "lease-w3",
            "generation": 1,
            "warm_epoch_id": "ep-w3",
            "warm_fingerprint": fp,
            **bad_fields,
        }
        assert is_warm_verified_for_job(job, instance_data=inst, lease_data=lease, executions_by_id={"exec-w3": exec_row}) is False
        ok, assessment = check_job_reuse_guard(
            job,
            now=1000.0,
            instance_data=inst,
            lease_data=lease,
            executions_by_id={"exec-w3": exec_row},
        )
        assert ok is False
        assert assessment.warm_verified is False
        assert assessment.need_seconds == pytest.approx(780.0)

    # Part (b): production completion path persists warm_epoch_id, instance_id, lease_id, generation, warm_fingerprint
    repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-prod", state="allocating"), record_id="job-prod")
    j_prod = repo.get("jobs", "job-prod", "owner-1")
    inst_p = service.allocate(j_prod)
    lease_p = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "instance_id": inst_p["id"],
            "owner_job_id": "job-prod",
            "runtime_id": "rt-prod",
            "generation": 2,
            "compatibility_signature": j_prod["compatibility_signature"],
            "deadline_epoch": 2000.0,
            "hard_deadline_at": 2000.0,
            "handle": {"runtime_name": "runtimes/rt-prod", "runtime_id": "rt-prod", "job_id": "job-prod"},
        },
        record_id="lease-prod",
    )
    service.bind_lease(j_prod, lease_p)
    service.mark_ready(j_prod)
    j_prod = repo.update("jobs", "job-prod", {"state": "loading", "lease_id": lease_p["id"], "instance_id": inst_p["id"]}, "owner-1")
    j_prod = repo.prepare_execution("job-prod", "owner-1", lease_p["id"], 1000.0)
    exec_p_id = j_prod["execution_id"]
    inst_busy = service.job_busy(repo.get("jobs", "job-prod", "owner-1"))
    prod_gen = inst_busy["generation"]
    repo.update("instances", inst_p["id"], {"hard_deadline_at": 2000.0}, "owner-1")
    repo.claim_execution_dispatch("job-prod", "owner-1", lease_id=lease_p["id"], execution_id=exec_p_id, now=1000.0)
    status_run, j_prod = repo.commit_running("job-prod", "owner-1", lease_id=lease_p["id"], execution_id=exec_p_id, now=1001.0)
    assert status_run == "ok"

    prod_proof = {
        "warm_saved": True,
        "warm_execution_id": exec_p_id,
        "worker_reported_execution_id": exec_p_id,
        "warm_epoch_id": "ep-prod",
        "warm_fingerprint": fp,
        "generation": prod_gen,
        "runtime_meta": {"python_mm": "3.12"},
        "compatibility_signature": j_prod["compatibility_signature"],
    }
    term_job = repo.commit_execution_terminal(
        "job-prod",
        "owner-1",
        execution_id=exec_p_id,
        terminal_state="completed",
        message="done",
        patch=None,
        proof=prod_proof,
        expected_generation=prod_gen,
        now=1002.0,
    )
    assert term_job["state"] == "completed"
    persisted_exec = repo.get("executions", exec_p_id, "owner-1")
    assert persisted_exec["warm_epoch_id"] == "ep-prod"
    assert persisted_exec["warm_fingerprint"] == fp
    assert persisted_exec["instance_id"] == inst_p["id"]
    assert persisted_exec["lease_id"] == lease_p["id"]
    assert persisted_exec["generation"] == prod_gen

    next_job = _calibrated_colab_job(job_id="job-next-prod", state="queued")
    assert (
        assess_job_reuse_window(
            next_job,
            now=1005.0,
            instance_data=repo.get("instances", inst_p["id"], "owner-1"),
            lease_data=repo.get("leases", lease_p["id"], "owner-1"),
            repo=repo,
            owner_id="owner-1",
        ).warm_verified
        is True
    )


# ---------------------------------------------------------------------------
# V4 Sign-Off Regression Suite (every P0 and P1 counterexample from V4)
# ---------------------------------------------------------------------------


def _alloc_colab_job(
    repo: Repository,
    service: InstanceService,
    job_id: str,
    *,
    final_state: str = "queued",
    estimated_units: float = 2.0,
) -> tuple[dict, dict]:
    job = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id=job_id, state="queued"), "estimated_units": estimated_units},
        record_id=job_id,
    )
    inst = service.allocate(job)
    if final_state != "queued":
        job = repo.update("jobs", job_id, {"state": final_state}, "owner-1")
    return job, inst


def test_v4_p0_1_canonical_endpoint_other_job_link_and_no_instance_claim(tmp_path):
    """V4 P0-1: colab://DUP/ vs dup share identity, job-only link conflicts, and no-instance cleanup acquires claim."""
    assert (
        normalize_tunnel_endpoint("colab://DUP/")
        == GoogleColabProvider._normalize_tunnel_endpoint("colab://DUP/")
        == normalize_tunnel_endpoint("dup")
        == "dup"
    )

    _, repo = _make_repo(tmp_path, "v4_p0_1.db")
    _setup_account(repo, provider="colab")

    # Case A: live lease-a ("colab://DUP/") vs orphan lease-b ("dup") with different runtime_ids.
    lease_live = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-live-a",
            "current_job_id": "job-live-a",
            "runtime_id": "rt-a",
            "tunnel_endpoint": "colab://DUP/",
            "handle": {"runtime_name": "runtimes/rt-a", "runtime_id": "rt-a", "tunnel_endpoint": "colab://DUP/"},
        },
        record_id="lease-live-a",
    )
    lease_orphan = repo.create(
        "leases",
        "owner-1",
        {
            "state": "cleanup_pending",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-orphan-b",
            "runtime_id": "rt-b",
            "tunnel_endpoint": "dup",
            "cleanup_next_epoch": 0,
            "handle": {
                "runtime_name": "runtimes/rt-b",
                "runtime_id": "rt-b",
                "tunnel_endpoint": "dup",
                "owner_id": "owner-1",
            },
        },
        record_id="lease-orphan-b",
    )
    delete_calls: list[dict] = []

    class TrackingProvider:
        def inspect(self, handle):
            return {"ready": True, "deleted": False}

        def delete(self, handle):
            delete_calls.append(dict(handle))
            return {"deleted": True}

    scheduler = Scheduler(repo, provider_factory=lambda _: TrackingProvider(), clock=lambda: 1000.0)
    assert lease_live["id"] != lease_orphan["id"]
    assert scheduler._cleanup_orphan_lease(lease_orphan) is False
    assert delete_calls == []

    # Case B: another lease whose ONLY link to tunnel_endpoint/instance_id is via its associated job.
    repo.update("leases", lease_live["id"], {"state": "deleted"}, "owner-1")
    lease_other = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-other-link",
            "runtime_id": "rt-other",
        },
        record_id="lease-other-link",
    )
    repo.create(
        "jobs",
        "owner-1",
        {
            **_calibrated_colab_job(job_id="job-other-link", state="running"),
            "lease_id": lease_other["id"],
            "tunnel_endpoint": "colab://DUP/",
            "handle": {"endpoint": "colab://DUP/"},
        },
        record_id="job-other-link",
    )
    assert scheduler._cleanup_orphan_lease(repo.get("leases", lease_orphan["id"], "owner-1")) is False
    assert delete_calls == []

    # Case C: no-instance legacy cleanup in _cleanup holds claim across provider I/O.
    repo.update("jobs", "job-other-link", {"state": "completed"}, "owner-1")
    repo.update("leases", lease_other["id"], {"state": "deleted"}, "owner-1")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())
    job_racer, _ = _alloc_colab_job(repo, service, "job-racer", final_state="queued")
    lease_racer = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "owner_job_id": "job-racer",
            "provider": "colab",
            "account_id": "acct-1",
            "runtime_id": "rt-new-dup",
        },
        record_id="lease-racer",
    )
    repo.update(
        "leases",
        lease_orphan["id"],
        {
            "state": "cleanup_pending",
            "cleanup_next_epoch": 0,
            "cleanup_claim_token": None,
            "create_attempts": 1,
            "ever_active": True,
            "request_id": "req-b",
            "owner_job_id": "job-orphan-b",
            "current_job_id": "job-orphan-b",
        },
        "owner-1",
    )
    repo.create(
        "jobs",
        "owner-1",
        {
            **_calibrated_colab_job(job_id="job-orphan-b", state="cleaning_up", lifecycle="ephemeral"),
            "lease_id": lease_orphan["id"],
            "instance_id": None,
            "cleanup_final_state": "failed",
        },
        record_id="job-orphan-b",
    )
    observed_during_delete: dict[str, object] = {}

    class ClaimCheckingProvider:
        def inspect(self, handle):
            return {"ready": False, "deleted": False}

        def delete(self, handle):
            cur = repo.get("leases", lease_orphan["id"], "owner-1")
            observed_during_delete["claim_token"] = cur.get("cleanup_claim_token")
            observed_during_delete["lease_state"] = cur.get("state")
            observed_during_delete["ep_bind_rejected"] = False
            try:
                service.bind_lease(
                    job_racer,
                    {**lease_racer, "tunnel_endpoint": "colab://DUP/"},
                )
            except InstanceConflict:
                observed_during_delete["ep_bind_rejected"] = True
            return True

    sched_claim = Scheduler(
        repo,
        provider_factory=lambda _: ClaimCheckingProvider(),
        clock=lambda: 1000.0,
        config=_colab_config(),
    )
    finished = sched_claim._cleanup(repo.get("jobs", "job-orphan-b", "owner-1"))
    assert finished["state"] == "failed"
    assert observed_during_delete["claim_token"]
    assert observed_during_delete["lease_state"] == "deleting"
    assert observed_during_delete["ep_bind_rejected"] is True
    after_lease = repo.get("leases", lease_orphan["id"], "owner-1")
    assert after_lease["state"] == "deleted"
    assert after_lease.get("cleanup_claim_token") is None


def test_v4_p0_2a_colab_fail_closed_on_page2_link_prev_and_cursor_zero(tmp_path):
    """V4 P0-2A: page=2/totalPages=2, Link rel=prev, and cursor='0' all fail closed as unknown (0 unassigns)."""
    cases = [
        ("page2", {"assignments": [], "hasPreviousPage": True, "page": 2, "totalPages": 2}, {}),
        ("link-prev", {"assignments": []}, {"Link": '</tokens/fst>; rel="prev"'}),
        ("cursor-zero", {"assignments": [], "nextPageToken": "0"}, {}),
    ]
    for name, payload, extra_headers in cases:
        _, repo = _make_repo(tmp_path, f"v4_p0_2a_{name}.db")
        _setup_account(repo, provider="colab")
        service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())
        _, inst = _alloc_colab_job(repo, service, f"job-{name}", final_state="queued")
        lease = repo.create(
            "leases",
            "owner-1",
            {
                "state": "cleanup_pending",
                "provider": "colab",
                "account_id": "acct-1",
                "owner_job_id": f"job-{name}",
                "instance_id": inst["id"],
                "runtime_id": f"rt-{name}",
                "tunnel_endpoint": f"m-s-{name}",
                "cleanup_next_epoch": 0,
                "handle": {
                    "runtime_name": f"runtimes/rt-{name}",
                    "runtime_id": f"rt-{name}",
                    "tunnel_endpoint": f"m-s-{name}",
                    "owner_id": "owner-1",
                    "owner_job_id": f"job-{name}",
                    "request_id": f"req-{name}",
                },
                "request_id": f"req-{name}",
            },
            record_id=f"lease-{name}",
        )
        repo.update("instances", inst["id"], {"lease_id": lease["id"], "state": "releasing", "current_job_id": None}, "owner-1")
        unassign_calls = 0

        def handler(request: httpx.Request, *, p=payload, h=extra_headers) -> httpx.Response:
            nonlocal unassign_calls
            if request.url.path == "/tun/m/assignments":
                return httpx.Response(200, text=_tunnel_body(p), headers=h)
            if "/unassign/" in request.url.path or request.method == "DELETE":
                unassign_calls += 1
            if request.url.path == "/v1beta/runtimes":
                v1_payload = {
                    "runtimes": [],
                    **{k: v for k, v in p.items() if k != "assignments"},
                }
                return httpx.Response(200, json=v1_payload, headers=h)
            raise AssertionError(f"unexpected path: {request.method} {request.url.path}")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = GoogleColabProvider(repo, "owner-1", account_id="acct-1", client=client, google=_tunnel_google())

        inspect_res = provider.inspect(lease["handle"])
        assert inspect_res["remote_verified"] is False
        assert inspect_res["allocation_unknown"] is True
        assert inspect_res["deleted"] is False

        with pytest.raises(ColabError):
            GoogleColabProvider(
                repo,
                "owner-1",
                account_id="acct-1",
                client=client,
                google=SimpleNamespace(access_token=lambda scope="colab": "tok", _credential=lambda *a, **k: {}),
            ).list_runtimes()

        scheduler = Scheduler(
            repo,
            provider_factory=lambda _: provider,
            clock=lambda: 1000.0,
            config=_colab_config(),
            instance_service=service,
        )
        assert scheduler._cleanup_orphan_lease(repo.get("leases", lease["id"], "owner-1")) is False
        assert unassign_calls == 0
        assert repo.get("leases", lease["id"], "owner-1")["state"] == "cleanup_pending"
        assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
        assert _budget_totals(service)[0] > 0.0
        client.close()


def test_v4_p0_2b_in_flight_rpc_never_expires_after_300s_and_late_handle_isolation(tmp_path):
    """V4 P0-2B: create_rpc_in_flight has no 300s timeout; late RPC on L_old never clobbers L_new/I_new."""
    _, repo = _make_repo(tmp_path, "v4_p0_2b.db")
    _setup_account(repo, provider="colab")
    service = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        clock=lambda: 2000.0,
        config=_colab_config(),
        instance_service=service,
    )

    _, inst_old = _alloc_colab_job(repo, service, "job-rpc", final_state="allocating")
    lease_old = repo.create(
        "leases",
        "owner-1",
        {
            "state": "creating",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-rpc",
            "instance_id": inst_old["id"],
            "create_rpc_instance_id": inst_old["id"],
            "create_attempts": 1,
            "create_rpc_in_flight": True,
            "create_rpc_started_at": 1000.0,
            "create_rpc_token": "tok-old",
            "request_id": "req-old",
        },
        record_id="lease-old",
    )
    service.bind_lease({"id": "job-rpc", "owner_id": "owner-1"}, lease_old)
    repo.update(
        "jobs",
        "job-rpc",
        {"state": "allocating", "lease_id": lease_old["id"], "instance_id": inst_old["id"]},
        "owner-1",
    )

    # Even at +301s and +1000s, in-flight create RPC must never locally expire into absence/deletion.
    assert Scheduler._create_rpc_is_in_flight(lease_old, 1301.0) is True
    assert Scheduler._create_rpc_is_in_flight(lease_old, 2000.0) is True
    ret_status, _, _ = repo.retire_absent_lease_cas("job-rpc", "owner-1", lease_old["id"], now=2000.0)
    assert ret_status == "in_flight"
    repo.update("leases", lease_old["id"], {"state": "cleanup_pending", "cleanup_next_epoch": 0}, "owner-1")
    repo.update("jobs", "job-rpc", {"lease_id": "lease-other"}, "owner-1")
    assert scheduler._cleanup_orphan_lease(repo.get("leases", lease_old["id"], "owner-1")) is False
    assert repo.get("leases", lease_old["id"], "owner-1")["state"] != "deleted"

    # Rebind job-rpc to L_new/I_new with rt-new active/busy, while slow RPC from L_old finishes with rt-old.
    repo.update("leases", lease_old["id"], {"state": "deleted", "cleanup_next_epoch": 0}, "owner-1")
    repo.update("instances", inst_old["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    job_rebind = repo.update(
        "jobs",
        "job-rpc",
        {"state": "queued", "lease_id": None, "instance_id": None, "alloc_generation": 1},
        "owner-1",
    )
    inst_new = service.allocate(job_rebind)
    job_rebind_inst = {**job_rebind, "instance_id": inst_new["id"]}
    lease_new = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-rpc",
            "current_job_id": "job-rpc",
            "instance_id": inst_new["id"],
            "runtime_id": "rt-new",
            "handle": {"runtime_name": "runtimes/rt-new", "runtime_id": "rt-new", "job_id": "job-rpc", "owner_id": "owner-1"},
        },
        record_id="lease-new",
    )
    service.bind_lease(job_rebind_inst, lease_new)
    service.mark_ready(job_rebind_inst)
    service.job_busy(job_rebind_inst)
    repo.update("instances", inst_new["id"], {"runtime_id": "rt-new"}, "owner-1")
    repo.update(
        "jobs",
        "job-rpc",
        {"state": "running", "lease_id": lease_new["id"], "instance_id": inst_new["id"]},
        "owner-1",
    )

    late_handle_old = {
        "runtime_name": "runtimes/rt-old",
        "runtime_id": "rt-old",
        "request_id": "req-old",
        "job_id": "job-rpc",
        "owner_id": "owner-1",
    }
    commit_status, _, recovered_old = repo.commit_create_handle_success(
        "job-rpc",
        "owner-1",
        lease_old["id"],
        handle=late_handle_old,
        rpc_token="tok-old",
        expected_attempt=1,
        expected_instance_id=inst_old["id"],
        started_at=1050.0,
    )
    assert commit_status == "fenced"
    assert recovered_old["state"] == "cleanup_pending"
    assert recovered_old["handle"]["runtime_id"] == "rt-old"

    # Also exercise recover_late_create_handle directly when job points to L_new/I_new.
    rec_status, _, rec_lease = repo.recover_late_create_handle(
        "job-rpc",
        "owner-1",
        lease_old["id"],
        handle=late_handle_old,
        instance_id=inst_old["id"],
        rpc_token="tok-old",
        expected_attempt=1,
        now=1060.0,
    )
    assert rec_status == "ok"
    assert rec_lease["state"] == "cleanup_pending"
    job_after = repo.get("jobs", "job-rpc", "owner-1")
    lease_new_after = repo.get("leases", lease_new["id"], "owner-1")
    inst_new_after = repo.get("instances", inst_new["id"], "owner-1")
    inst_old_after = repo.get("instances", inst_old["id"], "owner-1")
    assert job_after["state"] == "running"
    assert job_after["lease_id"] == lease_new["id"]
    assert job_after["instance_id"] == inst_new["id"]
    assert lease_new_after["state"] == "active"
    assert lease_new_after["handle"]["runtime_id"] == "rt-new"
    assert inst_new_after["state"] == "busy"
    assert inst_new_after["current_job_id"] == "job-rpc"
    assert inst_new_after["runtime_id"] == "rt-new"
    assert inst_old_after["state"] == "cleanup_pending"
    assert inst_old_after["runtime_id"] == "rt-old"


def test_v4_p0_3_placeholder_retire_guards_terminated_job_exec_and_usage_segments(tmp_path):
    """V4 P0-3: associated terminated job execution_id or positive usage segments block 0-RPC delete & keep quota."""
    # Subcase A: terminated job carrying execution_id (+ no deleted flag on inst/lease).
    _, repo_a = _make_repo(tmp_path, "v4_p0_3a.db")
    _setup_account(repo_a, provider="colab")
    svc_a = InstanceService(repo_a, clock=lambda: 1000.0, config=_colab_config())
    _, inst_a = _alloc_colab_job(repo_a, svc_a, "job-a", final_state="queued", estimated_units=3.0)
    lease_a = repo_a.create(
        "leases",
        "owner-1",
        {"state": "active", "provider": "colab", "account_id": "acct-1", "owner_job_id": "job-a", "instance_id": inst_a["id"], "cleanup_next_epoch": 0},
        record_id="lease-a",
    )
    svc_a.bind_lease({"id": "job-a", "owner_id": "owner-1"}, lease_a)
    repo_a.update("leases", lease_a["id"], {"state": "creating", "cleanup_next_epoch": 0}, "owner-1")
    repo_a.update("instances", inst_a["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    repo_a.update(
        "jobs",
        "job-a",
        {"state": "failed", "lease_id": lease_a["id"], "instance_id": inst_a["id"], "execution_id": "exec-ghost-a"},
        "owner-1",
    )
    cas_a, reason_a = repo_a.retire_pure_orphan_placeholder_cas("owner-1", lease_a["id"], instance_id=inst_a["id"], now=1001.0)
    assert cas_a == "not_pure"
    assert reason_a in {
        "job_execution_evidence",
        "associated_execution_exists",
        "execution_id_exists",
        "active_execution_exists",
    }
    with pytest.raises(InstanceConflict):
        svc_a.retire_pure_placeholder("owner-1", inst_a["id"], lease_id=lease_a["id"])
    assert _budget_totals(svc_a)[0] == pytest.approx(3.0)

    # Subcase B: positive GpuInstanceUsageSegmentRow blocks 0-RPC placeholder delete and preserves full 4-unit quota.
    _, repo_b = _make_repo(tmp_path, "v4_p0_3b.db")
    _setup_account(repo_b, provider="colab")
    svc_b = InstanceService(repo_b, clock=lambda: 1000.0, config=_colab_config())
    _, inst_b = _alloc_colab_job(repo_b, svc_b, "job-b", final_state="queued", estimated_units=3.0)
    lease_b = repo_b.create(
        "leases",
        "owner-1",
        {"state": "creating", "provider": "colab", "account_id": "acct-1", "owner_job_id": "job-b", "instance_id": inst_b["id"], "cleanup_next_epoch": 0},
        record_id="lease-b",
    )
    svc_b.bind_lease({"id": "job-b", "owner_id": "owner-1"}, lease_b)
    repo_b.update("instances", inst_b["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    repo_b.update(
        "jobs",
        "job-b",
        {"state": "allocating", "lease_id": lease_b["id"], "instance_id": inst_b["id"], "alloc_generation": 1},
        "owner-1",
    )
    repo_b.create(
        "instance_usage_segments",
        "owner-1",
        {"account_id": "acct-1", "instance_id": inst_b["id"], "units": 1.0, "billing_units": 1.0, "actual_units": 1.0},
        record_id="seg-1",
    )
    cas_b, _ = repo_b.retire_pure_orphan_placeholder_cas("owner-1", lease_b["id"], instance_id=inst_b["id"], now=1001.0)
    assert cas_b == "not_pure"
    retried_b, _ = repo_b.retire_absent_placeholder_for_retry(
        "job-b", "owner-1", expected_lease_id=lease_b["id"], expected_generation=1, total_attempts=1, now=1001.0
    )
    assert retried_b is False
    scheduler_b = Scheduler(
        repo_b,
        provider_factory=lambda _: MockColabProvider(),
        clock=lambda: 1001.0,
        config=_colab_config(),
        instance_service=svc_b,
    )
    assert scheduler_b._cleanup_orphan_lease(repo_b.get("leases", lease_b["id"], "owner-1")) is False
    assert repo_b.get("instances", inst_b["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(svc_b)[0] == pytest.approx(4.0)


def test_v4_p1_3_quarantine_cas_snapshot_op_guard_and_historical_endpoint_retire(tmp_path):
    """V4 P1-3: rebound job safe from old lease quarantine; ready probe never reverts running/op; confirmed absent endpoint retires."""
    # (1) Quarantine L_old must never fail job rebound to L_new/I_new even when L_old.version is unchanged.
    _, repo1 = _make_repo(tmp_path, "v4_p1_3a.db")
    _setup_account(repo1, provider="colab")
    svc1 = InstanceService(repo1, clock=lambda: 1000.0, config=_colab_config())
    job_q_row, inst_old = _alloc_colab_job(repo1, svc1, "job-q", final_state="queued")
    lease_old = repo1.create(
        "leases",
        "owner-1",
        {"state": "active", "provider": "colab", "account_id": "acct-1", "owner_job_id": "job-q", "instance_id": inst_old["id"]},
        record_id="lease-q-old",
    )
    svc1.bind_lease(job_q_row, lease_old)
    repo1.update("instances", inst_old["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    job_q_row = repo1.update(
        "jobs",
        "job-q",
        {"state": "queued", "lease_id": None, "instance_id": None, "alloc_generation": 1},
        "owner-1",
    )
    inst_new = svc1.allocate(job_q_row)
    job_q_new = {**job_q_row, "instance_id": inst_new["id"]}
    lease_new = repo1.create(
        "leases",
        "owner-1",
        {"state": "active", "provider": "colab", "account_id": "acct-1", "owner_job_id": "job-q", "instance_id": inst_new["id"], "runtime_id": "rt-q-new"},
        record_id="lease-q-new",
    )
    job_q_new = repo1.update(
        "jobs",
        "job-q",
        {"lease_id": lease_new["id"], "instance_id": inst_new["id"]},
        "owner-1",
    )
    svc1.bind_lease(job_q_new, lease_new)
    svc1.mark_ready(job_q_new)
    svc1.job_busy(job_q_new)
    job_live = repo1.update(
        "jobs",
        "job-q",
        {"state": "running", "lease_id": lease_new["id"], "instance_id": inst_new["id"]},
        "owner-1",
    )
    stale_snapshot = {**job_live, "lease_id": lease_old["id"], "instance_id": inst_old["id"]}
    sched1 = Scheduler(repo1, provider_factory=lambda _: MockColabProvider(), clock=lambda: 1000.0, instance_service=svc1)
    sched1._quarantine_missing_identity(stale_snapshot, repo1.get("leases", lease_old["id"], "owner-1"))
    assert repo1.get("leases", lease_old["id"], "owner-1")["state"] == "quarantined"
    assert repo1.get("jobs", "job-q", "owner-1")["state"] == "running"
    assert repo1.get("instances", inst_new["id"], "owner-1")["state"] == "busy"

    # (2) Concurrent recover_unknown_create must never overwrite op-new or revert running -> preparing.
    _, repo2 = _make_repo(tmp_path, "v4_p1_3b.db")
    _setup_account(repo2, provider="colab")
    lease_op = repo2.create(
        "leases",
        "owner-1",
        {
            "state": "unknown",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-rec",
            "runtime_id": "rt-rec",
            "operation_name": "operations/op-old",
            "create_attempts": 1,
            "handle": {"runtime_name": "runtimes/rt-rec", "runtime_id": "rt-rec", "owner_id": "owner-1", "job_id": "job-rec"},
        },
        record_id="lease-op",
    )
    job_rec = repo2.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-rec", state="allocating"), "lease_id": lease_op["id"]},
        record_id="job-rec",
    )

    class ConcurrentPromoteProvider:
        def inspect(self, handle):
            repo2.update("jobs", "job-rec", {"state": "running"}, "owner-1")
            repo2.update("leases", lease_op["id"], {"operation_name": "operations/op-new", "state": "active"}, "owner-1")
            return {"ready": True, "deleted": False, "operation_name": "operations/op-old"}

    sched2 = Scheduler(repo2, provider_factory=lambda _: ConcurrentPromoteProvider(), clock=lambda: 1000.0)
    outcome, _, _ = sched2._recover_unknown_create_with_snapshot(job_rec, lease_op, ConcurrentPromoteProvider())
    assert outcome == "stale"
    assert repo2.get("jobs", "job-rec", "owner-1")["state"] == "running"
    assert repo2.get("leases", lease_op["id"], "owner-1")["operation_name"] == "operations/op-new"

    # (3) Historical active lease without handle but WITH confirmed-absent tunnel_endpoint + nbh retires to deleted.
    _, repo3 = _make_repo(tmp_path, "v4_p1_3c.db")
    _setup_account(repo3, provider="colab")
    svc3 = InstanceService(repo3, clock=lambda: 1000.0, config=_colab_config())
    _, inst_h = _alloc_colab_job(repo3, svc3, "job-hist", final_state="allocating")
    lease_h = repo3.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-hist",
            "instance_id": inst_h["id"],
            "tunnel_endpoint": "colab://m-s-hist/",
            "tunnel_nbh": "nbh-hist",
            "create_attempts": 1,
            "create_rpc_in_flight": False,
            "ever_active": True,
        },
        record_id="lease-hist",
    )
    svc3.bind_lease({"id": "job-hist", "owner_id": "owner-1"}, lease_h)
    repo3.update(
        "jobs",
        "job-hist",
        {"state": "allocating", "lease_id": lease_h["id"], "instance_id": inst_h["id"]},
        "owner-1",
    )
    ret_h, _, retired_lease = repo3.retire_absent_lease_cas(
        "job-hist",
        "owner-1",
        lease_h["id"],
        expected_tunnel_endpoint="colab://m-s-hist/",
        expected_tunnel_nbh="nbh-hist",
        confirmed_absent=True,
        now=1005.0,
    )
    assert ret_h == "retired"
    assert retired_lease["state"] == "deleted"


def test_v4_p1_4_dispatch_poll_fail_closed_and_recovery_deadline(tmp_path):
    """V4 P1-4: missing/false marker or missing status state never defaults to running; sets deadline & fails closed."""
    exec_uuid = "11111111-1111-4111-8111-111111111111"
    code = build_poll_code(exec_uuid)
    assert '"state": "unknown"' in code
    assert '"state": "running"' not in code

    def _run_probe(sub_dir):
        patched = code.replace('"/content/ai-studio"', repr(str(sub_dir)))
        ns: dict[str, object] = {}
        exec(compile(patched, "<poll_probe>", "exec"), ns)
        return ns["_result"]

    # 1. Empty execution dir (no marker.json, no status.json) -> state="unknown", marker=False, status_present=False.
    d_empty = tmp_path / "probe_empty"
    (d_empty / exec_uuid).mkdir(parents=True)
    res_empty = _run_probe(d_empty)
    assert res_empty["state"] == "unknown"
    assert res_empty["marker"] is False
    assert res_empty["status_present"] is False

    # 2. marker.json with state="reserved" -> never "running".
    (d_empty / exec_uuid / "marker.json").write_text(json.dumps({"execution_id": exec_uuid, "state": "reserved"}))
    res_res = _run_probe(d_empty)
    assert res_res["state"] == "reserved"
    assert res_res["marker_state"] == "reserved"

    # 3. status.json omitting "state" -> never defaults to "running".
    (d_empty / exec_uuid / "status.json").write_text(json.dumps({"execution_id": exec_uuid}))
    res_nostate = _run_probe(d_empty)
    assert res_nostate["state"] == "unknown"
    assert res_nostate["status_present"] is True

    # 4. marker.json with state="launched", live pid, matching execution_id (and no status.json) -> "running".
    (d_empty / exec_uuid / "status.json").unlink()
    (d_empty / exec_uuid / "marker.json").write_text(json.dumps({"execution_id": exec_uuid, "state": "launched", "pid": os.getpid()}))
    res_launched = _run_probe(d_empty)
    assert res_launched["state"] == "running"
    assert res_launched["marker"] is True
    assert res_launched["launcher_execution_id"] == exec_uuid
    assert res_launched["worker_process_alive"] is True
    assert res_launched.get("worker_reported_execution_id") is None

    class DispatchPollExecutor(MockGPUExecutor):
        def __init__(self, payloads: list[dict]):
            super().__init__()
            self.payloads = list(payloads)

        def start(self, handle, job, execution_id):
            raise AssertionError("must not start a second execution during dispatching reconcile")

        def poll(self, handle, job):
            raw = self.payloads.pop(0)
            req_url = "https://mock-mcp.example/mcp"

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                method = body.get("method")
                if method == "initialize":
                    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"protocolVersion": "2024-11-05", "capabilities": {}}})
                if method == "notifications/initialized":
                    return httpx.Response(202, json={})
                params = body.get("params") or {}
                if params.get("name") == "create_session":
                    parent = f"{handle['runtime_name']}"
                    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"structuredContent": {"name": f"{parent}/sessions/sess-1"}}})
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(raw)}]}})

            client = httpx.Client(transport=httpx.MockTransport(handler))
            try:
                ex = MCPGPUExecutor(
                    repo,
                    "owner-1",
                    account_id="acct-1",
                    google=SimpleNamespace(access_token=lambda scope="colab": "tok", _credential=lambda *a, **k: {}),
                    client=client,
                    endpoint=req_url,
                )
                return ex.poll(handle, job)
            finally:
                client.close()

    _, repo = _make_repo(tmp_path, "v4_p1_4.db")
    _setup_account(repo, provider="mock")
    now_box = [1000.0]
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "mock",
            "owner_job_id": "job-disp",
            "current_job_id": "job-disp",
            "deadline_epoch": 5000.0,
            "generation": 1,
            "handle": {"runtime_name": "runtimes/rt-disp", "runtime_id": "rt-disp", "owner_id": "owner-1", "job_id": "job-disp"},
        },
        record_id="lease-disp",
    )
    job = repo.create(
        "jobs",
        "owner-1",
        {
            **_calibrated_colab_job(job_id="job-disp", state="loading"),
            "provider": "mock",
            "execution_policy": {"provider": "mock", "lifecycle": "retain"},
            "lease_id": lease["id"],
            "generation": 1,
        },
        record_id="job-disp",
    )
    job = repo.prepare_execution("job-disp", "owner-1", lease["id"], 1000.0)
    exec_id = job["execution_id"]
    job = seed_execution_binding(repo, job, lease)
    claim_status, job, lease = repo.claim_execution_dispatch("job-disp", "owner-1", lease_id=lease["id"], execution_id=exec_id, now=1000.0)
    assert claim_status == "ok"

    ambiguous_executor = DispatchPollExecutor(
        [
            {"execution_id": exec_id, "marker": False, "status_present": False},
            {"execution_id": exec_id, "marker": False, "status_present": True, "state": "running"},
            {"execution_id": exec_id, "marker": True, "state": "running", "worker_reported_execution_id": "wrong-exec"},
        ]
    )
    scheduler = Scheduler(
        repo,
        provider_factory=lambda _: MockColabProvider(),
        executor_factory=lambda _: ambiguous_executor,
        clock=lambda: now_box[0],
    )
    # Polls 1 and 2 receive ambiguous non-start evidence -> must NOT promote running; back off & persist deadline.
    for attempt in (1, 2):
        out = scheduler._reconcile_dispatching_load(repo.get("jobs", "job-disp", "owner-1"), lease, exec_id)
        assert out["state"] == "loading"
        assert out.get("dispatch_poll_attempts") == attempt
        assert out.get("dispatch_recovery_deadline_at") == pytest.approx(1030.0)
        now_box[0] += 1.0

    # Poll 3 returns worker_reported_execution_id mismatch -> fails cleanly without starting a 2nd run.
    out_mismatch = scheduler._reconcile_dispatching_load(repo.get("jobs", "job-disp", "owner-1"), lease, exec_id)
    assert out_mismatch["state"] in {"cleaning_up", "cleanup_pending", "failed"}
    assert out_mismatch.get("error_code") == "gpu_dispatch_timeout"


def test_v4_p1_5_strict_colab_runtime_name_validation_everywhere(tmp_path):
    """V4 P1-5: 'runtimes/rt-b/' (trailing slash) and siblings fail uniformly across Colab, repository, and scheduler."""
    bad_names = ["runtimes/rt-b/", "projects/p/locations/l/runtimes/rt-b", "runtimes/RT_B", " runtimes/rt-b "]
    _, repo = _make_repo(tmp_path, "v4_p1_5.db")
    provider = GoogleColabProvider(repo, "owner-1", account_id="acct-1", google=_tunnel_google())

    for bad_name in bad_names:
        handle = {"runtime_name": bad_name, "runtime_id": "rt-b", "owner_id": "owner-1", "job_id": "job-bad"}
        assert is_valid_runtime_handle(handle, owner_id="owner-1", provider="colab") is False
        with pytest.raises(ColabError):
            provider._owned(handle)

    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-bad",
            "current_job_id": "job-bad",
            "runtime_id": "rt-b",
            "generation": 1,
            "deadline_epoch": 5000.0,
            "handle": {"runtime_name": "runtimes/rt-b/", "runtime_id": "rt-b", "owner_id": "owner-1", "job_id": "job-bad"},
        },
        record_id="lease-bad-rt",
    )
    repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-bad", state="loading"), "lease_id": lease["id"], "generation": 1},
        record_id="job-bad",
    )
    with pytest.raises(ValueError):
        repo.prepare_execution("job-bad", "owner-1", lease["id"], 1000.0)

    exec_row = repo.create(
        "executions",
        "owner-1",
        {"state": "starting", "job_id": "job-bad", "lease_id": lease["id"], "generation": 1},
        record_id="exec-bad-rt",
    )
    repo.update("jobs", "job-bad", {"execution_id": exec_row["id"]}, "owner-1")
    d_status, _ = repo.claim_execution_dispatch("job-bad", "owner-1", lease_id=lease["id"], execution_id=exec_row["id"], now=1000.0)[:2]
    assert d_status in {"stale", "fenced"}

    lease2 = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-bad-2",
            "current_job_id": "job-bad-2",
            "runtime_id": "rt-b",
            "generation": 1,
            "deadline_epoch": 5000.0,
            "handle": {"runtime_name": "runtimes/rt-b/", "runtime_id": "rt-b", "owner_id": "owner-1", "job_id": "job-bad-2"},
        },
        record_id="lease-bad-rt2",
    )
    job2 = repo.create(
        "jobs",
        "owner-1",
        {**_calibrated_colab_job(job_id="job-bad-2", state="loading"), "lease_id": lease2["id"], "generation": 1},
        record_id="job-bad-2",
    )
    scheduler = Scheduler(repo, provider_factory=lambda _: MockColabProvider(), clock=lambda: 1000.0)
    res_job = scheduler._load(job2)
    assert res_job["state"] != "running"
    assert res_job.get("error_code") == "gpu_lease_invalid"


def test_v4_p1_6_non_null_execution_generation_required_across_all_writers(tmp_path):
    """V4 P1-6: missing/None execution.generation must reject diagnostics, progress, terminal, and retained cleanup without backfill."""
    _, repo = _make_repo(tmp_path, "v4_p1_6.db")
    _setup_account(repo, provider="colab")
    svc = InstanceService(repo, clock=lambda: 1000.0, config=_colab_config())
    _, inst = _alloc_colab_job(repo, svc, "job-g", final_state="queued")
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-g",
            "current_job_id": "job-g",
            "instance_id": inst["id"],
            "runtime_id": "rt-g",
            "generation": 1,
            "hard_deadline_at": 2000.0,
            "handle": {"runtime_name": "runtimes/rt-g", "runtime_id": "rt-g", "owner_id": "owner-1", "job_id": "job-g"},
        },
        record_id="lease-g",
    )
    svc.bind_lease({"id": "job-g", "owner_id": "owner-1"}, lease)
    svc.mark_ready({"id": "job-g", "owner_id": "owner-1"})
    svc.job_busy({"id": "job-g", "owner_id": "owner-1"})
    exec_g = repo.create(
        "executions",
        "owner-1",
        {
            "state": "running",
            "job_id": "job-g",
            "lease_id": lease["id"],
            "instance_id": inst["id"],
            "generation": None,
        },
        record_id="exec-null-gen",
    )
    job = repo.update(
        "jobs",
        "job-g",
        {
            "state": "running",
            "lease_id": lease["id"],
            "instance_id": inst["id"],
            "execution_id": exec_g["id"],
            "generation": 1,
        },
        "owner-1",
    )

    repo.record_gpu_diagnostics(job, {"devices": [{"name": "NVIDIA L4", "memory_total_mb": 24000}]})
    assert repo.get("jobs", "job-g", "owner-1").get("gpu_diagnostics") is None
    assert repo.get("executions", exec_g["id"], "owner-1").get("generation") is None

    prog_status, prog_job = repo.commit_run_progress(
        "job-g",
        "owner-1",
        execution_id=exec_g["id"],
        patch={"state": "uploading"},
        message="uploading",
        progress=0.9,
        expected_generation=1,
        now=1001.0,
    )
    assert prog_status == "stale"
    assert prog_job["state"] == "running"
    assert repo.get("executions", exec_g["id"], "owner-1").get("generation") is None

    term_out = repo.commit_execution_terminal(
        "job-g",
        "owner-1",
        execution_id=exec_g["id"],
        terminal_state="completed",
        message="done",
        patch=None,
        expected_generation=1,
        now=1002.0,
    )
    assert term_out["state"] == "running"
    assert repo.get("executions", exec_g["id"], "owner-1").get("generation") is None

    repo.update("instances", inst["id"], {"state": "retained_idle", "current_job_id": None, "idle_deadline_at": 1500.0}, "owner-1")
    fin_status, _, _ = repo.finalize_retained_cleanup(
        "job-g",
        "owner-1",
        lease_id=lease["id"],
        instance_id=inst["id"],
        final_state="completed",
        expected_instance_state="retained_idle",
        idle_deadline_at=1500.0,
        hard_deadline_at=2000.0,
        legacy_mock_idle=False,
        require_instance=True,
        now=1003.0,
    )
    assert fin_status == "aborted"
    assert repo.get("jobs", "job-g", "owner-1")["state"] != "completed"
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "idle"
    assert repo.get("executions", exec_g["id"], "owner-1").get("generation") is None


def test_v4_p1_7_non_legacy_finalize_retained_cleanup_requires_all_four_pointers(tmp_path):
    """V4 P1-7: non-legacy finalize_retained_cleanup must abort if L.instance_id or I.lease_id is None."""
    _, repo = _make_repo(tmp_path, "v4_p1_7.db")
    _setup_account(repo, provider="colab")
    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "retained_idle",
            "provider": "colab",
            "account_id": "acct-1",
            "lease_id": None,
            "generation": 1,
            "idle_deadline_at": 1500.0,
            "hard_deadline_at": 2000.0,
            "runtime_id": "rt-p7",
        },
        record_id="inst-p7",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "provider": "colab",
            "account_id": "acct-1",
            "owner_job_id": "job-p7",
            "instance_id": None,
            "generation": 1,
            "runtime_id": "rt-p7",
            "handle": {"runtime_name": "runtimes/rt-p7", "runtime_id": "rt-p7", "owner_id": "owner-1", "job_id": "job-p7"},
        },
        record_id="lease-p7",
    )
    exec_p7 = repo.create(
        "executions",
        "owner-1",
        {"state": "running", "job_id": "job-p7", "lease_id": lease["id"], "instance_id": inst["id"], "generation": 1},
        record_id="exec-p7",
    )
    repo.create(
        "jobs",
        "owner-1",
        {
            **_calibrated_colab_job(job_id="job-p7", state="cleaning_up"),
            "lease_id": lease["id"],
            "instance_id": inst["id"],
            "execution_id": exec_p7["id"],
            "generation": 1,
        },
        record_id="job-p7",
    )

    for l_inst, i_lease in [(None, None), (inst["id"], None), (None, lease["id"])]:
        repo.update("leases", lease["id"], {"instance_id": l_inst, "state": "active"}, "owner-1")
        repo.update("instances", inst["id"], {"lease_id": i_lease, "state": "retained_idle"}, "owner-1")
        status, job_row, lease_row = repo.finalize_retained_cleanup(
            "job-p7",
            "owner-1",
            lease_id=lease["id"],
            instance_id=inst["id"],
            final_state="completed",
            expected_instance_state="retained_idle",
            idle_deadline_at=1500.0,
            hard_deadline_at=2000.0,
            legacy_mock_idle=False,
            require_instance=True,
            now=1005.0,
        )
        assert status == "aborted"
        assert job_row["state"] != "completed"
        assert lease_row["state"] != "idle"


def test_v4_p1_8_warm_verified_rejects_false_warm_saved_and_partial_five_tuple():
    """V4 P1-8: requires warm_saved==True on both proofs and all 5 execution fields strictly matching."""
    job = _calibrated_colab_job(job_id="job-warm-next", state="queued")
    fp = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    assert fp is not None

    base_proof = {
        "warm_saved": True,
        "warm_execution_id": "exec-p8", "worker_reported_execution_id": "exec-p8",
        "warm_epoch_id": "ep-p8",
        "warm_fingerprint": fp,
        "generation": 2,
        "runtime_meta": {"python_mm": "3.12"},
    }
    base_exec = {
        "id": "exec-p8",
        "state": "completed",
        "job_id": "job-prev",
        "instance_id": "inst-p8",
        "lease_id": "lease-p8",
        "generation": 2,
        "warm_epoch_id": "ep-p8",
        "warm_fingerprint": fp,
    }
    inst = {"id": "inst-p8", "generation": 2, "warm_proof": dict(base_proof)}
    lease = {"id": "lease-p8", "generation": 2, "warm_proof": dict(base_proof)}

    # Control: complete pair + warm_saved=True + 5-tuple match -> warm_verified=True (450s allowed at +451s).
    assert is_warm_verified_for_job(job, instance_data=inst, lease_data=lease, executions_by_id={"exec-p8": base_exec}) is True
    assert check_job_reuse_guard(
        job, guard_mode="enforce", now=1000.0, instance_data=inst, lease_data=lease, deadlines=[1451.0], executions_by_id={"exec-p8": base_exec}
    )[0] is True

    # Any warm_saved=False (instance only, lease only, or both) forces cold-start (780s required -> +451s denied).
    for inst_saved, lease_saved in [(False, True), (True, False), (False, False)]:
        bad_inst = {**inst, "warm_proof": {**base_proof, "warm_saved": inst_saved}}
        bad_lease = {**lease, "warm_proof": {**base_proof, "warm_saved": lease_saved}}
        assert is_warm_verified_for_job(job, instance_data=bad_inst, lease_data=bad_lease, executions_by_id={"exec-p8": base_exec}) is False
        allowed, assessment = check_job_reuse_guard(
            job, guard_mode="enforce", now=1000.0, instance_data=bad_inst, lease_data=bad_lease, deadlines=[1451.0], executions_by_id={"exec-p8": base_exec}
        )
        assert allowed is False
        assert assessment is not None and assessment.need_seconds >= 780.0

    # Any missing ID or missing field in the 5-tuple falls back to cold-start (single-ID forbidden).
    for omit_key in ("instance_id", "lease_id", "warm_epoch_id", "generation", "warm_fingerprint"):
        incomplete_exec = {k: v for k, v in base_exec.items() if k != omit_key}
        assert is_warm_verified_for_job(
            job, instance_data=inst, lease_data=lease, executions_by_id={"exec-p8": incomplete_exec}
        ) is False
        allowed, assessment = check_job_reuse_guard(
            job, guard_mode="enforce", now=1000.0, instance_data=inst, lease_data=lease, deadlines=[1451.0], executions_by_id={"exec-p8": incomplete_exec}
        )
        assert allowed is False
        assert assessment is not None and assessment.need_seconds >= 780.0


# V5 uses real SQL transitions, real generated probes, and controlled provider I/O.
def seed_execution_binding(repo, job, lease, *, instance=None):
    """Build an explicit, valid persisted execution fixture; never infer identity in production writers."""
    owner = job["owner_id"]
    generation = job.get("generation") or lease.get("generation") or 1
    instance_id = instance["id"] if instance is not None else job.get("instance_id") or lease.get("instance_id")
    if instance is None and instance_id is None:
        instance_id = "fixture-i-" + job["id"]
        instance = repo.create("instances", owner, {"state": "busy", "provider": job.get("provider", "mock"),
            "account_id": job.get("account_id") or lease.get("account_id") or "fixture-account",
            "lease_id": lease["id"], "current_job_id": job["id"], "generation": generation}, record_id=instance_id)
    repo.update("leases", lease["id"], {"instance_id": instance_id}, owner)
    execution_id = job["execution_id"]
    binding = {"job_id": job["id"], "lease_id": lease["id"], "instance_id": instance_id, "generation": generation}
    if repo.get("executions", execution_id, owner) is None:
        repo.create("executions", owner, {"state": "running", **binding}, record_id=execution_id)
    else:
        repo.update("executions", execution_id, binding, owner)
    repo.update("leases", lease["id"], {"generation": generation}, owner)
    if instance is not None:
        repo.update("instances", instance_id, {"lease_id": lease["id"], "generation": generation}, owner)
    return repo.update("jobs", job["id"], {"lease_id": lease["id"], "instance_id": instance_id,
        "generation": generation}, owner)


def _v5_runtime(repo, *, suffix="old", state="allocating", with_handle=True, job_id="job-v5"):
    lease_id, inst_id = f"lease-v5-{suffix}", f"inst-v5-{suffix}"
    runtime_id = f"rt-{suffix}"
    handle = {"runtime_name": f"runtimes/{runtime_id}", "runtime_id": runtime_id,
              "owner_id": "owner-1", "job_id": job_id, "owner_job_id": job_id,
              "request_id": "11111111-1111-4111-8111-111111111111"}
    previous = repo.get("jobs", job_id, "owner-1")
    if previous and previous.get("instance_id") != inst_id:
        repo.update("instances", previous["instance_id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    inst = repo.create("instances", "owner-1", {
        "state": "busy" if with_handle else "allocating", "provider": "colab", "account_id": "acct-1",
        "current_job_id": job_id, "lease_id": lease_id, "budget_reserved": 3.0,
        "generation": 1, "compatibility_signature": "sig-ref2v", "hard_deadline_at": 10000.0,
    }, record_id=inst_id)
    lease = repo.create("leases", "owner-1", {
        "state": "active" if with_handle else "creating", "provider": "colab", "account_id": "acct-1",
        "owner_job_id": job_id, "current_job_id": job_id, "instance_id": inst_id,
        "runtime_id": runtime_id, "request_id": handle["request_id"], "generation": 1,
        "compatibility_signature": "sig-ref2v", "deadline_epoch": 10000.0,
        **({"handle": handle} if with_handle else {}),
    }, record_id=lease_id)
    data = {**_calibrated_colab_job(job_id=job_id, state=state, lifecycle="ephemeral"),
            "account_id": "acct-1", "lease_id": lease_id, "instance_id": inst_id,
            "generation": 1, "cleanup_final_state": "failed"}
    if repo.get("jobs", job_id, "owner-1"):
        job = repo.update("jobs", job_id, data, "owner-1")
    else:
        job = repo.create("jobs", "owner-1", data, record_id=job_id)
    return job, lease, inst


def _v5_local_executor(repo, root):
    """Execute the production MCP reader against real local worker files."""
    executor = object.__new__(MCPGPUExecutor)
    executor.store, executor.owner_id = repo, "owner-1"
    executor._session = lambda *args: "local-session"

    def execute(handle, session, probe_id, code):
        namespace = {}
        with contextlib.redirect_stdout(__import__("io").StringIO()):
            exec(compile(code.replace('"/content/ai-studio"', repr(str(root))), "v5-worker-probe", "exec"), namespace)
        return namespace["_result"]

    executor._exec_code_on_handle = execute
    return executor


def _v5_prepared(repo):
    job, lease, inst = _v5_runtime(repo, state="loading")
    job = repo.prepare_execution(job["id"], "owner-1", lease["id"], 1000.0, require_instance=True)
    status, job, lease = repo.claim_execution_dispatch(job["id"], "owner-1", lease_id=lease["id"],
        execution_id=job["execution_id"], now=1000.0, require_instance=True)
    assert status == "ok"
    return job, lease, repo.get("instances", inst["id"], "owner-1")


def _v5_rows(repo):
    return {kind: repo.list(kind, "owner-1") for kind in ("jobs", "leases", "instances", "executions", "events")}


def test_v5_p0_1_handoff_between_snapshot_and_cleanup_claim_has_zero_delete(monkeypatch):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, state="cleaning_up")
    next_job = repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-b"), record_id="job-b")
    original = repo.claim_job_cleanup_lease
    claims, deletes = [], []

    def interleave(*args, **kwargs):
        # Cleanup already read A's snapshot. Real handoff completes A and retargets the handle to B.
        repo.update("instances", inst["id"], {"state": "busy", "current_job_id": job["id"]}, "owner-1")
        assert repo.handoff_lease("owner-1", job["id"], next_job["id"], lease["id"], 1001.0, guard_mode="off")
        repo.update("jobs", "job-b", {"state": "running"}, "owner-1")
        repo.update("instances", inst["id"], {"state": "busy", "current_job_id": "job-b"}, "owner-1")
        result = original(*args, **kwargs)
        claims.append(result[0])
        return result

    monkeypatch.setattr(repo, "claim_job_cleanup_lease", interleave)
    provider = SimpleNamespace(inspect=lambda _: {"ready": True}, delete=lambda h: deletes.append(h))
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1000.0)
    scheduler._cleanup(job)
    assert claims == ["stale"]
    assert deletes == []
    assert repo.get("jobs", "job-b", "owner-1")["state"] == "running"
    assert repo.get("leases", lease["id"], "owner-1")["state"] == "active"
    assert repo.claim_orphan_lease("owner-1", lease["id"], inst["id"], "orphan", 1002.0) is None
    # Even with the job phase forged back, lease version/ownership and current instance binding fence A.
    repo.update("jobs", job["id"], {"state": "cleaning_up"}, "owner-1")
    assert original("owner-1", job["id"], lease["id"], "old", 1002.0,
                    expected_lease_version=lease["version"])[0] == "stale"


@pytest.mark.parametrize("headers,payload", [
    ({"X-Next-Page": "2"}, {}), ({"X-Previous-Cursor": "0"}, {}),
    ({"X-Previous-Page": "1"}, {}), ({"X-Next-Cursor": "   "}, {}),
    ({"Link": '<https://colab.test?page=1>; rel="prev"'}, {}),
    ({}, {"nextPageToken": "   "}), ({}, {"previousCursor": "0"}),
])
def test_v5_p0_2a_partial_inventory_never_releases_quota(headers, payload):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, state="failed")
    repo.update("instances", inst["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    lease = repo.update("leases", lease["id"], {"state": "cleanup_pending", "tunnel_endpoint": "live"}, "owner-1")
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/tun/m/assignments"
        return httpx.Response(200, text=_tunnel_body({"assignments": [], **payload}), headers=headers)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleColabProvider(repo, "owner-1", client=client, google=_tunnel_google())
        assert provider.inspect(lease["handle"])["remote_verified"] is False
        scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1001.0)
        assert scheduler._cleanup_orphan_lease(lease) is False
    assert all(method == "GET" for method, _ in requests)
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "deleted"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(scheduler.instance_service)[0] >= 3.0
    assert repo.get("jobs", job["id"], "owner-1")["state"] == "failed"


@pytest.mark.parametrize("status", [206, 404])
def test_v5_p0_2a_rest_partial_or_missing_list_is_not_complete_empty(status):
    _, repo = _make_repo()
    job, lease, _ = _v5_runtime(repo)
    google = SimpleNamespace(access_token=lambda _: "tok", _credential=lambda *a: {})
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status, json={}))) as client:
        provider = GoogleColabProvider(repo, "owner-1", client=client, google=google)
        with pytest.raises(ColabError):
            provider.list_runtimes()
        if status == 404:
            # Only single-resource GET retains its legitimate absence semantics.
            assert provider.inspect(lease["handle"])["remote_verified"] is True
    assert repo.get("jobs", job["id"], "owner-1")["state"] == "allocating"


@pytest.mark.parametrize("entry", ["commit", "recover"])
@pytest.mark.parametrize("mismatch", ["token", "attempt", "instance", "all"])
def test_v5_p0_2b_late_success_isolates_without_touching_pending_attempt(entry, mismatch):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"create_attempts": 2, "create_rpc_attempt": 2,
        "create_rpc_token": "token-2", "create_rpc_instance_id": inst["id"], "create_rpc_in_flight": True}, "owner-1")
    token, attempt, instance_id = "token-2", 2, inst["id"]
    if mismatch == "token":
        token = "token-1"
    elif mismatch == "attempt":
        attempt = 1
    elif mismatch == "instance":
        instance_id = "old-instance"
    else:
        token, attempt = "token-1", 1
    before = _v5_rows(repo)
    budget_before = _budget_totals(InstanceService(repo))[0]
    handle = {"runtime_name": "runtimes/late", "runtime_id": "late", "owner_id": "owner-1", "job_id": job["id"]}
    if entry == "commit":
        status, _, _ = repo.commit_create_handle_success(job["id"], "owner-1", lease["id"], handle=handle,
            rpc_token=token, expected_attempt=attempt, expected_instance_id=instance_id, started_at=1001.0)
    else:
        status, _, _ = repo.recover_late_create_handle(job["id"], "owner-1", lease["id"], handle=handle,
            rpc_token=token, expected_attempt=attempt, instance_id=instance_id, now=1001.0)
    assert status == "stale"
    for kind, row in (("jobs", job), ("leases", lease), ("instances", inst)):
        assert repo.get(kind, row["id"], "owner-1") == next(r for r in before[kind] if r["id"] == row["id"])
    isolated = [r for r in repo.list("leases", "owner-1") if r["id"] != lease["id"]]
    assert len(isolated) == 1 and isolated[0]["state"] == "quarantined"
    assert isolated[0]["handle"] == handle
    assert _budget_totals(InstanceService(repo))[0] > budget_before
    scheduler = Scheduler(repo, clock=lambda: 1002.0)
    scheduler._handle_fenced_alloc_result(job, lease)
    assert repo.get("leases", lease["id"], "owner-1")["create_rpc_in_flight"] is True


@pytest.mark.parametrize("failure", ["capacity", "unavailable", "network"])
@pytest.mark.parametrize("rebind", ["new-lease", "new-attempt"])
def test_v5_p0_2b_old_create_failure_never_cleans_new_running_binding(failure, rebind):
    from app.gpu.colab import ColabAllocationError
    from app.gpu.compute import ProviderUnavailable

    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"create_attempts": 1, "create_rpc_attempt": 1,
        "create_rpc_token": "old-token", "create_rpc_instance_id": inst["id"], "create_rpc_in_flight": True}, "owner-1")
    captured = {}

    def create(*args):
        if rebind == "new-lease":
            new_job, new_lease, new_inst = _v5_runtime(repo, suffix="new", state="running")
        else:
            new_job, new_inst = job, inst
            new_lease = repo.update("leases", lease["id"], {"create_attempts": 2, "create_rpc_attempt": 2,
                "create_rpc_token": "token-2", "create_rpc_in_flight": True}, "owner-1")
        captured.update(job=new_job, lease=new_lease, instance=new_inst, before=_v5_rows(repo))
        if failure == "capacity":
            raise ColabAllocationError(8, "NO_RUNTIMES")
        if failure == "unavailable":
            raise ProviderUnavailable("not available")
        raise OSError("lost response")

    deletes = []
    provider = SimpleNamespace(create=create, delete=lambda h: deletes.append(h))
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1001.0)
    out = scheduler._create_runtime_on_lease(job, lease, provider, MANIFEST_MODEL["compute_spec"])
    assert out["state"] == captured["job"]["state"] and out["lease_id"] == captured["lease"]["id"]
    assert _v5_rows(repo) == captured["before"]
    assert deletes == []
    assert scheduler._handle_fenced_alloc_result(job, lease)["state"] == captured["job"]["state"]
    assert _v5_rows(repo) == captured["before"]


@pytest.mark.parametrize("elapsed", [301, 3600, 86400])
@pytest.mark.parametrize("present", [True, False])
def test_v5_p0_2b_restart_reconciles_original_inflight_token_read_only(elapsed, present):
    _, repo = _make_repo()
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"create_attempts": 1, "create_rpc_attempt": 1,
        "create_rpc_token": "persisted-token", "create_rpc_instance_id": inst["id"],
        "create_rpc_started_at": 1000.0, "create_rpc_in_flight": True}, "owner-1")
    probes = []

    def inspect(handle):
        probes.append(handle)
        return {"ready": present, "deleted": not present, "remote_verified": True}

    provider = SimpleNamespace(inspect=inspect)
    job = repo.update("jobs", job["id"], {"deadline_epoch": 1100.0}, "owner-1")
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1000.0 + elapsed)
    out = scheduler._advance(job)
    assert len(probes) == 1
    fresh = repo.get("leases", lease["id"], "owner-1")
    assert fresh["create_rpc_token"] == "persisted-token"
    if present:
        assert out["state"] == "preparing" and fresh["state"] == "active"
        assert fresh["create_rpc_in_flight"] is False
    else:
        assert out["state"] == "allocating" and fresh == lease
        assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"


@pytest.mark.parametrize("field", ["total_duration_seconds", "total_duration_minutes"])
def test_v5_p0_3_production_segment_duration_prevents_local_retirement(field):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, state="failed", with_handle=False)
    service = InstanceService(repo, clock=lambda: 1060.0)
    segment = service.record_usage("owner-1", {"instance_id": inst["id"], "job_id": job["id"],
        "allocation_requested_at": 1000.0, "deletion_confirmed_at": 1060.0, "actual_units": None})
    assert segment[field] > 0
    from app.gpu.identity import row_has_positive_usage
    assert row_has_positive_usage({field: segment[field]})
    scheduler = Scheduler(repo, clock=lambda: 1060.0, instance_service=service)
    assert scheduler._lease_history_flags(lease, inst)["pure_placeholder"] is False
    assert repo.retire_pure_orphan_placeholder_cas("owner-1", lease["id"], instance_id=inst["id"], now=1060.0)[0] == "not_pure"
    with pytest.raises(InstanceConflict):
        service.retire_pure_placeholder("owner-1", inst["id"], lease_id=lease["id"])
    assert _budget_totals(service)[0] >= 3.0


@pytest.mark.parametrize("orphan", [False, True])
def test_v5_p0_3_usage_committed_after_history_snapshot_is_rechecked(monkeypatch, orphan):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, state="failed" if orphan else "cleaning_up", with_handle=False)
    if orphan:
        repo.update("instances", inst["id"], {"state": "releasing", "current_job_id": None}, "owner-1")
    scheduler = Scheduler(repo, clock=lambda: 1001.0)
    original = scheduler._lease_history_flags
    injected = []

    def interleave(*args, **kwargs):
        flags = original(*args, **kwargs)
        if flags["pure_placeholder"] and not injected:
            injected.append(repo.create("instance_usage_segments", "owner-1", {
                "instance_id": inst["id"], "job_id": job["id"], "actual_units": 1.0}))
        return flags

    monkeypatch.setattr(scheduler, "_lease_history_flags", interleave)
    if orphan:
        scheduler._cleanup_orphan_lease(lease)
    else:
        scheduler._cleanup(job)
    assert injected
    assert repo.get("leases", lease["id"], "owner-1")["state"] != "deleted"
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"
    assert _budget_totals(scheduler.instance_service)[0] >= 4.0


def test_v5_p0_4_old_absence_during_job_rebind_cannot_delete_new_runtime():
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"state": "active", "ever_active": True,
        "tunnel_endpoint": "old", "tunnel_nbh": "old-nbh", "create_attempts": 1}, "owner-1")
    calls, captured = [], {}

    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.url.path == "/tun/m/assignments" and request.method == "GET"
        _v5_runtime(repo, suffix="new", state="running")
        captured["rows"] = _v5_rows(repo)
        return httpx.Response(200, text=_tunnel_body({"assignments": []}))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleColabProvider(repo, "owner-1", client=client, google=_tunnel_google())
        scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1001.0)
        out = scheduler._handle_recovery(job, lease, provider)
        assert out["state"] == "running" and out["lease_id"] == "lease-v5-new"
        # A second tick must poll the successor, never clean it using the old absence proof.
        for kind, row_id in (("jobs", job["id"]), ("leases", "lease-v5-new"), ("instances", "inst-v5-new")):
            assert repo.get(kind, row_id, "owner-1") == next(r for r in captured["rows"][kind] if r["id"] == row_id)
        assert repo.get("leases", lease["id"], "owner-1")["state"] == "deleted"
    assert calls == [("GET", "/tun/m/assignments")]
    assert repo.get("instances", inst["id"], "owner-1")["state"] != "deleted"


def test_v5_p1_3b_cli_proxy_probe_is_read_only_and_recovery_uses_one_cas():
    _, repo = _make_repo()
    job, lease, _ = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"state": "unknown", "create_attempts": 1, "tunnel_endpoint": "live"}, "owner-1")
    payload = {"assignments": [{"endpoint": "live", "runtimeProxyInfo": {"url": "https://proxy.test/"}}]}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=_tunnel_body(payload)))) as client:
        provider = GoogleColabProvider(repo, "owner-1", client=client, google=_tunnel_google())
        scheduler = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1001.0)
        handle = scheduler._intent_handle(job, lease)
        for _ in range(3):
            assert provider.inspect(handle)["ready"] is True
            assert repo.get("leases", lease["id"], "owner-1") == lease
        assert scheduler._handle_recovery(job, lease, provider)["state"] == "preparing"
    recovered = repo.get("leases", lease["id"], "owner-1")
    assert recovered["version"] == lease["version"] + 1
    assert recovered["proxy_url"] == "https://proxy.test"
    assert recovered["handle"]["runtime_id"] == lease["runtime_id"]


@pytest.mark.parametrize("evidence", ["wrong-worker", "empty-marker", "lock-only"])
def test_v5_p1_4_unknown_worker_evidence_persists_dispatch_backoff(tmp_path, evidence):
    from app.gpu.mcp import build_bootstrap_code

    _, repo = _make_repo()
    job, lease, _ = _v5_prepared(repo)
    execution_id = job["execution_id"]
    root = tmp_path / "worker"
    directory = root / execution_id
    directory.mkdir(parents=True)
    if evidence == "lock-only":
        (directory / ".worker.lock").touch()
        # Execute the real bootstrap against an existing lock. It must not start a second process.
        code = build_bootstrap_code(execution_id, {}, worker_source="raise AssertionError('must not launch')")
        namespace = {}
        with contextlib.redirect_stdout(__import__("io").StringIO()):
            exec(compile(code.replace('"/content/ai-studio"', repr(str(root))), "v5-bootstrap", "exec"), namespace)
        assert namespace["_result"]["started"] is False
        assert not (directory / "marker.json").exists()
    else:
        (directory / "marker.json").write_text("{}")
        status = {"state": "running", "stage": "sampling"}
        if evidence == "wrong-worker":
            status["worker_reported_execution_id"] = "wrong-worker"
        (directory / "status.json").write_text(json.dumps(status))
    executor = _v5_local_executor(repo, root)
    assert executor.poll(lease["handle"], job)["state"] == "unknown"
    scheduler = Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: 1001.0)
    out = scheduler._reconcile_dispatching_load(job, lease, execution_id)
    assert out["state"] == "loading"
    assert out["dispatch_poll_attempts"] == 1 and 1001.0 < out["dispatch_next_epoch"] <= 1016.0
    assert out["dispatch_recovery_deadline_at"] == 1030.0
    # Foreign worker sampling can never mutate an already-running job.
    _, running = repo.commit_running(job["id"], "owner-1", lease_id=lease["id"], execution_id=execution_id,
        now=1002.0, require_instance=True)
    before = _v5_rows(repo)
    scheduler._run(running)
    assert _v5_rows(repo) == before


def test_v5_p1_4_lock_only_bootstrap_result_does_not_commit_running(tmp_path):
    _, repo = _make_repo()
    job, lease, _ = _v5_runtime(repo, state="loading")
    job = repo.prepare_execution(job["id"], "owner-1", lease["id"], 1000.0)
    root = tmp_path / "worker"
    directory = root / job["execution_id"]
    directory.mkdir(parents=True)
    (directory / ".worker.lock").touch()
    executor = _v5_local_executor(repo, root)
    executor._payload = lambda *args: {}
    # Both start() and poll() execute their actual trusted Python snippets.
    out = Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: 1001.0)._load(job)
    assert out["state"] == "loading" and out["dispatch_poll_attempts"] == 1
    assert out["dispatch_recovery_deadline_at"] == 1031.0


def test_v5_p1_5_bare_colab_name_rejected_at_prepare_dispatch_and_load():
    _, repo = _make_repo()
    job, lease, _ = _v5_prepared(repo)
    bad = {**lease["handle"], "runtime_name": lease["runtime_id"]}
    assert not is_valid_runtime_handle(bad, provider="colab")
    assert not is_valid_runtime_handle({**bad, "mock": True}, provider="colab")
    assert is_valid_runtime_handle(bad, provider="mock")
    repo.update("leases", lease["id"], {"handle": bad}, "owner-1")
    with pytest.raises(ValueError):
        repo.prepare_execution(job["id"], "owner-1", lease["id"], 1001.0)
    assert repo.claim_execution_dispatch(job["id"], "owner-1", lease_id=lease["id"],
        execution_id=job["execution_id"], now=1001.0, require_instance=True)[0] == "fenced"
    executor = MockGPUExecutor()
    Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: 1001.0)._load(job)
    assert executor.start_calls == executor.poll_calls == 0
    # An active Colab lease carrying an invalid bare runtime_name without endpoint must not block quarantine.
    q_status, _, q_lease = repo.quarantine_lease_cas(
        "owner-1", lease["id"], job_id=job["id"], instance_id=lease["instance_id"], now=1002.0
    )
    assert q_status == "ok" and q_lease["state"] == "quarantined"


@pytest.mark.parametrize("entry", ["diagnostics", "progress", "running", "terminal", "clear"])
@pytest.mark.parametrize("generation", [None, 0, -1, True, 1.0, "1", "all-missing", "wrong-job", "wrong-lease", "wrong-instance", "missing-execution", "missing-instance-id"])
def test_v5_p1_6_five_writers_reject_invalid_execution_identity_without_writes(entry, generation):
    _, repo = _make_repo()
    job, lease, inst = _v5_prepared(repo)
    execution_id = job["execution_id"]
    if generation == "missing-execution":
        with repo.store.db_lock(), repo.store.Session.begin() as session:
            session.delete(session.get(repo._model("executions"), execution_id))
        patch = None
    elif generation == "missing-instance-id":
        for kind, row in (("jobs", job), ("leases", lease)):
            repo.update(kind, row["id"], {"instance_id": None}, "owner-1")
        patch = {"instance_id": None}
    elif generation == "all-missing":
        for kind, row in (("jobs", job), ("leases", lease), ("instances", inst)):
            repo.update(kind, row["id"], {"generation": None, "active_execution": None}, "owner-1")
        patch = {"generation": None}
    elif isinstance(generation, str) and generation.startswith("wrong-"):
        key = {"wrong-job": "job_id", "wrong-lease": "lease_id", "wrong-instance": "instance_id"}[generation]
        patch = {key: "foreign"}
    else:
        patch = {"generation": generation}
    if patch is not None:
        repo.update("executions", execution_id, patch, "owner-1")
    if entry != "running":
        job = repo.update("jobs", job["id"], {"state": "running"}, "owner-1")
    before = _v5_rows(repo)
    if entry == "diagnostics":
        repo.record_gpu_diagnostics(job, {"devices": [{"name": "NVIDIA L4", "memory_mib": 24000}]})
    elif entry == "progress":
        assert repo.commit_run_progress(job["id"], "owner-1", execution_id=execution_id,
            patch={"stage": "late"}, now=1002.0)[0] == "stale"
    elif entry == "running":
        assert repo.commit_running(job["id"], "owner-1", lease_id=lease["id"],
            execution_id=execution_id, now=1002.0, require_instance=True)[0] == "stale"
    elif entry == "terminal":
        repo.commit_execution_terminal(job["id"], "owner-1", execution_id=execution_id,
            terminal_state="uploading", message="late", patch=None, now=1002.0)
    else:
        repo.clear_owned_execution_and_warm_proof("owner-1", job["id"], execution_id)
    assert _v5_rows(repo) == before


def test_v5_p1_7_nonlegacy_retention_requires_existing_instance_even_when_optional():
    _, repo = _make_repo()
    job, lease, inst = _v5_prepared(repo)
    repo.update("jobs", job["id"], {"state": "cleaning_up", "instance_id": "missing"}, "owner-1")
    lease = repo.update("leases", lease["id"], {"instance_id": "missing"}, "owner-1")
    repo.update("executions", job["execution_id"], {"instance_id": "missing"}, "owner-1")
    # Complete execution tuple and generation match; only the actual instance row is missing.
    before = _v5_rows(repo)
    status, _, _ = repo.finalize_retained_cleanup(job["id"], "owner-1", lease_id=lease["id"],
        instance_id="missing", legacy_mock_idle=False, require_instance=False, now=1001.0)
    assert status == "aborted" and _v5_rows(repo) == before
    assert repo.get("instances", inst["id"], "owner-1") is not None


@pytest.mark.parametrize("worker_id", [None, "matching"])
def test_v5_p1_8_launcher_marker_never_forges_worker_warm_proof(tmp_path, worker_id):
    _, repo = _make_repo()
    job, lease, inst = _v5_prepared(repo)
    execution_id = job["execution_id"]
    _, job = repo.commit_running(job["id"], "owner-1", lease_id=lease["id"], execution_id=execution_id,
        require_instance=True, now=1001.0)
    root = tmp_path / "worker"
    directory = root / execution_id
    directory.mkdir(parents=True)
    (directory / "marker.json").write_text(json.dumps({"execution_id": execution_id, "state": "launched"}))
    fingerprint = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    status = {"state": "completed", "warm_saved": True, "warm_epoch_id": "epoch-v5",
              "warm_fingerprint": fingerprint, "runtime_meta": {"python_mm": "3.12"}}
    if worker_id:
        status["worker_reported_execution_id"] = execution_id
    (directory / "status.json").write_text(json.dumps(status))
    (directory / "out.png").write_bytes(MockGPUExecutor().output)
    (directory / "meta.json").write_text(json.dumps({"outputs": [{"path": "out.png", "filename": "out.png", "mime": "image/png"}]}))
    executor = _v5_local_executor(repo, root)
    polled = executor.poll(lease["handle"], job)
    assert polled["launcher_execution_id"] == execution_id
    assert polled["worker_reported_execution_id"] == (execution_id if worker_id else None)
    out = Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: 1002.0)._run(job)
    assert out["state"] == "uploading"
    current_inst = repo.get("instances", inst["id"], "owner-1")
    current_lease = repo.get("leases", lease["id"], "owner-1")
    executions = {row["id"]: row for row in repo.list("executions", "owner-1")}
    next_job = _calibrated_colab_job(job_id="next")
    warm = is_warm_verified_for_job(next_job, instance_data=current_inst, lease_data=current_lease, executions_by_id=executions)
    allowed, assessment = check_job_reuse_guard(next_job, guard_mode="enforce", now=1002.0,
        instance_data=current_inst, lease_data=current_lease, executions_by_id=executions, deadlines=[1453.0])
    assert warm is bool(worker_id) and allowed is bool(worker_id)
    assert assessment.need_seconds == (450.0 if worker_id else 780.0)
    if not worker_id:
        assert current_inst.get("warm_proof") is None and current_lease.get("warm_proof") is None


@pytest.mark.parametrize("tunnel", [False, True])
def test_v5_p0_2b_real_adapter_late_success_cannot_overwrite_new_attempt(tunnel):
    _, repo = _make_repo()
    _setup_account(repo, provider="colab")
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"create_attempts": 1, "create_rpc_attempt": 1,
        "create_rpc_token": "token-1", "create_rpc_instance_id": inst["id"],
        "create_rpc_in_flight": True}, "owner-1")
    captured, calls = {}, []

    def handler(request):
        calls.append((request.method, request.url.path))
        repo.update("leases", lease["id"], {"create_attempts": 2, "create_rpc_attempt": 2,
            "create_rpc_token": "token-2", "operation_name": "operations/current",
            "tunnel_endpoint": "current" if tunnel else None}, "owner-1")
        captured["before"] = _v5_rows(repo)
        if tunnel:
            return httpx.Response(200, text=_tunnel_body({"endpoint": "late-endpoint", "runtimeProxyInfo": {
                "url": "https://late.test/", "token": "late-proxy-token"}}))
        return httpx.Response(200, json={"name": "operations/late", "done": True})

    google = _tunnel_google() if tunnel else SimpleNamespace(access_token=lambda _: "tok", _credential=lambda *a: {})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleColabProvider(repo, "owner-1", client=client, google=google)
        out = Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1001.0)._create_runtime_on_lease(
            job, lease, provider, MANIFEST_MODEL["compute_spec"])
    assert out["state"] == "allocating"
    for kind, row in (("jobs", job), ("leases", lease), ("instances", inst)):
        assert repo.get(kind, row["id"], "owner-1") == next(r for r in captured["before"][kind] if r["id"] == row["id"])
    isolated = [r for r in repo.list("leases", "owner-1") if r["id"] != lease["id"]]
    assert len(isolated) == 1 and isolated[0]["state"] == "quarantined"
    assert isolated[0]["handle"]["operation_name"].startswith("operations/tunnel-" if tunnel else "operations/late")
    if tunnel:
        assert isolated[0]["handle"]["tunnel_endpoint"] == "late-endpoint"
        assert isolated[0]["handle"]["connection_metadata"]["proxy_url"] == "https://late.test"
    assert _budget_totals(InstanceService(repo))[0] > 3.0
    assert len(calls) == 1 and not any("unassign" in path or method == "DELETE" for method, path in calls)


def test_v5_p0_1_expired_idle_claim_cannot_delete_concurrent_attachment(monkeypatch):
    _, repo = _make_repo()
    job, lease, inst = _v5_runtime(repo, state="completed")
    lease = repo.update("leases", lease["id"], {"state": "idle", "current_job_id": None,
        "idle_deadline_at": 1002.0}, "owner-1")
    repo.update("instances", inst["id"], {"state": "retained_idle", "current_job_id": None,
        "idle_deadline_at": 1002.0}, "owner-1")
    repo.create("jobs", "owner-1", _calibrated_colab_job(job_id="job-b", state="waiting_for_capacity"), record_id="job-b")
    original = repo.claim_orphan_lease
    captured, deletes = {}, []

    def interleave(*args, **kwargs):
        assert repo.attach_idle_lease("owner-1", "job-b", lease["id"], 1001.0, guard_mode="off")
        repo.update("jobs", "job-b", {"state": "running"}, "owner-1")
        captured["before"] = _v5_rows(repo)
        result = original(*args, **kwargs)
        assert result is None
        return result

    monkeypatch.setattr(repo, "claim_orphan_lease", interleave)
    provider = SimpleNamespace(delete=lambda h: deletes.append(h))
    Scheduler(repo, provider_factory=lambda _: provider, clock=lambda: 1003.0)._cleanup_orphan_lease(lease)
    assert captured and _v5_rows(repo) == captured["before"] and deletes == []
    assert repo.get("jobs", job["id"], "owner-1")["state"] == "completed"


def test_v5_p0_2b_inflight_probe_cannot_clear_token_replaced_during_get():
    _, repo = _make_repo()
    job, lease, inst = _v5_runtime(repo, with_handle=False)
    lease = repo.update("leases", lease["id"], {"create_attempts": 1, "create_rpc_attempt": 1,
        "create_rpc_token": "token-1", "create_rpc_instance_id": inst["id"],
        "create_rpc_in_flight": True}, "owner-1")
    captured = {}

    def inspect(handle):
        repo.update("leases", lease["id"], {"create_attempts": 2, "create_rpc_attempt": 2,
            "create_rpc_token": "token-2"}, "owner-1")
        captured["before"] = _v5_rows(repo)
        return {"ready": True, "remote_verified": True}

    scheduler = Scheduler(repo, provider_factory=lambda _: SimpleNamespace(inspect=inspect), clock=lambda: 1001.0)
    assert scheduler._advance(job)["state"] == "allocating"
    assert _v5_rows(repo) == captured["before"]


def test_v5_p1_4_live_launcher_without_worker_id_cannot_update_running_sampling(tmp_path):
    _, repo = _make_repo()
    job, lease, _ = _v5_prepared(repo)
    _, job = repo.commit_running(job["id"], "owner-1", lease_id=lease["id"],
        execution_id=job["execution_id"], require_instance=True, now=1001.0)
    root = tmp_path / "worker"
    directory = root / job["execution_id"]
    directory.mkdir(parents=True)
    (directory / "marker.json").write_text(json.dumps({"execution_id": job["execution_id"],
        "state": "launched", "pid": os.getpid()}))
    (directory / "status.json").write_text(json.dumps({"state": "running", "stage": "sampling"}))
    executor = _v5_local_executor(repo, root)
    result = executor.poll(lease["handle"], job)
    assert result["state"] == "running" and result["worker_process_alive"] is True
    assert result["worker_reported_execution_id"] is None
    before = _v5_rows(repo)
    Scheduler(repo, executor_factory=lambda _: executor, clock=lambda: 1002.0)._run(job)
    assert _v5_rows(repo) == before
