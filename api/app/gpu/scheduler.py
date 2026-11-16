"""Persistent short-tick scheduler for independent Colab GPU Studio API."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .colab import ColabAllocationError
from .compute import (
    ComputeError,
    DisabledProvider,
    GpuConfig,
    ProviderUnavailable,
    default_executor_factory,
    default_provider_factory,
)
from .estimator import check_job_reuse_guard, min_deadlines
from .google import GoogleError
from .identity import row_has_positive_usage
from .instances import InstanceServiceError
from .mcp import MCPError
from .newgpu.executor_plugins import TrustedExecutorAdapter
from .prep import PREP_SPEC_KEY, cache_key, create_prep_job, prep_idempotency_key, prep_jobs_for
from .repository import (
    extract_runtime_id,
    extract_runtime_name_id,
    extract_tunnel_endpoint,
    is_valid_runtime_handle,
    lease_has_active_origin,
    normalize_tunnel_endpoint,
)

TERMINAL_STATES = {"completed", "failed", "cancelled"}
ACTIVE_STATES = {
    "queued",
    "validating",
    "waiting_for_capacity",
    "allocating",
    "preparing",
    "loading",
    "running",
    "uploading",
    "cleaning_up",
    "cleanup_pending",
}
MAX_OUTPUT_BYTES = 100 * 1024 * 1024


# Stage markers written by runtime_worker.py into status.json while a job is running.
_STAGE_TEXT = {
    "restore_env": "正在恢复运行环境",
    "restore_weights": "正在恢复模型文件",
    "warm_reuse": "同一台机器已有模型，直接复用",
    "comfy_start": "正在启动 ComfyUI",
    "sampling": "采样中",
    "prep_env": "正在构建环境缓存",
    "prep_weights": "正在补齐模型缓存",
}


_PREP_TEXT = {
    "GPU job validating": "正在校验缓存准备任务",
    "GPU job validated": "缓存准备任务已校验",
    "GPU capacity available": "CPU 运行时可用",
    "GPU runtime allocated": "CPU 运行时已分配",
    "GPU runtime is ready": "CPU 运行时已就绪",
    "GPU execution started": "开始构建缓存",
    "GPU execution completed": "缓存构建完成",
    "GPU outputs persisted": "缓存已保存到 Drive",
    "GPU cleanup completed": "CPU 运行时已关闭（缓存保留在 Drive）",
    "GPU cleanup is pending": "正在关闭 CPU 运行时",
}


class SchedulerError(RuntimeError):
    pass


class Scheduler:
    """Advance one persisted phase per tick; never hold the lock over adapter I/O."""

    def __init__(
        self,
        store,
        provider_factory: Callable | None = None,
        executor_factory: Callable | None = None,
        storage_factory: Callable | None = None,
        config: GpuConfig | None = None,
        clock: Callable[[], float] | None = None,
        instance_service: Any | None = None,
        instance_service_factory: Callable | None = None,
    ):
        # Accept either the base Store or the GPU Repository wrapped around it.
        from .instances import InstanceService
        from .repository import Repository

        self.store = store if isinstance(store, Repository) else Repository(store)
        self.db_store = self.store.store
        self.config = config or GpuConfig.from_env()
        self.clock = clock or time.time
        if instance_service is not None:
            self.instance_service = instance_service
        elif instance_service_factory is not None:
            self.instance_service = self._factory(instance_service_factory, {})
        else:
            self.instance_service = InstanceService(self.store, self.clock, self.config)
        self.instances = self.instance_service
        self.provider_factory = provider_factory or (
            lambda job: default_provider_factory(
                job, self.config, store=self.store, owner_id=job.get("owner_id")
            )
        )
        self.executor_factory = executor_factory or (
            lambda job: default_executor_factory(
                job, self.config, store=self.store, owner_id=job.get("owner_id")
            )
        )
        self.storage_factory = storage_factory

    def _instance_call(self, method: str, *args, **kwargs):
        service = self.instance_service
        if service is None:
            return None
        # Callers may replace ``clock`` after construction; deadlines must share it.
        service.clock = self.clock
        return getattr(service, method)(*args, **kwargs)

    @staticmethod
    def _accepts_kwarg(fn: Any, name: str) -> bool:
        if not callable(fn):
            return False
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        if name in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _admission_lock(self, timeout: float = 1.0):
        lock = getattr(self.db_store, "admission_lock", None)
        if callable(lock):
            return lock(timeout=timeout)
        return nullcontext(True)

    def _clear_owned_execution(self, job: dict) -> None:
        if job.get("execution_id") and job.get("owner_id") and hasattr(self.store, "clear_owned_execution_and_warm_proof"):
            self.store.clear_owned_execution_and_warm_proof(job["owner_id"], job["id"], job["execution_id"])

    @staticmethod
    def _instance_id(instance: dict | None) -> str | None:
        if not isinstance(instance, dict):
            return None
        value = instance.get("instance_id") or instance.get("id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _instance_epoch(value: Any) -> float | None:
        return None if value is None else float(value)

    def _instance_patch(self, instance: dict | None) -> dict:
        if not isinstance(instance, dict):
            return {}
        patch = {}
        instance_id = self._instance_id(instance)
        if instance_id:
            patch["instance_id"] = instance_id
        for key in (
            "account_id",
            "drive_account_id",
            "account_alias_snapshot",
            "instance_label",
            "gpu_reported",
            "retention_mode",
            "retained_until",
            "idle_deadline_at",
            "hard_deadline_at",
            "selection_reason",
        ):
            if key in instance and instance[key] is not None:
                patch[key] = instance[key]
        if instance.get("state") is not None:
            patch["instance_state"] = instance["state"]
        hard_deadline = self._instance_epoch(instance.get("hard_deadline_at"))
        if hard_deadline is not None and hard_deadline > self.clock():
            patch["deadline_epoch"] = hard_deadline
        return patch

    def _merge_instance(self, job: dict, instance: dict | None) -> dict:
        patch = self._instance_patch(instance)
        if not patch:
            return job
        return self._update("jobs", job["id"], patch, job.get("owner_id"))

    def _service_instances(self) -> list[dict]:
        rows = self._list("instances")
        if rows:
            return rows
        if self.instance_service is None:
            return []
        from .instances import InstanceService

        if isinstance(self.instance_service, InstanceService):
            return []
        owners = {job.get("owner_id") for job in self._list("jobs") if job.get("owner_id")}
        return [item for owner in owners for item in self.instance_service.list(owner)]

    def _service_instance_for_job(self, job: dict) -> dict | None:
        if getattr(self, "store", None) is None:
            return None
        instance_id = job.get("instance_id")
        if not instance_id:
            return None
        row = self._get("instances", instance_id, job.get("owner_id"))
        if row is not None:
            return row
        svc = getattr(self, "instance_service", None)
        if svc is None:
            return None
        from .instances import InstanceService

        if isinstance(svc, InstanceService):
            return None
        return svc.get(job.get("owner_id"), instance_id)

    def _lifecycle(self, job: dict) -> str:
        policy = job.get("execution_policy") or {}
        lifecycle = policy.get("lifecycle") or policy.get("mode")
        if lifecycle in {"save", "batch", "smart", "retain"}:
            return lifecycle
        if policy.get("retention") in {True, "retain"} or job.get("retention_requested"):
            return "retain"
        return "save"

    def _instance_job(self, job: dict) -> dict:
        """Give InstanceService the legacy batch mode as an explicit lifecycle.

        Batch jobs created before v0.2 store ``mode=smart`` in their policy,
        while InstanceService deliberately reads ``lifecycle``.  Translate
        only the service-facing copy so the persisted legacy request remains
        unchanged and the instance service stays authoritative for retention.
        """
        lifecycle = self._lifecycle(job)
        policy = job.get("execution_policy") or {}
        if policy.get("lifecycle") == lifecycle:
            return job
        return {**job, "execution_policy": {**policy, "lifecycle": lifecycle}}

    def _is_v020_job(self, job: dict) -> bool:
        return getattr(self, "instance_service", None) is not None and not job.get("legacy_scheduler_only")

    def _has_sql_instance_service(self) -> bool:
        from .instances import InstanceService

        return isinstance(getattr(self, "instance_service", None), InstanceService)

    def _allocate_instance(self, job: dict) -> dict:
        if not self._is_v020_job(job) or job.get("instance_id"):
            return job
        fn = getattr(self.instance_service, "allocate", None)
        kwargs = {"atomic_lease": True} if self._accepts_kwarg(fn, "atomic_lease") else {}
        try:
            instance = self._instance_call("allocate", self._instance_job(job), **kwargs)
        except (InstanceServiceError, ValueError, KeyError) as exc:
            return self._begin_failure(job, "gpu_instance_allocation_failed", self._safe_error(exc))
        if instance is None:
            return job
        latest = self._get("jobs", job["id"], job.get("owner_id")) or job
        # Retarget a legacy idle lease before merging the instance metadata so
        # the lease handoff and job mirror cannot briefly disagree.
        latest = self._reuse_instance_lease(latest, instance)
        return self._merge_instance(latest, instance)

    def _guard_mode(self) -> str:
        cfg = getattr(self, "config", None)
        raw = cfg.get("reuse_lifetime_guard_mode") if isinstance(cfg, dict) else getattr(cfg, "reuse_lifetime_guard_mode", None)
        if raw is None or not str(raw).strip():
            return "enforce"
        return str(raw).strip().lower()

    def _reuse_instance_lease(self, job: dict, instance: dict | None) -> dict:
        """Retarget a retained lease under the same admission fence as claim/allocate."""
        with self._admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return job
            return self._reuse_instance_lease_locked(job, instance)

    def _reuse_instance_lease_locked(self, job: dict, instance: dict | None) -> dict:
        """Retarget the retained instance's lease instead of creating a second runtime."""
        from .repository import extract_runtime_id

        if not self._is_v020_job(job) or not isinstance(instance, dict):
            return job
        latest_job = self._get("jobs", job["id"], job.get("owner_id")) or job
        lease_id = instance.get("lease_id") or latest_job.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            return latest_job
        lease = self._get("leases", lease_id, latest_job.get("owner_id"))
        if not lease:
            return latest_job
        handle = lease.get("handle")
        if not isinstance(handle, dict) or not handle:
            return latest_job
        lease_rt = extract_runtime_id(lease) or extract_runtime_id(instance)
        if (
            self._conflicting_non_deleted_lease(
                latest_job,
                exclude_lease_id=lease_id,
                runtime_id=lease_rt,
            )
            is not None
        ):
            return latest_job
        effective_deadline = min_deadlines(
            lease.get("deadline_epoch"),
            lease.get("hard_deadline_at"),
            instance.get("hard_deadline_at"),
            latest_job.get("deadline_epoch"),
            latest_job.get("hard_deadline_at"),
        )
        # If allocate/_claim_existing already promoted this idle lease atomically in DB, just sync job mirror.
        if lease.get("state") == "active" and lease.get("owner_job_id") == latest_job["id"]:
            job_patch = {}
            if latest_job.get("lease_id") != lease_id:
                job_patch["lease_id"] = lease_id
            if effective_deadline is not None and latest_job.get("deadline_epoch") != effective_deadline:
                job_patch["deadline_epoch"] = effective_deadline
            return self._update("jobs", latest_job["id"], job_patch, latest_job.get("owner_id")) if job_patch else latest_job
        if lease.get("state") != "idle":
            return latest_job
        now = self.clock()
        idle_deadline = lease.get("idle_deadline_at")
        if idle_deadline is not None and now >= float(idle_deadline):
            return latest_job
        if effective_deadline is not None and now >= effective_deadline:
            return latest_job
        allowed, _ = check_job_reuse_guard(
            latest_job,
            guard_mode=self._guard_mode(),
            now=now,
            instance_data=instance,
            lease_data=lease,
            deadlines=(effective_deadline,),
            repo=self.store,
            owner_id=latest_job.get("owner_id"),
            observe_cb=lambda assessment: self._event(
                latest_job,
                "reuse_lifetime_observed",
                f"reuse window observed ({assessment.reason})",
            ),
        )
        if not allowed:
            return latest_job
        lease_patch = {
            "state": "active",
            "owner_job_id": latest_job["id"],
            "current_job_id": latest_job["id"],
            "idle_deadline_at": None,
            "last_active_at": now,
            "handle": {
                **handle,
                "job_id": latest_job["id"],
                "owner_job_id": latest_job["id"],
            },
        }
        self._update("leases", lease_id, lease_patch, latest_job.get("owner_id"))
        job_patch = {"lease_id": lease_id}
        if effective_deadline is not None:
            job_patch["deadline_epoch"] = effective_deadline
        return self._update("jobs", latest_job["id"], job_patch, latest_job.get("owner_id"))

    def _bind_instance_lease(self, job: dict, lease: dict) -> tuple[dict, dict]:
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job, lease
        already_bound = lease.get("instance_id") == job.get("instance_id") and bool(lease.get("instance_bound"))
        try:
            instance = self._instance_call("bind_lease", self._instance_job(job), lease)
        except Exception as exc:
            raise SchedulerError(f"instance lease binding failed: {self._safe_error(exc)}") from exc
        if not already_bound:
            patch = {"instance_id": job["instance_id"], "instance_bound": True}
            if job.get("account_id"):
                patch["account_id"] = job["account_id"]
            lease = self._update("leases", lease["id"], patch, job.get("owner_id"))
        if instance:
            job = self._merge_instance(job, instance)
        return job, lease

    def _mark_instance(self, method: str, job: dict, **kwargs) -> dict:
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        try:
            instance = self._instance_call(method, self._instance_job(job), **kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning("instance %s failed: %s", method, self._safe_error(exc))
            return job
        return self._merge_instance(job, instance)

    def _complete_instance(self, job: dict, success: bool) -> tuple[dict, dict | None]:
        """Let InstanceService decide whether a completed job is retained or released."""
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job, None
        existing = self._service_instance_for_job(job)
        if isinstance(existing, dict):
            if existing.get("current_job_id") not in (None, job.get("id")):
                return job, existing
            if existing.get("state") == "deleted":
                job = self._merge_instance(job, existing)
                job = self._update(
                    "jobs", job["id"], {"instance_completion_recorded": True}, job.get("owner_id")
                )
                return job, existing
        if job.get("instance_completion_recorded"):
            return job, existing
        lease = self._lease(job)
        lease_fenced = bool(
            lease is not None
            and (
                lease.get("state") in {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted"}
                or lease.get("cleanup_claim_token")
                or lease.get("manual_stop_requested")
                or lease.get("runtime_deleted")
                or lease.get("remote_deleted")
                or lease.get("owner_job_id") not in {None, job.get("id")}
                or lease.get("current_job_id") not in {None, job.get("id")}
                or not isinstance(lease.get("handle"), dict)
                or not lease.get("handle")
            )
        )
        try:
            instance = self._instance_call("complete", self._instance_job(job), success=(success and not lease_fenced))
        except Exception as exc:
            raise SchedulerError(f"instance completion failed: {self._safe_error(exc)}") from exc
        if isinstance(instance, dict) and instance.get("state") == "retained_idle":
            policy = job.get("execution_policy") or {}
            idle_seconds = policy.get("idle_seconds") if policy.get("mode") == "smart" else None
            if idle_seconds is not None:
                # Batch smart mode specifies seconds; apply it through the service's CAS path.
                instance = self._instance_call(
                    "retention",
                    job.get("owner_id"),
                    self._instance_id(instance),
                    "smart",
                    float(idle_seconds) / 60.0,
                    f"scheduler-smart:{job['id']}",
                )
        job = self._merge_instance(job, instance)
        job = self._update(
            "jobs", job["id"], {"instance_completion_recorded": True}, job.get("owner_id")
        )
        return job, instance if isinstance(instance, dict) else self._service_instance_for_job(job)

    def _record_instance_usage(self, job: dict, instance: dict | None, ended_at: float) -> None:
        """Close one idempotent instance-lifetime segment after remote deletion."""
        if not self._is_v020_job(job):
            return
        instance_id = job.get("instance_id") or self._instance_id(instance)
        if not instance_id:
            return
        instance = instance if isinstance(instance, dict) else {}
        started_at = (
            instance.get("allocation_requested_at")
            or job.get("started_epoch")
            or job.get("started_at")
        )
        confirmed_at = instance.get("deletion_confirmed_at") or ended_at
        started_epoch = self._instance_epoch(started_at)
        confirmed_epoch = self._instance_epoch(confirmed_at) or float(ended_at)
        elapsed_minutes = (
            max(0.0, confirmed_epoch - started_epoch) / 60.0
            if started_epoch is not None
            else 0.0
        )
        is_mock = self._provider_name(job) == "mock"
        rate = float(job.get("unit_rate") or self.config.unit_rate or 0.0)
        payload = {
            "instance_id": instance_id,
            "job_id": job.get("id"),
            "segment_key": f"instance-lifetime:{instance_id}",
            "allocation_requested_at": started_at,
            "deletion_confirmed_at": confirmed_at,
            "started_at": started_at,
            "ended_at": confirmed_at,
            "estimated_units": 0.0 if is_mock else elapsed_minutes * rate,
            # Provider billing is not available at deletion time; never invent CU.
            "actual_units": None,
        }
        self.instance_service.record_usage(job.get("owner_id"), payload)

    def _mark_instance_deleted(self, job: dict) -> dict:
        """Record deletion only after the provider has explicitly confirmed it."""
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        try:
            fn = getattr(self.instance_service, "mark_deleted", None)
            kwargs: dict[str, Any] = {
                "remote_deleted": True,
                "expected_version": job.get("_cleanup_instance_version"),
            }
            if self._accepts_kwarg(fn, "job_id") and not str(job.get("id") or "").startswith("orphan-instance:"):
                kwargs["job_id"] = job.get("id")
            instance = self._instance_call(
                "mark_deleted", job.get("owner_id"), job["instance_id"], **kwargs
            )
            self._record_instance_usage(job, instance, self.clock())
        except Exception as exc:
            raise SchedulerError(f"instance deletion confirmation failed: {self._safe_error(exc)}") from exc
        # Watchdog orphan cleanup has no job row to mirror; the ledger still
        # closes against the instance service and the lease can be finalized.
        if self._get("jobs", job["id"], job.get("owner_id")) is None:
            return job
        return self._merge_instance(job, instance)

    def _lease_history_flags(
        self,
        lease: dict,
        instance: dict | None = None,
        *,
        owner_id: str | None = None,
    ) -> dict[str, bool]:
        """Classify active, remote-identity, create-intent, and execution history on a lease."""
        owner = owner_id or lease.get("owner_id")
        instance_id = lease.get("instance_id") or (instance or {}).get("id")
        lease_id = lease.get("id")
        query_error = False
        executions: list[dict] = []
        jobs: list[dict] = []
        usage_segments: list[dict] = []
        usage_rows: list[dict] = []
        if owner:
            try:
                executions = self._list("executions", owner)
                jobs = self._list("jobs", owner)
                usage_segments = self._list("instance_usage_segments", owner)
                usage_rows = self._list("usage", owner)
            except Exception:
                query_error = True
                executions = []
                jobs = []
                usage_segments = []
                usage_rows = []
        assoc_job_ids: set[str] = set()
        if lease.get("owner_job_id"):
            assoc_job_ids.add(lease["owner_job_id"])
        if lease.get("current_job_id"):
            assoc_job_ids.add(lease["current_job_id"])
        if isinstance(instance, dict) and instance.get("current_job_id"):
            assoc_job_ids.add(instance["current_job_id"])
        has_job_exec_evidence = False
        has_job_intent = False
        for jr in jobs:
            if not isinstance(jr, dict):
                continue
            jr_inst = jr.get("instance_id")
            jr_lease = jr.get("lease_id")
            jr_intent = jr.get("start_intent")
            intent_lease_match = isinstance(jr_intent, dict) and lease_id and jr_intent.get("lease_id") == lease_id
            intent_inst_match = isinstance(jr_intent, dict) and instance_id and jr_intent.get("instance_id") == instance_id
            if (
                (lease_id and jr_lease == lease_id)
                or intent_lease_match
                or (instance_id and jr_inst == instance_id)
                or intent_inst_match
                or (jr.get("id") in assoc_job_ids and (not lease_id or jr_lease in {None, lease_id}))
            ):
                if jr.get("id"):
                    assoc_job_ids.add(jr["id"])
                if jr.get("execution_id") is not None or jr.get("active_execution") is not None:
                    has_job_exec_evidence = True
                if intent_lease_match or intent_inst_match:
                    has_job_intent = True
        has_execution = any(
            (lease_id and ex.get("lease_id") == lease_id)
            or (instance_id and ex.get("instance_id") == instance_id)
            or (ex.get("job_id") in assoc_job_ids)
            for ex in executions
        )
        has_positive_usage = any(
            row_has_positive_usage(row)
            and (
                (lease_id and row.get("lease_id") == lease_id)
                or (instance_id and row.get("instance_id") == instance_id)
                or (row.get("job_id") in assoc_job_ids)
            )
            for row in (*usage_segments, *usage_rows)
        )
        intent = lease.get("create_intent")
        from_active = lease_has_active_origin(lease, lease.get("state"))
        ever_active = bool(
            from_active
            or lease.get("state") in {"active", "busy"}
            or lease.get("ever_active")
            or lease.get("last_active_at") is not None
            or lease.get("active_execution") is not None
            or lease.get("warm_proof") is not None
            or has_execution
            or has_job_exec_evidence
            or has_job_intent
            or has_positive_usage
            or (
                isinstance(instance, dict)
                and (
                    instance.get("state") in {"ready", "busy", "retained_idle"}
                    or instance.get("ever_active")
                    or instance.get("runtime_created")
                    or instance.get("active_execution") is not None
                    or instance.get("warm_proof") is not None
                )
            )
        )
        has_remote_identity = bool(
            (isinstance(lease.get("handle"), dict) and lease.get("handle"))
            or lease.get("runtime_name")
            or extract_tunnel_endpoint(lease)
            or lease.get("tunnel_nbh") is not None
            or lease.get("operation_name")
            or lease.get("proxy_url")
            or lease.get("encrypted_proxy_token")
            or (
                isinstance(instance, dict)
                and (
                    instance.get("handle")
                    or instance.get("runtime_name")
                    or extract_tunnel_endpoint(instance)
                    or instance.get("tunnel_nbh") is not None
                    or instance.get("operation_name")
                )
            )
        )
        has_create_intent = bool(
            int(lease.get("create_attempts") or 0) > 0
            or lease.get("create_rpc_in_flight")
            or lease.get("create_rpc_started_at")
            or lease.get("create_started")
            or lease.get("selected_spec")
            or lease.get("create_intent_id")
            or intent is not None
        )
        pure_placeholder = (
            not query_error
            and lease.get("state") == "creating"
            and not ever_active
            and not has_remote_identity
            and not has_create_intent
        )
        return {
            "from_active": from_active,
            "ever_active": ever_active,
            "has_remote_identity": has_remote_identity,
            "has_create_intent": has_create_intent,
            "has_execution": has_execution or has_job_exec_evidence or has_job_intent or has_positive_usage,
            "query_error": query_error,
            "pure_placeholder": pure_placeholder,
        }

    def _quarantine_orphan_lease(
        self,
        owner_id: str,
        lease: dict,
        instance_id: str | None,
        *,
        from_active: bool,
        expected_version: int | None = None,
        expected_create_attempts: int | None = None,
    ) -> bool:
        eff_ver = lease.get("version") if expected_version is None else expected_version
        eff_attempts = int(lease.get("create_attempts") or 0) if expected_create_attempts is None else expected_create_attempts
        if hasattr(getattr(self, "store", None), "quarantine_lease_cas"):
            self.store.quarantine_lease_cas(
                owner_id,
                lease["id"],
                instance_id=instance_id,
                expected_lease_version=eff_ver,
                expected_create_attempts=eff_attempts,
                from_active=from_active,
                reason="gpu_runtime_identity_missing",
                message="gpu runtime identity missing",
                now=self.clock(),
            )
            return False
        with self._admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            fresh = self._get("leases", lease["id"], owner_id)
            if fresh is None or fresh.get("state") == "deleted":
                return False
            if eff_ver is not None and fresh.get("version") != eff_ver:
                return False
            if int(fresh.get("create_attempts") or 0) != int(eff_attempts):
                return False
            if fresh.get("state") in {"active", "busy"} and self._lease_handle(fresh):
                return False
            self._update(
                "leases",
                lease["id"],
                {
                    "state": "quarantined",
                    "from_active_missing_handle": bool(fresh.get("from_active_missing_handle")) or from_active,
                    "was_active_missing_handle": bool(fresh.get("was_active_missing_handle")) or from_active,
                    "cleanup_claim_token": None,
                    "cleanup_claimed_at": None,
                    "last_error": "gpu runtime identity missing",
                    "last_seen": self.clock(),
                },
                owner_id,
            )
        inst_row = self._get("instances", instance_id, owner_id) if instance_id else None
        if inst_row is None and instance_id and self.instance_service is not None and hasattr(self.instance_service, "get"):
            try:
                inst_row = self._instance_call("get", owner_id, instance_id)
            except Exception:
                inst_row = None
        owns_instance = bool(
            instance_id and (inst_row is None or inst_row.get("lease_id") in {None, lease["id"]})
        )
        if owns_instance and self.instance_service is not None and hasattr(self.instance_service, "quarantine"):
            try:
                self._instance_call(
                    "quarantine",
                    owner_id,
                    instance_id,
                    reason="gpu_runtime_identity_missing",
                )
            except Exception:
                pass
        elif owns_instance:
            if isinstance(inst_row, dict) and inst_row.get("state") not in {"deleted", "quarantined"}:
                self._update(
                    "instances",
                    instance_id,
                    {
                        "state": "quarantined",
                        "cleanup_claim_token": None,
                        "cleanup_claimed_at": None,
                        "quarantined_reason": "gpu_runtime_identity_missing",
                        "last_seen_at": self.clock(),
                    },
                    owner_id,
                )
        return False

    def _close_unstarted_instance(self, job: dict, lease: dict) -> dict:
        """任务在创建运行时之前就失败时，远端没有机器可删，实例直接收尾。

        否则实例会一直停在「释放中」，看起来像失败后还在继续跑。
        """
        if not self._is_v020_job(job) or not job.get("instance_id"):
            return job
        instance = self._service_instance_for_job(job)
        if not instance or instance.get("state") != "releasing":
            return job
        retire = getattr(self.store, "retire_pure_orphan_placeholder_cas", None)
        if retire is None:
            return job
        status, _ = retire(
            job["owner_id"], lease["id"], instance_id=instance["id"],
            cleanup_job_id=job["id"], expected_lease_version=lease.get("version"),
            expected_instance_version=instance.get("version"),
            allow_deleted_placeholder=True, now=self.clock(),
        )
        if status != "retired":
            return job
        return self._merge_instance(job, self._service_instance_for_job(job))

    @staticmethod
    def _is_from_active_lease(lease: dict | None) -> bool:
        """Return True if a lease is active or carries active/once-active origin history."""
        return lease_has_active_origin(lease, lease.get("state") if isinstance(lease, dict) else None)

    @staticmethod
    def _create_rpc_is_in_flight(lease: dict | None, now: float) -> bool:
        return bool(isinstance(lease, dict) and lease.get("create_rpc_in_flight"))

    def _conflicting_non_deleted_lease(
        self,
        job: dict,
        *,
        exclude_lease_id: str | None = None,
        runtime_id: str | None = None,
        tunnel_endpoint: str | None = None,
        handle: object = None,
        lease: dict | None = None,
    ) -> dict | None:
        """Return any other non-deleted lease/instance referencing this job's instance, runtime, or endpoint."""
        instance_id = job.get("instance_id")
        if hasattr(getattr(self, "store", None), "find_runtime_conflict_lease"):
            conflict, row = self.store.find_runtime_conflict_lease(
                instance_id=instance_id,
                runtime_id=runtime_id,
                tunnel_endpoint=tunnel_endpoint,
                handle=handle,
                lease=lease,
                job=job,
                exclude_lease_id=exclude_lease_id,
                exclude_instance_id=instance_id,
            )
            if conflict:
                return row or {"id": exclude_lease_id or "conflict", "instance_id": instance_id, "state": "conflict"}

        target_rts: set[str] = set()
        target_eps: set[str] = set()

        def _collect(obj: object) -> None:
            if not isinstance(obj, dict):
                return
            rt = extract_runtime_id(obj)
            if rt:
                target_rts.add(rt)
            rt_name_id = extract_runtime_name_id(obj.get("runtime_name"))
            if rt_name_id:
                target_rts.add(rt_name_id)
            ep = extract_tunnel_endpoint(obj)
            if ep:
                target_eps.add(ep)

        if runtime_id:
            target_rts.add(runtime_id)
        if tunnel_endpoint:
            norm_ep = normalize_tunnel_endpoint(tunnel_endpoint)
            if norm_ep:
                target_eps.add(norm_ep)
        _collect(handle if isinstance(handle, dict) else None)
        _collect(lease)
        _collect(job)

        instances_by_id = {
            row["id"]: row
            for row in self._list("instances")
            if isinstance(row, dict) and row.get("id")
        }
        if instance_id and instance_id in instances_by_id:
            _collect(instances_by_id[instance_id])
        eff_rt = runtime_id or extract_runtime_id(job)
        if eff_rt:
            target_rts.add(eff_rt)
        if not instance_id and not target_rts and not target_eps:
            return None
        for row in self._list("leases"):
            if row.get("state") == "deleted":
                continue
            if exclude_lease_id and row.get("id") == exclude_lease_id:
                continue
            row_inst = row.get("instance_id") or (row.get("data") or {}).get("instance_id")
            if instance_id and row_inst == instance_id:
                return row
            other_rts: set[str] = set()
            other_eps: set[str] = set()
            row_rt = extract_runtime_id(row)
            if row_rt:
                other_rts.add(row_rt)
            row_ep = extract_tunnel_endpoint(row)
            if row_ep:
                other_eps.add(row_ep)
            if row_inst and row_inst in instances_by_id:
                linked_inst = instances_by_id[row_inst]
                l_rt = extract_runtime_id(linked_inst)
                if l_rt:
                    other_rts.add(l_rt)
                l_ep = extract_tunnel_endpoint(linked_inst)
                if l_ep:
                    other_eps.add(l_ep)
            if (target_rts & other_rts) or (target_eps & other_eps):
                return row
        for inst_row in instances_by_id.values():
            if instance_id and inst_row.get("id") == instance_id:
                continue
            if exclude_lease_id and inst_row.get("lease_id") == exclude_lease_id:
                continue
            if inst_row.get("state") in {"deleted", "quarantined"}:
                continue
            inst_rts = set()
            inst_eps = set()
            i_rt = extract_runtime_id(inst_row)
            if i_rt:
                inst_rts.add(i_rt)
            i_ep = extract_tunnel_endpoint(inst_row)
            if i_ep:
                inst_eps.add(i_ep)
            if (target_rts & inst_rts) or (target_eps & inst_eps):
                return {"id": inst_row.get("lease_id") or inst_row["id"], "instance_id": inst_row["id"], "state": inst_row.get("state")}
        return None

    def _has_conflicting_runtime_in_store(
        self,
        owner_id: str | None,
        lease: dict | None,
        *,
        instance_id: str | None = None,
        runtime_id: str | None = None,
        tunnel_endpoint: str | None = None,
        handle: object = None,
        exclude_lease_id: str | None = None,
        job: dict | None = None,
    ) -> bool:
        eff_excl = exclude_lease_id or (lease.get("id") if isinstance(lease, dict) else None)
        if hasattr(getattr(self, "store", None), "find_runtime_conflict_lease"):
            conflict, _ = self.store.find_runtime_conflict_lease(
                instance_id=instance_id,
                runtime_id=runtime_id,
                tunnel_endpoint=tunnel_endpoint,
                handle=handle,
                lease=lease,
                job=job,
                exclude_lease_id=eff_excl,
                exclude_instance_id=instance_id,
            )
            if conflict:
                return True
        probe_job = dict(job) if isinstance(job, dict) else {"owner_id": owner_id, "instance_id": instance_id}
        if instance_id and not probe_job.get("instance_id"):
            probe_job["instance_id"] = instance_id
        return (
            self._conflicting_non_deleted_lease(
                probe_job,
                exclude_lease_id=eff_excl,
                runtime_id=runtime_id,
                tunnel_endpoint=tunnel_endpoint,
                handle=handle,
                lease=lease,
            )
            is not None
        )

    def _quarantine_missing_identity(
        self,
        job: dict,
        lease: dict | None,
        *,
        expected_version: int | None = None,
        expected_create_attempts: int | None = None,
    ) -> dict:
        """Freeze unidentifiable runtime/lease/instance via CAS without clobbering recovered leases."""
        if lease is not None:
            from_active = self._is_from_active_lease(lease)
            eff_ver = lease.get("version") if expected_version is None else expected_version
            eff_attempts = int(lease.get("create_attempts") or 0) if expected_create_attempts is None else expected_create_attempts
            if hasattr(getattr(self, "store", None), "quarantine_lease_cas"):
                status, updated_job, _ = self.store.quarantine_lease_cas(
                    job.get("owner_id"),
                    lease["id"],
                    job_id=job.get("id"),
                    instance_id=(job.get("instance_id") if job.get("lease_id") == lease["id"] else lease.get("instance_id")),
                    expected_lease_version=eff_ver,
                    expected_create_attempts=eff_attempts,
                    from_active=from_active,
                    reason="gpu_runtime_identity_missing",
                    error_code="gpu_runtime_identity_missing",
                    message="GPU runtime identity missing",
                    now=self.clock(),
                )
                fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or updated_job or job
                if status in {"stale", "locked", "missing"}:
                    return fresh_job
                if fresh_job.get("lease_id") == lease["id"]:
                    self._clear_owned_execution(fresh_job)
                return updated_job or fresh_job
            with self._admission_lock(timeout=1.0) as acquired:
                if not acquired:
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                fresh_lease = self._get("leases", lease["id"], job.get("owner_id"))
                if fresh_lease is None or fresh_lease.get("state") == "deleted":
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                if eff_ver is not None and fresh_lease.get("version") != eff_ver:
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                if int(fresh_lease.get("create_attempts") or 0) != int(eff_attempts):
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                if fresh_lease.get("state") in {"active", "busy"} and self._lease_handle(fresh_lease):
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                self._update(
                    "leases",
                    lease["id"],
                    {
                        "state": "quarantined",
                        "current_job_id": None,
                        "from_active_missing_handle": from_active,
                        "was_active_missing_handle": from_active,
                        "cleanup_claim_token": None,
                        "cleanup_claimed_at": None,
                        "last_error": "gpu runtime identity missing",
                        "last_seen": self.clock(),
                    },
                    job.get("owner_id"),
                )
        fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
        if lease is not None and fresh_job.get("lease_id") != lease["id"]:
            return fresh_job
        self._clear_owned_execution(fresh_job)
        if self._is_v020_job(fresh_job) and fresh_job.get("instance_id") and hasattr(self.instance_service, "quarantine"):
            existing = self._service_instance_for_job(fresh_job)
            if not existing or existing.get("state") != "quarantined":
                try:
                    instance = self._instance_call(
                        "quarantine",
                        fresh_job.get("owner_id"),
                        fresh_job["instance_id"],
                        reason="gpu_runtime_identity_missing",
                    )
                    fresh_job = self._merge_instance(fresh_job, instance)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "instance quarantine failed: %s", self._safe_error(exc)
                    )
        return self._set_state(
            fresh_job,
            "failed",
            "GPU runtime identity missing",
            requested_outcome="failed",
            error_code="gpu_runtime_identity_missing",
            error_message="GPU runtime identity missing",
        )

    def _factory(self, factory: Callable, job: dict):
        """Injected adapters take ``job`` and optionally the base Store."""
        import inspect

        parameters = inspect.signature(factory).parameters.values()
        positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if any(p.kind == p.VAR_POSITIONAL for p in parameters) or len(positional) >= 2:
            return factory(job, self.db_store)
        return factory(job)

    def _provider(self, job: dict):
        return self._factory(self.provider_factory, job)

    def _executor(self, job: dict):
        return self._factory(self.executor_factory, job)

    def _trusted_plugin_cleanup(self, job: dict, lease: dict | None) -> dict:
        """Run trusted-plugin lifecycle hooks without taking ownership from the provider.

        The provider remains authoritative for runtime deletion.  Plugin hook
        failures are therefore recorded as bounded internal status and never
        change the cleanup path or expose plugin exception text to the caller.
        """
        if not lease or not lease.get("handle") or not job.get("execution_id"):
            return job
        try:
            executor = self._executor(job)
        except Exception as exc:  # noqa: BLE001 - plugin discovery must not block provider cleanup
            logging.getLogger(__name__).warning(
                "trusted plugin executor unavailable during cleanup: %s", type(exc).__name__
            )
            return job
        if not isinstance(executor, TrustedExecutorAdapter):
            return job

        handle = self._lease_handle(lease)
        cancel_requested = bool(job.get("cancel_requested")) or job.get("requested_outcome") == "cancelled"
        if cancel_requested and job.get("trusted_plugin_cancel_succeeded") is not True:
            try:
                executor.cancel(handle, job)
                patch = {"trusted_plugin_cancel_succeeded": True, "trusted_plugin_cancel_error": None}
            except Exception as exc:  # noqa: BLE001 - provider cleanup remains authoritative
                logging.getLogger(__name__).warning("trusted plugin cancel failed: %s", type(exc).__name__)
                patch = {"trusted_plugin_cancel_error": "trusted_plugin_cancel_failed"}
            job = self._update("jobs", job["id"], patch, job.get("owner_id"))

        if job.get("trusted_plugin_cleanup_succeeded") is True:
            return job
        try:
            executor.cleanup(handle, job)
            patch = {"trusted_plugin_cleanup_succeeded": True, "trusted_plugin_cleanup_error": None}
        except Exception as exc:  # noqa: BLE001 - provider cleanup remains authoritative
            logging.getLogger(__name__).warning("trusted plugin cleanup failed: %s", type(exc).__name__)
            patch = {"trusted_plugin_cleanup_error": "trusted_plugin_cleanup_failed"}
        return self._update("jobs", job["id"], patch, job.get("owner_id"))

    def _list(self, kind: str, owner_id: str | None = None) -> list[dict]:
        return list(self.store.list(kind, owner_id=owner_id))

    def _get(self, kind: str, record_id: str, owner_id: str | None = None) -> dict | None:
        return self.store.get(kind, record_id, owner_id=owner_id)

    def _create(self, kind: str, owner_id: str, data: dict, record_id: str | None = None) -> dict:
        return self.store.create(kind, owner_id, data, record_id=record_id)

    def _update(self, kind: str, record_id: str, patch: dict, owner_id: str | None = None) -> dict:
        return self.store.update(kind, record_id, patch, owner_id=owner_id)

    def _event(self, job: dict, state: str, message: str, progress: float | None = None) -> None:
        self.store.event(job["id"], job.get("owner_id"), state, message, progress)

    def _set_state(self, job: dict, state: str, message: str, **patch) -> dict:
        if state in TERMINAL_STATES:
            # Worker and watchdog can both finish the same cleanup; only the first one records it.
            current = self.store.get("jobs", job["id"], job.get("owner_id")) or job
            if current.get("state") == state:
                return current
        if job.get("prep"):
            message = _PREP_TEXT.get(message, message.replace("GPU", "CPU"))
        return self.store.transition_job(
            job["id"], job.get("owner_id"), state, message, patch, self.clock(),
            expected_state=job.get("state"), expected_execution_id=job.get("execution_id"),
        )

    def _provider_name(self, job: dict) -> str:
        return job.get("provider") or (job.get("execution_policy") or {}).get("provider", "mock")

    def _effective_runtime(self, job: dict) -> int:
        requested = job.get("max_runtime_minutes")
        if requested is None:
            requested = (job.get("execution_policy") or {}).get("max_runtime_minutes")
        requested = self.config.max_runtime_minutes if requested is None else int(requested)
        if requested <= 0:
            raise SchedulerError("max runtime must be positive")
        return min(requested, self.config.max_runtime_minutes)

    def _usage_totals(self, owner_id: str) -> tuple[float, float]:
        now = self.clock()
        daily = monthly = 0.0
        for row in self._list("usage", owner_id):
            units = float(row.get("units") or 0)
            stamp = float(row.get("created_epoch") or row.get("epoch") or 0)
            monthly += units if stamp >= now - 31 * 86400 else 0
            daily += units if stamp >= now - 86400 else 0
        for row in self._list("jobs", owner_id):
            if row.get("budget_reserved") and row.get("state") not in TERMINAL_STATES:
                units = float(row["budget_reserved"])
                daily += units
                monthly += units
        return daily, monthly

    def _reserve_budget(self, job: dict, provider: Any) -> tuple[bool, str | None, dict]:
        # The requested provider remains authoritative even when a test double
        # is injected for a colab job.
        is_mock = self._provider_name(job) == "mock"
        raw_units = job.get("budget_units", job.get("estimated_units"))
        if raw_units is None and self.config.unit_rate is not None:
            raw_units = self._effective_runtime(job) * self.config.unit_rate
        if raw_units is None:
            if is_mock:
                raw_units = 0.0
            else:
                return False, "gpu_budget_unknown", {"error_message": "real GPU task has no unit estimate"}
        try:
            units = float(raw_units)
        except (TypeError, ValueError):
            return False, "gpu_budget_invalid", {"error_message": "estimated units are invalid"}
        if units < 0 or units > self.config.max_job_units:
            return False, "gpu_budget_exceeded", {"error_message": "job unit limit exceeded"}
        if is_mock:
            return True, None, {"simulated_units": units}
        daily, monthly = self._usage_totals(job["owner_id"])
        if daily + units > self.config.daily_units or monthly + units > self.config.monthly_units:
            return False, "gpu_budget_exceeded", {"error_message": "GPU budget exceeded"}
        return True, None, {"budget_reserved": units}

    def _validate(self, job: dict) -> dict:
        if not self.config.enabled:
            return self._set_state(
                job,
                "failed",
                "GPU scheduler is disabled",
                error_code="gpu_disabled",
                error_message="GPU is disabled",
            )
        try:
            runtime = self._effective_runtime(job)
            provider = self._provider(job)
            if self._provider_name(job) != "mock" and job.get("requested_spec"):
                try:
                    self._select_spec(job, provider)
                except ProviderUnavailable as exc:
                    return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
                except Exception as exc:
                    return self._begin_failure(job, "gpu_spec_invalid", self._safe_error(exc))
            ok, code, patch = self._reserve_budget(job, provider)
        except Exception as exc:
            return self._set_state(
                job,
                "failed",
                "GPU provider unavailable",
                error_code="gpu_provider_unavailable",
                error_message=self._safe_error(exc),
            )
        if not ok:
            return self._set_state(job, "failed", "GPU budget rejected", error_code=code, **patch)
        patch.update({"max_runtime_minutes": runtime, "validated_epoch": self.clock()})
        return self._set_state(job, "waiting_for_capacity", "GPU job validated", **patch)

    def _leases_for(self, owner_id: str | None = None) -> list[dict]:
        return self._list("leases", owner_id)

    @staticmethod
    def _lease_blocks(lease: dict) -> bool:
        return lease.get("state") in {
            "creating",
            "active",
            "busy",
            "idle",
            "unknown",
            "cleanup_pending",
            "quarantined",
            "releasing",
            "deleting",
        }

    def _has_capacity_blocker(self, job: dict) -> bool:
        return any(self._lease_blocks(x) and x.get("owner_job_id") != job["id"] for x in self._leases_for())

    def _await_prep(self, job: dict) -> bool:
        """A GPU job waits for its workflow's CPU cache prep; the prep job is created on the first miss.

        One failed or cancelled prep stops the waiting (the job then runs uncached); it is never retried here.
        """
        snapshot = job.get("snapshot") or {}
        sha = snapshot.get("workflow_sha256")
        if (
            self._provider_name(job) != "colab"
            or snapshot.get("prep")
            or snapshot.get("system_smoke")
            or not sha
            or not (snapshot.get("manifest") or {}).get("weight_files")
            or os.getenv("GPU_PREP", "on").strip().lower() == "off"
        ):
            return False
        drive_account_id = job.get("drive_account_id")
        if not drive_account_id and job.get("account_id"):
            from .accounts import AccountService

            drive_account_id = AccountService(self.store, job["owner_id"]).resolve_drive_account_id(job["account_id"])
        preps = prep_jobs_for(self.store, job["owner_id"], snapshot, drive_account_id=drive_account_id)
        done = next((p for p in preps if p["state"] == "completed"), None)
        if done:
            if done["id"] == job.get("prep_job_id") and job.get("prep_seconds") is None:
                # The prep ran for this job: show it as this job's first stage.
                spent = round(sum((done.get("worker_timings") or {}).values()), 1)
                self._update("jobs", job["id"], {"prep_seconds": spent}, job.get("owner_id"))
            return False
        active = next((p for p in preps if p["state"] in ACTIVE_STATES), None)
        if active:
            if job.get("prep_job_id") != active["id"]:
                self._update("jobs", job["id"], {"prep_job_id": active["id"]}, job.get("owner_id"))
            return True
        if preps:
            return False
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return True
            meta = {
                "workflow_sha256": sha,
                "model_id": job.get("model_id"),
                "workflow_key": job.get("workflow_key"),
                "workflow_version": job.get("workflow_version"),
                "account_id": job.get("account_id") or job.get("requested_account_id"),
                "drive_account_id": drive_account_id,
            }
            prep = create_prep_job(
                self.store, job["owner_id"], job.get("project_id"), snapshot, meta,
                prep_idempotency_key(cache_key(snapshot), drive_account_id or ""), for_job_id=job["id"],
            )
        self._update("jobs", job["id"], {"prep_job_id": prep["id"]}, job.get("owner_id"))
        self._event(job, "waiting_for_capacity", "准备缓存（CPU）：为这个任务自动准备环境和模型缓存，完成后自动开始", None)
        return True

    def _lease(self, job: dict) -> dict | None:
        def owned(row: dict | None) -> bool:
            if not row or row.get("owner_job_id") != job["id"]:
                return False
            current = row.get("current_job_id")
            return current in (None, job["id"])

        if job.get("lease_id"):
            lease = self._get("leases", job["lease_id"], job.get("owner_id"))
            return lease if owned(lease) else None
        skip_deleted = bool(job.get("attempt_retry_keys") or int(job.get("alloc_generation") or 0) > 0)
        return next(
            (
                x
                for x in self._leases_for(job.get("owner_id"))
                if owned(x) and not (skip_deleted and x.get("state") == "deleted")
            ),
            None,
        )

    def _ensure_lease(self, job: dict) -> dict:
        current = self._lease(job)
        if current:
            return current
        if job.get("lease_id"):
            raise SchedulerError("GPU job already references an absent or unowned lease")
        conflict = self._conflicting_non_deleted_lease(job)
        if conflict is not None:
            raise SchedulerError("GPU instance already has an active or non-deleted lease")
        request_id, runtime_id = str(uuid.uuid4()), f"studio-{uuid.uuid4().hex}"
        now = self.clock()
        deadline = job.get("deadline_epoch")
        if deadline is None:
            deadline = now + self._effective_runtime(job) * 60
        payload = {
            "owner_job_id": job["id"],
            "provider": self._provider_name(job),
            "state": "creating",
            "request_id": request_id,
            "runtime_id": runtime_id,
            "created_epoch": now,
            "last_seen": now,
            "delete_attempts": 0,
            "create_attempts": 0,
            "deadline_epoch": deadline,
        }
        for key in ("instance_id", "account_id"):
            if job.get(key) is not None:
                payload[key] = job[key]
        return self._create("leases", job["owner_id"], payload)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        """Expose only messages from deliberately safe adapter exception classes."""
        if not isinstance(exc, (ComputeError, GoogleError, MCPError)):
            logging.getLogger(__name__).warning("unexpected GPU error: %s: %s", type(exc).__name__, str(exc)[:300])
            return f"GPU operation failed ({type(exc).__name__})"
        message = str(exc).replace("token", "[redacted]").replace("secret", "[redacted]")
        return message[:1400] or "GPU operation failed"

    @staticmethod
    def _runtime_id(provider: str, runtime_id: str) -> str:
        if provider == "colab":
            return f"runtimes/{runtime_id}"
        return f"mock-runtime-{runtime_id}"

    def _lease_handle(self, lease: dict) -> dict | None:
        """The lease's runtime handle, always owned by the lease's current owner job.

        Lease reuse (handoff / idle attach) moves owner_job_id; providers verify the handle against it,
        so readers retarget a copy; only ownership transactions persist a changed handle.
        """
        handle = lease.get("handle")
        owner = lease.get("owner_job_id")
        if lease.get("current_job_id") not in (None, owner):
            return handle
        if not isinstance(handle, dict) or not handle or not isinstance(owner, str):
            return handle
        if handle.get("job_id") == owner and handle.get("owner_job_id") == owner:
            return handle
        fixed = {**handle, "job_id": owner, "owner_job_id": owner}
        return fixed

    def _intent_handle(self, job: dict, lease: dict) -> dict | None:
        """Build only a lease-owned handle; never infer a platform runtime name."""
        runtime_id = lease.get("runtime_id")
        if not (
            isinstance(runtime_id, str)
            and lease.get("owner_job_id") == job.get("id")
            and isinstance(lease.get("request_id"), str)
            and lease.get("request_id")
        ):
            return None
        handle = {
            "runtime_name": self._runtime_id(self._provider_name(job), runtime_id),
            "runtime_id": runtime_id,
            "request_id": lease["request_id"],
            "job_id": job["id"],
            "owner_job_id": job["id"],
            "owner_id": job.get("owner_id"),
            "intent": "gpu-studio",
        }
        if lease.get("operation_name"):
            handle["operation_name"] = lease["operation_name"]
        if lease.get("tunnel_endpoint"):
            handle["tunnel_endpoint"] = lease["tunnel_endpoint"]
        if self._provider_name(job) == "mock":
            handle["mock"] = True
        return handle

    @staticmethod
    def _spec_key(spec: dict) -> str | None:
        """Colab runtimespecs are identified by their ``key`` (variant/accelerator/shape)."""
        value = spec.get("key")
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return value or None

    @staticmethod
    def _has_gpu(spec: dict) -> bool:
        key = spec.get("key")
        return isinstance(key, dict) and key.get("variant") in {"GPU", "VARIANT_GPU"}

    def _select_spec(self, job: dict, provider: Any) -> dict:
        try:
            specs = provider.list_specs()
        except Exception as exc:
            raise ComputeError(self._safe_error(exc)) from exc
        requested = job.get("requested_spec") or (job.get("execution_policy") or {}).get("requested_spec")
        requested_key = requested if isinstance(requested, str) else self._spec_key(requested or {})
        snapshot = job.get("snapshot") or {}
        smoke = snapshot.get("system_smoke") is True
        prep = snapshot.get("prep") is True
        # 官方 runtimespecs 只返回 key(variant/accelerator/shape)。经审查的工作流在
        # manifest.compute_spec 写明已验证过的规格；普通任务只允许这个规格，
        # 管理员冒烟任务按其请求的规格运行，缓存准备只用固定的 CPU 规格。
        reviewed_key = self._spec_key({"key": (snapshot.get("manifest") or {}).get("compute_spec")})
        if prep:
            requested_key = reviewed_key = self._spec_key({"key": PREP_SPEC_KEY})
        if requested_key is None:
            requested_key = reviewed_key
        if not (smoke or snapshot.get("workflow_smoke") is True or reviewed_key is not None):
            raise ComputeError("没有符合工作流runtime的GPU规格")
        if smoke and requested_key is None:
            raise ComputeError("system smoke必须绑定已选择的GPU规格")
        for spec in specs:
            if spec.get("eligible") is not True or self._has_gpu(spec) == prep:
                continue
            if requested_key is None or self._spec_key(spec) == requested_key:
                return spec
        raise ComputeError("没有符合工作流runtime的GPU规格")

    def _mark_lease_unknown(
        self,
        lease: dict,
        job: dict,
        *,
        from_active: bool,
        error: str | None = None,
        expected_version: int | None = None,
        expected_create_attempts: int | None = None,
    ) -> dict:
        now = self.clock()
        was_active = self._is_from_active_lease(lease) or bool(from_active)
        intent_dict = lease.get("create_intent") if isinstance(lease.get("create_intent"), dict) else {}
        existing_origin = next(
            (
                v
                for v in (lease.get("origin"), intent_dict.get("origin"), lease.get("create_intent_origin"))
                if isinstance(v, str) and v.strip().lower() in {"active", "busy", "once_active", "from_active"}
            ),
            lease.get("create_intent_origin") or lease.get("origin"),
        )
        if was_active:
            next_origin = (
                existing_origin
                if isinstance(existing_origin, str)
                and existing_origin.strip().lower() in {"active", "busy", "once_active", "from_active"}
                else "once_active"
            )
        else:
            next_origin = existing_origin or "new"
        patch: dict[str, Any] = {
            "state": "unknown",
            "from_active_missing_handle": was_active,
            "was_active_missing_handle": was_active,
            "create_intent_origin": next_origin,
            "last_seen": now,
        }
        if was_active:
            patch["ever_active"] = True
        if error:
            patch["last_error"] = error
        eff_ver = lease.get("version") if expected_version is None else expected_version
        eff_attempts = lease.get("create_attempts") if expected_create_attempts is None else expected_create_attempts
        if hasattr(getattr(self, "store", None), "update_lease_cas"):
            status, current = self.store.update_lease_cas(
                lease["id"],
                job.get("owner_id"),
                patch,
                expected_version=eff_ver,
                expected_create_attempts=eff_attempts,
                preserve_active_history=True,
                refuse_downgrade_to_unknown=True,
                refuse_if_create_in_flight=True,
                now=now,
            )
            return current if current is not None else lease
        with self._admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return lease
            fresh = self._get("leases", lease["id"], job.get("owner_id")) or lease
            if eff_ver is not None and fresh.get("version") != eff_ver:
                return fresh
            if eff_attempts is not None and int(fresh.get("create_attempts") or 0) != int(eff_attempts):
                return fresh
            if self._create_rpc_is_in_flight(fresh, now):
                return fresh
            if fresh.get("state") in {"deleted", "idle", "cleanup_pending", "releasing", "deleting", "quarantined"}:
                return fresh
            if fresh.get("state") in {"active", "busy"} and self._lease_handle(fresh):
                return fresh
            if self._is_from_active_lease(fresh) or fresh.get("was_active_missing_handle"):
                patch["from_active_missing_handle"] = True
                patch["was_active_missing_handle"] = True
                patch["ever_active"] = True
                if fresh.get("create_intent_origin") in {"active", "busy", "once_active", "from_active"}:
                    patch["create_intent_origin"] = fresh["create_intent_origin"]
                elif fresh.get("origin") in {"active", "busy", "once_active", "from_active"}:
                    patch["create_intent_origin"] = fresh["origin"]
                elif patch.get("create_intent_origin") not in {"active", "busy", "once_active", "from_active"}:
                    patch["create_intent_origin"] = "once_active"
            return self._update("leases", lease["id"], patch, job.get("owner_id"))

    @staticmethod
    def _inspect_confirms_absent(status: Any) -> bool:
        """Return True only when inspect() gives an unambiguous, remotely verified deletion proof."""
        if not isinstance(status, dict):
            return False
        if any(
            status.get(flag)
            for flag in (
                "allocation_unknown",
                "operation_pending",
                "ambiguous",
                "pending",
                "identity_unverifiable",
            )
        ):
            return False
        return status.get("deleted") is True and status.get("remote_verified") is True

    @classmethod
    def _delete_confirms_absent(cls, response: Any) -> bool:
        """Accept True (verified provider contract) or an unambiguously verified delete proof dict."""
        if response is True:
            return True
        if isinstance(response, dict):
            return cls._inspect_confirms_absent(response)
        return False

    def _recover_unknown_create(self, job: dict, lease: dict, provider: Any) -> str | bool:
        """Reconcile a create that may have succeeded before its response was lost."""
        outcome, _, _ = self._recover_unknown_create_with_snapshot(job, lease, provider)
        return outcome

    def _recover_unknown_create_with_snapshot(
        self, job: dict, lease: dict, provider: Any
    ) -> tuple[str | bool, int | None, int]:
        """Reconcile unknown create while preserving the pre-probe (version, create_attempts) snapshot."""
        from_active = self._is_from_active_lease(lease)
        probe_started_at = self.clock()
        probe_version = lease.get("version")
        probe_attempts = int(lease.get("create_attempts") or 0)
        pre_rt = extract_runtime_id(lease)
        pre_ep = extract_tunnel_endpoint(lease)
        lease_handle_raw = lease.get("handle") if isinstance(lease.get("handle"), dict) else {}
        pre_op = lease.get("operation_name")
        pre_handle_op = lease_handle_raw.get("operation_name")
        in_flight = self._create_rpc_is_in_flight(lease, probe_started_at)
        handle = self._lease_handle(lease) or self._intent_handle(job, lease)
        if handle is None:
            if in_flight:
                return "in_flight", probe_version, probe_attempts
            self._mark_lease_unknown(
                lease,
                job,
                from_active=from_active,
                error="lease intent incomplete",
                expected_version=probe_version,
                expected_create_attempts=probe_attempts,
            )
            return "identity_unverifiable", probe_version, probe_attempts
        try:
            if hasattr(provider, "recover_handle"):
                handle = provider.recover_handle(handle)
            status = provider.inspect(handle)
        except Exception as exc:
            if in_flight:
                return "in_flight", probe_version, probe_attempts
            self._mark_lease_unknown(
                lease,
                job,
                from_active=from_active,
                error=self._safe_error(exc),
                expected_version=probe_version,
                expected_create_attempts=probe_attempts,
            )
            return False, probe_version, probe_attempts
        if in_flight and (
            not isinstance(status, dict) or status.get("ready") is not True
            or status.get("deleted") or status.get("allocation_failed")
            or status.get("allocation_unknown") or status.get("remote_verified") is False
        ):
            # Empty/unknown listings while a POST may still be running prove nothing.
            return "in_flight", probe_version, probe_attempts
        if not isinstance(status, dict) or (status.get("deleted") is None and "ready" not in status and not status.get("allocation_failed")):
            self._mark_lease_unknown(
                lease,
                job,
                from_active=from_active,
                error="provider inspect returned empty status",
                expected_version=probe_version,
                expected_create_attempts=probe_attempts,
            )
            return False, probe_version, probe_attempts
        if status.get("identity_unverifiable"):
            return "identity_unverifiable", probe_version, probe_attempts
        if self._inspect_confirms_absent(status):
            now = self.clock()
            intent_dict = lease.get("create_intent") if isinstance(lease.get("create_intent"), dict) else {}
            existing_origin = next(
                (
                    v
                    for v in (lease.get("origin"), intent_dict.get("origin"), lease.get("create_intent_origin"))
                    if isinstance(v, str) and v.strip().lower() in {"active", "busy", "once_active", "from_active"}
                ),
                lease.get("create_intent_origin") or lease.get("origin"),
            )
            if from_active:
                absent_origin = (
                    existing_origin
                    if isinstance(existing_origin, str)
                    and existing_origin.strip().lower() in {"active", "busy", "once_active", "from_active"}
                    else "once_active"
                )
            else:
                absent_origin = existing_origin or "new"
            absent_patch: dict[str, Any] = {
                "from_active_missing_handle": from_active,
                "was_active_missing_handle": from_active,
                "create_intent_origin": absent_origin,
                "last_seen": now,
            }
            if from_active:
                absent_patch["ever_active"] = True
            if hasattr(getattr(self, "store", None), "update_lease_cas"):
                # Pre-check timestamp: if a create RPC finished AFTER inspect began, treat as stale.
                fresh_pre = self._get("leases", lease["id"], job.get("owner_id"))
                if fresh_pre is not None:
                    if self._create_rpc_is_in_flight(fresh_pre, now):
                        return "in_flight", probe_version, probe_attempts
                    for ts_key in ("create_rpc_finished_at", "create_started"):
                        ts_val = fresh_pre.get(ts_key)
                        if ts_val is not None:
                            try:
                                if float(probe_started_at) < float(ts_val):
                                    return "stale", probe_version, probe_attempts
                            except (TypeError, ValueError):
                                return "stale", probe_version, probe_attempts
                cas_status, updated_lease = self.store.update_lease_cas(
                    lease["id"],
                    job.get("owner_id"),
                    absent_patch,
                    expected_version=probe_version,
                    expected_create_attempts=probe_attempts,
                    preserve_active_history=True,
                    refuse_if_create_in_flight=True,
                    now=now,
                )
                if cas_status != "ok" or updated_lease is None:
                    return "stale", probe_version, probe_attempts
                probe_version = updated_lease.get("version")
            else:
                with self._admission_lock(timeout=1.0) as acquired:
                    if not acquired:
                        return "stale", probe_version, probe_attempts
                    fresh_lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
                    if (
                        (probe_version is not None and fresh_lease.get("version") != probe_version)
                        or int(fresh_lease.get("create_attempts") or 0) != probe_attempts
                        or self._create_rpc_is_in_flight(fresh_lease, now)
                    ):
                        return "stale", probe_version, probe_attempts
                    updated_lease = self._update("leases", lease["id"], absent_patch, job.get("owner_id"))
                    probe_version = updated_lease.get("version")
            return "absent", probe_version, probe_attempts
        if (
            status.get("allocation_unknown")
            or status.get("operation_pending")
            or status.get("ambiguous")
            or status.get("pending")
            or status.get("remote_verified") is False
            or status.get("deleted")
        ):
            self._mark_lease_unknown(
                lease,
                job,
                from_active=from_active,
                expected_version=probe_version,
                expected_create_attempts=probe_attempts,
            )
            return False, probe_version, probe_attempts
        if (
            not status.get("allocation_failed")
            and not status.get("error")
            and not status.get("error_code")
        ):
            recovered_handle = dict(handle)
            if isinstance(status.get("handle"), dict):
                recovered_handle.update(status["handle"])
            recovered_handle.update(
                {
                    "runtime_id": lease.get("runtime_id") or recovered_handle.get("runtime_id"),
                    "request_id": lease.get("request_id") or recovered_handle.get("request_id"),
                    "job_id": job["id"],
                    "owner_job_id": job["id"],
                    "owner_id": job.get("owner_id"),
                }
            )
            discovered_op = status.get("operation_name") if isinstance(status, dict) else None
            if hasattr(getattr(self, "store", None), "recover_create_present_cas"):
                rec_status, _, _ = self.store.recover_create_present_cas(
                    job["id"],
                    job.get("owner_id"),
                    lease["id"],
                    handle=recovered_handle,
                    expected_lease_version=probe_version,
                    expected_create_attempts=probe_attempts,
                    expected_runtime_id=pre_rt,
                    expected_tunnel_endpoint=pre_ep,
                    expected_operation_name=pre_op,
                    operation_name=discovered_op,
                    deadline_epoch=lease.get("deadline_epoch") or job.get("deadline_epoch"),
                    check_snapshot=True,
                    connection_metadata=status.get("connection_metadata"),
                    expected_rpc_token=lease.get("create_rpc_token") if in_flight else None,
                    expected_rpc_instance_id=lease.get("create_rpc_instance_id") if in_flight else None,
                    now=self.clock(),
                )
                if rec_status == "ok":
                    return "present", probe_version, probe_attempts
                return "stale", probe_version, probe_attempts
            with self._admission_lock(timeout=1.0) as acquired:
                if not acquired:
                    return False, probe_version, probe_attempts
                fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
                fresh_lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
                if (
                    fresh_job.get("state") != "allocating"
                    or (
                        fresh_job.get("lease_id") is not None
                        and fresh_job.get("lease_id") != fresh_lease["id"]
                    )
                    or (
                        fresh_job.get("lease_id") is None
                        and fresh_lease.get("owner_job_id") != fresh_job["id"]
                    )
                    or (probe_version is not None and fresh_lease.get("version") != probe_version)
                    or int(fresh_lease.get("create_attempts") or 0) != probe_attempts
                    or extract_runtime_id(fresh_lease) != pre_rt
                    or extract_tunnel_endpoint(fresh_lease) != pre_ep
                    or (fresh_lease.get("operation_name") or None) != (pre_op or None)
                    or (
                        (
                            fresh_lease.get("handle").get("operation_name")
                            if isinstance(fresh_lease.get("handle"), dict)
                            else None
                        )
                        != pre_handle_op
                    )
                ):
                    return "stale", probe_version, probe_attempts
                inst = self._service_instance_for_job(fresh_job)
                if inst is None and fresh_job.get("instance_id"):
                    inst = self._get("instances", fresh_job["instance_id"], fresh_job.get("owner_id"))
                recovered_handle["runtime_id"] = fresh_lease.get("runtime_id") or recovered_handle.get("runtime_id")
                recovered_handle["request_id"] = fresh_lease.get("request_id") or recovered_handle.get("request_id")
                has_conflict = self._has_conflicting_runtime_in_store(
                    fresh_job.get("owner_id"),
                    fresh_lease,
                    instance_id=fresh_job.get("instance_id"),
                    handle=recovered_handle,
                    exclude_lease_id=fresh_lease["id"],
                    job=fresh_job,
                )
                fenced = (
                    has_conflict
                    or fresh_job.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}
                    or self._fence_reason(fresh_job, fresh_lease, inst, allow_unknown=True) is not None
                    or fresh_lease.get("owner_id") != fresh_job.get("owner_id")
                    or fresh_job.get("lease_id") not in {None, fresh_lease["id"]}
                )
                if fenced:
                    return "stale", probe_version, probe_attempts
                now_rec = self.clock()
                lease_patch = {
                    "state": "active",
                    "handle": recovered_handle,
                    "from_active_missing_handle": False,
                    "was_active_missing_handle": False,
                    "ever_active": True,
                    "create_rpc_in_flight": False,
                    "create_rpc_finished_at": now_rec,
                    "last_seen": now_rec,
                }
                if discovered_op and not fresh_lease.get("operation_name"):
                    lease_patch["operation_name"] = discovered_op
                self._update(
                    "leases",
                    fresh_lease["id"],
                    lease_patch,
                    fresh_job.get("owner_id"),
                )
                if (
                    fresh_job.get("instance_id")
                    and inst is not None
                    and inst.get("state") != "deleted"
                    and self._get("instances", fresh_job["instance_id"], fresh_job.get("owner_id")) is not None
                ):
                    self._update(
                        "instances",
                        fresh_job["instance_id"],
                        {"runtime_created": True},
                        fresh_job.get("owner_id"),
                    )
                return "present", probe_version, probe_attempts
        now = self.clock()
        err_msg = self._safe_error(status.get("error") or status.get("message") or "") if isinstance(status, dict) else ""
        claim_token: str | None = None
        claimed_lease_version: int | None = None
        if hasattr(getattr(self, "store", None), "acquire_failed_create_cleanup_cas"):
            claim_status, claimed_lease, claim_token = self.store.acquire_failed_create_cleanup_cas(
                job["id"],
                job.get("owner_id"),
                lease["id"],
                handle=handle,
                expected_lease_version=probe_version,
                expected_create_attempts=probe_attempts,
                expected_runtime_id=pre_rt,
                expected_tunnel_endpoint=pre_ep,
                expected_operation_name=pre_op,
                from_active=from_active,
                last_error=err_msg or None,
                now=now,
            )
            if claim_status in {"stale", "conflict", "locked", "missing"} or claimed_lease is None:
                return "stale", probe_version, probe_attempts
            claimed_lease_version = claimed_lease.get("version")
            probe_version = claimed_lease_version
        else:
            with self._admission_lock(timeout=1.0) as acquired:
                if not acquired:
                    return "stale", probe_version, probe_attempts
                fresh_lease = self._get("leases", lease["id"], job.get("owner_id"))
                if (
                    fresh_lease is None
                    or fresh_lease.get("state") in {"deleted", "idle", "active", "busy"}
                    or (probe_version is not None and fresh_lease.get("version") != probe_version)
                    or int(fresh_lease.get("create_attempts") or 0) != probe_attempts
                    or self._create_rpc_is_in_flight(fresh_lease, now)
                    or self._has_conflicting_runtime_in_store(
                        job.get("owner_id"),
                        fresh_lease,
                        instance_id=job.get("instance_id"),
                        handle=handle,
                        exclude_lease_id=lease["id"],
                        job=job,
                    )
                ):
                    return "stale", probe_version, probe_attempts
                claim_token = uuid.uuid4().hex
                updated_lease = self._update(
                    "leases",
                    lease["id"],
                    {
                        "handle": handle,
                        "state": "cleanup_pending",
                        "from_active_missing_handle": from_active,
                        "was_active_missing_handle": from_active,
                        "cleanup_claim_token": claim_token,
                        "cleanup_claimed_at": now,
                        "last_seen": now,
                    },
                    job.get("owner_id"),
                )
                claimed_lease_version = updated_lease.get("version")
                probe_version = claimed_lease_version
        try:
            confirmed = self._delete_confirms_absent(provider.delete(handle))
        except Exception as exc:
            confirmed = False
            error = self._safe_error(exc)
        else:
            error = None
        if confirmed:
            del_now = self.clock()
            del_patch = {
                "state": "deleted",
                "cleanup_claim_token": None,
                "cleanup_claimed_at": None,
                "last_seen": del_now,
                "remote_deleted": True,
                "remote_deleted_at": del_now,
                "remote_delete_source": "cleanup_provider_confirmation",
            }
            if hasattr(getattr(self, "store", None), "update_lease_cas"):
                cas_del, updated_del = self.store.update_lease_cas(
                    lease["id"],
                    job.get("owner_id"),
                    del_patch,
                    expected_version=claimed_lease_version,
                    expected_cleanup_token=claim_token,
                    preserve_active_history=True,
                    now=del_now,
                )
                if cas_del != "ok":
                    return "stale", probe_version, probe_attempts
                if updated_del is not None:
                    probe_version = updated_del.get("version")
            else:
                self._update("leases", lease["id"], del_patch, job.get("owner_id"))
            return "cleaned", probe_version, probe_attempts
        err_patch = {
            "last_error": error or "delete not confirmed",
            "cleanup_claim_token": None,
            "cleanup_claimed_at": None,
            "last_seen": self.clock(),
        }
        if hasattr(getattr(self, "store", None), "update_lease_cas"):
            self.store.update_lease_cas(
                lease["id"],
                job.get("owner_id"),
                err_patch,
                expected_version=claimed_lease_version,
                expected_cleanup_token=claim_token,
                preserve_active_history=True,
                now=self.clock(),
            )
        else:
            self._update("leases", lease["id"], err_patch, job.get("owner_id"))
        return "delete_pending", probe_version, probe_attempts

    def _retire_absent_lease_if_fresh(
        self,
        job: dict,
        lease: dict,
        *,
        probe_version: int | None,
        probe_attempts: int,
        from_active: bool,
        confirmed_absent: bool = False,
        expected_tunnel_endpoint: str | None = None,
        expected_tunnel_nbh: object = None,
        probe_started_at: float | None = None,
        now: float,
    ) -> str:
        if hasattr(getattr(self, "store", None), "retire_absent_lease_cas"):
            status, _, _ = self.store.retire_absent_lease_cas(
                job["id"],
                job["owner_id"],
                lease["id"],
                expected_lease_version=probe_version,
                expected_create_attempts=probe_attempts,
                expected_tunnel_endpoint=expected_tunnel_endpoint,
                expected_tunnel_nbh=expected_tunnel_nbh,
                from_active=from_active,
                confirmed_absent=confirmed_absent,
                probe_started_at=probe_started_at,
                source="recovery_probe_absent",
                now=now,
            )
            return status
        with self._admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked"
            fresh_lease = self._get("leases", lease["id"], job.get("owner_id"))
            if fresh_lease is None:
                return "missing"
            if fresh_lease.get("state") == "deleted":
                return "already_deleted"
            if self._create_rpc_is_in_flight(fresh_lease, now):
                return "in_flight"
            has_handle = isinstance(fresh_lease.get("handle"), dict) and bool(fresh_lease.get("handle"))
            cur_ep = extract_tunnel_endpoint(fresh_lease)
            exp_ep = normalize_tunnel_endpoint(expected_tunnel_endpoint)
            has_ep = bool(fresh_lease.get("tunnel_endpoint") or cur_ep)
            has_nbh = fresh_lease.get("tunnel_nbh") is not None
            ep_disallowed = (has_ep or has_nbh) and not (
                confirmed_absent
                and exp_ep is not None
                and cur_ep == exp_ep
                and (not has_nbh or fresh_lease.get("tunnel_nbh") == expected_tunnel_nbh)
            )
            if (
                (probe_version is not None and fresh_lease.get("version") != probe_version)
                or int(fresh_lease.get("create_attempts") or 0) != probe_attempts
                or fresh_lease.get("state") == "idle"
                or (fresh_lease.get("state") in {"active", "busy"} and not confirmed_absent)
                or has_handle
                or ep_disallowed
            ):
                return "stale"
            patch = {
                "state": "deleted",
                "current_job_id": None,
                "remote_deleted": True,
                "remote_deleted_at": now,
                "remote_delete_source": "recovery_probe_absent",
                "last_seen": now,
            }
            if from_active:
                patch["from_active_missing_handle"] = True
                patch["was_active_missing_handle"] = True
                patch["ever_active"] = True
            self._update("leases", lease["id"], patch, job.get("owner_id"))
            return "retired"

    def _handle_recovery(self, job: dict, lease: dict, provider: Any) -> dict:
        """Execute R1 -> R1.5 -> R2 -> R3 recovery for missing-handle or unknown-create leases."""
        probe_started_at = self.clock()
        from_active = self._is_from_active_lease(lease)
        pre_ep = extract_tunnel_endpoint(lease)
        pre_nbh = lease.get("tunnel_nbh")
        recovered, probe_version, probe_attempts = self._recover_unknown_create_with_snapshot(job, lease, provider)
        lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
        from_active = from_active or self._is_from_active_lease(lease)
        if recovered in {"stale", "in_flight"}:
            return self._get("jobs", job["id"], job.get("owner_id")) or job
        if recovered == "present":
            fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
            if fresh_job.get("state") != "allocating":
                return fresh_job
            fenced = self._enforce_stage_fence(fresh_job)
            if fenced is not None:
                return fenced
            if fresh_job.get("lease_id") not in {None, lease.get("id")}:
                return fresh_job
            return self._set_state(fresh_job, "preparing", "GPU runtime allocation recovered")
        if recovered == "identity_unverifiable":
            return self._quarantine_missing_identity(
                job,
                lease,
                expected_version=probe_version,
                expected_create_attempts=probe_attempts,
            )
        if recovered == "cleaned":
            return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
        if recovered == "delete_pending":
            return self._begin_failure(
                job,
                "gpu_allocation_aborted",
                "GPU allocation entered cleanup before execution; retry after release",
            )
        if recovered == "absent":
            now = self.clock()
            if from_active:
                retire_status = self._retire_absent_lease_if_fresh(
                    job,
                    lease,
                    probe_version=probe_version,
                    probe_attempts=probe_attempts,
                    from_active=True,
                    confirmed_absent=True,
                    expected_tunnel_endpoint=pre_ep,
                    expected_tunnel_nbh=pre_nbh,
                    probe_started_at=probe_started_at,
                    now=now,
                )
                if hasattr(self.store, "retire_absent_lease_cas") or retire_status in {"stale", "locked", "missing", "in_flight", "retired_detached"}:
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                try:
                    job = self._mark_instance_deleted(job)
                except SchedulerError:
                    pass
                return self._begin_failure(job, "gpu_runtime_missing", "GPU runtime disappeared before execution")
            inst = self._service_instance_for_job(job)
            if inst is None and job.get("instance_id"):
                inst = self._get("instances", job["instance_id"], job.get("owner_id"))
            if isinstance(inst, dict) and inst.get("state") not in {"creating", "allocating", "requested"}:
                retire_status = self._retire_absent_lease_if_fresh(
                    job,
                    lease,
                    probe_version=probe_version,
                    probe_attempts=probe_attempts,
                    from_active=False,
                    confirmed_absent=True,
                    expected_tunnel_endpoint=pre_ep,
                    expected_tunnel_nbh=pre_nbh,
                    probe_started_at=probe_started_at,
                    now=now,
                )
                if hasattr(self.store, "retire_absent_lease_cas") or retire_status in {"stale", "locked", "missing", "in_flight", "retired_detached"}:
                    return self._get("jobs", job["id"], job.get("owner_id")) or job
                try:
                    job = self._mark_instance_deleted(job)
                except SchedulerError:
                    pass
                return self._begin_failure(job, "gpu_create_failed", "GPU instance state incompatible with create retry")
            try:
                if isinstance(provider, DisabledProvider):
                    raise ProviderUnavailable(provider.reason)
                spec = (
                    {"provider": self._provider_name(job)}
                    if self._provider_name(job) == "mock"
                    else self._select_spec(job, provider)
                )
            except ProviderUnavailable as exc:
                return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
            except Exception as exc:
                return self._begin_failure(job, "gpu_spec_invalid", self._safe_error(exc))
            if hasattr(getattr(self, "store", None), "retry_absent_lease_create"):
                retry_kwargs: dict[str, Any] = {
                    "spec": spec,
                    "max_retries": self.config.create_retries,
                    "now": now,
                    "expected_lease_version": probe_version,
                    "expected_create_attempts": probe_attempts,
                }
                if self._accepts_kwarg(self.store.retry_absent_lease_create, "require_instance"):
                    retry_kwargs["require_instance"] = self._has_sql_instance_service()
                status, fresh_job, fresh_lease = self.store.retry_absent_lease_create(
                    job["id"],
                    job["owner_id"],
                    lease["id"],
                    **retry_kwargs,
                )
                if status in {"locked", "stale", "in_flight"} or fresh_job is None or fresh_lease is None:
                    return fresh_job or (self._get("jobs", job["id"], job.get("owner_id")) or job)
                if status == "fenced":
                    return self._handle_fenced_alloc_result(fresh_job, fresh_lease)
                if status == "has_handle":
                    if fresh_job.get("state") == "allocating" and fresh_job.get("lease_id") in {None, fresh_lease.get("id")}:
                        return self._set_state(fresh_job, "preparing", "GPU runtime allocation recovered")
                    return fresh_job
                if status == "create":
                    return self._create_runtime_on_lease(
                        fresh_job, fresh_lease, provider, spec, allow_recovery=False
                    )
            else:
                with self._admission_lock(timeout=1.0) as acquired:
                    if not acquired:
                        return job
                    fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
                    fresh_lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
                    attempts = int(fresh_lease.get("create_attempts") or 0)
                    if (
                        (probe_version is not None and fresh_lease.get("version") != probe_version)
                        or attempts != probe_attempts
                        or self._create_rpc_is_in_flight(fresh_lease, now)
                    ):
                        return fresh_job
                    if (
                        not self._is_from_active_lease(fresh_lease)
                        and attempts <= self.config.create_retries
                    ):
                        fresh_lease = self._update(
                            "leases",
                            fresh_lease["id"],
                            {
                                "state": "creating",
                                "from_active_missing_handle": False,
                                "create_attempts": attempts + 1,
                                "create_started": now,
                                "create_rpc_in_flight": True,
                                "create_rpc_token": uuid.uuid4().hex,
                                "create_rpc_attempt": attempts + 1,
                                "create_rpc_instance_id": fresh_lease.get("instance_id") or fresh_job.get("instance_id"),
                                "create_rpc_started_at": now,
                                "create_rpc_finished_at": None,
                                "selected_spec": spec,
                                "last_seen": now,
                            },
                            fresh_job.get("owner_id"),
                        )
                        return self._create_runtime_on_lease(
                            fresh_job, fresh_lease, provider, spec, allow_recovery=False
                        )
            retire_status = self._retire_absent_lease_if_fresh(
                job,
                lease,
                probe_version=probe_version,
                probe_attempts=probe_attempts,
                from_active=False,
                confirmed_absent=True,
                expected_tunnel_endpoint=pre_ep,
                expected_tunnel_nbh=pre_nbh,
                probe_started_at=probe_started_at,
                now=now,
            )
            if retire_status in {"stale", "locked", "missing", "in_flight"}:
                return self._get("jobs", job["id"], job.get("owner_id")) or job
            try:
                job = self._mark_instance_deleted(job)
            except SchedulerError:
                pass
            return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
        return self._get("jobs", job["id"], job.get("owner_id")) or job

    def _reconcile_rejected_allocation(self, job: dict, *, message: str | None = None) -> dict:
        """Abort lease creation when retirement/bind is rejected and reconcile safely."""
        if job.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
            return job
        if job.get("cancel_requested") or job.get("requested_outcome") == "cancelled":
            return self._cancel_or_cleanup(job)
        fenced = self._enforce_stage_fence(job, allow_unknown=True)
        if fenced is not None:
            return fenced
        lease = self._lease(job)
        exclude_id = lease["id"] if lease else job.get("lease_id")
        runtime_id = lease.get("runtime_id") if lease else None
        conflict = self._conflicting_non_deleted_lease(
            job,
            exclude_lease_id=exclude_id,
            runtime_id=runtime_id,
        )
        if conflict is not None or lease is None:
            if lease is not None:
                inst = self._service_instance_for_job(job)
                flags = self._lease_history_flags(lease, inst, owner_id=job.get("owner_id"))
                if flags["pure_placeholder"] and not flags["ever_active"] and not flags["has_remote_identity"]:
                    self._update(
                        "leases",
                        lease["id"],
                        {"state": "deleted", "current_job_id": None, "last_seen": self.clock()},
                        job.get("owner_id"),
                    )
            return self._set_state(
                job,
                "cleanup_pending",
                message or "GPU allocation conflict detected; entered reconciliation",
                requested_outcome="failed",
                cleanup_final_state="failed",
                error_code="gpu_instance_conflict",
                error_message=message or "GPU instance already bound to an active or non-deleted lease",
            )
        if lease.get("state") == "quarantined" and getattr(self, "store", None) is not None:
            return self._quarantine_missing_identity(job, lease)
        if lease.get("state") in {"cleanup_pending", "releasing", "deleting", "deleted"}:
            return self._handle_fenced_alloc_result(job, lease)
        if lease.get("state") != "unknown" and isinstance(lease.get("handle"), dict) and lease.get("handle"):
            return self._set_state(job, "preparing", "GPU runtime allocation recovered")
        try:
            provider = self._provider(job)
        except Exception as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        if (
            lease.get("state") in {"active", "busy", "unknown"}
            or self._is_from_active_lease(lease)
            or int(lease.get("create_attempts") or 0) > 0
            or lease.get("tunnel_endpoint")
            or lease.get("operation_name")
        ):
            return self._handle_recovery(job, lease, provider)
        if (
            self._is_v020_job(job)
            and job.get("instance_id")
            and job.get("state") == "allocating"
            and lease.get("state") == "creating"
            and int(lease.get("create_attempts") or 0) == 0
            and not lease.get("create_started")
            and not lease.get("selected_spec")
            and not lease.get("create_intent_id")
            and not lease.get("handle")
            and not lease.get("tunnel_endpoint")
            and lease.get("tunnel_nbh") is None
            and not lease.get("operation_name")
            and not lease.get("proxy_url")
            and not lease.get("warm_proof")
            and not lease.get("active_execution")
            and not self._is_from_active_lease(lease)
            and lease.get("owner_job_id") == job.get("id")
            and lease.get("current_job_id") in {None, job.get("id")}
            and hasattr(self.instance_service, "retire_pure_placeholder")
        ):
            try:
                retired_inst = self._instance_call(
                    "retire_pure_placeholder",
                    job.get("owner_id"),
                    job["instance_id"],
                    lease_id=lease["id"],
                )
            except Exception:
                retired_inst = None
            if isinstance(retired_inst, dict) and retired_inst.get("state") == "deleted":
                self._update(
                    "leases",
                    lease["id"],
                    {"state": "deleted", "current_job_id": None, "last_seen": self.clock()},
                    job.get("owner_id"),
                )
                return self._set_state(
                    self._merge_instance(job, retired_inst),
                    "failed",
                    message or "GPU allocation aborted before runtime creation",
                    requested_outcome="failed",
                    error_code="gpu_instance_conflict",
                    error_message=message or "GPU allocation aborted before runtime creation",
                )
        return self._set_state(
            job,
            "cleanup_pending",
            message or "GPU allocation aborted; entered reconciliation",
            requested_outcome="failed",
            cleanup_final_state="failed",
            error_code="gpu_instance_conflict",
            error_message=message or "GPU placeholder retirement rejected; reconciliation required",
        )

    def _allocation(self, job: dict) -> dict:
        fenced = self._enforce_stage_fence(job, allow_unknown=True)
        if fenced is not None:
            return fenced
        if (
            hasattr(getattr(self, "store", None), "retire_absent_placeholder_for_retry")
            and self._is_v020_job(job)
            and job.get("instance_id")
        ):
            bound_lease = self._get("leases", job["lease_id"], job.get("owner_id")) if job.get("lease_id") else None
            existing_lease = bound_lease or self._lease(job)
            dangling_lease_ref = bool(job.get("lease_id")) and bound_lease is None
            incomplete_unstarted_lease = bool(
                existing_lease is not None
                and existing_lease.get("state") == "creating"
                and int(existing_lease.get("create_attempts") or 0) == 0
                and not existing_lease.get("handle")
                and not existing_lease.get("tunnel_endpoint")
                and not existing_lease.get("operation_name")
                and not existing_lease.get("proxy_url")
                and not self._is_from_active_lease(existing_lease)
                and not existing_lease.get("create_intent_id")
                and not existing_lease.get("create_started")
                and not existing_lease.get("selected_spec")
                and (not existing_lease.get("request_id") or not existing_lease.get("runtime_id"))
            )
            if dangling_lease_ref or incomplete_unstarted_lease:
                retired, updated_job = self.store.retire_absent_placeholder_for_retry(
                    job["id"],
                    job["owner_id"],
                    expected_lease_id=job.get("lease_id"),
                    expected_generation=int(job.get("alloc_generation") or 0),
                    total_attempts=0,
                    now=self.clock(),
                )
                if retired and updated_job is not None:
                    return self._allocation(updated_job)
                if updated_job is None:
                    return job
                if int(updated_job.get("alloc_generation") or 0) != int(job.get("alloc_generation") or 0):
                    return self._allocation(updated_job)
                return self._reconcile_rejected_allocation(updated_job)
        if (
            self._is_v020_job(job)
            and not job.get("instance_id")
            and int(job.get("alloc_generation") or 0) > 0
        ):
            job = self._allocate_instance(job)
            if job.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
                return job
        # Persist intent under the short lock, then release it before provider I/O.
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return job
            try:
                lease = self._ensure_lease(job)
            except SchedulerError as exc:
                return self._reconcile_rejected_allocation(
                    self._get("jobs", job["id"], job.get("owner_id")) or job,
                    message=str(exc),
                )
        if job.get("lease_id") != lease["id"]:
            job = self._update("jobs", job["id"], {"lease_id": lease["id"]}, job.get("owner_id"))
        if lease.get("state") == "quarantined" and getattr(self, "store", None) is not None:
            return self._quarantine_missing_identity(job, lease)
        if lease.get("state") in {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted"}:
            return self._begin_failure(
                job, "gpu_allocation_aborted", "GPU allocation entered cleanup before execution; retry after release"
            )
        conflict = self._conflicting_non_deleted_lease(
            job,
            exclude_lease_id=lease["id"],
            runtime_id=lease.get("runtime_id"),
        )
        if conflict is not None:
            return self._reconcile_rejected_allocation(
                job, message="GPU instance already bound to an active or non-deleted lease"
            )
        try:
            job, lease = self._bind_instance_lease(job, lease)
        except SchedulerError as exc:
            return self._reconcile_rejected_allocation(job, message=str(exc))
        try:
            provider = self._provider(job)
        except Exception as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        if lease.get("state") != "unknown" and isinstance(lease.get("handle"), dict) and lease.get("handle"):
            return self._set_state(job, "preparing", "GPU runtime allocation recovered")
        needs_recovery = (
            lease.get("state") in {"active", "busy", "unknown"}
            or self._is_from_active_lease(lease)
            or int(lease.get("create_attempts") or 0) > 0
            or bool(lease.get("tunnel_endpoint"))
            or bool(lease.get("operation_name"))
        )
        if needs_recovery:
            if hasattr(self.store, "claim_create_intent"):
                claim_kwargs: dict[str, Any] = {
                    "spec": {},
                    "max_retries": self.config.create_retries,
                    "now": self.clock(),
                }
                if self._accepts_kwarg(self.store.claim_create_intent, "require_instance"):
                    claim_kwargs["require_instance"] = self._has_sql_instance_service()
                status, fresh_job, fresh_lease = self.store.claim_create_intent(
                    job["id"],
                    job["owner_id"],
                    lease["id"],
                    **claim_kwargs,
                )
                if status in {"locked", "in_flight"} or fresh_job is None or fresh_lease is None:
                    return fresh_job or job
                job, lease = fresh_job, fresh_lease
                if status == "fenced":
                    return self.store.abort_allocation_cas(job, lease, self.clock()) or job
                if status == "has_handle":
                    return self._set_state(job, "preparing", "GPU runtime allocation recovered")
                if status == "exhausted":
                    return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
            return self._handle_recovery(job, lease, provider)
        try:
            if isinstance(provider, DisabledProvider):
                raise ProviderUnavailable(provider.reason)
            spec = (
                {"provider": self._provider_name(job)}
                if self._provider_name(job) == "mock"
                else self._select_spec(job, provider)
            )
        except ProviderUnavailable as exc:
            return self._begin_failure(job, "gpu_provider_unavailable", self._safe_error(exc))
        except Exception as exc:
            return self._begin_failure(job, "gpu_spec_invalid", self._safe_error(exc))
        if hasattr(self.store, "claim_create_intent"):
            claim_kwargs2: dict[str, Any] = {
                "spec": spec,
                "max_retries": self.config.create_retries,
                "now": self.clock(),
            }
            if self._accepts_kwarg(self.store.claim_create_intent, "require_instance"):
                claim_kwargs2["require_instance"] = self._has_sql_instance_service()
            status, fresh_job, fresh_lease = self.store.claim_create_intent(
                job["id"],
                job["owner_id"],
                lease["id"],
                **claim_kwargs2,
            )
            if status in {"locked", "in_flight"} or fresh_job is None or fresh_lease is None:
                return fresh_job or job
            job, lease = fresh_job, fresh_lease
            if status == "fenced":
                return self.store.abort_allocation_cas(job, lease, self.clock()) or job
            if status == "has_handle":
                return self._set_state(job, "preparing", "GPU runtime allocation recovered")
            if status == "recover":
                return self._handle_recovery(job, lease, provider)
            if status == "exhausted":
                return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
        else:
            now_alloc = self.clock()
            if self._create_rpc_is_in_flight(lease, now_alloc):
                return job
            attempts = int(lease.get("create_attempts") or 0)
            if attempts > self.config.create_retries:
                return self._begin_failure(job, "gpu_create_failed", "GPU runtime allocation failed")
            attempts += 1
            lease = self._update(
                "leases",
                lease["id"],
                {
                    "create_attempts": attempts,
                    "create_started": now_alloc,
                    "create_rpc_in_flight": True,
                    "create_rpc_token": uuid.uuid4().hex,
                    "create_rpc_attempt": attempts,
                    "create_rpc_instance_id": lease.get("instance_id") or job.get("instance_id"),
                    "create_rpc_started_at": now_alloc,
                    "create_rpc_finished_at": None,
                    "selected_spec": spec,
                    "last_seen": now_alloc,
                },
                job.get("owner_id"),
            )
        return self._create_runtime_on_lease(job, lease, provider, spec, allow_recovery=True)

    def _create_runtime_on_lease(
        self,
        job: dict,
        lease: dict,
        provider: Any,
        spec: dict,
        *,
        allow_recovery: bool = True,
    ) -> dict:
        if not all(callable(getattr(self.store, name, None)) for name in (
            "commit_create_handle_success", "recover_late_create_handle", "finish_create_rpc_failure",
        )):
            raise SchedulerError("atomic create transitions are required before provider I/O")
        rpc_token = lease.get("create_rpc_token")
        rpc_attempt = lease.get("create_rpc_attempt") or lease.get("create_attempts")
        rpc_inst_id = (
            lease.get("create_rpc_instance_id")
            or lease.get("instance_id")
            or (job.get("instance_id") if job.get("lease_id") in {None, lease["id"]} else None)
        )

        def _finish_failure(code=None, message=None):
            if hasattr(self.store, "finish_create_rpc_failure"):
                return self.store.finish_create_rpc_failure(
                    job["id"], job["owner_id"], lease["id"], rpc_token=rpc_token,
                    expected_attempt=rpc_attempt, expected_instance_id=rpc_inst_id,
                    now=self.clock(), code=code, message=message,
                )
            return "stale", self._get("jobs", job["id"], job.get("owner_id")), lease

        try:
            handle = provider.create(spec, lease["request_id"], job["id"], lease["runtime_id"])
            if not isinstance(handle, dict) or not handle.get("runtime_name"):
                raise ComputeError("provider returned an unknown runtime handle")
            if hasattr(provider, "bind_owner"):
                handle = provider.bind_owner(handle, job.get("owner_id"))
            handle = {
                **handle,
                "runtime_id": lease["runtime_id"],
                "request_id": lease["request_id"],
                "job_id": job["id"],
                "owner_job_id": job["id"],
                "owner_id": job.get("owner_id"),
            }
        except (ProviderUnavailable, ColabAllocationError) as exc:
            code = "gpu_provider_unavailable" if isinstance(exc, ProviderUnavailable) else (
                "gpu_capacity_unavailable" if exc.reason == "NO_RUNTIMES" else "gpu_allocation_failed"
            )
            _, current_job, _ = _finish_failure(code, self._safe_error(exc))
            return current_job or job
        except Exception as exc:
            status, latest_job, latest_lease = _finish_failure()
            if status != "ok":
                return latest_job or job
            if allow_recovery:
                return self._handle_recovery(latest_job, latest_lease, provider)
            self._mark_lease_unknown(latest_lease, latest_job, from_active=False, error=self._safe_error(exc))
            return self._get("jobs", job["id"], job.get("owner_id")) or latest_job

        now_done = self.clock()
        if hasattr(getattr(self, "store", None), "commit_create_handle_success"):
            commit_kwargs: dict[str, Any] = {
                "handle": handle,
                "rpc_token": rpc_token,
                "started_at": now_done,
                "deadline_epoch": lease.get("deadline_epoch") or job.get("deadline_epoch"),
            }
            if self._accepts_kwarg(self.store.commit_create_handle_success, "expected_attempt"):
                commit_kwargs["expected_attempt"] = rpc_attempt
            if self._accepts_kwarg(self.store.commit_create_handle_success, "expected_instance_id"):
                commit_kwargs["expected_instance_id"] = rpc_inst_id
            commit_status, committed_job, committed_lease = self.store.commit_create_handle_success(
                job["id"],
                job.get("owner_id"),
                lease["id"],
                **commit_kwargs,
            )
            if commit_status == "ok":
                return committed_job or (self._get("jobs", job["id"], job.get("owner_id")) or job)
            if commit_status == "fenced":
                return self._handle_fenced_alloc_result(committed_job or job, committed_lease or lease)
            if hasattr(self.store, "recover_late_create_handle"):
                rec_kwargs: dict[str, Any] = {
                    "handle": handle,
                    "instance_id": rpc_inst_id,
                    "now": self.clock(),
                }
                if self._accepts_kwarg(self.store.recover_late_create_handle, "rpc_token"):
                    rec_kwargs["rpc_token"] = rpc_token
                if self._accepts_kwarg(self.store.recover_late_create_handle, "expected_attempt"):
                    rec_kwargs["expected_attempt"] = rpc_attempt
                rec_status, rec_job, rec_lease = self.store.recover_late_create_handle(
                    job.get("id"),
                    job.get("owner_id"),
                    lease["id"],
                    **rec_kwargs,
                )
                if rec_status == "ok":
                    return self._handle_fenced_alloc_result(rec_job or job, rec_lease or lease)

            return self._get("jobs", job["id"], job.get("owner_id")) or job

    def _handle_fenced_alloc_result(self, job: dict, lease: dict) -> dict:
        # SQL repository has already committed the original RPC's scoped disposition.
        # Never derive a failure/cleanup transition from a possibly rebound job here.
        return self._get("jobs", job["id"], job.get("owner_id")) or job

    def _begin_failure(self, job: dict, code: str, message: str) -> dict:
        with self._admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return job
            current = self.store.get("jobs", job["id"], job.get("owner_id"))
            if current and (
                current.get("state") != job.get("state")
                or any(current.get(key) != job.get(key) for key in ("lease_id", "instance_id", "execution_id"))
            ):
                return current
            if current and current.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
                return current
            if current and (current.get("cancel_requested") or current.get("requested_outcome") == "cancelled"):
                return self._cancel_or_cleanup(current)
            target = current or job
            self._clear_owned_execution(target)
            lease = self._lease(target)
            if lease:
                return self._set_state(
                    target,
                    "cleaning_up",
                    message,
                    requested_outcome="failed",
                    error_code=code,
                    error_message=message,
                    cleanup_final_state="failed",
                )
            return self._set_state(
                target, "failed", message, requested_outcome="failed", error_code=code, error_message=message
            )

    def _allocation_failed(self, job: dict, exc: ColabAllocationError) -> dict:
        code = "gpu_capacity_unavailable" if exc.reason == "NO_RUNTIMES" else "gpu_allocation_failed"
        return self._begin_failure(job, code, str(exc))

    def _expired(self, job: dict) -> bool:
        lease = self._lease(job)
        deadline = job.get("deadline_epoch") or (lease or {}).get("deadline_epoch")
        return deadline is not None and self.clock() > float(deadline)

    def _cancel_or_cleanup(self, job: dict) -> dict:
        self._clear_owned_execution(job)
        if self._lease(job):
            return self._set_state(
                job,
                "cleaning_up",
                "GPU cancellation requested",
                requested_outcome="cancelled",
                cleanup_final_state="cancelled",
            )
        return self._set_state(job, "cancelled", "GPU job cancelled", requested_outcome="cancelled")

    def _fence_reason(
        self,
        job: dict,
        lease: dict | None = None,
        instance: dict | None = None,
        *,
        allow_unknown: bool = False,
    ) -> tuple[str, str, str] | None:
        if job.get("cancel_requested") or job.get("requested_outcome") == "cancelled":
            return ("cancelled", "gpu_cancelled", "GPU job cancelled")
        if job.get("manual_stop_requested") or job.get("cleanup_claim_token") or job.get("cleanup_claim"):
            return ("failed", "gpu_instance_stopped", "GPU job stopped or claimed for cleanup")
        blocked_lease_states = {"cleanup_pending", "releasing", "deleting", "deleted"}
        blocked_instance_states = {"deleted", "deleting", "releasing", "cleanup_pending"}
        if not allow_unknown:
            blocked_lease_states.add("unknown")
            blocked_instance_states.add("unknown")
        if lease is not None:
            lstate = lease.get("state")
            if lstate == "quarantined":
                return ("failed", "gpu_runtime_identity_missing", "GPU runtime identity missing")
            if (
                lstate in blocked_lease_states
                or lease.get("manual_stop_requested")
                or lease.get("cleanup_claim_token")
                or lease.get("cleanup_claim")
                or lease.get("runtime_deleted")
                or lease.get("remote_deleted")
            ):
                return ("failed", "gpu_allocation_aborted", "GPU allocation entered cleanup or stopped")
            if lease.get("owner_job_id") not in (None, job["id"]) or lease.get("current_job_id") not in (
                None,
                job["id"],
            ):
                return ("failed", "gpu_lease_conflict", "GPU lease ownership changed")
        if isinstance(instance, dict):
            istate = instance.get("state")
            if istate == "quarantined":
                return ("failed", "gpu_runtime_identity_missing", "GPU runtime identity missing")
            legacy_stub_unknown = (
                istate == "unknown"
                and bool(instance.get("legacy_lease_id"))
                and not instance.get("allocation_key")
                and lease is not None
                and lease.get("state") in {"active", "creating"}
            )
            eff_blocked_instance = blocked_instance_states - {"unknown"} if legacy_stub_unknown else blocked_instance_states
            if (
                istate in eff_blocked_instance
                or instance.get("manual_stop_requested")
                or instance.get("runtime_deleted")
                or instance.get("remote_deleted")
                or instance.get("cleanup_claim_token")
                or instance.get("cleanup_claim")
                or instance.get("current_job_id") not in (None, job["id"])
            ):
                return ("failed", "gpu_instance_stopped", "GPU instance stopped or released")
        return None

    def _enforce_stage_fence(self, job: dict, *, allow_unknown: bool = False) -> dict | None:
        store = getattr(self, "store", None)
        fresh = (self._get("jobs", job["id"], job.get("owner_id")) if store is not None else None) or job
        if fresh.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
            return fresh
        if store is not None:
            raw_lease = (
                self._get("leases", fresh["lease_id"], fresh.get("owner_id"))
                if fresh.get("lease_id")
                else self._lease(fresh)
            )
            inst = self._service_instance_for_job(fresh)
            if inst is None and fresh.get("instance_id"):
                inst = self._get("instances", fresh["instance_id"], fresh.get("owner_id"))
        else:
            raw_lease = self.__dict__["_lease"](fresh) if "_lease" in self.__dict__ else None
            inst = None
        reason = self._fence_reason(fresh, raw_lease, inst, allow_unknown=allow_unknown)
        if reason is None:
            return None
        final_state, code, message = reason
        if code == "gpu_runtime_identity_missing":
            return self._quarantine_missing_identity(fresh, raw_lease)
        if final_state == "cancelled":
            with self._admission_lock(timeout=1.0) as acquired:
                if not acquired:
                    return fresh
                latest = self._get("jobs", fresh["id"], fresh.get("owner_id")) or fresh
                if latest.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
                    return latest
                return self._cancel_or_cleanup(latest)
        return self._begin_failure(fresh, code, message)

    def _prepare(self, job: dict) -> dict:
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        lease = self._lease(job)
        if not lease or not lease.get("handle"):
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        try:
            status = self._provider(job).inspect(self._lease_handle(lease))
        except Exception as exc:
            return self._begin_failure(job, "gpu_runtime_inspect_failed", self._safe_error(exc))
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        if status.get("identity_unverifiable"):
            return self._quarantine_missing_identity(job, lease)
        if status.get("allocation_failed"):
            return self._allocation_failed(job, ColabAllocationError(
                status.get("allocation_error_code"), status.get("allocation_failure_reason")))
        if status.get("deleted"):
            return self._begin_failure(job, "gpu_runtime_missing", "GPU runtime disappeared")
        if status.get("ready"):
            if self._is_v020_job(job) and job.get("instance_id"):
                try:
                    instance = self._instance_call("mark_ready", self._instance_job(job))
                    job = self._merge_instance(job, instance)
                except Exception as exc:
                    return self._begin_failure(job, "gpu_instance_stopped", self._safe_error(exc))
            return self._set_state(job, "loading", "GPU runtime is ready")
        return job

    def _commit_running_locked(self, job: dict, lease: dict, execution_id: str) -> dict:
        """Linearize job_busy and the loading -> running commit under admission_lock."""
        fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
        fresh_lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
        inst = self._service_instance_for_job(fresh_job)
        if inst is None and fresh_job.get("instance_id"):
            inst = self._get("instances", fresh_job["instance_id"], fresh_job.get("owner_id"))
        pre_fenced = self._fence_reason(fresh_job, fresh_lease, inst)
        if pre_fenced is None:
            execution = self._get("executions", execution_id, fresh_job.get("owner_id"))
            j_intent = fresh_job.get("start_intent")
            e_intent = execution.get("start_intent") if isinstance(execution, dict) else None
            if (
                (isinstance(j_intent, dict) and j_intent.get("state") == "aborted")
                or (isinstance(e_intent, dict) and e_intent.get("state") == "aborted")
                or execution is None
                or execution.get("state") in {"failed", "aborted", "cancelled"}
            ):
                pre_fenced = ("failed", "gpu_execution_aborted", "Execution aborted before running")
        if hasattr(getattr(self, "store", None), "commit_running"):
            fn = self.store.commit_running
            kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
            status, committed = fn(
                fresh_job["id"],
                fresh_job["owner_id"],
                lease_id=fresh_lease["id"],
                execution_id=execution_id,
                now=self.clock(),
                **kwargs,
            )
            if (
                status == "ok"
                and self.instance_service is not None
                and self._is_v020_job(committed)
                and committed.get("instance_id")
                and self._get("instances", committed["instance_id"], committed.get("owner_id")) is None
            ):
                try:
                    instance = self._instance_call("job_busy", self._instance_job(committed))
                    committed = self._merge_instance(committed, instance)
                except Exception as exc:
                    self._clear_owned_execution(committed)
                    return self._begin_failure(committed, "gpu_instance_stopped", self._safe_error(exc))
            if status in {"ok", "fenced", "stale"}:
                return committed
            return fresh_job
        if pre_fenced is None and self._is_v020_job(fresh_job) and fresh_job.get("instance_id"):
            try:
                instance = self._instance_call("job_busy", self._instance_job(fresh_job))
                fresh_job = self._merge_instance(fresh_job, instance)
            except Exception as exc:
                self._clear_owned_execution(fresh_job)
                return self._begin_failure(fresh_job, "gpu_instance_stopped", self._safe_error(exc))
        fenced = self._enforce_stage_fence(fresh_job)
        if fenced is not None:
            return fenced
        return self._set_state(fresh_job, "running", "GPU execution started")

    def _load(self, job: dict) -> dict:
        from .repository import extract_runtime_id, is_valid_runtime_handle

        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        lease = self._lease(job)
        if not lease:
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        inst = self._service_instance_for_job(job)
        if inst is None and job.get("instance_id"):
            inst = self._get("instances", job["instance_id"], job.get("owner_id"))
        expected_lease_states = {"active"} if inst is None else {"active", "busy"}
        if (
            lease.get("state") not in expected_lease_states
            or lease.get("owner_job_id") != job["id"]
            or not is_valid_runtime_handle(
                lease.get("handle"),
                owner_id=job.get("owner_id"),
                lease_runtime_id=extract_runtime_id(lease),
                instance_runtime_id=extract_runtime_id(inst) if isinstance(inst, dict) else None,
                provider=lease.get("provider") or self._provider_name(job),
            )
        ):
            self._clear_owned_execution(job)
            return self._begin_failure(
                job,
                "gpu_lease_invalid",
                "GPU lease is not active or handle lacks valid runtime identity",
            )
        execution_id = job.get("execution_id")
        if execution_id:
            execution = self._get("executions", execution_id, job.get("owner_id"))
            j_intent = job.get("start_intent")
            e_intent = execution.get("start_intent") if isinstance(execution, dict) else None
            if (
                execution is None
                or execution.get("state") in {"failed", "aborted", "cancelled"}
                or (isinstance(j_intent, dict) and j_intent.get("state") == "aborted")
                or (isinstance(e_intent, dict) and e_intent.get("state") == "aborted")
            ):
                self._clear_owned_execution(job)
                return self._begin_failure(
                    job, "gpu_execution_aborted", "Execution missing or aborted before running"
                )
            if execution.get("state") != "starting":
                return self._commit_running_locked(job, lease, execution_id)
        # Complete slow model-cache and handle prep BEFORE acquiring dispatch send-right.
        drive_cache = self._model_cache(job)
        handle = self._lease_handle(lease)
        if not execution_id:
            try:
                fn = self.store.prepare_execution
                kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
                job = fn(job["id"], job["owner_id"], lease["id"], self.clock(), **kwargs)
            except (KeyError, ValueError) as exc:
                # prepare_execution raises these only with its own fixed identity-check messages.
                # The lease is owned by this job (see _lease), so cleaning it up here is safe.
                fresh = self._get("jobs", job["id"], job.get("owner_id")) or job
                if fresh.get("cancel_requested") or fresh.get("requested_outcome") == "cancelled":
                    return self._begin_failure(fresh, "gpu_cancelled", "GPU job cancelled")
                return self._begin_failure(job, "gpu_execution_prepare_failed", f"Execution registration refused: {exc}")
            execution_id = job["execution_id"]
        if hasattr(self.store, "claim_execution_dispatch"):
            fn = self.store.claim_execution_dispatch
            kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
            claim_status, claimed_job, claimed_lease = fn(
                job["id"],
                job["owner_id"],
                lease_id=lease["id"],
                execution_id=execution_id,
                now=self.clock(),
                **kwargs,
            )
            if claim_status == "locked":
                return job
            if claim_status == "in_flight":
                return self._reconcile_dispatching_load(claimed_job or job, claimed_lease or lease, execution_id)
            if claim_status in {"stale", "fenced"}:
                return claimed_job or job
            job = claimed_job or job
            if claimed_lease is not None:
                lease = claimed_lease
                handle = claimed_lease.get("handle") or handle
        else:
            fenced = self._enforce_stage_fence(job)
            if fenced is not None:
                return fenced
        try:
            # The Drive token is attached to an in-memory copy only; it is never persisted.
            executor_job = {
                **job,
                "drive_cache": drive_cache,
                "deadline_epoch": job.get("deadline_epoch")
                if job.get("deadline_epoch") is not None
                else lease.get("deadline_epoch"),
            }
            result = self._executor(job).start(handle, executor_job, execution_id)
        except Exception as exc:
            return self._begin_failure(job, "gpu_start_failed", self._safe_error(exc))
        if isinstance(result, dict) and result.get("execution_id") not in {None, execution_id}:
            return self._begin_failure(job, "gpu_execution_id_mismatch", "executor changed execution ID")
        if isinstance(result, dict) and ("remote" in result or self._provider_name(job) == "colab"):
            return self._reconcile_dispatching_load(job, lease, execution_id)
        return self._commit_running_locked(job, lease, execution_id)

    @staticmethod
    def _matching_worker_evidence(result: dict, execution_id: str) -> bool:
        reported = result.get("worker_reported_execution_id")
        if reported is not None:
            return reported == execution_id
        return (
            result.get("worker_process_alive") is True
            and result.get("launcher_execution_id") == execution_id
        )

    def _reconcile_dispatching_load(self, job: dict, lease: dict | None, execution_id: str) -> dict:
        """Reconcile an in-flight `dispatching` start_intent without resending executor.start or starving the queue."""
        now = self.clock()
        if not isinstance(lease, dict):
            lease = self._lease(job)
        handle = self._lease_handle(lease) if isinstance(lease, dict) else None
        if not isinstance(lease, dict) or not handle or not is_valid_runtime_handle(
            handle,
            owner_id=job.get("owner_id"),
            lease_runtime_id=extract_runtime_id(lease),
            provider=lease.get("provider") or self._provider_name(job),
        ):
            self._clear_owned_execution(job)
            return self._begin_failure(
                job,
                "gpu_lease_invalid",
                "GPU lease is invalid or missing handle during dispatch reconciliation",
            )
        executor_job = {
            **job,
            "execution_id": execution_id,
            "deadline_epoch": job.get("deadline_epoch")
            if job.get("deadline_epoch") is not None
            else lease.get("deadline_epoch"),
        }
        attempts = int(job.get("dispatch_poll_attempts") or 0) + 1
        intent = job.get("start_intent") if isinstance(job.get("start_intent"), dict) else {}
        dispatched_at = intent.get("dispatched_at") or intent.get("claimed_at") or now
        try:
            elapsed = now - float(dispatched_at)
        except (TypeError, ValueError):
            elapsed = 0.0
        raw_deadline = job.get("dispatch_recovery_deadline_at")
        try:
            deadline_at = float(raw_deadline) if raw_deadline is not None else (float(dispatched_at) + 30.0)
        except (TypeError, ValueError):
            deadline_at = now + 30.0
        try:
            result = self._executor(job).poll(handle, executor_job)
        except Exception as exc:
            err_msg = self._safe_error(exc)
            non_recoverable = isinstance(exc, ComputeError) and "not recoverable" in str(exc).lower()
            if non_recoverable or attempts >= 3 or elapsed >= 30.0 or now >= deadline_at:
                self._clear_owned_execution(job)
                return self._begin_failure(
                    job,
                    "gpu_dispatch_failed",
                    f"GPU execution dispatch not recoverable: {err_msg}" if err_msg else "GPU execution dispatch failed",
                )
            delay = min(15.0, float(2 ** min(attempts, 3)))
            next_epoch = now + delay
            if hasattr(getattr(self, "store", None), "record_dispatch_poll_backoff"):
                _, updated = self.store.record_dispatch_poll_backoff(
                    job["id"],
                    job["owner_id"],
                    execution_id=execution_id,
                    next_epoch=next_epoch,
                    attempts=attempts,
                    now=now,
                    deadline_at=deadline_at,
                )
                return updated or job
            return self._update(
                "jobs",
                job["id"],
                {
                    "dispatch_next_epoch": next_epoch,
                    "dispatch_poll_attempts": attempts,
                    "dispatch_last_poll_at": now,
                    "dispatch_recovery_deadline_at": deadline_at,
                },
                job.get("owner_id"),
            )
        if isinstance(result, dict):
            identity_matches = result.get("execution_id") in {None, execution_id}
            remote_state = str(result.get("state") or "").strip().lower()
            positive_start = (
                identity_matches and remote_state in {"running", "uploading", "completed"}
                and self._matching_worker_evidence(result, execution_id)
            )
            if positive_start:
                return self._commit_running_locked(job, lease, execution_id)
            if (identity_matches and self._matching_worker_evidence(result, execution_id)
                    and remote_state in {"failed", "cancelled", "aborted", "error"}):
                detail = str(result.get("message") or "")[:1400]
                self._clear_owned_execution(job)
                return self._begin_failure(
                    job,
                    result.get("error_code") or "gpu_start_failed",
                    f"GPU execution failed during dispatch: {detail}" if detail else "GPU execution failed during dispatch",
                )
        if attempts >= 3 or elapsed >= 30.0 or now >= deadline_at:
            self._clear_owned_execution(job)
            return self._begin_failure(
                job,
                "gpu_dispatch_timeout",
                "GPU execution stuck in dispatching after reconciliation attempts",
            )
        delay = min(15.0, float(2 ** min(attempts, 3)))
        next_epoch = now + delay
        if hasattr(getattr(self, "store", None), "record_dispatch_poll_backoff"):
            _, updated = self.store.record_dispatch_poll_backoff(
                job["id"],
                job["owner_id"],
                execution_id=execution_id,
                next_epoch=next_epoch,
                attempts=attempts,
                now=now,
                deadline_at=deadline_at,
            )
            return updated or job
        return self._update(
            "jobs",
            job["id"],
            {
                "dispatch_next_epoch": next_epoch,
                "dispatch_poll_attempts": attempts,
                "dispatch_last_poll_at": now,
                "dispatch_recovery_deadline_at": deadline_at,
            },
            job.get("owner_id"),
        )

    def _model_cache(self, job: dict) -> dict | None:
        """Drive cache for reviewed model weights; None means download from Hugging Face."""
        if os.getenv("GPU_MODEL_CACHE", "drive").strip().lower() != "drive":
            return None
        if self._provider_name(job) != "colab":
            return None
        manifest = (job.get("snapshot") or {}).get("manifest") or {}
        if not manifest.get("weight_files"):
            return None
        try:
            from .drive import DriveStorage

            return DriveStorage(job, self.store).model_cache()
        except Exception as exc:  # noqa: BLE001 - the cache is an optimisation, not a requirement
            logging.getLogger(__name__).warning("model cache unavailable: %s", self._safe_error(exc))
            return None

    @staticmethod
    def _extract_warm_proof_evidence(result: dict) -> dict:
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
        return {
            "worker_reported_execution_id": result.get("worker_reported_execution_id"),
            "warm_saved": result.get("warm_saved"),
            "warm_epoch_id": result.get("warm_epoch_id"),
            "warm_fingerprint": result.get("warm_fingerprint"),
            "runtime_meta": result.get("runtime_meta"),
            "timings": timings,
        }

    def _expected_execution_generation(self, job: dict) -> int | None:
        execution_id = job.get("execution_id")
        if not execution_id:
            return None
        execution = self._get("executions", execution_id, job.get("owner_id"))
        return execution.get("generation") if isinstance(execution, dict) else None

    def _commit_run_failure(
        self,
        job: dict,
        lease: dict,
        code: str,
        message: str,
        *,
        result: dict | None = None,
        timings: dict | None = None,
    ) -> dict:
        execution_id = job.get("execution_id")
        if execution_id and (lease.get("handle") or int(lease.get("create_attempts") or 0)):
            proof = self._extract_warm_proof_evidence(result) if isinstance(result, dict) else {"warm_saved": False}
            patch = {
                "requested_outcome": "failed",
                "error_code": code,
                "error_message": message,
                "cleanup_final_state": "failed",
            }
            if timings:
                patch["worker_timings"] = timings
            fn = self.store.commit_execution_terminal
            kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
            return fn(
                job["id"],
                job["owner_id"],
                execution_id=execution_id,
                terminal_state="cleaning_up",
                message=message,
                patch=patch,
                now=self.clock(),
                proof=proof,
                expected_state=job.get("state"),
                expected_generation=self._expected_execution_generation(job),
                **kwargs,
            )
        return self._begin_failure(job, code, message)

    def _commit_run_success(
        self,
        job: dict,
        *,
        outputs: list[dict],
        output_bytes: int,
        timings: dict | None,
        proof: dict,
    ) -> dict:
        execution_id = job.get("execution_id")
        patch = {
            "outputs": outputs,
            "output_bytes": output_bytes,
            "worker_timings": timings or {},
        }
        if execution_id:
            fn = self.store.commit_execution_terminal
            kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
            return fn(
                job["id"],
                job["owner_id"],
                execution_id=execution_id,
                terminal_state="uploading",
                message="GPU execution completed",
                patch=patch,
                now=self.clock(),
                proof=proof,
                expected_state=job.get("state"),
                expected_generation=self._expected_execution_generation(job),
                **kwargs,
            )
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        return self._set_state(job, "uploading", "GPU execution completed", **patch)

    def _commit_run_step(
        self,
        job: dict,
        *,
        execution_id: str | None,
        expected_gen: int | None,
        patch: dict[str, Any] | None = None,
        message: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[bool, dict]:
        if hasattr(getattr(self, "store", None), "commit_run_progress"):
            fn = self.store.commit_run_progress
            kwargs = {"require_instance": self._has_sql_instance_service()} if self._accepts_kwarg(fn, "require_instance") else {}
            status, updated = fn(
                job["id"],
                job["owner_id"],
                execution_id=execution_id,
                now=self.clock(),
                patch=patch,
                message=message,
                diagnostics=diagnostics,
                expected_state="running",
                expected_generation=expected_gen,
                **kwargs,
            )
            return status == "ok", updated
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return False, fenced
        updated = job
        if message is not None:
            updated = self._set_state(job, "running", message, **(patch or {}))
        elif patch:
            updated = self._update("jobs", job["id"], patch, job.get("owner_id"))
        if diagnostics is not None:
            updated = self.store.record_gpu_diagnostics(updated, diagnostics) or updated
        return True, updated

    def _run(self, job: dict) -> dict:
        if float(job.get('poll_retry_at') or 0) > self.clock():
            return job
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        lease = self._lease(job)
        if not lease:
            return self._begin_failure(job, "gpu_lease_missing", "GPU lease is missing")
        execution_id = job.get("execution_id")
        expected_gen = self._expected_execution_generation(job)
        try:
            executor_job = {
                **job,
                "deadline_epoch": job.get("deadline_epoch")
                if job.get("deadline_epoch") is not None
                else lease.get("deadline_epoch"),
            }
            result = self._executor(job).poll(self._lease_handle(lease), executor_job)
        except Exception as exc:
            from .mcp import MCPTransient

            retries = int(job.get('poll_retry_count') or 0) + 1
            if isinstance(exc, MCPTransient) and retries <= 5:
                _, updated = self._commit_run_step(
                    job,
                    execution_id=execution_id,
                    expected_gen=expected_gen,
                    patch={"poll_retry_count": retries, "poll_retry_at": self.clock() + min(2**retries, 30)},
                    message="GPU connection interrupted; retry pending",
                )
                return updated
            return self._commit_run_failure(job, lease, "gpu_poll_failed", self._safe_error(exc))
        reported = result.get("worker_reported_execution_id")
        if (reported is not None and reported != execution_id) or (
            result.get("state") == "running"
            and self._provider_name(job) == "colab"
            and reported != execution_id
        ) or result.get("state") == "unknown":
            return self._get("jobs", job["id"], job.get("owner_id")) or job
        if job.get('poll_retry_count') or job.get('poll_retry_at'):
            ok, updated = self._commit_run_step(
                job,
                execution_id=execution_id,
                expected_gen=expected_gen,
                patch={"poll_retry_count": 0, "poll_retry_at": 0},
                message="GPU connection recovered",
            )
            if not ok:
                return updated
            job = updated
        if result.get('execution_id') not in {None, job.get('execution_id')}:
            return self._commit_run_failure(
                job, lease, 'gpu_execution_id_mismatch', 'poll execution identity mismatch', result=result
            )
        if result.get("gpu_diagnostics") and not job.get("gpu_reported"):
            ok, job = self._commit_run_step(
                job,
                execution_id=execution_id,
                expected_gen=expected_gen,
                diagnostics=result["gpu_diagnostics"],
            )
            if not ok:
                return job
        state = result.get("state")
        timings = result.get("timings") if isinstance(result.get("timings"), dict) else None
        if state == "running":
            stage = result.get("stage")
            if stage in _STAGE_TEXT and stage != job.get("stage"):
                _, updated = self._commit_run_step(
                    job,
                    execution_id=execution_id,
                    expected_gen=expected_gen,
                    patch={"stage": stage, "worker_timings": timings or {}},
                    message=_STAGE_TEXT[stage],
                )
                return updated
            _, updated = self._commit_run_step(job, execution_id=execution_id, expected_gen=expected_gen)
            return updated
        if state == "failed":
            detail = str(result.get("message") or "")[:1400]
            return self._commit_run_failure(
                job,
                lease,
                result.get("error_code", "gpu_execution_failed"),
                "GPU execution failed" + (": " + detail if detail else ""),
                result=result,
                timings=timings,
            )
        if state != "completed":
            return self._commit_run_failure(
                job, lease, "gpu_invalid_poll", "invalid GPU poll state", result=result
            )
        execution_id = job.get("execution_id")
        proof = self._extract_warm_proof_evidence(result)
        if 'artifacts' in result:
            from .output_protocol import verify_output
            try:
                manifest = result['artifacts']
                if not isinstance(manifest, list) or not 1 <= len(manifest) <= 100:
                    raise ComputeError('invalid artifact manifest count')
                if result.get('execution_id') != execution_id:
                    raise ComputeError('artifact execution mismatch')
                verified = [verify_output(job, item)[0] for item in manifest]
                return self._commit_run_success(
                    job,
                    outputs=verified,
                    output_bytes=sum(item['size_bytes'] for item in verified),
                    timings=timings,
                    proof=proof,
                )
            except Exception as exc:
                return self._commit_run_failure(
                    job, lease, 'gpu_output_invalid', self._safe_error(exc), result=result, timings=timings
                )
        raw_outputs = result.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            return self._commit_run_failure(
                job, lease, "gpu_empty_outputs", "GPU execution produced no outputs", result=result, timings=timings
            )
        # Job snapshots are JSON columns. Keep binary provider outputs durable
        # without ever putting raw bytes or more than 100MB into the database.
        outputs, total = [], 0
        try:
            for output in raw_outputs:
                if not isinstance(output, dict):
                    raise ComputeError("GPU output format invalid")
                item = dict(output)
                content = item.get("content", b"")
                if item.get("content_encoding") == "base64":
                    content = base64.b64decode(content, validate=True)
                elif isinstance(content, str):
                    content = content.encode()
                if not isinstance(content, (bytes, bytearray)) or not content:
                    raise ComputeError("GPU output is empty")
                total += len(content)
                if total > MAX_OUTPUT_BYTES:
                    raise ComputeError("GPU output exceeds 100MB limit")
                item["content"] = base64.b64encode(bytes(content)).decode("ascii")
                item["content_encoding"] = "base64"
                outputs.append(item)
        except (ValueError, TypeError, ComputeError) as exc:
            return self._commit_run_failure(
                job, lease, "gpu_output_invalid", self._safe_error(exc), result=result, timings=timings
            )
        return self._commit_run_success(
            job,
            outputs=outputs,
            output_bytes=total,
            timings=timings,
            proof=proof,
        )

    def _result_metadata(self, job: dict) -> dict:
        snapshot = job.get("snapshot") if isinstance(job.get("snapshot"), dict) else {}
        raw = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return {
            "job_id": job["id"],
            "snapshot_sha256": snapshot.get("workflow_sha256") or hashlib.sha256(raw).hexdigest(),
            "workflow_key": job.get("workflow_key"),
            "workflow_version": job.get("workflow_version"),
            "model_id": job.get("model_id"),
            "model_revision": snapshot.get("model_revision"),
            "resolved_params": snapshot.get("resolved_params", {}),
            "system_smoke": snapshot.get("system_smoke") is True,
            "gpu_reported": job.get("gpu_reported"),
            "gpu_diagnostics": job.get("gpu_diagnostics"),
        }

    def _write_local_metadata(self, job: dict, metadata: dict) -> dict:
        from app.config import settings

        root = Path(settings.data_dir) / "gpu-results" / job["owner_id"] / job["id"]
        root.mkdir(parents=True, exist_ok=True)
        target = root / "metadata.json"
        content = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        tmp = target.with_suffix(".partial")
        tmp.write_bytes(content)
        tmp.replace(target)
        return {
            "filename": "metadata.json",
            "storage_key": str(target),
            "sha256": hashlib.sha256(content).hexdigest(),
            "mime": "application/json",
            "metadata": metadata,
        }

    def _upload(self, job: dict) -> dict:
        fenced = self._enforce_stage_fence(job)
        if fenced is not None:
            return fenced
        outputs, assets = job.get("outputs") or [], []
        report = None
        if self.storage_factory:
            storage = self._factory(self.storage_factory, job)
        elif os.getenv("GPU_STORAGE", "local").strip().lower() == "drive":
            from .drive import DriveStorage

            storage = DriveStorage(job, self.store)
        else:
            storage = None
        result_metadata = self._result_metadata(job)
        metadata_record = None
        try:
            if job.get('prep'):
                for output in outputs:
                    if 'storage' in output:
                        from .output_protocol import read_diagnostic_report
                        report=read_diagnostic_report(job,output)
                    elif isinstance(output.get('metadata'),dict):
                        report=output['metadata']
                if not isinstance(report,dict):
                    raise ComputeError('prep output missing verified diagnostic report')
                outputs=[]
                result_metadata['prep_report']=report
            for index, output in enumerate(outputs):
                if 'storage' in output:
                    from .output_protocol import commit_output
                    assets.append(commit_output(self.db_store,job,output,result_metadata,storage=storage))
                    continue
                content = output.get("content", b"")
                if output.get("content_encoding") == "base64":
                    content = base64.b64decode(content, validate=True)
                elif isinstance(content, str):
                    content = content.encode()
                if (
                    not isinstance(content, (bytes, bytearray))
                    or not content
                    or len(content) > MAX_OUTPUT_BYTES
                ):
                    raise ComputeError("GPU output exceeds 100MB or is empty")
                metadata = {**result_metadata, **dict(output.get("metadata") or {})}
                if self._provider_name(job) == "mock":
                    metadata["mock"] = True
                else:
                    metadata.pop("mock", None)
                if storage is not None:
                    key = f"outputs/{job['id']}/{index}-{output.get('filename', 'output')}"
                    cloud_record = storage.put(
                        key, bytes(content), output.get("mime", "application/octet-stream")
                    )
                    if not storage.verify(cloud_record, bytes(content)):
                        raise ComputeError("stored GPU output failed verification")
                    metadata = {**metadata, "cloud_record": cloud_record}
                from app.storage import save_asset

                assets.append(save_asset(
                    self.db_store,
                    job["owner_id"],
                    job.get("project_id"),
                    bytes(content),
                    output.get("filename", "output"),
                    job["id"],
                    metadata,
                ))
            metadata_content = json.dumps(
                result_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            if storage is not None:
                metadata_record = storage.put(
                    f"outputs/{job['id']}/metadata.json", metadata_content, "application/json"
                )
                if not storage.verify(metadata_record, metadata_content):
                    raise ComputeError("stored GPU metadata failed verification")
            else:
                metadata_record = self._write_local_metadata(job, result_metadata)
        except Exception as exc:
            attempts = int(job.get("upload_attempts") or 0) + 1
            if attempts >= 3:
                return self._begin_failure(job, "gpu_upload_failed", self._safe_error(exc))
            return self._update("jobs", job["id"], {"upload_attempts": attempts}, job.get("owner_id"))
        # Drive metadata file IDs are not authenticated application asset IDs.
        output_asset_ids = [a.get("id") for a in assets if a.get("id")]
        if metadata_record:
            assets.append(metadata_record)
        return self._set_state(
            job,
            "cleaning_up",
            "GPU outputs persisted",
            output_asset_ids=output_asset_ids,
            persisted_outputs=assets,
            **({'prep_report':report} if report is not None else {}),
            cleanup_final_state="completed",
        )

    def _measure_spend(self, job: dict, lease: dict | None) -> dict:
        """Measure the real Google spend for this job when the balance is observable.

        Attribution is only exclusive when no other lease on the same account was
        active, otherwise the delta may include a sibling job and is flagged.
        """
        if self._provider_name(job) == "mock" or not job.get("account_id"):
            return {}
        owner_id = job.get("owner_id")
        if not owner_id:
            return {}
        from . import ccu

        before = job.get("ccu_before")
        after = ccu.maybe_sample_balance(self.store, owner_id, job.get("account_id"))
        contended = ccu.active_lease_count(
            self.store,
            owner_id,
            job.get("account_id"),
            exclude_lease_id=(lease or {}).get("id"),
        ) > 0
        patch: dict = {"ccu_measured_at": self.clock()}
        if after:
            patch["ccu_after"] = after["balance"]
            patch["ccu_after_at"] = after["at"]
        measured = ccu.reconcile(before, patch.get("ccu_after"))
        if measured is not None:
            patch["actual_units"] = measured
            patch["ccu_attribution"] = "contended" if contended else "exclusive"
            patch["ccu_source"] = "google_balance"
        elif before is None:
            patch["ccu_source"] = "baseline_missing"
        else:
            patch["ccu_source"] = "after_missing"
        return patch

    def _record_usage(self, job: dict, lease: dict, stopped_at: float) -> None:
        if any(row.get("job_id") == job.get("id") and row.get("execution_id") == job.get("execution_id")
               for row in self._list("usage", job.get("owner_id"))):
            return
        started = float(job.get("started_epoch") or lease.get("created_epoch") or stopped_at)
        elapsed_minutes = max(0.0, stopped_at - started) / 60.0
        is_mock = self._provider_name(job) == "mock"
        rate = float(job.get("unit_rate") or self.config.unit_rate or 0)
        estimate = 0.0 if is_mock else elapsed_minutes * rate
        spend: dict = {}
        if not is_mock:
            try:
                spend = self._measure_spend(job, lease)
            except Exception as exc:  # noqa: BLE001 - never fail a finished job on accounting
                logging.getLogger(__name__).warning("ccu reconciliation failed: %s", type(exc).__name__)
                spend = {}
        measured = spend.get("actual_units")
        payload = {
            "job_id": job["id"],
            "execution_id": job.get('execution_id'),
            "provider": self._provider_name(job),
            "mock": is_mock,
            "units": estimate,
            "units_estimate": estimate,
            "units_actual": measured if not is_mock else 0.0,
            "ccu_before": job.get("ccu_before"),
            "ccu_after": spend.get("ccu_after"),
            "ccu_attribution": spend.get("ccu_attribution"),
            "ccu_source": spend.get("ccu_source", "not_sampled" if not is_mock else "simulated"),
            "source": "google_balance" if measured is not None else ("simulated" if is_mock else "administrator_estimate"),
            "unit_rate": rate,
            "elapsed_minutes": elapsed_minutes,
            "started_at": started,
            "stopped_at": stopped_at,
            "created_epoch": stopped_at,
            "simulated": is_mock,
            "simulated_units": float(job.get("simulated_units") or 0),
        }
        self.store.record_usage_once(job["owner_id"], payload)
        self._update(
            "jobs",
            job["id"],
            {
                "actual_units": 0.0 if is_mock else measured,
                "stopped_at": stopped_at,
                "runtime_deleted": True,
                **{key: value for key, value in spend.items() if key.startswith("ccu_")},
            },
            job.get("owner_id"),
        )

    def _finish_cleanup(self, job: dict, state: str, message: str, **patch) -> dict:
        if state in {"completed", "cancelled"} and job.get("error_code") in {
            "gpu_cleanup_pending", "gpu_instance_completion_pending",
        }:
            patch.update(error_code=None, error_message=None)
        return self._set_state(job, state, message, cleanup_next_epoch=0, **patch)

    def _cleanup(self, job: dict) -> dict:
        lease = self._lease(job)
        if lease and lease.get("state") == "quarantined":
            return self._quarantine_missing_identity(job, lease)
        job = self._trusted_plugin_cleanup(job, lease)
        if lease and job.get("batch_id") and job.get("cleanup_final_state") == "completed":
            waiting = [j for j in self._list("jobs", job["owner_id"])
                       if j.get("batch_id") == job["batch_id"] and j["state"] not in TERMINAL_STATES
                       and j["id"] != job["id"] and not j.get("cancel_requested")]
            waiting.sort(key=lambda j: j.get("queue_order", 0))
            next_job = waiting[0] if waiting else None
            if next_job:
                with self.db_store.dispatch_lock() as acquired:
                    if not acquired:
                        return job
                    transferred = self.store.handoff_lease(
                        job["owner_id"],
                        job["id"],
                        next_job["id"],
                        lease["id"],
                        self.clock(),
                        guard_mode=self._guard_mode(),
                    )
                if transferred:
                    self._event(job,'completed','Mock outputs verified; lease reused')
                    return self._get('jobs',job['id'],job['owner_id'])
        final_state = job.get("cleanup_final_state") or job.get("requested_outcome") or "failed"
        if lease is not None or (self._is_v020_job(job) and job.get("instance_id")):
            conflicting = self._conflicting_non_deleted_lease(
                job,
                exclude_lease_id=(lease or {}).get("id") or job.get("lease_id"),
                runtime_id=extract_runtime_id(lease) if isinstance(lease, dict) else None,
                tunnel_endpoint=extract_tunnel_endpoint(lease) if isinstance(lease, dict) else None,
                handle=lease.get("handle") if isinstance(lease, dict) else None,
                lease=lease if isinstance(lease, dict) else None,
            )
            if conflicting is not None:
                return self._set_state(
                    job,
                    "cleanup_pending",
                    "GPU instance is bound to another non-deleted lease; waiting for reconciliation",
                    error_code=job.get("error_code") or "gpu_instance_conflict",
                    cleanup_next_epoch=self.clock() + 5.0,
                )
        if self._is_v020_job(job) and job.get("instance_id"):
            if job.get("error_code") == "gpu_instance_conflict" and lease is None:
                return self._finish_cleanup(job, final_state, "GPU allocation conflict closed")
            try:
                job, instance = self._complete_instance(job, final_state == "completed")
            except SchedulerError as exc:
                return self._set_state(
                    job,
                    "cleanup_pending",
                    "GPU instance completion is pending",
                    error_code=job.get("error_code") or "gpu_instance_completion_pending",
                    error_message=job.get("error_message") or str(exc),
                )
            if isinstance(instance, dict) and instance.get("state") == "retained_idle":
                fresh_lease = self._lease(job)
                if isinstance(fresh_lease, dict) and hasattr(self.store, "finalize_retained_cleanup"):
                    finalize_kwargs: dict[str, Any] = {
                        "lease_id": fresh_lease["id"],
                        "now": self.clock(),
                        "final_state": final_state,
                        "instance_id": job.get("instance_id"),
                        "expected_instance_state": instance.get("state"),
                        "idle_deadline_at": instance.get("idle_deadline_at"),
                        "hard_deadline_at": instance.get("hard_deadline_at"),
                        "legacy_mock_idle": False,
                    }
                    if self._accepts_kwarg(self.store.finalize_retained_cleanup, "require_instance"):
                        finalize_kwargs["require_instance"] = self._has_sql_instance_service()
                    r_status, finalized_job, r_lease = self.store.finalize_retained_cleanup(
                        job["id"],
                        job["owner_id"],
                        **finalize_kwargs,
                    )
                    if r_status == "locked":
                        return job
                    if r_status in {"ok", "already_terminal"} and finalized_job is not None:
                        return finalized_job
                    if r_status in {"aborted", "expired"}:
                        if r_lease is not None:
                            lease = r_lease
                        else:
                            lease = self._lease(job) or lease
                        fresh_inst = self._service_instance_for_job(finalized_job or job)
                        if r_status == "aborted" and (
                            (isinstance(lease, dict) and (lease.get("cleanup_claim_token") or lease.get("cleanup_claim") or lease.get("state") == "deleting"))
                            or (isinstance(fresh_inst, dict) and fresh_inst.get("cleanup_claim_token"))
                        ):
                            return self._set_state(
                                finalized_job or job,
                                "cleanup_pending",
                                "GPU cleanup claimed before retention finalized",
                                error_code="gpu_cleanup_pending",
                                cleanup_next_epoch=self.clock() + 5.0,
                            )
                else:
                    with self._admission_lock(timeout=1.0) as acquired:
                        if not acquired:
                            return job
                        latest_job = self._get("jobs", job["id"], job.get("owner_id")) or job
                        latest_lease = self._lease(latest_job)
                        latest_inst = self._service_instance_for_job(latest_job) or instance
                        retain_deadline = (
                            latest_inst.get("hard_deadline_at")
                            if isinstance(latest_inst, dict) and latest_inst.get("hard_deadline_at") is not None
                            else (latest_lease.get("deadline_epoch") if isinstance(latest_lease, dict) else None)
                        )
                        lease_can_idle = bool(
                            isinstance(latest_lease, dict)
                            and isinstance(latest_inst, dict)
                            and latest_inst.get("state") == "retained_idle"
                            and not latest_inst.get("cleanup_claim_token")
                            and not latest_inst.get("manual_stop_requested")
                            and not latest_inst.get("runtime_deleted")
                            and not latest_inst.get("remote_deleted")
                            and latest_lease.get("owner_id") == latest_job.get("owner_id")
                            and latest_lease.get("state") not in {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted", "unknown"}
                            and not latest_lease.get("cleanup_claim_token")
                            and not latest_lease.get("cleanup_claim")
                            and not latest_lease.get("manual_stop_requested")
                            and not latest_lease.get("runtime_deleted")
                            and not latest_lease.get("remote_deleted")
                            and latest_lease.get("owner_job_id") == latest_job["id"]
                            and latest_lease.get("current_job_id") in {None, latest_job["id"]}
                            and isinstance(latest_lease.get("handle"), dict)
                            and latest_lease.get("handle")
                            and (retain_deadline is None or self.clock() < float(retain_deadline))
                        )
                        if lease_can_idle:
                            lease_patch = {"state": "idle", "current_job_id": None, "last_seen": self.clock()}
                            if latest_inst.get("idle_deadline_at") is not None:
                                lease_patch["idle_deadline_at"] = latest_inst["idle_deadline_at"]
                            if latest_inst.get("hard_deadline_at") is not None:
                                lease_patch["deadline_epoch"] = latest_inst["hard_deadline_at"]
                            self._update("leases", latest_lease["id"], lease_patch, latest_job.get("owner_id"))
                            if latest_job.get("execution_id"):
                                self._update(
                                    "executions",
                                    latest_job["execution_id"],
                                    {"state": "completed", "finished_at": self.clock()},
                                    latest_job.get("owner_id"),
                                )
                            return self._finish_cleanup(
                                latest_job,
                                final_state,
                                "GPU cleanup completed; instance retained",
                                runtime_deleted=False,
                            )
                        lease = latest_lease or lease
        policy = job.get("execution_policy") or {}
        if (
            lease
            and job.get("batch_id")
            and final_state == "completed"
            and policy.get("mode") == "smart"
            and self._provider_name(job) == "mock"
            and not self._is_v020_job(job)
            and lease.get("state") in {"active", "busy"}
            and not lease.get("cleanup_claim_token")
            and not lease.get("cleanup_claim")
            and not lease.get("manual_stop_requested")
            and not lease.get("runtime_deleted")
            and not lease.get("remote_deleted")
            and lease.get("owner_job_id") == job["id"]
            and lease.get("current_job_id") in {None, job["id"]}
            and isinstance(lease.get("handle"), dict)
            and lease.get("handle")
            and self.clock() < float(lease.get("deadline_epoch") or 0)
        ):
            idle_dl = min_deadlines(
                self.clock() + int(policy.get("idle_seconds", 120)),
                lease.get("deadline_epoch"),
                lease.get("hard_deadline_at"),
            )
            if hasattr(self.store, "finalize_retained_cleanup"):
                r_status, finalized_job, r_lease = self.store.finalize_retained_cleanup(
                    job["id"],
                    job["owner_id"],
                    lease_id=lease["id"],
                    now=self.clock(),
                    final_state="completed",
                    idle_deadline_at=idle_dl,
                    compatibility_signature=job.get("compatibility_signature"),
                    legacy_mock_idle=True,
                )
                if r_status == "locked":
                    return job
                if r_status in {"ok", "already_terminal"} and finalized_job is not None:
                    return finalized_job
                if r_lease is not None:
                    lease = r_lease
                if r_status == "aborted" and lease and (
                    lease.get("cleanup_claim_token") or lease.get("cleanup_claim") or lease.get("state") == "deleting"
                ):
                    return self._set_state(
                        finalized_job or job,
                        "cleanup_pending",
                        "GPU cleanup claimed before retention finalized",
                        error_code="gpu_cleanup_pending",
                        cleanup_next_epoch=self.clock() + 5.0,
                    )
            else:
                with self._admission_lock(timeout=1.0) as acquired:
                    if not acquired:
                        return job
                    fresh_job = self._get("jobs", job["id"], job.get("owner_id")) or job
                    fresh_lease = self._get("leases", lease["id"], job.get("owner_id")) or lease
                    if (
                        fresh_lease.get("state") in {"active", "busy"}
                        and not fresh_lease.get("cleanup_claim_token")
                        and not fresh_lease.get("cleanup_claim")
                        and not fresh_lease.get("manual_stop_requested")
                        and not fresh_lease.get("runtime_deleted")
                        and not fresh_lease.get("remote_deleted")
                        and fresh_lease.get("owner_job_id") == fresh_job["id"]
                        and fresh_lease.get("current_job_id") in {None, fresh_job["id"]}
                    ):
                        self._update(
                            "leases",
                            fresh_lease["id"],
                            {
                                "state": "idle",
                                "current_job_id": None,
                                "compatibility_signature": fresh_job.get("compatibility_signature"),
                                "idle_deadline_at": idle_dl,
                                "last_active_at": self.clock(),
                            },
                            fresh_job["owner_id"],
                        )
                        if fresh_job.get('execution_id'):
                            self._update('executions', fresh_job['execution_id'], {'state': 'completed', 'finished_at': self.clock()}, fresh_job['owner_id'])
                        return self._set_state(fresh_job, 'completed', 'Mock outputs verified; idle lease retained')
                    lease = fresh_lease
        if not lease:
            return self._finish_cleanup(job, final_state, "GPU cleanup completed")
        if lease.get("state") == "deleted":
            job = self._close_unstarted_instance(job, lease)
            if lease.get("remote_deleted") and not self._is_from_active_lease(lease):
                existing_inst = self._service_instance_for_job(job)
                if existing_inst and existing_inst.get("state") in {"allocating", "unknown", "releasing", "cleanup_pending"}:
                    try:
                        job = self._mark_instance_deleted(job)
                    except SchedulerError:
                        pass
            self._record_usage(job, lease, self.clock())
            return self._finish_cleanup(job, final_state, "GPU cleanup completed")
        next_epoch = float(lease.get("cleanup_next_epoch") or 0)
        if self.clock() < next_epoch:
            return job
        flags = self._lease_history_flags(lease, self._service_instance_for_job(job), owner_id=job.get("owner_id"))
        if flags["pure_placeholder"]:
            retired, _ = self.store.retire_pure_orphan_placeholder_cas(
                job["owner_id"], lease["id"], instance_id=lease.get("instance_id") or job.get("instance_id"),
                cleanup_job_id=job["id"], expected_lease_version=lease.get("version"), now=self.clock(),
            )
            if retired == "retired":
                lease = self._get("leases", lease["id"], job.get("owner_id"))
                self._record_usage(job, lease, self.clock())
                return self._finish_cleanup(job, final_state, "GPU cleanup completed")
            if retired in {"locked", "stale", "missing", "conflict"}:
                return self._get("jobs", job["id"], job.get("owner_id")) or job
            # Evidence committed after the initial snapshot: reconcile remotely below.
        provider = self._provider(job)
        handle = self._lease_handle(lease) or self._intent_handle(job, lease)
        if handle is None:
            self._update(
                "leases",
                lease["id"],
                {
                    "state": "quarantined",
                    "cleanup_claim_token": None,
                    "cleanup_claimed_at": None,
                    "last_error": "lease intent is incomplete",
                    "last_seen": self.clock(),
                },
                job.get("owner_id"),
            )
            return self._set_state(job, "failed", "GPU cleanup was refused", error_code="gpu_cleanup_refused")
        claim_token = None
        if hasattr(getattr(self, "store", None), "claim_job_cleanup_lease"):
            claim_token = uuid.uuid4().hex
            claim_status, claimed_lease = self.store.claim_job_cleanup_lease(
                job.get("owner_id"),
                job["id"],
                lease["id"],
                claim_token,
                self.clock(),
                expected_lease_version=lease.get("version"),
            )
            if claim_status == "already_deleted":
                lease = claimed_lease or self._lease(job) or {**lease, "state": "deleted"}
                job = self._close_unstarted_instance(job, lease)
                self._record_usage(job, lease, self.clock())
                return self._finish_cleanup(job, final_state, "GPU cleanup completed")
            if claim_status == "conflict":
                return self._set_state(
                    job,
                    "cleanup_pending",
                    "GPU instance is bound to another non-deleted lease; waiting for reconciliation",
                    error_code=job.get("error_code") or "gpu_instance_conflict",
                    cleanup_next_epoch=self.clock() + 5.0,
                )
            if claim_status in {"locked", "stale", "missing", "in_flight"}:
                return self._get("jobs", job["id"], job.get("owner_id")) or job
            if isinstance(claimed_lease, dict):
                lease = claimed_lease
                handle = self._lease_handle(lease) or self._intent_handle(job, lease)
                if handle is None:
                    self.store.finish_job_cleanup_lease(
                        job.get("owner_id"),
                        job["id"],
                        lease["id"],
                        claim_token,
                        self.clock(),
                        confirmed=False,
                        error="lease intent is incomplete",
                        attempts=int(lease.get("delete_attempts") or 0),
                    )
                    self._update(
                        "leases",
                        lease["id"],
                        {
                            "state": "quarantined",
                            "cleanup_claim_token": None,
                            "cleanup_claimed_at": None,
                            "last_error": "lease intent is incomplete",
                            "last_seen": self.clock(),
                        },
                        job.get("owner_id"),
                    )
                    return self._set_state(job, "failed", "GPU cleanup was refused", error_code="gpu_cleanup_refused")
        try:
            try:
                status = provider.inspect(handle)
            except Exception:
                status = {}
            if isinstance(status, dict) and status.get("identity_unverifiable"):
                return self._quarantine_missing_identity(job, lease)
            confirmed = self._inspect_confirms_absent(status)
            if not confirmed:
                response = provider.delete(handle)
                # delete() must return True (per ComputeProvider contract) or an unambiguously verified proof dict.
                confirmed = self._delete_confirms_absent(response)
            error = None
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)
        attempts = int(lease.get("delete_attempts") or 0) + 1
        if confirmed:
            try:
                job = self._mark_instance_deleted(job)
            except SchedulerError as exc:
                confirmed = False
                error = str(exc)
        delay = min(
            self.config.cleanup_retry_max_seconds,
            self.config.cleanup_retry_base_seconds * (2 ** max(0, attempts - 1)),
        )
        if claim_token is not None and hasattr(getattr(self, "store", None), "finish_job_cleanup_lease"):
            fin_status, fin_lease = self.store.finish_job_cleanup_lease(
                job.get("owner_id"),
                job["id"],
                lease["id"],
                claim_token,
                self.clock(),
                confirmed=confirmed,
                error=error,
                attempts=attempts,
                retry_delay=delay,
            )
            if fin_status == "deleted":
                self._record_usage(job, fin_lease or lease, self.clock())
                return self._finish_cleanup(job, final_state, "GPU cleanup completed")
            if fin_status in {"locked", "stale", "missing"}:
                return self._get("jobs", job["id"], job.get("owner_id")) or job
            return self._set_state(
                job,
                "cleanup_pending",
                "GPU cleanup is pending",
                cleanup_attempts=attempts,
                cleanup_next_epoch=self.clock() + delay,
                error_code=job.get("error_code") or "gpu_cleanup_pending",
            )
        if not confirmed:
            self._update(
                "leases",
                lease["id"],
                {
                    "state": "cleanup_pending",
                    "delete_attempts": attempts,
                    "cleanup_next_epoch": self.clock() + delay,
                    "last_error": error or "delete not confirmed",
                    "last_seen": self.clock(),
                },
                job.get("owner_id"),
            )
            return self._set_state(
                job,
                "cleanup_pending",
                "GPU cleanup is pending",
                cleanup_attempts=attempts,
                cleanup_next_epoch=self.clock() + delay,
                error_code=job.get("error_code") or "gpu_cleanup_pending",
            )
        self._update(
            "leases",
            lease["id"],
            {"state": "deleted", "delete_attempts": attempts, "last_seen": self.clock(),
             "remote_deleted": True, "remote_deleted_at": self.clock(),
             "remote_delete_source": "cleanup_provider_confirmation"},
            job.get("owner_id"),
        )
        self._record_usage(job, lease, self.clock())
        return self._finish_cleanup(job, final_state, "GPU cleanup completed")

    def _advance(self, job: dict) -> dict:
        state = job.get("state")
        if state in TERMINAL_STATES:
            return job
        if state == "allocating":
            lease = self._lease(job)
            if self._create_rpc_is_in_flight(lease, self.clock()):
                # Restart recovery precedes timeout/cancellation: elapsed time
                # cannot establish whether the original POST created a runtime.
                return self._handle_recovery(job, lease, self._provider(job))
        if job.get("cancel_requested") and state not in {"cleaning_up", "cleanup_pending"}:
            if self._lease(job):
                job = self._set_state(
                    job,
                    "cleaning_up",
                    "GPU cancellation requested",
                    requested_outcome="cancelled",
                    cleanup_final_state="cancelled",
                )
            else:
                return self._set_state(job, "cancelled", "GPU job cancelled", requested_outcome="cancelled")
        if job.get("state") == "cleanup_pending":
            return self._cleanup(job)
        if self._expired(job) and job.get("state") not in {
            "queued",
            "validating",
            "waiting_for_capacity",
            "cleaning_up",
            "cleanup_pending",
        }:
            return self._begin_failure(job, "gpu_timeout", "GPU job exceeded its runtime limit")
        state = job.get("state")
        if state == "queued":
            return self._set_state(job, "validating", "GPU job validating")
        if state == "validating":
            return self._validate(job)
        if state == "waiting_for_capacity":
            # CPU cache preparation owns the first wait.  Do not allocate a
            # durable/runtime instance until the prep decision is settled.
            if self._await_prep(job):
                return job if job.get("awaiting_prep") else self._update("jobs", job["id"], {"awaiting_prep": True}, job.get("owner_id"))
            if self._is_v020_job(job):
                allocated = self._allocate_instance(job)
                if allocated.get("state") in TERMINAL_STATES | {"cleaning_up", "cleanup_pending"}:
                    return allocated
                job = self._reuse_instance_lease(allocated, allocated)
                if not job.get("instance_id"):
                    return job
            elif job.get('compatibility_signature'):
                with self.db_store.dispatch_lock() as acquired:
                    if not acquired:
                        return job
                    for idle in self._leases_for(job["owner_id"]):
                        if self.store.attach_idle_lease(
                            job["owner_id"],
                            job["id"],
                            idle["id"],
                            self.clock(),
                            guard_mode=self._guard_mode(),
                        ):
                            job = self._get("jobs", job["id"], job["owner_id"])
                            break
            if not self._is_v020_job(job) and self._has_capacity_blocker(job):
                return job
            return self._set_state(job, "allocating", "GPU capacity available", awaiting_prep=False)
        if state == "allocating":
            return self._allocation(job)
        if state in {"preparing", "loading", "running", "uploading"}:
            fenced = self._enforce_stage_fence(job)
            if fenced is not None:
                return fenced
        if state == "preparing":
            return self._prepare(job)
        if state == "loading":
            return self._load(job)
        if state == "running":
            return self._run(job)
        if state == "uploading":
            return self._upload(job)
        if state == "cleaning_up":
            return self._cleanup(job)
        return self._begin_failure(job, "gpu_invalid_state", f"unknown GPU state: {state}")

    def _claim_next_job(self) -> tuple[dict | None, str | None]:
        now = self.clock()
        jobs = self._list("jobs")
        jobs.sort(
            key=lambda x: (
                # A job parked on its cache prep or awaiting dispatch poll backoff goes last
                # so it cannot starve queued jobs or the prep job it waits on.
                0
                if x.get("state") in {"cleanup_pending", "cleaning_up"}
                else 3
                if (x.get("state") == "loading" and int(x.get("dispatch_poll_attempts") or 0) > 0)
                else 2
                if x.get("awaiting_prep")
                else 1,
                x.get("created_epoch", x.get("created_at", "")),
            )
        )
        for job in jobs:
            if job.get("state") not in ACTIVE_STATES:
                continue
            if job.get("state") in {"cleanup_pending", "cleaning_up"} and float(job.get("cleanup_next_epoch") or 0) > now:
                continue
            if job.get("state") == "loading" and float(job.get("dispatch_next_epoch") or 0) > now:
                continue
            until = float(job.get("claim_until") or 0)
            if job.get("claim_token") and until > now:
                continue
            token = uuid.uuid4().hex
            claimed = self._update(
                "jobs",
                job["id"],
                {"claim_token": token, "claim_until": now + self.config.claim_ttl_seconds},
                job.get("owner_id"),
            )
            return claimed, token
        return None, None

    def _release_claim(self, job_id: str, token: str | None, owner_id: str | None = None) -> None:
        if not token:
            return
        current = self._get("jobs", job_id, owner_id)
        if current and current.get("claim_token") == token:
            self._update("jobs", job_id, {"claim_token": None, "claim_until": None}, owner_id)

    def _next_job(self) -> dict | None:
        now = self.clock()
        return next(
            (
                x
                for x in self._list("jobs")
                if x.get("state") in ACTIVE_STATES
                and not x.get("claim_token")
                and not (
                    x.get("state") in {"cleanup_pending", "cleaning_up"}
                    and float(x.get("cleanup_next_epoch") or 0) > now
                )
                and not (
                    x.get("state") == "loading"
                    and float(x.get("dispatch_next_epoch") or 0) > now
                )
            ),
            None,
        )

    def tick(self) -> dict:
        if not self.config.enabled:
            return {"acquired": True, "advanced": False, "reason": "disabled"}
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return {"acquired": False, "advanced": False, "reason": "busy"}
            job, claim = self._claim_next_job()
        if job is None:
            return {"acquired": True, "advanced": False}
        before = job.get("state")
        try:
            after = self._advance(job) or job
        finally:
            self._release_claim(job["id"], claim, job.get("owner_id"))
        return {
            "acquired": True,
            "advanced": before != after.get("state"),
            "job_id": job["id"],
            "state": after.get("state"),
        }

    def _cleanup_orphan_lease(self, lease: dict) -> bool:
        owner_job_id = lease.get("owner_job_id")
        owner_id = lease.get("owner_id")
        instance_id = lease.get("instance_id")
        if not instance_id:
            instance_id = (lease.get("data") or {}).get("instance_id")
        account_id = lease.get("account_id")
        if not account_id:
            account_id = (lease.get("data") or {}).get("account_id")
        if not (isinstance(owner_job_id, str) and isinstance(owner_id, str)):
            return False
        if lease.get("state") in {"deleted", "quarantined"}:
            return False
        if self.clock() < float(lease.get("cleanup_next_epoch") or 0):
            return False
        if self._has_conflicting_runtime_in_store(
            owner_id,
            lease,
            instance_id=instance_id,
            exclude_lease_id=lease.get("id"),
            job={"owner_id": owner_id, "instance_id": instance_id},
        ):
            return False

        lease_version = lease.get("version")
        lease_attempts = int(lease.get("create_attempts") or 0)
        durable_instance = self._get("instances", instance_id, owner_id) if instance_id else None
        flags = self._lease_history_flags(lease, durable_instance, owner_id=owner_id)
        if flags.get("query_error"):
            return self._quarantine_orphan_lease(
                owner_id,
                lease,
                instance_id,
                from_active=flags["from_active"] or flags["ever_active"],
                expected_version=lease_version,
                expected_create_attempts=lease_attempts,
            )
        job = {
            "id": owner_job_id,
            "owner_id": owner_id,
            "provider": lease.get("provider", "mock"),
            "instance_id": instance_id,
            "account_id": account_id,
        }
        if not job.get("account_id") and isinstance(durable_instance, dict):
            job["account_id"] = durable_instance.get("account_id") or (durable_instance.get("data") or {}).get("account_id")

        has_req_rt = bool(lease.get("request_id") and lease.get("runtime_id"))
        handle = (
            (self._lease_handle(lease) or self._intent_handle(job, lease))
            if has_req_rt
            else (self._lease_handle(lease) if isinstance(lease.get("handle"), dict) and lease.get("handle") else None)
        )
        if not flags["pure_placeholder"] and handle is None:
            if (
                flags.get("query_error")
                or flags["ever_active"]
                or flags["has_remote_identity"]
                or flags["has_create_intent"]
            ):
                return self._quarantine_orphan_lease(
                    owner_id,
                    lease,
                    instance_id,
                    from_active=flags["from_active"] or flags["ever_active"],
                    expected_version=lease_version,
                    expected_create_attempts=lease_attempts,
                )
            return False
        if flags["pure_placeholder"] and not has_req_rt:
            return False

        # Only a genuinely pure placeholder without any active history, remote identity,
        # create intent, or execution records may be closed locally without provider verification,
        # and all checks + local deletion must run in ONE atomic transaction before any claim.
        if flags["pure_placeholder"] and not flags.get("query_error"):
            if hasattr(self.store, "retire_pure_orphan_placeholder_cas"):
                try:
                    retire_status, _ = self.store.retire_pure_orphan_placeholder_cas(
                        owner_id,
                        lease["id"],
                        instance_id=instance_id,
                        expected_lease_version=lease_version,
                        expected_instance_version=(
                            durable_instance.get("version") if isinstance(durable_instance, dict) else None
                        ),
                        now=self.clock(),
                    )
                except Exception:
                    retire_status = "error"
                if retire_status == "retired":
                    if not self._has_sql_instance_service() and self.instance_service is not None:
                        try:
                            self._mark_instance_deleted(job)
                        except SchedulerError:
                            pass
                    return True
                if retire_status in {"stale", "locked", "missing", "conflict"}:
                    return False
                fresh_lease = self._get("leases", lease["id"], owner_id)
                if (
                    fresh_lease is None
                    or fresh_lease.get("state") in {"deleted", "quarantined", "idle", "active", "busy"}
                    or self._lease_handle(fresh_lease)
                    or self._create_rpc_is_in_flight(fresh_lease, self.clock())
                ):
                    return False
                fresh_flags = self._lease_history_flags(fresh_lease, owner_id=owner_id)
                return self._quarantine_orphan_lease(
                    owner_id,
                    fresh_lease,
                    instance_id,
                    from_active=fresh_flags["from_active"] or fresh_flags["ever_active"],
                    expected_version=fresh_lease.get("version"),
                    expected_create_attempts=int(fresh_lease.get("create_attempts") or 0),
                )
            with self._admission_lock(timeout=1.0) as acquired:
                if not acquired:
                    return False
                fresh_lease = self._get("leases", lease["id"], owner_id)
                if fresh_lease is None or fresh_lease.get("version") != lease_version:
                    return False
                if fresh_lease.get("state") in {"deleted", "quarantined", "idle", "active", "busy"}:
                    return False
                if self._has_conflicting_runtime_in_store(
                    owner_id,
                    fresh_lease,
                    instance_id=instance_id,
                    exclude_lease_id=fresh_lease.get("id"),
                    job=job,
                ):
                    return False
                fresh_inst = self._get("instances", instance_id, owner_id) if instance_id else durable_instance
                fresh_flags = self._lease_history_flags(fresh_lease, fresh_inst, owner_id=owner_id)
                if not fresh_flags["pure_placeholder"] or fresh_flags.get("query_error"):
                    if self._lease_handle(fresh_lease) or fresh_lease.get("state") in {"active", "busy"}:
                        return False
                    return self._quarantine_orphan_lease(
                        owner_id,
                        fresh_lease,
                        instance_id,
                        from_active=fresh_flags["from_active"] or fresh_flags["ever_active"],
                        expected_version=fresh_lease.get("version"),
                        expected_create_attempts=int(fresh_lease.get("create_attempts") or 0),
                    )
                try:
                    self._mark_instance_deleted(job)
                except SchedulerError:
                    return False
                self._update("leases", fresh_lease["id"], {"state": "deleted", "last_seen": self.clock()}, owner_id)
                return True

        token = uuid.uuid4().hex
        # Expired idle leases also need the atomic claim: attachment may have
        # transferred ownership since the watchdog captured this snapshot.
        is_retained_idle = (
            lease.get("state") == "idle"
            or (isinstance(durable_instance, dict) and durable_instance.get("state") == "retained_idle")
        )
        use_claim = hasattr(getattr(self, "store", None), "claim_orphan_lease")
        claimed = self.store.claim_orphan_lease(
            owner_id, lease["id"], instance_id, token, self.clock(),
            expected_lease_version=lease.get("version"), allow_retained_idle=is_retained_idle,
        ) if use_claim else None
        if use_claim and not claimed:
            return False
        if claimed:
            lease = claimed[0]
            instance = claimed[1]
            lease_version = lease.get("version")
            lease_attempts = int(lease.get("create_attempts") or 0)
            if instance_id is not None:
                job["instance_id"] = instance_id
            if not job.get("account_id"):
                job["account_id"] = (
                    lease.get("account_id")
                    or (instance.get("account_id") if isinstance(instance, dict) else None)
                )
            if instance and instance.get("version") is not None:
                job["_cleanup_instance_version"] = instance["version"]
            flags = self._lease_history_flags(lease, instance, owner_id=owner_id)
            if flags.get("query_error"):
                return self._quarantine_orphan_lease(
                    owner_id,
                    lease,
                    instance_id,
                    from_active=flags["from_active"] or flags["ever_active"],
                    expected_version=lease_version,
                    expected_create_attempts=lease_attempts,
                )
            handle = self._lease_handle(lease) or self._intent_handle(job, lease)

        def finish(confirmed, error=None, attempts=0):
            if confirmed:
                if claimed:
                    return bool(self.store.finish_orphan_lease(
                        owner_id, lease["id"], instance_id, token, self.clock(),
                        confirmed=True, error=error, attempts=attempts,
                    ))
                self._update("leases", lease["id"], {"state": "deleted", "last_seen": self.clock()}, owner_id)
                return True
            delay = min(
                self.config.cleanup_retry_max_seconds,
                self.config.cleanup_retry_base_seconds * (2 ** max(0, attempts - 1)),
            )
            message = error or "delete not confirmed"
            finish_error = {"message": message, "retry_after": delay}
            if claimed:
                return bool(self.store.finish_orphan_lease(
                    owner_id, lease["id"], instance_id, token, self.clock(),
                    confirmed=False, error=finish_error, attempts=attempts,
                ))
            self._update("leases", lease["id"], {
                "state": "cleanup_pending", "delete_attempts": attempts,
                "cleanup_next_epoch": self.clock() + delay, "last_error": message,
                "last_seen": self.clock(),
            }, owner_id)
            return False

        if self._has_conflicting_runtime_in_store(
            owner_id,
            lease,
            instance_id=instance_id,
            handle=handle,
            exclude_lease_id=lease.get("id"),
            job=job,
        ):
            finish(False, "runtime acquired by another live lease")
            return False

        if handle is None:
            return self._quarantine_orphan_lease(
                owner_id,
                lease,
                instance_id,
                from_active=flags["from_active"] or flags["ever_active"],
                expected_version=lease_version,
                expected_create_attempts=lease_attempts,
            )

        try:
            provider = self._provider(job)
            if hasattr(provider, "recover_handle"):
                try:
                    handle = provider.recover_handle(handle)
                except ComputeError:
                    if flags["ever_active"] and not flags["has_remote_identity"]:
                        return self._quarantine_orphan_lease(
                            owner_id,
                            lease,
                            instance_id,
                            from_active=True,
                            expected_version=lease_version,
                            expected_create_attempts=lease_attempts,
                        )
                    raise
            try:
                status = provider.inspect(handle)
            except ComputeError as exc:
                if flags["ever_active"] and not flags["has_remote_identity"] and "identity mismatch" in str(exc):
                    return self._quarantine_orphan_lease(
                        owner_id,
                        lease,
                        instance_id,
                        from_active=True,
                        expected_version=lease_version,
                        expected_create_attempts=lease_attempts,
                    )
                status = {}
            except Exception:
                status = {}
            if isinstance(status, dict) and status.get("identity_unverifiable"):
                return self._quarantine_orphan_lease(
                    owner_id,
                    lease,
                    instance_id,
                    from_active=flags["from_active"] or flags["ever_active"],
                    expected_version=lease_version,
                    expected_create_attempts=lease_attempts,
                )
            confirmed = self._inspect_confirms_absent(status)
            # Recheck right before remote delete across authoritative runtime & endpoint (not just instance_id).
            if not confirmed and self._has_conflicting_runtime_in_store(
                owner_id,
                lease,
                instance_id=instance_id,
                handle=handle,
                exclude_lease_id=lease.get("id"),
                job=job,
            ):
                finish(False, "runtime acquired by another live lease")
                return False
            if not confirmed:
                response = provider.delete(handle)
                confirmed = self._delete_confirms_absent(response)
            error = None
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)

        attempts = int(lease.get("delete_attempts") or 0) + 1
        if confirmed:
            try:
                self._mark_instance_deleted(job)
            except SchedulerError as exc:
                confirmed = False
                error = str(exc)
        if confirmed:
            return finish(True, attempts=attempts)
        finish(False, error or "delete not confirmed", attempts)
        return False

    def _historical_orphan_evidence(self, instance: dict, lease: dict, leases: list[dict], jobs: list[dict]):
        """Return a lease-owned inspection job only for an unambiguous old handoff."""
        owner = instance.get("owner_id")
        instance_id = self._instance_id(instance)
        if (
            not isinstance(owner, str)
            or not owner
            or not instance_id
            or instance.get("state") not in {"releasing", "cleanup_pending"}
            or lease.get("state") != "deleted"
            or lease.get("owner_id") != owner
            or lease.get("remote_delete_source") not in (None, "")
        ):
            return None
        account = instance.get("account_id")
        if not account or lease.get("account_id") != account:
            return None
        provider = lease.get("provider")
        if provider not in {"mock", "colab"} or instance.get("provider") not in {None, provider}:
            return None
        runtime_id = lease.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id or not lease.get("request_id"):
            return None
        if instance.get("runtime_id") not in (None, runtime_id):
            return None
        current_job_id = instance.get("current_job_id")
        owner_job_id = lease.get("owner_job_id")
        by_id = {row.get("id"): row for row in jobs}
        if not owner_job_id or not by_id.get(owner_job_id):
            return None
        if by_id[owner_job_id].get("owner_id") != owner or by_id[owner_job_id].get("state") not in TERMINAL_STATES:
            return None
        if current_job_id:
            current = by_id.get(current_job_id)
            if not current or current.get("owner_id") != owner or current.get("state") not in TERMINAL_STATES:
                return None
        lease_instance_id = lease.get("instance_id")
        if lease_instance_id in ("",):
            lease_instance_id = None
        lease_rebound = lease_instance_id not in (None, instance_id)
        owner_job = by_id[owner_job_id]
        if lease_rebound:
            successor_instance_id = owner_job.get("instance_id")
            if successor_instance_id != lease_instance_id or successor_instance_id == instance_id:
                return None
        related = [
            row for row in jobs
            if row.get("instance_id") == instance_id or row.get("id") in {current_job_id, owner_job_id}
        ]
        if any(row.get("owner_id") != owner or row.get("state") not in TERMINAL_STATES for row in related):
            return None
        for row in related:
            row_provider = row.get("provider") or (row.get("execution_policy") or {}).get("provider")
            allowed_instances = {None, instance_id}
            if row.get("id") == owner_job_id and lease_rebound:
                allowed_instances.add(lease_instance_id)
            if (
                row.get("account_id") not in (None, account)
                or row.get("instance_id") not in allowed_instances
                or row_provider not in (None, provider)
            ):
                return None
        for row in leases:
            if row.get("id") == lease.get("id") or row.get("state") == "deleted":
                continue
            if row.get("instance_id") == instance_id or row.get("runtime_id") == runtime_id:
                return None
        job = {
            "id": owner_job_id,
            "owner_id": owner,
            "provider": provider,
            "account_id": account,
            "instance_id": instance_id,
        }
        handle = self._intent_handle(job, lease)
        if not handle:
            return None
        if isinstance(lease.get("handle"), dict):
            handle = {**lease["handle"], **handle}
        return job, handle

    def _cleanup_historical_orphan_instance(self, instance: dict, lease: dict, leases: list[dict], jobs: list[dict]) -> bool:
        """Inspect, then persist deletion, for a legacy lease missing provenance."""
        now = self.clock()
        if any(now < float(row.get("cleanup_next_epoch") or 0) for row in (instance, lease)):
            return False
        evidence = self._historical_orphan_evidence(instance, lease, leases, jobs)
        if evidence is None:
            return False
        token = uuid.uuid4().hex
        claimed = self.store.claim_orphan_instance(
            instance["owner_id"], self._instance_id(instance), lease["id"], token, self.clock(),
            expected_lease_version=lease.get("version"), expected_instance_version=instance.get("version"),
        )
        if not claimed:
            return False
        claim_instance, claim_lease = claimed
        job, handle = evidence
        try:
            provider = self._provider(job)
            if hasattr(provider, "recover_handle"):
                handle = provider.recover_handle(handle)
            status = provider.inspect(handle)
            confirmed = (
                isinstance(status, dict)
                and status.get("deleted") is True
                and status.get("remote_verified") is not False
                and not any(
                    status.get(key)
                    for key in (
                        "allocation_unknown",
                        "operation_pending",
                        "ambiguous",
                        "pending",
                        "identity_unverifiable",
                    )
                )
            )
            error = None if confirmed else "remote deletion not unambiguously confirmed"
        except Exception as exc:
            confirmed, error = False, self._safe_error(exc)
        return bool(self.store.finish_orphan_instance(
            instance["owner_id"], self._instance_id(instance), lease["id"], token, self.clock(),
            confirmed=confirmed, error=error, expected_version=claim_instance.get("version"),
            expected_lease_version=claim_lease.get("version"),
        )) and confirmed

    def _reconcile_deleted_lease_instances(self, leases: list[dict]) -> int:
        """Converge old releasing rows only with durable remote-delete evidence."""
        if self.instance_service is None:
            return 0
        by_lease = {row.get("id"): row for row in leases if row.get("id")}
        recovered = 0
        service_instances = self._service_instances()
        jobs = None
        for instance in service_instances:
            if instance.get("state") not in {"releasing", "cleanup_pending"}:
                continue
            lease_id = instance.get("lease_id")
            lease = by_lease.get(lease_id)
            if not lease or lease.get("state") != "deleted":
                continue
            source = lease.get("remote_delete_source")
            if source in (None, ""):
                if jobs is None:
                    jobs = self._list("jobs")
                if self._cleanup_historical_orphan_instance(instance, lease, leases, jobs):
                    recovered += 1
                continue
            if lease.get("remote_deleted") is not True or source not in {
                "cleanup_provider_confirmation", "orphan_cleanup_provider_confirmation",
            }:
                continue
            if lease.get("owner_id") != instance.get("owner_id"):
                continue
            current_instance_id = self._instance_id(instance)
            lease_instance_json = (lease.get("data") or {}).get("instance_id")
            lease_instance_typed = lease.get("instance_id")
            if (
                lease_instance_json not in (None, "")
                and lease_instance_typed not in (None, "")
                and lease_instance_json != lease_instance_typed
            ):
                continue
            lease_instance_id = lease_instance_json or lease_instance_typed
            if not current_instance_id or lease_instance_id != current_instance_id:
                continue
            lease_account = lease.get("account_id")
            instance_account = instance.get("account_id")
            if (not lease_account or not instance_account or lease_account != instance_account):
                continue
            job = {
                "id": lease.get("owner_job_id") or f"orphan-instance:{instance.get('id')}",
                "owner_id": instance.get("owner_id"),
                "provider": lease.get("provider", "mock"),
                "instance_id": instance.get("id"),
                "_cleanup_instance_version": instance.get("version"),
            }
            try:
                self._mark_instance_deleted(job)
            except SchedulerError:
                continue
            recovered += 1
        return recovered

    def watchdog(self) -> dict:
        """Independent process entrypoint for timeout, orphan, and cleanup compensation."""
        with self.db_store.dispatch_lock() as acquired:
            if not acquired:
                return {"acquired": False, "recovered": 0, "reason": "busy"}
            all_jobs = self._list("jobs")
            jobs = [
                x
                for x in all_jobs
                if x.get("state") in ACTIVE_STATES
                and (x.get("state") in {"cleanup_pending", "cleaning_up"} or self._expired(x))
            ]
            job_by_id = {x["id"]: x for x in all_jobs}
            all_leases = self._leases_for()
            leases = [
                x
                for x in all_leases
                if x.get("state") not in {"deleted", "quarantined"}
                and (
                    x.get("owner_job_id") not in job_by_id
                    or (
                        job_by_id.get(x.get("owner_job_id"), {}).get("state") in TERMINAL_STATES
                        and x.get("owner_id") == job_by_id.get(x.get("owner_job_id"), {}).get("owner_id")
                    )
                )
            ]
        recovered = 0
        for job in jobs:
            before = job.get("state")
            self._advance(job)
            current = self._get("jobs", job["id"], job.get("owner_id"))
            if current and current.get("state") != before:
                recovered += 1
        for lease in leases:
            instance = next((row for row in self._service_instances() if self._instance_id(row) == lease.get("instance_id")), None)
            stopping = bool(instance and (instance.get("manual_stop_requested") or instance.get("state") in {"releasing", "cleanup_pending"}))
            if lease.get('state')=='idle' and not stopping and self.clock()<min(
                    float(lease.get('idle_deadline_at') or 0),float(lease.get('deadline_epoch') or 0)):
                continue
            associated = job_by_id.get(lease.get("owner_job_id"))
            expired = float(lease.get("deadline_epoch") or 0) < self.clock()
            if (
                associated is None and expired
            ) or (
                associated
                and associated.get("state") in TERMINAL_STATES
                and lease.get("owner_id") == associated.get("owner_id")
            ):
                if self._cleanup_orphan_lease(lease):
                    recovered += 1
        recovered += self._reconcile_deleted_lease_instances(all_leases)
        return {"acquired": True, "recovered": recovered}

    def cancel(self, owner_id: str, job_id: str) -> dict:
        if hasattr(self.store, "request_job_cancel"):
            return self.store.request_job_cancel(owner_id, job_id, self.clock())
        job = self._get("jobs", job_id, owner_id)
        if not job:
            raise KeyError(job_id)
        if job.get("state") in TERMINAL_STATES:
            return job
        if job.get("state") == "queued" and not self._lease(job):
            return self._set_state(
                job, "cancelled", "GPU job cancelled", cancel_requested=True, requested_outcome="cancelled"
            )
        return self._update(
            "jobs", job_id, {"cancel_requested": True, "requested_outcome": "cancelled"}, owner_id
        )

    def stop(self, owner_id: str) -> dict:
        count = 0
        for job in self._list("jobs", owner_id):
            if job.get("state") not in TERMINAL_STATES:
                self.cancel(owner_id, job["id"])
                count += 1
        return {"owner_id": owner_id, "cancelled": count}


__all__ = ["Scheduler", "SchedulerError", "TERMINAL_STATES", "ACTIVE_STATES"]
