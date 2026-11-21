"""Independent GPU repository backed by app.db; no Atelier imports."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Mapping

from sqlalchemy import String, UniqueConstraint, inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Mapped, mapped_column

from app import db as db_module
from app.db import Base, CommonMixin, Store, utc_isoformat, utcnow

from .estimator import (
    check_job_reuse_guard,
    is_legacy_mock_job,
    load_session_history,
    min_deadlines,
)
from .identity import (
    collect_runtime_identities as collect_runtime_identities,
    extract_runtime_id as extract_runtime_id,
    extract_runtime_name_id,
    extract_tunnel_endpoint as extract_tunnel_endpoint,
    is_valid_runtime_handle,
    normalize_tunnel_endpoint,
    retarget_handle as retarget_handle,
    row_has_positive_usage,
)
from .manifest import digest


class GpuModelRow(CommonMixin, Base):
    __tablename__ = "gpu_models"


class GpuWorkflowRow(CommonMixin, Base):
    __tablename__ = "gpu_workflows"
    workflow_key: Mapped[str] = mapped_column(String(255), nullable=False)
    # Stored as text so both legacy integer versions and semver versions have
    # one database identity.  _row converts digit-only legacy values back to int.
    workflow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "workflow_key", "workflow_version", name="uq_gpu_workflows_owner_key_version"
        ),
    )


class GpuJobRow(CommonMixin, Base):
    __tablename__ = "gpu_jobs"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key", name="uq_gpu_jobs_owner_idempotency"),)


class GpuBatchRow(CommonMixin, Base):
    __tablename__ = "gpu_batches"
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key", name="uq_gpu_batches_owner_key"),)


class GpuExecutionRow(CommonMixin, Base):
    __tablename__ = "gpu_executions"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuLeaseRow(CommonMixin, Base):
    __tablename__ = "gpu_leases"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuEventRow(CommonMixin, Base):
    __tablename__ = "gpu_events"


class GpuPresetRow(CommonMixin, Base):
    __tablename__ = "gpu_presets"


class GpuUsageRow(CommonMixin, Base):
    __tablename__ = "gpu_usage"


class GpuFileRow(CommonMixin, Base):
    __tablename__ = "gpu_files"


class GpuCredentialRow(CommonMixin, Base):
    __tablename__ = "gpu_credentials"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuAccountRow(CommonMixin, Base):
    __tablename__ = "gpu_accounts"


class GpuInstanceRow(CommonMixin, Base):
    __tablename__ = "gpu_instances"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    current_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (
        UniqueConstraint("owner_id", "current_job_id", name="uq_gpu_instances_owner_current_job"),
    )


class GpuInstanceUsageSegmentRow(CommonMixin, Base):
    __tablename__ = "gpu_instance_usage_segments"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuAllocationCursorRow(CommonMixin, Base):
    __tablename__ = "gpu_allocation_cursors"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


class GpuOperationRow(CommonMixin, Base):
    __tablename__ = "gpu_operations"
    account_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)


_GPU_MODELS = {
    "batches": GpuBatchRow,
    "executions": GpuExecutionRow,
    "models": GpuModelRow,
    "workflows": GpuWorkflowRow,
    "jobs": GpuJobRow,
    "leases": GpuLeaseRow,
    "events": GpuEventRow,
    "presets": GpuPresetRow,
    "usage": GpuUsageRow,
    "files": GpuFileRow,
    "credentials": GpuCredentialRow,
    "accounts": GpuAccountRow,
    "instances": GpuInstanceRow,
    "instance_usage_segments": GpuInstanceUsageSegmentRow,
    "allocation_cursors": GpuAllocationCursorRow,
    "operations": GpuOperationRow,
}
_GPU_TABLES = [model.__table__ for model in _GPU_MODELS.values()]
for _kind, _model in _GPU_MODELS.items():
    db_module.MODELS.setdefault("gpu_" + _kind, _model)


def ensure_schema(store: Store) -> None:
    for attempt in range(6):
        try:
            with store.db_lock():
                Base.metadata.create_all(store.engine, tables=_GPU_TABLES, checkfirst=True)
                # create_all does not migrate a database made by an older
                # build.  Add the two identity columns and the unique index
                # idempotently before any workflow can be written.
                columns = {column["name"] for column in inspect(store.engine).get_columns("gpu_workflows")}
                with store.engine.begin() as connection:
                    if "workflow_key" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_key VARCHAR(255)")
                        )
                    if "workflow_version" not in columns:
                        connection.execute(
                            text("ALTER TABLE gpu_workflows ADD COLUMN workflow_version VARCHAR(64)")
                        )
                    if store.engine.dialect.name in {"sqlite", "postgresql"}:
                        connection.execute(
                            text(
                                "CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_workflows_owner_key_version ON gpu_workflows (owner_id, workflow_key, workflow_version)"
                            )
                        )
                    for table, column_specs in {
                        "gpu_jobs": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_executions": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_leases": {"account_id": "VARCHAR(64)", "instance_id": "VARCHAR(64)"},
                        "gpu_credentials": {"account_id": "VARCHAR(64)"},
                    }.items():
                        existing = {column["name"] for column in inspect(store.engine).get_columns(table)}
                        for column, column_type in column_specs.items():
                            if column not in existing:
                                connection.execute(
                                    text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
                                )
                        for column in column_specs:
                            connection.execute(
                                text(
                                    f"CREATE INDEX IF NOT EXISTS ix_{table}_{column}_v020 "
                                    f"ON {table} ({column})"
                                )
                            )
                    connection.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_instances_owner_current_job "
                            "ON gpu_instances (owner_id, current_job_id)"
                        )
                    )
                    # DDL alone cannot reconstruct the v0.2 identity graph.
                    # Keep startup safe and local: this helper only repairs
                    # owner/account/instance identity metadata and never calls
                    # OAuth, Colab, or Drive.
                    from .legacy_migration import upgrade_legacy_identity

                    upgrade_legacy_identity(connection)
            return
        except OperationalError:
            if attempt == 5:
                raise
            time.sleep(0.03 * (attempt + 1))


_ACTIVE_ORIGINS = {"active", "busy", "once_active", "from_active", "active_lease"}
_STOP_DELETE_FLAGS = (
    "manual_stop_requested",
    "stop_requested",
    "runtime_deleted",
    "remote_deleted",
    "deleted",
    "cleanup_claim_token",
    "cleanup_claim",
)
_CONFLICTING_INSTANCE_STATES = frozenset(
    {"creating", "allocating", "ready", "busy", "idle", "retained_idle", "cleanup_pending", "deleting"}
)


# Endpoint & strict runtime handle validators live in `.identity` and are
# re-exported here so DB/provider callers share one canonical definition.


def lease_has_active_origin(obj: object, state: object = None) -> bool:
    """Return True if any active marker or active origin field exists (never shadowed by 'new')."""
    if obj is None:
        return False
    eff_state = state
    if eff_state is None:
        eff_state = obj.get("state") if isinstance(obj, Mapping) else getattr(obj, "state", None)
    if eff_state in {"active", "busy"}:
        return True

    def _dict_has_active(mapping: Mapping[str, object] | None) -> bool:
        if not isinstance(mapping, Mapping):
            return False
        if (
            mapping.get("from_active_missing_handle")
            or mapping.get("was_active_missing_handle")
            or mapping.get("from_active")
            or mapping.get("was_active")
            or mapping.get("once_active")
            or mapping.get("ever_active")
            or mapping.get("last_active_at") is not None
            or mapping.get("active_execution") is not None
        ):
            return True
        for origin_key in ("origin", "create_intent_origin"):
            raw_origin = mapping.get(origin_key)
            if isinstance(raw_origin, str) and raw_origin.strip().lower() in _ACTIVE_ORIGINS:
                return True
        intent = mapping.get("create_intent")
        if isinstance(intent, Mapping):
            intent_origin = intent.get("origin")
            if isinstance(intent_origin, str) and intent_origin.strip().lower() in _ACTIVE_ORIGINS:
                return True
            if (
                intent.get("from_active")
                or intent.get("was_active")
                or intent.get("once_active")
                or intent.get("ever_active")
            ):
                return True
        return False

    if isinstance(obj, Mapping):
        if _dict_has_active(obj):
            return True
        data = obj.get("data")
        return _dict_has_active(data if isinstance(data, Mapping) else None)
    data = getattr(obj, "data", None)
    return _dict_has_active(data if isinstance(data, Mapping) else None)


_collect_runtime_identities = collect_runtime_identities


def resolve_authoritative_runtime(
    session,
    *,
    lease: object = None,
    instance: object = None,
    job: object = None,
    runtime_id: object = None,
    runtime_name: object = None,
    tunnel_endpoint: object = None,
    handle: object = None,
) -> tuple[set[str], set[str]]:
    """Resolve authoritative runtime IDs and normalized endpoints from entity rows AND linked DB rows."""
    runtime_ids: set[str] = set()
    endpoints: set[str] = set()
    if isinstance(runtime_id, str) and runtime_id.strip():
        runtime_ids.add(runtime_id.strip())
    name_rt = extract_runtime_name_id(runtime_name)
    if name_rt:
        runtime_ids.add(name_rt)
    norm_ep = normalize_tunnel_endpoint(tunnel_endpoint)
    if norm_ep:
        endpoints.add(norm_ep)
    if isinstance(handle, Mapping):
        _collect_runtime_identities(handle, runtime_ids, endpoints)

    def _row_field(obj: object, key: str) -> str | None:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            data = obj.get("data")
            if isinstance(data, Mapping):
                dval = data.get(key)
                if isinstance(dval, str) and dval.strip():
                    return dval.strip()
            return None
        val = getattr(obj, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
        data = getattr(obj, "data", None)
        if isinstance(data, Mapping):
            dval = data.get(key)
            if isinstance(dval, str) and dval.strip():
                return dval.strip()
        return None

    all_jobs = session.scalars(select(GpuJobRow)).all() if session is not None else []
    all_leases = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.state != "deleted")).all() if session is not None else []
    jobs_by_id = {j.id: j for j in all_jobs}

    seen_jobs: set[str] = set()
    seen_leases: set[str] = set()
    seen_insts: set[str] = set()

    def _visit_job(jobj: object) -> None:
        if jobj is None:
            return
        jid = _row_field(jobj, "id")
        if jid:
            if jid in seen_jobs:
                return
            seen_jobs.add(jid)
        _collect_runtime_identities(jobj, runtime_ids, endpoints)
        if session is not None:
            if lease is None:
                j_lease_id = _row_field(jobj, "lease_id")
                if j_lease_id and j_lease_id not in seen_leases:
                    _visit_lease(session.get(GpuLeaseRow, j_lease_id))
            if instance is None:
                j_inst_id = _row_field(jobj, "instance_id")
                if j_inst_id and j_inst_id not in seen_insts:
                    _visit_instance(session.get(GpuInstanceRow, j_inst_id))

    def _visit_lease(lobj: object) -> None:
        if lobj is None:
            return
        lid = _row_field(lobj, "id")
        if lid:
            if lid in seen_leases:
                return
            seen_leases.add(lid)
        _collect_runtime_identities(lobj, runtime_ids, endpoints)
        if session is None:
            return
        l_inst_id = _row_field(lobj, "instance_id")
        if l_inst_id and l_inst_id not in seen_insts:
            _visit_instance(session.get(GpuInstanceRow, l_inst_id))
        if lid:
            owner_jid = _row_field(lobj, "owner_job_id")
            cur_jid = _row_field(lobj, "current_job_id")
            for cand_jid in (owner_jid, cur_jid):
                if cand_jid and cand_jid in jobs_by_id:
                    cand_j = jobs_by_id[cand_jid]
                    cand_j_lease = _row_field(cand_j, "lease_id")
                    if cand_j_lease in {None, lid}:
                        _visit_job(cand_j)
            for cand_j in all_jobs:
                jdata = cand_j.data or {}
                j_lease = jdata.get("lease_id")
                j_intent = jdata.get("start_intent") if isinstance(jdata.get("start_intent"), Mapping) else {}
                if j_lease == lid or (
                    j_lease is None
                    and cand_j.state not in {"completed", "failed", "cancelled"}
                    and j_intent.get("lease_id") == lid
                ):
                    _visit_job(cand_j)

    def _visit_instance(iobj: object) -> None:
        if iobj is None:
            return
        iid = _row_field(iobj, "id")
        if iid:
            if iid in seen_insts:
                return
            seen_insts.add(iid)
        _collect_runtime_identities(iobj, runtime_ids, endpoints)
        if session is None:
            return
        i_lease_id = _row_field(iobj, "lease_id")
        if i_lease_id and i_lease_id not in seen_leases:
            _visit_lease(session.get(GpuLeaseRow, i_lease_id))
        elif iid:
            for linked in all_leases:
                if _row_field(linked, "instance_id") == iid:
                    _visit_lease(linked)
        if iid:
            cur_jid = _row_field(iobj, "current_job_id")
            if cur_jid and cur_jid in jobs_by_id:
                cand_j = jobs_by_id[cur_jid]
                if _row_field(cand_j, "instance_id") in {None, iid}:
                    _visit_job(cand_j)
            for cand_j in all_jobs:
                j_inst = cand_j.instance_id or (cand_j.data or {}).get("instance_id")
                if j_inst == iid:
                    _visit_job(cand_j)

    _visit_job(job)
    _visit_lease(lease)
    _visit_instance(instance)

    return runtime_ids, endpoints


_resolve_authoritative_runtime = resolve_authoritative_runtime


def _lease_associated_instance_ids(session, lease: object, jobs: list[GpuJobRow]) -> set[str]:
    """Return all instance IDs directly on a lease or on jobs bound to that lease."""
    inst_ids: set[str] = set()
    lid = lease.get("id") if isinstance(lease, Mapping) else getattr(lease, "id", None)
    direct_inst = lease.get("instance_id") if isinstance(lease, Mapping) else getattr(lease, "instance_id", None)
    if isinstance(direct_inst, str) and direct_inst:
        inst_ids.add(direct_inst)
    ldata = lease if isinstance(lease, Mapping) else (getattr(lease, "data", None) or {})
    if isinstance(ldata, Mapping):
        data_inst = ldata.get("instance_id")
        if isinstance(data_inst, str) and data_inst:
            inst_ids.add(data_inst)
    if lid:
        owner_jid = ldata.get("owner_job_id") if isinstance(ldata, Mapping) else None
        cur_jid = ldata.get("current_job_id") if isinstance(ldata, Mapping) else None
        for jrow in jobs:
            jdata = jrow.data or {}
            j_lease = jdata.get("lease_id")
            if j_lease == lid or (j_lease is None and jrow.id in {owner_jid, cur_jid}):
                j_inst = jrow.instance_id or jdata.get("instance_id")
                if isinstance(j_inst, str) and j_inst:
                    inst_ids.add(j_inst)
    return inst_ids


def find_conflicting_non_deleted_lease(
    session,
    *,
    instance_id: str | None = None,
    runtime_id: str | None = None,
    runtime_name: str | None = None,
    tunnel_endpoint: str | None = None,
    handle: object = None,
    lease: object = None,
    instance: object = None,
    job: object = None,
    exclude_lease_id: str | None = None,
) -> GpuLeaseRow | None:
    """Return any other non-deleted lease referencing the same instance_id, runtime_id, or normalized endpoint."""
    eff_excl_lease = exclude_lease_id
    if not eff_excl_lease and lease is not None:
        eff_excl_lease = lease.get("id") if isinstance(lease, Mapping) else getattr(lease, "id", None)
    target_inst_ids: set[str] = set()
    if isinstance(instance_id, str) and instance_id:
        target_inst_ids.add(instance_id)
    if instance is not None:
        iid = instance.get("id") if isinstance(instance, Mapping) else getattr(instance, "id", None)
        if isinstance(iid, str) and iid:
            target_inst_ids.add(iid)
    if not target_inst_ids:
        for src in (lease, job):
            if src is not None:
                sid = src.get("instance_id") if isinstance(src, Mapping) else getattr(src, "instance_id", None)
                if not sid and hasattr(src, "data") and isinstance(src.data, Mapping):
                    sid = src.data.get("instance_id")
                if isinstance(sid, str) and sid:
                    target_inst_ids.add(sid)

    target_rts, target_eps = resolve_authoritative_runtime(
        session,
        lease=lease,
        instance=instance,
        job=job,
        runtime_id=runtime_id,
        runtime_name=runtime_name,
        tunnel_endpoint=tunnel_endpoint,
        handle=handle,
    )
    if not target_inst_ids and not target_rts and not target_eps:
        return None
    others = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.state != "deleted").with_for_update()).all()
    all_jobs = session.scalars(select(GpuJobRow)).all()
    for other in others:
        if eff_excl_lease and other.id == eff_excl_lease:
            continue
        other_inst_ids = _lease_associated_instance_ids(session, other, all_jobs)
        if target_inst_ids & other_inst_ids:
            return other
        other_rts, other_eps = resolve_authoritative_runtime(session, lease=other)
        if (target_rts & other_rts) or (target_eps & other_eps):
            return other
    return None


def has_conflicting_active_runtime(
    session,
    *,
    instance_id: str | None = None,
    runtime_id: str | None = None,
    runtime_name: str | None = None,
    tunnel_endpoint: str | None = None,
    handle: object = None,
    lease: object = None,
    instance: object = None,
    job: object = None,
    exclude_lease_id: str | None = None,
    exclude_instance_id: str | None = None,
) -> tuple[bool, str | None]:
    """Check both non-deleted leases AND active/cleanup-pending instances for authoritative runtime conflicts."""
    conflict_lease = find_conflicting_non_deleted_lease(
        session,
        instance_id=instance_id,
        runtime_id=runtime_id,
        runtime_name=runtime_name,
        tunnel_endpoint=tunnel_endpoint,
        handle=handle,
        lease=lease,
        instance=instance,
        job=job,
        exclude_lease_id=exclude_lease_id,
    )
    if conflict_lease is not None:
        return True, f"lease:{conflict_lease.id}:{conflict_lease.state}"

    eff_excl_inst = exclude_instance_id or instance_id
    if not eff_excl_inst and instance is not None:
        eff_excl_inst = instance.get("id") if isinstance(instance, Mapping) else getattr(instance, "id", None)
    eff_excl_lease = exclude_lease_id
    if not eff_excl_lease and lease is not None:
        eff_excl_lease = lease.get("id") if isinstance(lease, Mapping) else getattr(lease, "id", None)

    target_rts, target_eps = resolve_authoritative_runtime(
        session,
        lease=lease,
        instance=instance,
        job=job,
        runtime_id=runtime_id,
        runtime_name=runtime_name,
        tunnel_endpoint=tunnel_endpoint,
        handle=handle,
    )
    if not target_rts and not target_eps:
        return False, None
    other_instances = session.scalars(select(GpuInstanceRow).where(GpuInstanceRow.state != "deleted")).all()
    for other_inst in other_instances:
        if eff_excl_inst and other_inst.id == eff_excl_inst:
            continue
        idata = other_inst.data or {}
        if eff_excl_lease and eff_excl_lease in {idata.get("lease_id"), idata.get("legacy_lease_id")}:
            continue
        if other_inst.state not in _CONFLICTING_INSTANCE_STATES:
            has_live_claim = bool(idata.get("cleanup_claim_token"))
            unconfirmed_unknown = (
                other_inst.state == "unknown"
                and not idata.get("runtime_deleted")
                and not idata.get("remote_deleted")
            )
            if not (has_live_claim or unconfirmed_unknown):
                continue
        other_rts, other_eps = resolve_authoritative_runtime(session, instance=other_inst)
        if (target_rts & other_rts) or (target_eps & other_eps):
            return True, f"instance:{other_inst.id}:{other_inst.state}"
    return False, None


class Repository:
    _CLEANUP_CLAIM_TTL_SECONDS = 300.0
    _CLEANUP_RETRY_BASE_SECONDS = 2.0
    _CLEANUP_RETRY_MAX_SECONDS = 300.0

    def __init__(self, store: Store):
        self.store = store
        ensure_schema(store)

    def dispatch_lock(self):
        """Expose the Store's cross-process tick lock to scheduler callers."""
        return self.store.dispatch_lock()

    @staticmethod
    def _model(kind):
        try:
            return _GPU_MODELS[kind]
        except KeyError as exc:
            raise ValueError(f"未知GPU实体类型: {kind}") from exc

    @staticmethod
    def _row(row):
        if row is None:
            return None
        reserved = {"id", "owner_id", "project_id", "state", "version", "created_at", "updated_at"}
        result = {
            "id": row.id,
            "owner_id": row.owner_id,
            "project_id": row.project_id,
            "created_at": utc_isoformat(row.created_at),
            "updated_at": utc_isoformat(row.updated_at),
        }
        result.update({k: copy.deepcopy(v) for k, v in (row.data or {}).items() if k not in reserved})
        # v0.2 stores identity in typed columns as well as JSON.  Old rows may
        # have only the typed value after Alembic 0003; expose and reconcile it
        # at the repository boundary instead of silently dropping ownership.
        for field in ("account_id", "instance_id", "current_job_id"):
            if hasattr(row, field):
                value = getattr(row, field)
                if value is not None:
                    result[field] = copy.deepcopy(value)
        if isinstance(row, GpuWorkflowRow):
            workflow_version = row.workflow_version
            if isinstance(workflow_version, str) and workflow_version.isdigit():
                workflow_version = int(workflow_version)
            result["workflow_key"] = row.workflow_key
            result["workflow_version"] = workflow_version
        if row.state is not None:
            result["state"] = row.state
        if row.version is not None:
            result["version"] = row.version
        if isinstance(row, GpuJobRow):
            result.update(idempotency_key=row.idempotency_key, body_hash=row.body_hash)
        return result

    def create(self, kind, owner_id, data, project_id=None, record_id=None):
        model = self._model(kind)
        payload = copy.deepcopy(dict(data))
        rid = record_id or str(uuid.uuid4())
        effective_project = project_id if project_id is not None else payload.get("project_id")
        workflow_key = workflow_version = None
        if kind == "workflows":
            workflow_key = payload.get("workflow_key")
            workflow_version = payload.get("workflow_version")
            if not isinstance(workflow_key, str) or not workflow_key:
                raise ValueError("workflow_key必填")
            if not (
                (
                    isinstance(workflow_version, int)
                    and not isinstance(workflow_version, bool)
                    and workflow_version > 0
                )
                or (isinstance(workflow_version, str) and workflow_version)
            ):
                raise ValueError("workflow_version格式无效")
        if kind == "files" and effective_project is None:
            storage_key = payload.get("storage_key")
            parts = storage_key.split("/") if isinstance(storage_key, str) else []
            if len(parts) >= 2 and parts[0] == "gpu":
                source_job = self.get("jobs", parts[1], owner_id)
                effective_project = source_job.get("project_id") if source_job else None
        kwargs = {
            "id": rid,
            "owner_id": owner_id,
            "project_id": effective_project,
            "state": payload.get("state"),
            "data": payload,
            "version": 0,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        for field in ("account_id", "instance_id", "current_job_id"):
            if hasattr(model, field) and field in payload:
                kwargs[field] = payload[field]
        if model is GpuWorkflowRow:
            kwargs.update(workflow_key=workflow_key, workflow_version=str(workflow_version))
        if model is GpuJobRow:
            key = payload.get("idempotency_key") or f"__direct__:{rid}"
            kwargs.update(
                idempotency_key=key,
                body_hash=payload.get("body_hash") or digest(payload.get("request_body", payload)),
            )
        try:
            with self.store.db_lock(), self.store.Session.begin() as session:
                if kind in {"usage", "instance_usage_segments", "executions"}:
                    for parent, key in ((GpuLeaseRow, "lease_id"), (GpuInstanceRow, "instance_id"), (GpuJobRow, "job_id")):
                        if payload.get(key):
                            session.scalar(select(parent).where(
                                parent.id == payload[key], parent.owner_id == owner_id
                            ).with_for_update())
                row = model(**kwargs)
                session.add(row)
                session.flush()
                return self._row(row)
        except IntegrityError as exc:
            raise ValueError("GPU记录已存在或违反唯一约束") from exc

    def get(self, kind, id, owner_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            return self._row(session.scalar(stmt))

    def list(self, kind, owner_id=None, project_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session() as session:
            stmt = select(model).order_by(model.created_at.desc())
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            if project_id is not None:
                stmt = stmt.where(model.project_id == project_id)
            return [self._row(row) for row in session.scalars(stmt).all()]

    @staticmethod
    def _has_stop_or_delete_flag(data: Mapping[str, object] | None) -> bool:
        if not isinstance(data, Mapping):
            return False
        return any(bool(data.get(flag)) for flag in _STOP_DELETE_FLAGS)

    @staticmethod
    def _valid_generation_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return value

    @classmethod
    def _execution_matches_bindings_and_generation(
        cls,
        *,
        owner_id: str,
        job: GpuJobRow,
        execution: GpuExecutionRow | None,
        lease: GpuLeaseRow | None = None,
        instance: GpuInstanceRow | None = None,
        lease_id: str | None = None,
        instance_id: str | None = None,
        active: Mapping[str, object] | None = None,
        expected_generation: int | None = None,
    ) -> tuple[bool, int | None]:
        """Verify strict non-null ownership (job/lease/instance) and generation across execution writers."""
        jdata = job.data or {}
        ldata = (lease.data or {}) if lease is not None else {}
        idata = (instance.data or {}) if instance is not None else {}

        eff_lease_id = lease.id if lease is not None else (lease_id or jdata.get("lease_id"))
        eff_inst_id = (
            instance.id
            if instance is not None
            else (instance_id or jdata.get("instance_id") or ldata.get("instance_id") or (lease.instance_id if lease is not None else None))
        )

        active_map = active if isinstance(active, Mapping) else None
        agen = active_map.get("generation") if active_map is not None else None
        jgen = jdata.get("generation")
        lgen = ldata.get("generation") if lease is not None else None
        igen = idata.get("generation") if instance is not None else None
        egen = (execution.data or {}).get("generation") if execution is not None else None

        # Execution rows are mandatory, even when every generation source is absent.
        if execution is None or execution.owner_id != owner_id:
            return False, None
        edata = execution.data or {}
        exec_gen = cls._valid_generation_int(egen)
        markers = [expected_generation, jgen, lgen, igen]
        if active_map is not None:
            if cls._valid_generation_int(agen) is None:
                return False, None
            markers.append(agen)
        current = [cls._valid_generation_int(v) for v in markers if v is not None]
        if (not any(v is not None for v in (jgen, lgen, igen, agen))
            or exec_gen is None or not current or any(v != exec_gen for v in current)):
            return False, None
        if (
            not eff_lease_id
            or not eff_inst_id
            or edata.get("job_id") != job.id
            or edata.get("lease_id") != eff_lease_id
            or (execution.instance_id or edata.get("instance_id")) != eff_inst_id
            or (execution.instance_id and edata.get("instance_id") not in {None, execution.instance_id})
            or jdata.get("lease_id") != eff_lease_id
            or (job.instance_id or jdata.get("instance_id")) != eff_inst_id
            or (jdata.get("instance_id") is not None and jdata.get("instance_id") != eff_inst_id)
            or (ldata.get("instance_id") is not None and ldata.get("instance_id") != eff_inst_id)
            or (lease is not None and (lease.owner_id != owner_id or (lease.instance_id or ldata.get("instance_id")) != eff_inst_id))
            or (instance is not None and (instance.owner_id != owner_id or idata.get("lease_id") != eff_lease_id))
        ):
            return False, None
        resolved_gen = exec_gen

        return True, resolved_gen

    @classmethod
    def _collect_associated_jobs_for_scope(
        cls,
        session,
        owner_id: str,
        *,
        lease_id: str | None = None,
        instance_id: str | None = None,
        lease_data: Mapping[str, object] | None = None,
        inst_data: Mapping[str, object] | None = None,
    ) -> list[GpuJobRow]:
        """Collect all owner jobs referencing this lease or instance in any state."""
        seed_job_ids: set[str] = set()
        if isinstance(lease_data, Mapping):
            for key in ("owner_job_id", "current_job_id"):
                val = lease_data.get(key)
                if isinstance(val, str) and val:
                    seed_job_ids.add(val)
        if isinstance(inst_data, Mapping):
            val = inst_data.get("current_job_id")
            if isinstance(val, str) and val:
                seed_job_ids.add(val)
        rows = session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id).with_for_update()).all()
        matched: list[GpuJobRow] = []
        for jrow in rows:
            jdata = jrow.data or {}
            j_inst = jrow.instance_id or jdata.get("instance_id")
            j_lease = jdata.get("lease_id")
            j_intent = jdata.get("start_intent") if isinstance(jdata.get("start_intent"), Mapping) else {}
            if (
                (lease_id and j_lease == lease_id)
                or (lease_id and j_intent.get("lease_id") == lease_id)
                or (instance_id and j_inst == instance_id)
                or (instance_id and j_intent.get("instance_id") == instance_id)
                or (jrow.id in seed_job_ids and (lease_id is None or j_lease in {None, lease_id}))
            ):
                matched.append(jrow)
        return matched

    @classmethod
    def _has_positive_usage_or_segments(
        cls,
        session,
        owner_id: str,
        *,
        lease_id: str | None = None,
        instance_id: str | None = None,
        job_ids: set[str] | None = None,
    ) -> bool:
        """Return True if any usage or usage-segment row shows positive consumption for this runtime scope."""
        jids = set(job_ids or ())
        segments = session.scalars(
            select(GpuInstanceUsageSegmentRow).where(GpuInstanceUsageSegmentRow.owner_id == owner_id).with_for_update()
        ).all()
        for seg in segments:
            if not row_has_positive_usage(seg):
                continue
            sdata = seg.data or {}
            seg_inst = seg.instance_id or sdata.get("instance_id")
            if (
                (instance_id and seg_inst == instance_id)
                or (lease_id and sdata.get("lease_id") == lease_id)
                or (sdata.get("job_id") in jids)
            ):
                return True
        usage_rows = session.scalars(select(GpuUsageRow).where(GpuUsageRow.owner_id == owner_id).with_for_update()).all()
        for urow in usage_rows:
            if not row_has_positive_usage(urow):
                continue
            udata = urow.data or {}
            if (
                (lease_id and udata.get("lease_id") == lease_id)
                or (instance_id and udata.get("instance_id") == instance_id)
                or (udata.get("job_id") in jids)
            ):
                return True
        return False

    def _three_entity_fenced(
        self,
        session,
        job: GpuJobRow,
        owner_id: str,
        lease: GpuLeaseRow | None,
        instance: GpuInstanceRow | None,
        *,
        lease_id: str | None = None,
        instance_id: str | None = None,
        require_instance: bool = False,
        allow_legacy_unknown_stub: bool = False,
    ) -> bool:
        """Shared 3-entity (job, lease, instance) status, ownership, and delete/stop fence."""
        jdata = job.data or {}
        job_fenced = (
            job.owner_id != owner_id
            or bool(jdata.get("cancel_requested"))
            or jdata.get("requested_outcome") == "cancelled"
            or self._has_stop_or_delete_flag(jdata)
        )
        lease_data = (lease.data or {}) if lease is not None else {}
        lease_fenced = (lease_id is not None and lease is None) or (
            lease is not None
            and (
                lease.owner_id != owner_id
                or jdata.get("lease_id") not in {None, lease.id}
                or lease.state in {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted", "unknown"}
                or self._has_stop_or_delete_flag(lease_data)
                or lease_data.get("owner_job_id") not in {None, job.id}
                or lease_data.get("current_job_id") not in {None, job.id}
                or (
                    instance is not None
                    and (lease.instance_id or lease_data.get("instance_id")) not in {None, instance.id}
                )
            )
        )
        inst_data = (instance.data or {}) if instance is not None else {}
        stub_lease = lease
        if allow_legacy_unknown_stub and stub_lease is None and instance is not None and inst_data.get("legacy_lease_id"):
            stub_lease = session.get(GpuLeaseRow, inst_data["legacy_lease_id"])
            if stub_lease is not None and stub_lease.owner_id != owner_id:
                stub_lease = None
        legacy_stub_unknown = (
            allow_legacy_unknown_stub
            and instance is not None
            and instance.state == "unknown"
            and bool(inst_data.get("legacy_lease_id"))
            and not inst_data.get("allocation_key")
            and stub_lease is not None
            and stub_lease.state in {"active", "creating"}
        )
        blocked_inst_states = {"deleted", "deleting", "releasing", "cleanup_pending", "quarantined"}
        if not legacy_stub_unknown:
            blocked_inst_states.add("unknown")
        inst_fenced = (require_instance and instance_id is not None and instance is None) or (
            instance is not None
            and (
                instance.owner_id != owner_id
                or jdata.get("instance_id") not in {None, instance.id}
                or instance.state in blocked_inst_states
                or self._has_stop_or_delete_flag(inst_data)
                or inst_data.get("current_job_id") not in {None, job.id}
                or (
                    lease is not None
                    and inst_data.get("lease_id") not in {None, lease.id}
                )
            )
        )
        return bool(job_fenced or lease_fenced or inst_fenced)

    def record_gpu_diagnostics(self, job, diagnostics):
        """Save worker-measured devices on the running job, its lease and instance."""
        from .diagnostics import gpu_diagnostics

        clean = gpu_diagnostics(diagnostics)
        if clean is None or not isinstance(job, Mapping) or not job.get("id") or not job.get("owner_id"):
            return None
        patch = {"gpu_diagnostics": clean, "gpu_reported": ", ".join(d["name"] for d in clean["devices"])}
        owner_id = job["owner_id"]
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                row = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job["id"], GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if row is None or row.owner_id != owner_id or row.state != "running":
                    return None
                jdata = row.data or {}
                exec_id = job.get("execution_id") or jdata.get("execution_id")
                if jdata.get("execution_id") and exec_id != jdata.get("execution_id"):
                    return self._row(row)
                lease_id = jdata.get("lease_id") or job.get("lease_id")
                if jdata.get("lease_id") and lease_id != jdata.get("lease_id"):
                    return self._row(row)
                lease = session.get(GpuLeaseRow, lease_id) if lease_id else None
                inst_id = (
                    jdata.get("instance_id")
                    or ((lease.data or {}).get("instance_id") if lease is not None else None)
                    or (lease.instance_id if lease is not None else None)
                    or job.get("instance_id")
                )
                if jdata.get("instance_id") and inst_id != jdata.get("instance_id"):
                    return self._row(row)
                inst = session.get(GpuInstanceRow, inst_id) if inst_id else None
                inst_active = (inst.data or {}).get("active_execution") if inst is not None else None
                lease_active = (lease.data or {}).get("active_execution") if lease is not None else None
                if isinstance(inst_active, Mapping) and isinstance(lease_active, Mapping) and inst_active != lease_active:
                    return self._row(row)
                active = inst_active if isinstance(inst_active, Mapping) else lease_active
                if isinstance(active, Mapping) and (
                    active.get("active_execution_id") != exec_id
                    or (lease_id is not None and active.get("lease_id") != lease_id)
                    or (inst_id is not None and active.get("instance_id") not in {None, inst_id})
                ):
                    return self._row(row)
                execution = session.get(GpuExecutionRow, exec_id) if exec_id else None
                if execution is None:
                    return self._row(row)
                if execution is not None:
                    exec_ok, _ = self._execution_matches_bindings_and_generation(
                        owner_id=owner_id,
                        job=row,
                        execution=execution,
                        lease=lease,
                        instance=inst,
                        lease_id=lease_id,
                        instance_id=inst_id,
                        active=active if isinstance(active, Mapping) else None,
                    )
                    if not exec_ok or execution.state in {"failed", "aborted", "cancelled"}:
                        return self._row(row)
                if self._three_entity_fenced(
                    session,
                    row,
                    owner_id,
                    lease,
                    inst,
                    lease_id=lease_id,
                    instance_id=inst_id,
                    require_instance=bool(inst_id),
                    allow_legacy_unknown_stub=True,
                ):
                    return self._row(row)
                rows = [row]
                if lease is not None:
                    rows.append(lease)
                if inst is not None:
                    rows.append(inst)
                for item in rows:
                    item.data = {**(item.data or {}), **copy.deepcopy(patch)}
                    item.version += 1
                    item.updated_at = utcnow()
                session.flush()
                return self._row(row)

    def update(self, kind, id, patch, owner_id=None):
        model = self._model(kind)
        with self.store.db_lock(), self.store.Session.begin() as session:
            stmt = select(model).where(model.id == id)
            if owner_id is not None:
                stmt = stmt.where(model.owner_id == owner_id)
            row = session.scalar(stmt.with_for_update())
            if row is None:
                raise KeyError(id)
            if kind == "workflows" and row.state == "published" and set(patch) - {"active"}:
                raise ValueError("已发布workflow版本不可覆写")
            changes = copy.deepcopy(dict(patch))
            row.data = {**(row.data or {}), **changes}
            for field in ("account_id", "instance_id", "current_job_id"):
                if hasattr(model, field) and field in changes:
                    setattr(row, field, changes[field])
            if "state" in changes:
                row.state = changes["state"]
            if "project_id" in changes:
                row.project_id = changes["project_id"]
            if kind == "workflows":
                if "workflow_key" in changes:
                    if not isinstance(changes["workflow_key"], str) or not changes["workflow_key"]:
                        raise ValueError("workflow_key格式无效")
                    row.workflow_key = changes["workflow_key"]
                if "workflow_version" in changes:
                    value = changes["workflow_version"]
                    if not (
                        (isinstance(value, int) and not isinstance(value, bool) and value > 0)
                        or (isinstance(value, str) and value)
                    ):
                        raise ValueError("workflow_version格式无效")
                    row.workflow_version = str(value)
            row.version += 1
            row.updated_at = utcnow()
            if kind=='jobs' and row.data.get('batch_id'):
                self._refresh_batch(session,row.data['batch_id'],row.owner_id)
            session.flush()
            return self._row(row)

    def update_credential_if_version(
        self, owner_id: str, credential_id: str, expected_version: int, patch: Mapping[str, object]
    ) -> dict | None:
        """Patch one live credential; SQLite needs a write transaction before the version read."""
        with self.store.db_lock(), self.store.Session.begin() as session:
            if self.store.engine.dialect.name == "sqlite":
                session.execute(text("BEGIN IMMEDIATE"))
            row = session.scalar(
                select(GpuCredentialRow)
                .where(GpuCredentialRow.id == credential_id, GpuCredentialRow.owner_id == owner_id)
                .with_for_update()
            )
            if row is None or row.version != expected_version or (row.data or {}).get("revoked") is True:
                return None
            row.data = {**(row.data or {}), **copy.deepcopy(dict(patch))}
            row.version += 1
            row.updated_at = utcnow()
            session.flush()
            return self._row(row)

    def create_job(self, owner_id, body, idempotency_key, *, admitted=None):
        raw = copy.deepcopy(dict(body))
        body_hash = digest(raw)
        for attempt in range(8):
            try:
                with self.store.db_lock(), self.store.Session.begin() as session:
                    existing = session.scalar(
                        select(GpuJobRow).where(
                            GpuJobRow.owner_id == owner_id, GpuJobRow.idempotency_key == idempotency_key
                        )
                    )
                    if existing:
                        if existing.body_hash != body_hash:
                            raise ValueError("同一owner的idempotency_key已用于不同body")
                        return self._row(existing), False
                    policy = raw.get("execution_policy") or {}
                    payload = {
                        **raw,
                        "request_body": raw,
                        "idempotency_key": idempotency_key,
                        "body_hash": body_hash,
                        "state": "queued",
                        "provider": policy.get("provider", "mock"),
                        "workflow_key": raw.get("workflow_key"),
                        "workflow_version": raw.get("workflow_version"),
                        "model_id": raw.get("model_id"),
                        "params": copy.deepcopy(raw.get("params", {})),
                        "resolved_seed": None,
                        "snapshot": None,
                        "max_runtime_minutes": policy.get("max_runtime_minutes"),
                        "estimated_units": None,
                        "unit_rate": None,
                        "requested_outcome": None,
                        "cancel_requested": False,
                        "output_asset_ids": [],
                    }
                    if admitted is not None:
                        payload.update(copy.deepcopy(admitted))
                    row = GpuJobRow(
                        id=str(uuid.uuid4()),
                        owner_id=owner_id,
                        project_id=raw.get("project_id"),
                        state="queued",
                        data=payload,
                        version=0,
                        idempotency_key=idempotency_key,
                        body_hash=body_hash,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(row)
                    session.flush()
                    return self._row(row), True
            except IntegrityError:
                if attempt == 7:
                    raise ValueError("GPU job创建竞争失败，请重试")
                time.sleep(0.01 * (attempt + 1))
            except OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise RuntimeError("GPU job创建未完成")

    def create_batch(self, owner_id, jobs, idempotency_key, policy, request_body=None):
        for attempt in range(8):
            try:
                return self._create_batch_once(owner_id,jobs,idempotency_key,policy,request_body)
            except IntegrityError:
                if attempt==7:
                    raise ValueError('batch creation contention; retry')
            except OperationalError as exc:
                if 'locked' not in str(exc).lower() or attempt==7:
                    raise
            time.sleep(0.02*(attempt+1))

    def _refresh_batch(self, session, batch_id, owner_id):
        session.flush()
        batch=session.get(GpuBatchRow,batch_id)
        if not batch or batch.owner_id!=owner_id:
            return
        jobs=list(session.scalars(select(GpuJobRow).where(
            GpuJobRow.owner_id==owner_id,GpuJobRow.id.in_(batch.data.get('job_ids',[])))))
        counts={state:sum(job.state==state for job in jobs) for state in ('completed','failed','cancelled')}
        finished=sum(counts.values())
        if jobs and finished==len(jobs):
            state=next((name for name,count in counts.items() if count==len(jobs)),'partial')
        else:
            state='running' if any(job.state!='queued' for job in jobs) else 'queued'
        batch.state=state
        batch.data={**batch.data,'state':state,'finished_jobs':finished,'counts':counts}
        batch.version+=1
        batch.updated_at=utcnow()

    def _create_batch_once(self, owner_id, jobs, idempotency_key, policy, request_body=None):
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= 100:
            raise ValueError("batch requires 1..100 jobs")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
            raise ValueError("invalid idempotency key")
        if policy.get("mode", "save") not in {"save", "smart", "batch"}:
            raise ValueError("invalid lifecycle mode")
        body_hash = digest(request_body if request_body is not None else {"jobs": jobs, "policy": policy})
        with self.store.db_lock(), self.store.Session.begin() as session:
            existing = session.scalar(select(GpuBatchRow).where(
                GpuBatchRow.owner_id == owner_id, GpuBatchRow.idempotency_key == idempotency_key))
            if existing:
                if existing.body_hash != body_hash:
                    raise ValueError("batch idempotency conflict")
                return self._row(existing), False
            batch_id = str(uuid.uuid4())
            payload = {"policy_snapshot": copy.deepcopy(policy), "total_jobs": len(jobs),
                       "finished_jobs": 0, "mode": policy.get("mode", "save"), "job_ids": [],
                       "request_key":idempotency_key,"request_hash":body_hash}
            for index, job in enumerate(jobs):
                job_id = str(uuid.uuid4())
                data = copy.deepcopy(job)
                raw_jobs=(request_body or {}).get('jobs',[])
                data['request_body']=copy.deepcopy(raw_jobs[index] if index<len(raw_jobs) else {
                    k:v for k,v in job.items() if k in {'project_id','model_id','workflow_key','workflow_version','params','asset_ids','execution_policy','preset_id','loras'}})
                data.update(batch_id=batch_id, queue_order=index, state="queued",
                            execution_policy={**data.get("execution_policy", {}), **policy},
                            provider=data.get("execution_policy", {}).get("provider", "mock"))
                if data['provider'] not in {'mock', 'colab'}:
                    raise ValueError('batch provider is not authorized')
                data['compatibility_signature'] = digest({
                    'provider':data['provider'],
                    'environment':{k:v for k,v in (data.get('snapshot') or {}).items()
                                   if k not in {'resolved_params','bound_graph','input_files'}},
                    'workflow_key':data.get('workflow_key'), 'workflow_version':data.get('workflow_version'),
                    'requested_spec':data.get('requested_spec')})
                row = GpuJobRow(id=job_id, owner_id=owner_id, project_id=data.get('project_id'),
                                state='queued', data=data, version=0,
                                idempotency_key=f'batch:{batch_id}:{index}', body_hash=digest(job),
                                created_at=utcnow(), updated_at=utcnow())
                session.add(row)
                payload['job_ids'].append(job_id)
            batch = GpuBatchRow(id=batch_id, owner_id=owner_id,
                                project_id=jobs[0].get('project_id'), state='queued', version=0,
                                data=payload, idempotency_key=idempotency_key, body_hash=body_hash,
                                created_at=utcnow(), updated_at=utcnow())
            session.add(batch)
            session.flush()
            return self._row(batch), True

    def _record_reuse_observation(
        self,
        session,
        owner_id: str,
        project_id: str | None,
        job_id: str,
        assessment,
        *,
        instance_id: str | None = None,
        lease_id: str | None = None,
    ) -> None:
        event_data = {
            "event": "reuse_lifetime_observed",
            "job_id": job_id,
            "instance_id": instance_id,
            "lease_id": lease_id,
            "state": "reuse_lifetime_observed",
            "remaining_s": assessment.remaining_seconds,
            "need_s": assessment.need_seconds,
            "warm_verified": assessment.warm_verified,
            "evidence_source": assessment.evidence_source,
            "sampling_source": assessment.sampling_source,
            "would_allow": assessment.can_reuse,
            "reason": assessment.reason,
            "message": f"reuse window observed ({assessment.reason})",
        }
        now_dt = utcnow()
        session.add(
            GpuEventRow(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                project_id=project_id,
                state="reuse_lifetime_observed",
                version=0,
                data=event_data,
                created_at=now_dt,
                updated_at=now_dt,
            )
        )

    def _check_reuse_in_session(
        self,
        session,
        owner_id: str,
        job_row: GpuJobRow,
        lease_row: GpuLeaseRow,
        instance_row: GpuInstanceRow | None,
        now: float,
        guard_mode: str,
    ) -> tuple[bool, float | None]:
        inst_data = {**(instance_row.data or {}), "id": instance_row.id} if instance_row is not None else {}
        lease_data = {**(lease_row.data or {}), "id": lease_row.id}
        job_data = job_row.data or {}
        effective_deadline = min_deadlines(
            lease_data.get("deadline_epoch"),
            lease_data.get("hard_deadline_at"),
            inst_data.get("hard_deadline_at") if instance_row is not None else None,
            job_data.get("deadline_epoch"),
            job_data.get("hard_deadline_at"),
        )
        if effective_deadline is not None and now >= effective_deadline:
            return False, effective_deadline
        if str(guard_mode or "enforce").strip().lower() == "off" or is_legacy_mock_job(job_data):
            return True, effective_deadline
        history_jobs, executions_by_id, usage_by_job = load_session_history(
            session, owner_id, GpuJobRow, GpuExecutionRow, GpuUsageRow
        )
        allowed, _ = check_job_reuse_guard(
            job_data,
            guard_mode=guard_mode,
            now=now,
            instance_data=inst_data if instance_row is not None else None,
            lease_data=lease_data,
            deadlines=(effective_deadline,),
            history_jobs=history_jobs,
            executions_by_id=executions_by_id,
            usage_by_job=usage_by_job,
            observe_cb=lambda assessment: self._record_reuse_observation(
                session,
                owner_id,
                job_row.project_id,
                job_row.id,
                assessment,
                instance_id=instance_row.id if instance_row is not None else None,
                lease_id=lease_row.id,
            ),
        )
        return allowed, effective_deadline

    def attach_idle_lease(self, owner_id, job_id, lease_id, now, *, guard_mode="enforce"):
        """Atomically attach a compatible idle lease to a waiting job under admission_lock."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            return self._attach_idle_lease_locked(owner_id, job_id, lease_id, now, guard_mode=guard_mode)

    def _attach_idle_lease_locked(self, owner_id, job_id, lease_id, now, *, guard_mode="enforce"):
        with self.store.db_lock(), self.store.Session.begin() as session:
            job = session.get(GpuJobRow, job_id)
            lease = session.get(GpuLeaseRow, lease_id)
            if not job or not lease or job.owner_id != owner_id or lease.owner_id != owner_id:
                return False
            jdata = job.data or {}
            ldata = lease.data or {}
            if (
                lease.state != "idle"
                or ldata.get("current_job_id") not in {None, job_id}
                or jdata.get("lease_id") not in {None, lease_id}
                or job.state != "waiting_for_capacity"
                or jdata.get("cancel_requested")
                or jdata.get("requested_outcome") == "cancelled"
                or self._has_stop_or_delete_flag(jdata)
                or self._has_stop_or_delete_flag(ldata)
                or not jdata.get("compatibility_signature")
                or jdata["compatibility_signature"] != ldata.get("compatibility_signature")
            ):
                return False
            handle = ldata.get("handle")
            if not isinstance(handle, Mapping) or not handle:
                return False
            idle_deadline = ldata.get("idle_deadline_at")
            if idle_deadline is not None and now >= float(idle_deadline):
                return False
            instance_id = jdata.get("instance_id") or lease.instance_id or ldata.get("instance_id")
            instance = session.get(GpuInstanceRow, instance_id) if instance_id else None
            if (jdata.get("instance_id") and (lease.instance_id or ldata.get("instance_id")) not in {None, instance_id}):
                return False
            if instance is not None:
                idata = instance.data or {}
                if (
                    instance.owner_id != owner_id
                    or instance.state in {"deleted", "deleting", "unknown", "releasing", "cleanup_pending", "quarantined"}
                    or self._has_stop_or_delete_flag(idata)
                    or idata.get("current_job_id") not in {None, job_id}
                    or idata.get("lease_id") not in {None, lease_id}
                ):
                    return False
            conflicting, _ = has_conflicting_active_runtime(
                session,
                instance_id=instance_id,
                lease=lease,
                instance=instance,
                job=job,
                exclude_lease_id=lease_id,
                exclude_instance_id=instance_id,
            )
            if conflicting:
                return False
            allowed, effective_deadline = self._check_reuse_in_session(
                session, owner_id, job, lease, instance, now, guard_mode
            )
            if not allowed:
                return False
            lease.state = "active"
            lease.data = {
                **lease.data,
                "state": "active",
                "owner_job_id": job_id,
                "current_job_id": job_id,
                "last_active_at": now,
                "idle_deadline_at": None,
                "handle": retarget_handle(handle, job_id),
            }
            next_job_data = {**job.data, "lease_id": lease_id}
            if effective_deadline is not None:
                next_job_data["deadline_epoch"] = effective_deadline
            elif lease.data.get("deadline_epoch") is not None:
                next_job_data["deadline_epoch"] = lease.data["deadline_epoch"]
            job.data = next_job_data
            for row in (lease, job):
                row.version += 1
                row.updated_at = utcnow()
            return True

    def handoff_lease(self, owner_id, job_id, next_job_id, lease_id, now, *, guard_mode="enforce"):
        """Transfer a lease under the same admission fence as allocation/bind."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            return self._handoff_lease_locked(owner_id, job_id, next_job_id, lease_id, now, guard_mode=guard_mode)

    def _handoff_lease_locked(self, owner_id, job_id, next_job_id, lease_id, now, *, guard_mode="enforce"):
        """Finish old job and transfer exclusive lease ownership in one DB transaction."""
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.get(GpuJobRow,job_id)
            following=session.get(GpuJobRow,next_job_id)
            lease=session.get(GpuLeaseRow,lease_id)
            def association(row, key):
                data_value = (row.data or {}).get(key)
                typed_value = getattr(row, key, None)
                if data_value not in (None, '') and typed_value not in (None, '') and data_value != typed_value:
                    return False, None
                return True, data_value if data_value not in (None, '') else typed_value

            valid = True
            associations = {}
            for name, row in (("job", job), ("following", following), ("lease", lease)):
                for key in ("account_id", "instance_id"):
                    ok, value = association(row, key) if row is not None else (False, None)
                    valid = valid and ok
                    associations[f"{name}_{key}"] = value
            instance_id = associations.get('job_instance_id') or associations.get('lease_instance_id')
            instance = session.get(GpuInstanceRow, instance_id) if instance_id else None
            if instance is not None:
                for key in ("account_id", "current_job_id"):
                    ok, value = association(instance, key)
                    valid = valid and ok
                    associations[f"instance_{key}"] = value
            if not valid or not all(row is not None and row.owner_id==owner_id for row in (job,following,lease)):
                raise ValueError('lease handoff association conflict')
            if instance_id and (instance is None or instance.owner_id != owner_id):
                raise ValueError('lease handoff instance ownership mismatch')
            if associations.get('job_instance_id') not in {None, instance_id} or associations.get('lease_instance_id') not in {None, instance_id}:
                raise ValueError('lease handoff instance mismatch')
            lease_account = associations.get('lease_account_id')
            if associations.get('job_account_id') not in {None, lease_account} or associations.get('following_account_id') not in {None, lease_account}:
                return False
            if instance is not None and (not associations.get('instance_account_id') or lease_account not in {None, associations['instance_account_id']}):
                return False
            if instance is not None and associations.get('instance_current_job_id') not in {None, job_id}:
                return False
            if job.state!='cleaning_up' or following.state not in {'queued','validating','waiting_for_capacity'}:
                return False
            if following.data.get('lease_id') not in {None,lease_id}:
                return False
            others=session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id==owner_id)).all()
            if any(row.id!=lease_id and row.data.get('owner_job_id')==next_job_id and row.state!='deleted' for row in others):
                return False
            if (
                has_conflicting_active_runtime(
                    session,
                    instance_id=instance_id,
                    lease=lease,
                    instance=instance,
                    job=job,
                    exclude_lease_id=lease_id,
                    exclude_instance_id=instance_id,
                )[0]
            ):
                return False
            if following.data.get('cancel_requested') or lease.data.get('owner_job_id')!=job_id:
                return False
            lease_account = (lease.data or {}).get('account_id') or lease.account_id
            job_account = (job.data or {}).get('account_id') or job.account_id
            following_account = (following.data or {}).get('account_id') or following.account_id
            if any(value is not None and value != lease_account for value in (job_account, following_account)):
                return False
            instance_data = copy.deepcopy(instance.data or {}) if instance is not None else {}
            account_id = instance_data.get('account_id') or (instance.account_id if instance is not None else None)
            if instance is not None:
                lease_account = (lease.data or {}).get('account_id') or lease.account_id
                job_account = (job.data or {}).get('account_id') or job.account_id
                following_account = (following.data or {}).get('account_id') or following.account_id
                if not account_id or any(value not in {None, account_id} for value in (lease_account, job_account, following_account)):
                    return False
                if instance.current_job_id not in {None, job_id}:
                    return False
                if instance.state in {'deleted', 'unknown', 'releasing', 'cleanup_pending', 'quarantined'}:
                    return False
                if instance_data.get('manual_stop_requested') or instance_data.get('runtime_deleted') or instance_data.get('remote_deleted') or instance_data.get('cleanup_claim_token'):
                    return False
            if (
                job.data.get("compatibility_signature") != following.data.get("compatibility_signature")
                or not job.data.get("compatibility_signature")
                or lease.state not in {"active", "busy"}
            ):
                return False
            allowed, effective_deadline = self._check_reuse_in_session(
                session, owner_id, following, lease, instance, now, guard_mode
            )
            if not allowed:
                return False
            handoff_deadline = (
                effective_deadline if effective_deadline is not None else lease.data.get("deadline_epoch")
            )
            execution_id = job.data.get('execution_id')
            execution=session.get(GpuExecutionRow, execution_id) if execution_id else None
            next_execution_id = following.data.get('execution_id')
            next_execution = session.get(GpuExecutionRow, next_execution_id) if next_execution_id else None
            def execution_matches(execution, expected_job_id):
                if execution is None or execution.owner_id != owner_id:
                    return False
                data = execution.data or {}
                for key, expected in (("job_id", expected_job_id), ("lease_id", lease_id),
                                      ("instance_id", instance_id), ("account_id", associations.get('instance_account_id') or lease_account)):
                    json_value = data.get(key)
                    typed_value = getattr(execution, key, None)
                    if json_value not in (None, '') and typed_value not in (None, '') and json_value != typed_value:
                        return False
                    value = json_value if json_value not in (None, '') else typed_value
                    if value not in (None, '') and value != expected:
                        return False
                return True

            if execution is not None and not execution_matches(execution, job_id):
                return False
            if next_execution is not None and not execution_matches(next_execution, next_job_id):
                return False
            job.data={**job.data,'state':'completed','runtime_deleted':False,'stopped_at':now,
                      **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
            job.state='completed'
            if instance is not None:
                job.instance_id = instance.id
                job.account_id = account_id
            following.data={**following.data,'lease_id':lease_id,
                            **({'deadline_epoch':handoff_deadline} if handoff_deadline is not None else {}),
                            **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
            if instance is not None:
                following.instance_id = instance.id
                following.account_id = account_id
            handle=lease.data.get('handle')
            # Handle ownership must follow the lease owner for every provider (real ones verify it).
            lease.data={**lease.data,'owner_job_id':next_job_id,'current_job_id':next_job_id,
                        'last_active_at':now,'state':'active',
                        **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {}),
                        **({'handle':retarget_handle(handle,next_job_id)} if isinstance(handle,dict) else {})}
            lease.state='active'
            if instance is not None:
                lease.instance_id = instance.id
                lease.account_id = account_id
                instance_data.update(current_job_id=next_job_id, instance_id=instance.id, account_id=account_id,
                                     lease_id=lease_id, last_seen_at=now, runtime_deleted=False, state='allocating')
                instance.data = instance_data
                instance.current_job_id = next_job_id
                instance.account_id = account_id
                instance.state = 'allocating'
            execution_id = job.data.get('execution_id')
            execution=session.get(GpuExecutionRow, execution_id) if execution_id else None
            if execution is not None:
                execution.state='completed'
                execution.data={**execution.data,'state':'completed','job_id':job_id,'lease_id':lease_id,
                                'finished_at':now,
                                **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
                if instance is not None:
                    execution.instance_id=instance.id
                    execution.account_id=account_id
                execution.updated_at=utcnow()
                execution.version += 1
            next_execution_id = following.data.get('execution_id')
            next_execution = session.get(GpuExecutionRow, next_execution_id) if next_execution_id else None
            if next_execution is not None:
                next_execution.data={**next_execution.data,'job_id':next_job_id,'lease_id':lease_id,
                                     **({'instance_id':instance.id,'account_id':account_id} if instance is not None else {})}
                if instance is not None:
                    next_execution.instance_id=instance.id
                    next_execution.account_id=account_id
                next_execution.updated_at=utcnow()
                next_execution.version += 1
            for row in (job,following,lease):
                row.version+=1
                row.updated_at=utcnow()
            if instance is not None:
                instance.version += 1
                instance.updated_at=utcnow()
            if job.data.get('batch_id'):
                self._refresh_batch(session,job.data['batch_id'],owner_id)
            return True

    @staticmethod
    def _cleanup_blocking(state):
        return state in {'creating', 'active', 'busy', 'idle', 'cleanup_pending', 'quarantined', 'releasing', 'deleting', 'unknown'}

    @classmethod
    def _claim_is_live(cls, data, now):
        claimed_at = data.get('cleanup_claimed_at')
        try:
            return claimed_at is not None and float(now) - float(claimed_at) < cls._CLEANUP_CLAIM_TTL_SECONDS
        except (TypeError, ValueError):
            return True

    def claim_orphan_lease(self, owner_id, lease_id, instance_id, token, now, *,
                           expected_lease_version=None, allow_retained_idle=False):
        """Atomically fence an orphan lease (and optional instance) before provider I/O.

        Admission is intentionally used here instead of dispatch: allocation and
        lease binding take the same short admission lock, while the provider call
        happens after this transaction commits.  Legacy/no-instance orphan leases
        (`instance_id is None`) acquire the same lease-level atomic cleanup claim.
        """
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update())
                lease_instance_json = (lease.data or {}).get('instance_id') if lease is not None else None
                lease_instance_typed = lease.instance_id if lease is not None else None
                if (lease_instance_json not in (None, '') and lease_instance_typed not in (None, '')
                        and lease_instance_json != lease_instance_typed):
                    return None
                lease_instance_id = lease_instance_json or lease_instance_typed
                if lease is None or lease.state == 'deleted' or lease_instance_id != instance_id:
                    return None
                lease_data = lease.data or {}
                if expected_lease_version is not None and lease.version != expected_lease_version:
                    return None
                if allow_retained_idle and lease.state != "idle":
                    return None
                if lease_data.get('cleanup_claim_token') not in {None, token} and self._claim_is_live(lease_data, now):
                    return None
                instance = (
                    session.scalar(select(GpuInstanceRow).where(
                        GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                    ).with_for_update())
                    if instance_id is not None
                    else None
                )
                related = self._collect_associated_jobs_for_scope(
                    session, owner_id, lease_id=lease.id, instance_id=instance_id,
                    lease_data=lease_data, inst_data=instance.data if instance is not None else None,
                )
                if self._create_rpc_live_in_flight(lease_data, now) or any(
                    row.state not in {"completed", "failed", "cancelled"} for row in related
                ):
                    return None
                if instance is not None and (
                    (instance.data or {}).get("lease_id") != lease.id
                    or instance.current_job_id not in {None, lease_data.get("owner_job_id")}
                    or (instance.data or {}).get("current_job_id") not in {None, lease_data.get("owner_job_id")}
                ):
                    return None
                if instance is None:
                    if instance_id is not None and session.scalar(
                        select(GpuInstanceRow.id).where(GpuInstanceRow.id == instance_id)
                    ) is not None:
                        return None
                    if (
                        has_conflicting_active_runtime(
                            session,
                            instance_id=instance_id,
                            lease=lease,
                            exclude_lease_id=lease_id,
                            exclude_instance_id=instance_id,
                        )[0]
                    ):
                        return None
                    lease_data = copy.deepcopy(lease.data or {})
                    lease_data.update(cleanup_claim_token=token, cleanup_claimed_at=now, state='deleting')
                    lease.data = lease_data
                    lease.state = 'deleting'
                    lease.version += 1
                    lease.updated_at = utcnow()
                    session.flush()
                    return self._row(lease), None
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=instance_id,
                        lease=lease,
                        instance=instance,
                        exclude_lease_id=lease_id,
                        exclude_instance_id=instance_id,
                    )[0]
                ):
                    return None
                instance_data = copy.deepcopy(instance.data or {})
                instance_account_json = instance_data.get('account_id')
                instance_account_typed = instance.account_id
                lease_account_json = lease_data.get('account_id')
                lease_account_typed = lease.account_id
                if (instance_account_json in (None, '') or instance_account_typed in (None, '')
                        or lease_account_json in (None, '') or lease_account_typed in (None, '')
                        or instance_account_json != instance_account_typed
                        or lease_account_json != lease_account_typed
                        or lease_account_json != instance_account_json):
                    return None
                current_job = instance_data.get('current_job_id') or instance.current_job_id
                if current_job or instance.state in {'allocating', 'ready', 'busy', 'deleted', 'unknown'} or (instance.state == 'retained_idle' and not allow_retained_idle):
                    return None
                if instance_data.get('cleanup_claim_token') not in {None, token} and self._claim_is_live(instance_data, now):
                    return None
                instance_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    runtime_deleted=False,
                )
                instance.state = 'cleanup_pending'
                instance.version += 1
                instance.updated_at = utcnow()
                instance.data = instance_data
                lease_data = copy.deepcopy(lease.data or {})
                lease_data.update(cleanup_claim_token=token, cleanup_claimed_at=now, state='deleting')
                lease.data = lease_data
                lease.state = 'deleting'
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return self._row(lease), self._row(instance)

    def finish_orphan_lease(self, owner_id, lease_id, instance_id, token, now, *, confirmed, error=None, attempts=0):
        """Commit or release a previously fenced orphan cleanup claim."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id))
                lease_instance_id = (lease.data or {}).get('instance_id') if lease is not None else None
                lease_instance_id = lease_instance_id or (lease.instance_id if lease is not None else None)
                if lease is None or lease_instance_id != instance_id:
                    return False
                data = copy.deepcopy(lease.data or {})
                if data.get('cleanup_claim_token') != token or lease.state != 'deleting':
                    return False
                if confirmed:
                    instance = (
                        session.scalar(
                            select(GpuInstanceRow).where(
                                GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                            )
                        )
                        if instance_id is not None
                        else None
                    )
                    if (
                        has_conflicting_active_runtime(
                            session,
                            instance_id=instance_id,
                            lease=lease,
                            instance=instance,
                            exclude_lease_id=lease_id,
                            exclude_instance_id=instance_id,
                        )[0]
                    ):
                        return False
                    data.update(
                        state='deleted', last_seen=now, cleanup_claim_token=None, cleanup_claimed_at=None,
                        remote_deleted=True, remote_deleted_at=now,
                        remote_delete_source='orphan_cleanup_provider_confirmation',
                    )
                    lease.state = 'deleted'
                else:
                    if isinstance(error, dict):
                        try:
                            delay = max(0.0, float(error.get("retry_after")))
                        except (TypeError, ValueError):
                            delay = min(
                                self._CLEANUP_RETRY_MAX_SECONDS,
                                self._CLEANUP_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                            )
                        last_error = error.get("message") or error.get("error") or str(error)
                    else:
                        delay = min(
                            self._CLEANUP_RETRY_MAX_SECONDS,
                            self._CLEANUP_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                        )
                        last_error = error or "delete not confirmed"
                    data.update(
                        state="cleanup_pending", delete_attempts=attempts,
                        cleanup_next_epoch=now + delay, last_error=last_error,
                        last_seen=now, cleanup_claim_token=None, cleanup_claimed_at=None,
                    )
                    lease.state = "cleanup_pending"
                    instance = (
                        session.scalar(
                            select(GpuInstanceRow).where(
                                GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                            )
                        )
                        if instance_id is not None
                        else None
                    )
                    if instance is not None:
                        instance_data = copy.deepcopy(instance.data or {})
                        if instance_data.get("cleanup_claim_token") == token:
                            instance_data.update(cleanup_claim_token=None, cleanup_claimed_at=None)
                            instance.data = instance_data
                            instance.version += 1
                            instance.updated_at = utcnow()
                lease.data = data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return True

    def claim_job_cleanup_lease(
        self,
        owner_id: str,
        job_id: str,
        lease_id: str,
        token: str,
        now: float,
        *,
        expected_lease_version: int | None = None,
    ) -> tuple[str, dict | None]:
        """Atomically fence a job's lease (and instance if bound) before cleanup provider I/O."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(
                    select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update()
                )
                if job is None or lease is None:
                    return "missing", None
                if lease.state == "deleted":
                    return "already_deleted", self._row(lease)
                ldata = copy.deepcopy(lease.data or {})
                jdata = job.data or {}
                if (
                    job.state not in {"cleaning_up", "cleanup_pending"}
                    or jdata.get("lease_id") != lease.id
                    or (expected_lease_version is not None and lease.version != expected_lease_version)
                    or ldata.get("owner_job_id") != job.id
                    or ldata.get("current_job_id") not in {None, job.id}
                ):
                    return "stale", self._row(lease)
                if ldata.get("cleanup_claim_token") not in {None, token} and self._claim_is_live(ldata, now):
                    return "stale", self._row(lease)
                if self._create_rpc_live_in_flight(ldata, now):
                    return "in_flight", self._row(lease)
                eff_inst_id = lease.instance_id or ldata.get("instance_id") or job.instance_id or jdata.get("instance_id")
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if eff_inst_id
                    else None
                )
                if eff_inst_id and (instance is None or (job.instance_id or jdata.get("instance_id")) != eff_inst_id):
                    return "stale", self._row(lease)
                if instance is not None:
                    idata = copy.deepcopy(instance.data or {})
                    if (
                        idata.get("lease_id") != lease.id
                        or idata.get("current_job_id") not in {None, job.id}
                        or instance.current_job_id not in {None, job.id}
                    ):
                        return "stale", self._row(lease)
                    if (
                        idata.get("cleanup_claim_token") not in {None, token}
                        and self._claim_is_live(idata, now)
                    ):
                        return "stale", self._row(lease)
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=eff_inst_id,
                        lease=lease,
                        instance=instance,
                        job=job,
                        exclude_lease_id=lease.id,
                        exclude_instance_id=eff_inst_id,
                    )[0]
                ):
                    return "conflict", self._row(lease)
                if isinstance(ldata.get("handle"), Mapping):
                    ldata["handle"] = retarget_handle(ldata["handle"], job.id)
                ldata.update(
                    state="deleting",
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    last_seen=now,
                )
                lease.state = "deleting"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                if instance is not None and instance.state not in {"deleted", "retained_idle"}:
                    idata = copy.deepcopy(instance.data or {})
                    if idata.get("lease_id") in {None, lease.id}:
                        idata.update(cleanup_claim_token=token, cleanup_claimed_at=now)
                        instance.data = idata
                        instance.version += 1
                        instance.updated_at = utcnow()
                session.flush()
                return "ok", self._row(lease)

    def finish_job_cleanup_lease(
        self,
        owner_id: str,
        job_id: str,
        lease_id: str,
        token: str,
        now: float,
        *,
        confirmed: bool,
        error: object = None,
        attempts: int = 0,
        retry_delay: float | None = None,
    ) -> tuple[str, dict | None]:
        """Commit or release a previously fenced job cleanup lease claim."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(
                    select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update()
                )
                if lease is None:
                    return "missing", None
                ldata = copy.deepcopy(lease.data or {})
                if ldata.get("cleanup_claim_token") != token or lease.state != "deleting":
                    return "stale", self._row(lease)
                eff_inst_id = lease.instance_id or ldata.get("instance_id")
                if not eff_inst_id and job is not None:
                    eff_inst_id = job.instance_id or (job.data or {}).get("instance_id")
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if eff_inst_id
                    else None
                )
                effective_confirmed = bool(confirmed)
                effective_error = error
                if effective_confirmed:
                    if (
                        has_conflicting_active_runtime(
                            session,
                            instance_id=eff_inst_id,
                            lease=lease,
                            instance=instance,
                            job=job,
                            exclude_lease_id=lease.id,
                            exclude_instance_id=eff_inst_id,
                        )[0]
                    ):
                        effective_confirmed = False
                        effective_error = "runtime acquired by another live lease"
                if effective_confirmed:
                    ldata.update(
                        state="deleted",
                        delete_attempts=attempts,
                        last_seen=now,
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="cleanup_provider_confirmation",
                    )
                    lease.state = "deleted"
                else:
                    delay = (
                        max(0.0, float(retry_delay))
                        if retry_delay is not None
                        else min(
                            self._CLEANUP_RETRY_MAX_SECONDS,
                            self._CLEANUP_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)),
                        )
                    )
                    ldata.update(
                        state="cleanup_pending",
                        delete_attempts=attempts,
                        cleanup_next_epoch=now + delay,
                        last_error=effective_error or "delete not confirmed",
                        last_seen=now,
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                    )
                    lease.state = "cleanup_pending"
                if instance is not None:
                    idata = copy.deepcopy(instance.data or {})
                    if idata.get("cleanup_claim_token") == token:
                        idata.update(cleanup_claim_token=None, cleanup_claimed_at=None)
                        instance.data = idata
                        instance.version += 1
                        instance.updated_at = utcnow()
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return ("deleted" if effective_confirmed else "pending"), self._row(lease)

    def claim_orphan_instance(self, owner_id, instance_id, lease_id, token, now, *,
                              expected_lease_version=None, expected_instance_version=None):
        """Fence a historical instance before checking a deleted lease remotely.

        Old handoff rows may have a deleted lease without the provenance marker
        written by newer cleanup code.  The short admission transaction only
        claims a fully owned, terminal, unambiguous instance; provider I/O is
        deliberately performed after it commits.
        """
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                instance = session.scalar(select(GpuInstanceRow).where(
                    GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                ).with_for_update())
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id
                ).with_for_update())
                if instance is None or lease is None or instance.state not in {"releasing", "cleanup_pending"}:
                    return None
                if ((expected_lease_version is not None and lease.version != expected_lease_version)
                    or (expected_instance_version is not None and instance.version != expected_instance_version)):
                    return None
                if lease.state != "deleted":
                    return None
                instance_data = copy.deepcopy(instance.data or {})
                lease_data = copy.deepcopy(lease.data or {})

                def identity(row, data, key):
                    json_value = data.get(key)
                    typed_value = getattr(row, key, None)
                    if json_value not in (None, "") and typed_value not in (None, "") and json_value != typed_value:
                        return None, False
                    return (json_value if json_value not in (None, "") else typed_value), True

                instance_identity = {}
                lease_identity = {}
                for key in ("account_id", "instance_id", "current_job_id"):
                    value, valid = identity(instance, instance_data, key)
                    if not valid:
                        return None
                    instance_identity[key] = value
                for key in ("account_id", "instance_id"):
                    value, valid = identity(lease, lease_data, key)
                    if not valid:
                        return None
                    lease_identity[key] = value
                if (
                    not instance_identity["account_id"]
                    or instance_identity["account_id"] != lease_identity["account_id"]
                    or instance_identity["instance_id"] not in (None, instance_id)
                    or lease_data.get("provider") not in {"mock", "colab"}
                    or instance_data.get("provider") not in {None, lease_data.get("provider")}
                    or not lease_data.get("runtime_id")
                    or not lease_data.get("request_id")
                ):
                    return None
                if lease_data.get("remote_delete_source") not in (None, ""):
                    return None
                for row_data in (instance_data, lease_data):
                    retry_at = row_data.get("cleanup_next_epoch")
                    if retry_at is not None:
                        try:
                            if float(now) < float(retry_at):
                                return None
                        except (TypeError, ValueError):
                            return None
                current_job_id = instance_identity["current_job_id"]
                owner_job_id = lease_data.get("owner_job_id")
                if not owner_job_id:
                    return None
                jobs = session.scalars(select(GpuJobRow).where(GpuJobRow.owner_id == owner_id).with_for_update()).all()
                by_id = {row.id: row for row in jobs}
                related = [row for row in jobs if row.id in {current_job_id, owner_job_id}
                           or ((row.data or {}).get("instance_id") or row.instance_id) == instance_id]
                if any(row.state not in {"completed", "failed", "cancelled"} for row in related):
                    return None
                provider = lease_data.get("provider")
                owner_job = by_id.get(owner_job_id)
                if owner_job is None or owner_job.state not in {"completed", "failed", "cancelled"}:
                    return None
                lease_instance_id = lease_identity["instance_id"]
                owner_job_data = owner_job.data or {}
                owner_job_instance = owner_job_data.get("instance_id") or owner_job.instance_id
                if lease_instance_id not in (None, instance_id):
                    # A completed handoff may have rebound this lease to the
                    # successor instance.  Accept it only when the lease and
                    # its current owner job agree exactly on that successor.
                    if owner_job_instance != lease_instance_id or lease_instance_id == instance_id:
                        return None
                for row in related:
                    row_data = row.data or {}
                    row_account = row_data.get("account_id") or row.account_id
                    row_instance = row_data.get("instance_id") or row.instance_id
                    row_provider = row_data.get("provider") or (row_data.get("execution_policy") or {}).get("provider")
                    allowed_instances = {None, instance_id}
                    if row.id == owner_job_id and lease_instance_id not in (None, instance_id):
                        allowed_instances.add(lease_instance_id)
                    if (
                        row_account not in (None, instance_identity["account_id"])
                        or row_instance not in allowed_instances
                        or row_provider not in (None, provider)
                    ):
                        return None
                if current_job_id and current_job_id not in by_id:
                    return None
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=instance_id,
                        lease=lease,
                        instance=instance,
                        exclude_lease_id=lease_id,
                        exclude_instance_id=instance_id,
                    )[0]
                ):
                    return None
                claimed = instance_data.get("cleanup_claim_token")
                if claimed not in (None, token) and self._claim_is_live(instance_data, now):
                    return None
                instance_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    cleanup_claim_lease_id=lease_id,
                    cleanup_claim_instance_id=instance_id,
                    cleanup_claim_current_job_id=current_job_id,
                    runtime_deleted=False,
                )
                instance.state = "cleanup_pending"
                instance.data = instance_data
                instance.version += 1
                instance.updated_at = utcnow()
                lease_data.update(
                    cleanup_claim_token=token,
                    cleanup_claimed_at=now,
                    cleanup_claim_lease_id=lease_id,
                    cleanup_claim_instance_id=lease_identity["instance_id"],
                    cleanup_claim_owner_job_id=owner_job_id,
                    cleanup_claim_runtime_id=lease_data.get("runtime_id"),
                )
                lease.data = lease_data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return self._row(instance), self._row(lease)

    def finish_orphan_instance(
        self, owner_id, instance_id, lease_id, token, now, *, confirmed, error=None, expected_version=None,
        expected_lease_version=None
    ):
        """Persist or release a fenced historical-instance reconciliation."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False
            with self.store.db_lock(), self.store.Session.begin() as session:
                instance = session.scalar(select(GpuInstanceRow).where(
                    GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id
                ))
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id
                ))
                if instance is None or lease is None or lease.state != "deleted":
                    return False
                if expected_lease_version is None or lease.version != expected_lease_version:
                    return False
                if expected_version is None or instance.version != expected_version or instance.state != "cleanup_pending":
                    return False
                data = copy.deepcopy(instance.data or {})
                if (
                    data.get("cleanup_claim_token") != token
                    or data.get("cleanup_claim_lease_id") != lease_id
                    or data.get("cleanup_claim_instance_id") != instance_id
                ):
                    return False
                lease_data = copy.deepcopy(lease.data or {})
                if (
                    lease_data.get("cleanup_claim_token") != token
                    or lease_data.get("cleanup_claim_lease_id") != lease_id
                    or lease_data.get("cleanup_claim_owner_job_id") != lease_data.get("owner_job_id")
                    or lease_data.get("cleanup_claim_runtime_id") != lease_data.get("runtime_id")
                ):
                    return False
                lease_instance_json = lease_data.get("instance_id")
                lease_instance_typed = lease.instance_id
                if (
                    lease_instance_json not in (None, "")
                    and lease_instance_typed not in (None, "")
                    and lease_instance_json != lease_instance_typed
                ):
                    return False
                if (lease_instance_json or lease_instance_typed) != lease_data.get("cleanup_claim_instance_id"):
                    return False
                owner_job_id = lease_data.get("owner_job_id")
                owner_job = session.scalar(select(GpuJobRow).where(
                    GpuJobRow.id == owner_job_id, GpuJobRow.owner_id == owner_id
                ))
                if owner_job is None or owner_job.state not in {"completed", "failed", "cancelled"}:
                    return False
                claimed_current_job = data.get("cleanup_claim_current_job_id")
                current_job_id = data.get("current_job_id")
                if current_job_id != claimed_current_job:
                    return False
                if current_job_id:
                    current_job = session.scalar(select(GpuJobRow).where(
                        GpuJobRow.id == current_job_id, GpuJobRow.owner_id == owner_id
                    ))
                    if current_job is None or current_job.state not in {"completed", "failed", "cancelled"}:
                        return False
                if confirmed:
                    leases = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                    runtime_id = lease_data.get("runtime_id")
                    if any(
                        row.id != lease_id
                        and row.state != "deleted"
                        and (
                            ((row.data or {}).get("instance_id") or row.instance_id) == instance_id
                            or (row.data or {}).get("runtime_id") == runtime_id
                        )
                        for row in leases
                    ):
                        return False
                    data.update(
                        current_job_id=None,
                        runtime_deleted=True,
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="orphan_reconciliation_provider_confirmation",
                        retention_reserved=0.0,
                        deleted_at=now,
                        deletion_confirmed_at=now,
                        last_seen_at=now,
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                        cleanup_claim_lease_id=None,
                        cleanup_claim_instance_id=None,
                        cleanup_claim_current_job_id=None,
                    )
                    instance.state = "deleted"
                    instance.current_job_id = None
                else:
                    data.update(
                        cleanup_claim_token=None,
                        cleanup_claimed_at=None,
                        cleanup_next_epoch=now + self._CLEANUP_RETRY_BASE_SECONDS,
                        last_error=error or "remote deletion not confirmed",
                        last_seen_at=now,
                        cleanup_claim_lease_id=None,
                        cleanup_claim_instance_id=None,
                        cleanup_claim_current_job_id=None,
                    )
                    instance.state = "cleanup_pending"
                instance.data = data
                instance.version += 1
                instance.updated_at = utcnow()
                lease_data.pop("cleanup_claim_token", None)
                lease_data.pop("cleanup_claimed_at", None)
                lease_data.pop("cleanup_claim_lease_id", None)
                lease_data.pop("cleanup_claim_instance_id", None)
                lease_data.pop("cleanup_claim_owner_job_id", None)
                lease_data.pop("cleanup_claim_runtime_id", None)
                if confirmed:
                    lease_data.update(
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="orphan_reconciliation_provider_confirmation",
                    )
                lease.data = lease_data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return True

    def record_usage_once(self, owner_id, payload):
        # Deterministic primary key provides DB-enforced dedupe across processes.
        identity=digest([owner_id,payload['job_id'],payload.get('execution_id')])
        try:
            return self.create('usage',owner_id,payload,record_id=identity)
        except (IntegrityError, ValueError):
            existing=self.get('usage',identity,owner_id)
            if existing is None:
                raise
            return existing

    def retry_attempt(self, job_id, owner_id, key):
        """Requeue a cleaned failed Mock job; immutable snapshot and executions survive."""
        if not key or len(key)>200:
            raise ValueError('idempotency key required')
        with self.store.db_lock(), self.store.Session.begin() as session:
            job=session.scalar(select(GpuJobRow).where(GpuJobRow.id==job_id,GpuJobRow.owner_id==owner_id).with_for_update())
            if not job:
                raise KeyError(job_id)
            data=copy.deepcopy(job.data)
            keys=data.get('attempt_retry_keys') or []
            if key in keys:
                return self._row(job)
            if job.state not in {'failed','cancelled'}:
                raise ValueError('job must be failed or cancelled')
            if data.get('provider',(data.get('execution_policy') or {}).get('provider','mock'))!='mock':
                raise ValueError('same-job retry is currently Mock only')
            snapshot=data.get('snapshot') or {}
            if data.get('prep') or any(snapshot.get(k) or data.get(k) for k in ('system_smoke','workflow_smoke','prep')) or (data.get('request_body') or {}).get('kind'):
                raise ValueError('special job requires its confirmation entry')
            leases=session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id==owner_id)).all()
            if any(row.data.get('owner_job_id')==job_id and row.state!='deleted' for row in leases):
                raise ValueError('lease cleanup must be confirmed before retry')
            history=data.get('attempt_history') or []
            history.append({k:copy.deepcopy(v) for k,v in data.items() if k not in {'snapshot','request_body','attempt_history','attempt_retry_keys'}})
            transient={
                'execution_id','lease_id','outputs','output_asset_ids','persisted_outputs','error_code','error_message',
                'cancel_requested','requested_outcome','cleanup_final_state','cleanup_attempts','cleanup_next_epoch',
                'upload_attempts','started_epoch','stopped_at','deadline_epoch','validated_epoch','budget_reserved',
                'runtime_deleted','actual_units','prep_report','progress','awaiting_prep',
                'instance_completion_recorded','instance_id','instance_state','instance_bound','account_id',
                'requested_instance_id',
                'account_alias_snapshot','selection_reason','allocation_key','attempt_id','runtime_id','request_id',
                'stage','worker_timings','output_bytes','started_at','finished_at','poll_retry_count','poll_retry_at',
                'gpu_reported','gpu_diagnostics',
            }
            for k in transient:
                data.pop(k,None)
            policy=data.get('execution_policy')
            if isinstance(policy, Mapping) and ('instance_id' in policy or 'requested_instance_id' in policy):
                data['execution_policy']={**policy,'instance_id':None,'requested_instance_id':None}
            data.update(state='queued',attempt_history=history,attempt_retry_keys=[*keys,key],attempt_id=key)
            job.data=data
            job.state='queued'
            job.instance_id = None
            job.account_id = None
            job.version+=1
            job.updated_at=utcnow()
            session.add(GpuEventRow(id=str(uuid.uuid4()),owner_id=owner_id,project_id=job.project_id,state='queued',
                version=0,created_at=utcnow(),updated_at=utcnow(),data={'job_id':job_id,'state':'queued','message':'Mock retry attempt queued; previous evidence retained'}))
            if data.get('batch_id'):
                self._refresh_batch(session,data['batch_id'],owner_id)
            session.flush()
            return self._row(job)

    @staticmethod
    def _validate_warm_proof(
        job_data: Mapping[str, object],
        execution_id: str | None,
        terminal_state: str,
        proof: Mapping[str, object] | None,
        *,
        generation: int | None = None,
        now: float = 0.0,
    ) -> dict | None:
        """Accept warm_proof only from a completed execution with matching identity and aligned fingerprint."""
        if terminal_state not in {"uploading", "completed"} or not execution_id:
            return None
        if not isinstance(proof, Mapping) or proof.get("warm_saved") is not True:
            return None
        reported_id = proof.get("worker_reported_execution_id")
        if not isinstance(reported_id, str) or reported_id != execution_id:
            return None
        epoch_id = proof.get("warm_epoch_id")
        fingerprint = proof.get("warm_fingerprint")
        runtime_meta = proof.get("runtime_meta")
        if not isinstance(epoch_id, str) or not epoch_id.strip():
            return None
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            return None
        if not isinstance(runtime_meta, Mapping):
            return None
        python_mm = runtime_meta.get("python_mm")
        if not isinstance(python_mm, str) or not python_mm.strip():
            return None
        snapshot = job_data.get("snapshot") if isinstance(job_data.get("snapshot"), Mapping) else {}
        manifest = snapshot.get("manifest") if isinstance(snapshot, Mapping) else None
        from .runtime_worker import compute_manifest_warm_fingerprint

        expected_fp = compute_manifest_warm_fingerprint(manifest, python_mm=python_mm.strip())
        if expected_fp is None:
            is_mock_job = (
                job_data.get("provider") == "mock"
                and isinstance(job_data.get("execution_policy"), Mapping)
                and job_data["execution_policy"].get("provider") == "mock"
                and isinstance(manifest, Mapping)
                and manifest.get("mock") is True
                and bool(job_data.get("compatibility_signature"))
            )
            if not is_mock_job or fingerprint != job_data.get("compatibility_signature"):
                return None
        elif fingerprint != expected_fp:
            return None
        if generation is None or isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            return None
        return {
            "warm_execution_id": execution_id,
            "worker_reported_execution_id": reported_id,
            "warm_saved": True,
            "warm_epoch_id": epoch_id,
            "warm_fingerprint": fingerprint,
            "runtime_meta": copy.deepcopy(dict(runtime_meta)),
            "compatibility_signature": job_data.get("compatibility_signature"),
            "generation": generation,
            "verified_at": now,
        }

    def _validate_lease_runtime_access(
        self,
        job: GpuJobRow,
        lease: GpuLeaseRow | None,
        instance: GpuInstanceRow | None,
        owner_id: str,
        *,
        instance_id: str | None = None,
        require_instance: bool = False,
        allow_retarget_handle_job: bool = False,
    ) -> str | None:
        """Shared strict runtime, handle, and 3-entity ownership/fence validator for start/resume."""
        jdata = job.data or {}
        if (
            jdata.get("cancel_requested")
            or jdata.get("requested_outcome") == "cancelled"
            or self._has_stop_or_delete_flag(jdata)
        ):
            return "job cancellation or stop requested"
        if lease is None or lease.owner_id != owner_id or jdata.get("lease_id") not in {None, lease.id}:
            return "job lease identity mismatch"
        lease_data = lease.data or {}
        if lease_data.get("owner_job_id") != job.id:
            return "lease does not belong to job"
        if lease_data.get("current_job_id") not in {None, job.id}:
            return "lease is currently occupied by another job"
        if self._has_stop_or_delete_flag(lease_data):
            return "lease is stopping or deleted"
        job_account = jdata.get("account_id") or job.account_id
        lease_account = lease_data.get("account_id") or lease.account_id
        if job_account is not None and lease_account is not None and job_account != lease_account:
            return "lease account mismatch"
        job_instance = jdata.get("instance_id") or job.instance_id
        lease_instance = lease_data.get("instance_id") or lease.instance_id
        if job_instance is not None and lease_instance is not None and job_instance != lease_instance:
            return "lease instance mismatch"
        if require_instance and instance_id and instance is None:
            return "instance missing or unavailable"
        if instance is None:
            if lease.state != "active":
                return "lease is not active"
        else:
            if lease.state not in {"active", "busy"}:
                return "lease is not active"
            inst_data = instance.data or {}
            if (
                instance.owner_id != owner_id
                or jdata.get("instance_id") not in {None, instance.id}
                or instance.state not in {"allocating", "ready", "busy"}
                or self._has_stop_or_delete_flag(inst_data)
                or inst_data.get("current_job_id") not in {None, job.id}
                or inst_data.get("lease_id") not in {None, lease.id}
            ):
                return "instance is stopping or unavailable"
        if not is_valid_runtime_handle(
            lease_data.get("handle"),
            job_id=None if allow_retarget_handle_job else job.id,
            owner_id=owner_id,
            lease_runtime_id=extract_runtime_id(lease),
            instance_runtime_id=extract_runtime_id(instance) if instance is not None else None,
            provider=lease_data.get("provider") or jdata.get("provider"),
        ):
            return "lease handle missing valid runtime identity"
        return None

    def prepare_execution(self, job_id, owner_id, lease_id, now, *, require_instance: bool = False):
        """Persist execution and start_intent identity before start, atomically under admission fence."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                raise ValueError('admission lock unavailable')
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.get(GpuLeaseRow, lease_id)
                if not job or not lease or lease.owner_id != owner_id:
                    raise KeyError(job_id)
                if job.state != 'loading':
                    raise ValueError('job is not ready to start')
                lease_data = copy.deepcopy(lease.data or {})
                job_account = job.data.get('account_id') or job.account_id
                lease_account = lease_data.get('account_id') or lease.account_id
                job_instance = job.data.get('instance_id') or job.instance_id
                lease_instance = lease_data.get('instance_id') or lease.instance_id
                instance_id = job_instance or lease_instance
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if instance_id
                    else None
                )
                refusal = self._validate_lease_runtime_access(
                    job,
                    lease,
                    instance,
                    owner_id,
                    instance_id=instance_id,
                    require_instance=require_instance,
                    allow_retarget_handle_job=not bool(job.data.get('execution_id')),
                )
                if refusal is not None:
                    raise ValueError(refusal)
                account_id = job_account or lease_account
                existing_execution_id = job.data.get('execution_id')
                if existing_execution_id:
                    execution = session.get(GpuExecutionRow, existing_execution_id)
                    if execution is None or execution.owner_id != owner_id:
                        raise ValueError('execution association mismatch')
                    expected = {
                        'job_id': job_id,
                        'lease_id': lease_id,
                        'instance_id': job_instance or lease_instance,
                        'account_id': job_account or lease_account,
                    }
                    for key, expected_value in expected.items():
                        json_value = (execution.data or {}).get(key)
                        typed_value = getattr(execution, key, None)
                        if json_value not in (None, '') and typed_value not in (None, '') and json_value != typed_value:
                            raise ValueError('execution association conflict')
                        value = json_value if json_value not in (None, '') else typed_value
                        if expected_value not in (None, '') and value != expected_value:
                            raise ValueError('execution association mismatch')
                    job_intent = (job.data or {}).get('start_intent') or {}
                    exec_intent = (execution.data or {}).get('start_intent') or {}
                    if (
                        execution.state not in {'starting', 'running'}
                        or job_intent.get('state') == 'aborted'
                        or exec_intent.get('state') == 'aborted'
                    ):
                        raise ValueError('start intent aborted')
                    return self._row(job)
                execution_id = str(uuid.uuid4())
                attempt = int(job.data.get('attempt_number') or 0) + 1
                inst_data = copy.deepcopy(instance.data or {}) if instance is not None else {}
                generation = max(int(inst_data.get("generation") or 0), int(lease_data.get("generation") or 0)) + 1
                active_execution = {
                    "active_execution_id": execution_id,
                    "execution_id": execution_id,
                    "job_id": job_id,
                    "lease_id": lease_id,
                    "instance_id": instance_id,
                    "generation": generation,
                }
                start_intent = {
                    "intent_id": str(uuid.uuid4()),
                    "execution_id": execution_id,
                    "state": "starting",
                    "job_id": job_id,
                    "lease_id": lease_id,
                    "instance_id": instance_id,
                    "created_at": now,
                }
                # B taking the stage immediately revokes A's warm_proof on both lease and instance.
                if isinstance(lease_data.get("handle"), Mapping):
                    lease_data["handle"] = retarget_handle(lease_data["handle"], job_id)
                lease_data.update(
                    generation=generation,
                    active_execution=copy.deepcopy(active_execution),
                    warm_proof=None,
                    restored_signature=None,
                )
                lease.data = lease_data
                lease.version += 1
                lease.updated_at = utcnow()
                if instance is not None:
                    inst_data.update(
                        generation=generation,
                        active_execution=copy.deepcopy(active_execution),
                        warm_proof=None,
                    )
                    if isinstance(lease_data.get("handle"), Mapping) and lease_data.get("handle"):
                        inst_data["runtime_created"] = True
                    instance.data = inst_data
                    instance.version += 1
                    instance.updated_at = utcnow()
                job.data = {
                    **job.data,
                    'execution_id': execution_id,
                    'lease_id': lease_id,
                    'generation': generation,
                    'attempt_number': attempt,
                    'start_intent': copy.deepcopy(start_intent),
                }
                job.version += 1
                job.updated_at = utcnow()
                session.add(
                    GpuExecutionRow(
                        id=execution_id,
                        owner_id=owner_id,
                        project_id=job.project_id,
                        account_id=account_id,
                        instance_id=instance_id,
                        state='starting',
                        version=0,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                        data={
                            'job_id': job_id,
                            'lease_id': lease_id,
                            'attempt': attempt,
                            'generation': generation,
                            'worker_session_id': lease_id,
                            'started_at': now,
                            'state': 'starting',
                            'instance_id': instance_id,
                            'account_id': account_id,
                            'start_intent': copy.deepcopy(start_intent),
                        },
                    )
                )
                session.flush()
                return self._row(job)

    def _apply_job_transition(self, session, job, owner_id, state, message, patch, now):
        terminal = {'completed', 'failed', 'cancelled'}
        progress = (patch or {}).get('progress')
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1
        ):
            raise ValueError('progress必须在0到1之间')
        job.state = state
        job.data = {**job.data, **copy.deepcopy(patch or {}), 'state': state}
        job.version += 1
        job.updated_at = utcnow()
        execution_id = job.data.get('execution_id')
        if execution_id and state in terminal | {'running', 'uploading', 'cleaning_up'}:
            execution = session.get(GpuExecutionRow, execution_id)
            if execution and execution.owner_id != owner_id:
                raise ValueError('execution owner mismatch')
            if execution and execution.data.get('job_id') not in {None, job.id}:
                raise ValueError('execution job identity mismatch')
            if execution and state in terminal | {'running', 'uploading'}:
                changes = {'state': state}
                if state in terminal:
                    changes.update(finished_at=now, error_code=job.data.get('error_code'))
                execution.state = state
                execution.data = {**execution.data, **changes}
                execution.version += 1
                execution.updated_at = utcnow()
        data = {'job_id': job.id, 'state': state, 'message': message}
        if progress is not None:
            data['progress'] = progress
        session.add(
            GpuEventRow(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                project_id=job.project_id,
                state=state,
                data=data,
                version=0,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        if job.data.get('batch_id'):
            self._refresh_batch(session, job.data['batch_id'], owner_id)
        session.flush()
        return self._row(job)

    def _evaluate_execution_fence_in_session(
        self,
        session,
        job: GpuJobRow,
        owner_id: str,
        *,
        execution_id: str | None,
        expected_state: str | None,
        expected_generation: int | None,
        require_instance: bool = False,
    ):
        terminal = {"completed", "failed", "cancelled"}
        if expected_state is not None and job.state != expected_state:
            return "stale", None, None, None, None
        if job.data.get("execution_id") != execution_id or not execution_id:
            return "stale", None, None, None, None
        if job.state in terminal:
            return "stale", None, None, None, None

        lease_id = job.data.get("lease_id")
        lease = session.get(GpuLeaseRow, lease_id) if lease_id else None
        if lease is not None and lease.owner_id != owner_id:
            return "stale", None, None, None, None
        instance_id = (
            job.data.get("instance_id")
            or (lease.data.get("instance_id") if lease else None)
            or (lease.instance_id if lease else None)
        )
        instance = session.get(GpuInstanceRow, instance_id) if instance_id else None
        if instance is not None and instance.owner_id != owner_id:
            return "stale", None, None, None, None
        execution = session.get(GpuExecutionRow, execution_id)
        inst_active = (instance.data or {}).get("active_execution") if instance is not None else None
        lease_active = (lease.data or {}).get("active_execution") if lease is not None else None
        if inst_active is not None and lease_active is not None and inst_active != lease_active:
            return "stale", None, None, None, None
        active = inst_active if inst_active is not None else lease_active
        require_exec_row = (
            isinstance(active, Mapping)
            or expected_generation is not None
            or (job.data or {}).get("generation") is not None
            or ((lease.data or {}).get("generation") is not None if lease is not None else False)
            or ((instance.data or {}).get("generation") is not None if instance is not None else False)
        )
        if require_exec_row and execution is None:
            return "stale", None, None, None, None
        if active is not None:
            if not isinstance(active, Mapping):
                return "stale", None, None, None, None
            if active.get("active_execution_id") != execution_id:
                return "stale", None, None, None, None
            if active.get("lease_id") != lease_id:
                return "stale", None, None, None, None
            if instance_id and active.get("instance_id") not in {None, instance_id}:
                return "stale", None, None, None, None
        else:
            if lease is not None and lease.data.get("owner_job_id") not in {None, job.id}:
                return "stale", None, None, None, None
            if instance is not None and instance.data.get("current_job_id") not in {None, job.id}:
                return "stale", None, None, None, None

        exec_ok, active_gen = self._execution_matches_bindings_and_generation(
            owner_id=owner_id,
            job=job,
            execution=execution,
            lease=lease,
            instance=instance,
            lease_id=lease_id,
            instance_id=instance_id,
            active=active if isinstance(active, Mapping) else None,
            expected_generation=expected_generation,
        )
        if not exec_ok:
            return "stale", None, None, None, None

        if execution is not None and execution.state in {"failed", "aborted", "cancelled"}:
            return "fenced", lease, instance, execution, active_gen
        if self._three_entity_fenced(
            session,
            job,
            owner_id,
            lease,
            instance,
            lease_id=lease_id,
            instance_id=instance_id,
            require_instance=require_instance,
            allow_legacy_unknown_stub=True,
        ):
            return "fenced", lease, instance, execution, active_gen
        return "ok", lease, instance, execution, active_gen

    def _apply_fenced_cleanup_in_session(
        self,
        session,
        job: GpuJobRow,
        owner_id: str,
        execution_id: str | None,
        lease: GpuLeaseRow | None,
        instance: GpuInstanceRow | None,
        now: float,
    ) -> dict:
        if job.state in {"completed", "failed", "cancelled", "cleaning_up", "cleanup_pending"}:
            return self._row(job)
        if execution_id:
            target_lease = lease if (lease is not None and self._owned_execution_id(lease.data) == execution_id) else None
            target_instance = (
                instance if (instance is not None and self._owned_execution_id(instance.data) == execution_id) else None
            )
            if target_lease is not None or target_instance is not None:
                self._persist_runtime_warm_proof(target_lease, target_instance, job.data or {}, None)
        inst_data = (instance.data or {}) if instance is not None else {}
        is_cancel = (
            bool((job.data or {}).get("cancel_requested"))
            or (job.data or {}).get("requested_outcome") == "cancelled"
            or bool(inst_data.get("manual_stop_requested"))
        )
        outcome = "cancelled" if is_cancel else ((job.data or {}).get("requested_outcome") or "failed")
        patch = {
            "requested_outcome": outcome,
            "cleanup_final_state": "cancelled" if is_cancel else "failed",
        }
        owned_lease = (
            lease is not None
            and (lease.data or {}).get("owner_job_id") in {None, job.id}
            and (lease.data or {}).get("current_job_id") in {None, job.id}
        )
        if owned_lease:
            msg = "GPU cancellation requested" if is_cancel else "GPU execution fenced before completion"
            return self._apply_job_transition(session, job, owner_id, "cleaning_up", msg, patch, now)
        final_state = "cancelled" if is_cancel else "failed"
        msg = "GPU job cancelled" if is_cancel else "GPU execution fenced before completion"
        return self._apply_job_transition(session, job, owner_id, final_state, msg, patch, now)

    def commit_run_progress(
        self,
        job_id: str,
        owner_id: str,
        *,
        execution_id: str | None,
        now: float,
        patch: Mapping[str, object] | None = None,
        message: str | None = None,
        progress: float | None = None,
        diagnostics: Mapping[str, object] | None = None,
        expected_state: str | None = "running",
        expected_generation: int | None = None,
        require_instance: bool = False,
    ) -> tuple[str, dict]:
        """Atomically verify run fences and record poll progress/backoff/diagnostics in one tx."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                raise RuntimeError("admission lock unavailable")
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                status, lease, instance, _, _ = self._evaluate_execution_fence_in_session(
                    session,
                    job,
                    owner_id,
                    execution_id=execution_id,
                    expected_state=expected_state,
                    expected_generation=expected_generation,
                    require_instance=require_instance,
                )
                if status == "stale":
                    return "stale", self._row(job)
                if status == "fenced":
                    return "fenced", self._row(job)
                next_patch = copy.deepcopy(dict(patch or {}))
                diag_patch = None
                if isinstance(diagnostics, Mapping) and diagnostics and not (job.data or {}).get("gpu_reported"):
                    from .diagnostics import gpu_diagnostics

                    clean_diag = gpu_diagnostics(diagnostics)
                    if clean_diag is not None:
                        reported = ", ".join(d["name"] for d in clean_diag["devices"])
                        diag_patch = {"gpu_diagnostics": clean_diag, "gpu_reported": reported}
                        next_patch.update(copy.deepcopy(diag_patch))
                if diag_patch is not None:
                    for rel, blocked_states, check_owner in (
                        (lease, {"deleted", "deleting", "quarantined"}, True),
                        (instance, {"deleted", "deleting", "quarantined", "releasing", "cleanup_pending"}, False),
                    ):
                        if rel is None or rel.state in blocked_states:
                            continue
                        rdata = rel.data or {}
                        if check_owner and rdata.get("owner_job_id") not in {None, job_id}:
                            continue
                        if not check_owner and (
                            rdata.get("cleanup_claim_token")
                            or rdata.get("manual_stop_requested")
                            or rdata.get("runtime_deleted")
                            or rdata.get("remote_deleted")
                        ):
                            continue
                        if rdata.get("current_job_id") not in {None, job_id}:
                            continue
                        rel.data = {**rdata, **copy.deepcopy(diag_patch)}
                        rel.version += 1
                        rel.updated_at = utcnow()
                if next_patch or message is not None or progress is not None:
                    if progress is not None:
                        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1:
                            raise ValueError("progress必须在0到1之间")
                        next_patch["progress"] = progress
                    if message is not None:
                        row = self._apply_job_transition(
                            session, job, owner_id, job.state, message, next_patch, now
                        )
                        return "ok", row
                    job.data = {**job.data, **next_patch}
                    job.version += 1
                    job.updated_at = utcnow()
                    session.flush()
                elif diag_patch is not None:
                    session.flush()
                return "ok", self._row(job)

    def _evaluate_start_fence_in_session(
        self,
        session,
        job: GpuJobRow,
        owner_id: str,
        *,
        lease_id: str | None,
        execution_id: str,
        allowed_intent_states: set[str],
        require_instance: bool = False,
    ) -> tuple[bool, GpuLeaseRow | None, GpuInstanceRow | None, GpuExecutionRow | None, dict | None]:
        """Validate job, intent, lease, execution, and with-instance vs no-instance recovery fences."""
        lease = session.get(GpuLeaseRow, lease_id) if lease_id else None
        execution = session.get(GpuExecutionRow, execution_id)
        instance_id = (
            (job.data or {}).get("instance_id")
            or (lease.data.get("instance_id") if lease else None)
            or (lease.instance_id if lease else None)
        )
        instance = session.get(GpuInstanceRow, instance_id) if instance_id else None

        jdata = job.data or {}
        job_intent = jdata.get("start_intent")
        exec_intent = (execution.data or {}).get("start_intent") if execution is not None else None
        intent_bad = (
            not isinstance(job_intent, Mapping)
            or not isinstance(job_intent.get("intent_id"), str)
            or not job_intent.get("intent_id")
            or job_intent.get("execution_id") != execution_id
            or job_intent.get("state") not in allowed_intent_states
            or not isinstance(exec_intent, Mapping)
            or exec_intent.get("intent_id") != job_intent.get("intent_id")
            or exec_intent.get("execution_id") != execution_id
            or exec_intent.get("state") not in allowed_intent_states
        )
        refusal = self._validate_lease_runtime_access(
            job,
            lease,
            instance,
            owner_id,
            instance_id=instance_id,
            require_instance=require_instance,
        )
        exec_data = (execution.data or {}) if execution is not None else {}
        exec_bad = (
            execution is None
            or execution.owner_id != owner_id
            or exec_data.get("job_id") not in {None, job.id}
            or (lease is not None and exec_data.get("lease_id") != lease.id)
            or (instance_id is not None and (execution.instance_id or exec_data.get("instance_id")) not in {None, instance_id})
            or execution.state not in {"starting", "running"}
        )
        fenced = intent_bad or (refusal is not None) or exec_bad
        intent_dict = dict(job_intent) if isinstance(job_intent, Mapping) else None
        return fenced, lease, instance, execution, intent_dict

    def claim_execution_dispatch(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_id: str | None,
        execution_id: str,
        now: float,
        require_instance: bool = False,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically claim send-right (starting/prepared -> dispatching) immediately before executor.start."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                if job.state != "loading" or (job.data or {}).get("execution_id") != execution_id:
                    return "stale", self._row(job), None
                fenced, lease, instance, execution, job_intent = self._evaluate_start_fence_in_session(
                    session,
                    job,
                    owner_id,
                    lease_id=lease_id,
                    execution_id=execution_id,
                    allowed_intent_states={"starting", "prepared", "dispatching"},
                    require_instance=require_instance,
                )
                if not fenced and isinstance(job_intent, Mapping) and job_intent.get("state") == "dispatching":
                    return "in_flight", self._row(job), (self._row(lease) if lease is not None else None)
                exec_ok, _ = self._execution_matches_bindings_and_generation(
                    owner_id=owner_id, job=job, execution=execution, lease=lease, instance=instance,
                    active=(lease.data or {}).get("active_execution") if lease is not None else None,
                )
                if not exec_ok:
                    return "stale", self._row(job), None
                if fenced:
                    if isinstance((job.data or {}).get("start_intent"), Mapping) and job.data["start_intent"].get("state") in {
                        "starting",
                        "prepared",
                        "dispatching",
                    }:
                        job.data = {
                            **job.data,
                            "start_intent": {**job.data["start_intent"], "state": "aborted", "aborted_at": now},
                        }
                        job.version += 1
                        job.updated_at = utcnow()
                    if execution is not None and isinstance((execution.data or {}).get("start_intent"), Mapping):
                        edata = copy.deepcopy(execution.data or {})
                        if edata["start_intent"].get("state") in {"starting", "prepared", "dispatching"}:
                            edata["start_intent"] = {**edata["start_intent"], "state": "aborted", "aborted_at": now}
                            execution.data = edata
                            execution.version += 1
                            execution.updated_at = utcnow()
                    return "fenced", self._apply_fenced_cleanup_in_session(
                        session, job, owner_id, execution_id, lease, instance, now
                    ), None
                dispatch_intent = {**job_intent, "state": "dispatching", "dispatched_at": now}
                job.data = {**job.data, "start_intent": copy.deepcopy(dispatch_intent)}
                job.version += 1
                job.updated_at = utcnow()
                execution.data = {**execution.data, "start_intent": copy.deepcopy(dispatch_intent)}
                execution.version += 1
                execution.updated_at = utcnow()
                session.flush()
                return "ok", self._row(job), (self._row(lease) if lease is not None else None)

    def commit_running(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_id: str | None,
        execution_id: str,
        now: float,
        require_instance: bool = False,
    ) -> tuple[str, dict]:
        """Atomically verify all start/stop/delete fences and commit transition into running."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                raise RuntimeError("admission lock unavailable")
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                if job.state != "loading" or job.data.get("execution_id") != execution_id:
                    return "stale", self._row(job)
                fenced, lease, instance, execution, job_intent = self._evaluate_start_fence_in_session(
                    session,
                    job,
                    owner_id,
                    lease_id=lease_id,
                    execution_id=execution_id,
                    allowed_intent_states={"starting", "prepared", "dispatching"},
                    require_instance=require_instance,
                )
                exec_ok, _ = self._execution_matches_bindings_and_generation(
                    owner_id=owner_id, job=job, execution=execution, lease=lease, instance=instance,
                    active=(lease.data or {}).get("active_execution") if lease is not None else None,
                )
                if not exec_ok:
                    return "stale", self._row(job)
                if fenced:
                    return "fenced", self._apply_fenced_cleanup_in_session(
                        session, job, owner_id, execution_id, lease, instance, now
                    )
                running_intent = {**job_intent, "state": "running", "started_at": now}
                job.data = {**job.data, "start_intent": copy.deepcopy(running_intent)}
                execution.state = "running"
                execution.data = {**execution.data, "state": "running", "start_intent": copy.deepcopy(running_intent)}
                execution.version += 1
                execution.updated_at = utcnow()
                if instance is not None:
                    idata = copy.deepcopy(instance.data or {})
                    idata.update(current_job_id=job_id, last_seen_at=now)
                    if lease is not None and isinstance((lease.data or {}).get("handle"), Mapping) and (lease.data or {}).get("handle"):
                        idata["runtime_created"] = True
                    instance.state = "busy"
                    instance.current_job_id = job_id
                    instance.data = idata
                    instance.version += 1
                    instance.updated_at = utcnow()
                    session.add(
                        GpuEventRow(
                            id=str(uuid.uuid4()),
                            owner_id=owner_id,
                            project_id=instance.project_id,
                            state="busy",
                            version=0,
                            data={"instance_id": instance.id, "job_id": job_id, "state": "busy", "message": "job started"},
                            created_at=utcnow(),
                            updated_at=utcnow(),
                        )
                    )
                return "ok", self._apply_job_transition(
                    session, job, owner_id, "running", "GPU execution started", {}, now
                )

    def request_job_cancel(self, owner_id: str, job_id: str, now: float) -> dict:
        """Atomically mark a job cancel_requested and abort any pending start_intent under admission_lock."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                raise RuntimeError("GPU 准入锁繁忙，请稍后重试取消")
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                if job.state in {"completed", "failed", "cancelled"}:
                    return self._row(job)
                data = copy.deepcopy(job.data or {})
                if job.state == "queued":
                    lease_id = data.get("lease_id")
                    leases = session.scalars(select(GpuLeaseRow).where(GpuLeaseRow.owner_id == owner_id)).all()
                    has_lease = any(
                        (row.id == lease_id or (row.data or {}).get("owner_job_id") == job_id)
                        and (row.data or {}).get("current_job_id") in {None, job_id}
                        for row in leases
                    )
                    if not has_lease:
                        return self._apply_job_transition(
                            session,
                            job,
                            owner_id,
                            "cancelled",
                            "GPU job cancelled",
                            {"cancel_requested": True, "requested_outcome": "cancelled"},
                            now,
                        )
                data["cancel_requested"] = True
                data["requested_outcome"] = "cancelled"
                if isinstance(data.get("start_intent"), Mapping) and data["start_intent"].get("state") in {
                    "starting",
                    "prepared",
                    "dispatching",
                }:
                    data["start_intent"] = {**data["start_intent"], "state": "aborted", "aborted_at": now}
                job.data = data
                job.version += 1
                job.updated_at = utcnow()
                execution_id = data.get("execution_id")
                if execution_id:
                    execution = session.get(GpuExecutionRow, execution_id)
                    if execution is not None and execution.owner_id == owner_id:
                        edata = copy.deepcopy(execution.data or {})
                        if isinstance(edata.get("start_intent"), Mapping) and edata["start_intent"].get("state") in {
                            "starting",
                            "prepared",
                            "dispatching",
                        }:
                            edata["start_intent"] = {**edata["start_intent"], "state": "aborted", "aborted_at": now}
                            execution.data = edata
                            execution.version += 1
                            execution.updated_at = utcnow()
                session.flush()
                return self._row(job)

    @staticmethod
    def _lease_has_active_origin(ldata: Mapping[str, object], state: str | None = None) -> bool:
        """Return True if the lease is currently active or originated from a once-active lease."""
        return lease_has_active_origin(ldata, state)

    def _is_create_fenced_in_session(
        self,
        session,
        job: GpuJobRow,
        lease: GpuLeaseRow,
        owner_id: str,
        job_id: str,
        *,
        allowed_lease_states: set[str] | None = None,
        require_instance: bool = False,
    ) -> tuple[bool, bool]:
        """Re-read and fence job, lease, AND instance inside create claim/retry lock."""
        jdata = job.data or {}
        ldata = lease.data or {}
        if (
            job.state != "allocating"
            or jdata.get("cancel_requested")
            or jdata.get("requested_outcome") == "cancelled"
            or self._has_stop_or_delete_flag(jdata)
            or jdata.get("lease_id") not in {None, lease.id}
        ):
            return True, False
        blocked_lease = {"cleanup_pending", "releasing", "deleting", "quarantined", "deleted", "idle"}
        if lease.state in blocked_lease or (allowed_lease_states is not None and lease.state not in allowed_lease_states):
            return True, False
        if (
            self._has_stop_or_delete_flag(ldata)
            or ldata.get("owner_job_id") != job_id
            or ldata.get("current_job_id") not in {None, job_id}
        ):
            return True, False
        instance_id = jdata.get("instance_id") or ldata.get("instance_id") or lease.instance_id
        has_handle = isinstance(ldata.get("handle"), Mapping) and bool(ldata.get("handle"))
        if instance_id:
            instance = session.scalar(
                select(GpuInstanceRow)
                .where(GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id)
                .with_for_update()
            )
            if instance is None:
                if require_instance:
                    return True, has_handle
            else:
                idata = instance.data or {}
                if (
                    instance.state in {"deleted", "deleting", "releasing", "cleanup_pending", "quarantined", "unknown", "retained_idle"}
                    or self._has_stop_or_delete_flag(idata)
                    or idata.get("current_job_id") not in {None, job_id}
                    or idata.get("lease_id") not in {None, lease.id}
                ):
                    return True, has_handle
                if has_handle:
                    return False, True
                if instance.state not in {"creating", "allocating", "requested"}:
                    return True, False
        return False, has_handle

    @classmethod
    def _create_rpc_live_in_flight(cls, ldata: Mapping[str, object], now: float | None = None) -> bool:
        # Never expire an in-flight create RPC locally; timeout/elapsed age may trigger
        # remote reconciliation, never local deletion or retirement while create RPC runs.
        return bool(ldata.get("create_rpc_in_flight"))

    def update_lease_cas(
        self,
        lease_id: str,
        owner_id: str,
        patch: Mapping[str, object],
        *,
        expected_version: int | None = None,
        expected_create_attempts: int | None = None,
        preserve_active_history: bool = True,
        refuse_downgrade_to_unknown: bool = False,
        refuse_if_create_in_flight: bool = False,
        expected_cleanup_token: str | None = None,
        expected_create_identity: tuple | None = None,
        now: float | None = None,
    ) -> tuple[str, dict | None]:
        """Conditionally update a lease inside admission_lock + db_lock with monotonic active history."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id:
                    return "missing", None
                ldata = copy.deepcopy(lease.data or {})
                if expected_create_identity is not None and not self._create_attempt_matches(ldata, *expected_create_identity):
                    return "stale", self._row(lease)
                if expected_version is not None and int(lease.version) != int(expected_version):
                    return "stale", self._row(lease)
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(
                    expected_create_attempts
                ):
                    return "stale", self._row(lease)
                if expected_cleanup_token is not None and ldata.get("cleanup_claim_token") != expected_cleanup_token:
                    return "stale", self._row(lease)
                if refuse_if_create_in_flight and self._create_rpc_live_in_flight(ldata, now):
                    return "stale", self._row(lease)
                if refuse_downgrade_to_unknown and (
                    lease.state in {"deleted", "idle", "cleanup_pending", "releasing", "deleting", "quarantined"}
                    or (
                        lease.state in {"active", "busy"}
                        and is_valid_runtime_handle(
                            ldata.get("handle"),
                            owner_id=owner_id,
                            provider=self._resolve_lease_provider_in_session(session, ldata),
                        )
                    )
                ):
                    return "stale", self._row(lease)
                next_patch = copy.deepcopy(dict(patch))
                if preserve_active_history:
                    ci = ldata.get("create_intent") if isinstance(ldata.get("create_intent"), Mapping) else {}
                    active_origin_val = next(
                        (
                            v.strip().lower()
                            for v in (ldata.get("origin"), ldata.get("create_intent_origin"), ci.get("origin"))
                            if isinstance(v, str) and v.strip().lower() in _ACTIVE_ORIGINS
                        ),
                        None,
                    )
                    if self._lease_has_active_origin(ldata, lease.state):
                        next_patch["from_active_missing_handle"] = True
                        next_patch["was_active_missing_handle"] = True
                        next_patch["ever_active"] = True
                        if active_origin_val:
                            next_patch["create_intent_origin"] = active_origin_val
                            next_patch["origin"] = active_origin_val
                        elif str(next_patch.get("create_intent_origin") or "").strip().lower() not in _ACTIVE_ORIGINS:
                            next_patch["create_intent_origin"] = "once_active"
                new_state = str(next_patch.get("state") or lease.state)
                next_data = {**ldata, **next_patch, "state": new_state}
                lease.state = new_state
                lease.data = next_data
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return "ok", self._row(lease)

    def retire_absent_lease_cas(
        self,
        job_id: str,
        owner_id: str,
        lease_id: str,
        *,
        expected_lease_version: int | None = None,
        expected_create_attempts: int | None = None,
        expected_tunnel_endpoint: str | None = None,
        expected_tunnel_nbh: object = None,
        from_active: bool = False,
        confirmed_absent: bool = False,
        probe_started_at: float | None = None,
        source: str = "recovery_probe_absent",
        now: float,
    ) -> tuple[str, dict | None, dict | None]:
        """Retire a lease only if (version, create_attempts) still match and no live handle or in-flight RPC exists."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if not job or not lease or lease.owner_id != owner_id:
                    return "missing", (self._row(job) if job else None), None
                ldata = copy.deepcopy(lease.data or {})
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", self._row(job), self._row(lease)
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(
                    expected_create_attempts
                ):
                    return "stale", self._row(job), self._row(lease)
                if lease.state == "deleted":
                    return "already_deleted", self._row(job), self._row(lease)
                if self._create_rpc_live_in_flight(ldata, now):
                    return "in_flight", self._row(job), self._row(lease)
                if probe_started_at is not None:
                    for ts_key in ("create_rpc_finished_at", "create_started"):
                        ts_val = ldata.get(ts_key)
                        if ts_val is not None:
                            try:
                                if float(probe_started_at) < float(ts_val):
                                    return "stale", self._row(job), self._row(lease)
                            except (TypeError, ValueError):
                                return "stale", self._row(job), self._row(lease)
                has_handle = isinstance(ldata.get("handle"), Mapping) and bool(ldata.get("handle"))
                if has_handle:
                    return "stale", self._row(job), self._row(lease)
                cur_ep = extract_tunnel_endpoint(lease)
                exp_ep = normalize_tunnel_endpoint(expected_tunnel_endpoint)
                has_ep = bool(ldata.get("tunnel_endpoint") or cur_ep)
                has_nbh = ldata.get("tunnel_nbh") is not None
                if has_ep or has_nbh:
                    allow_known_absent_ep = (
                        confirmed_absent
                        and exp_ep is not None
                        and cur_ep == exp_ep
                        and (not has_nbh or ldata.get("tunnel_nbh") == expected_tunnel_nbh)
                    )
                    if not allow_known_absent_ep:
                        return "stale", self._row(job), self._row(lease)
                if lease.state == "idle":
                    return "stale", self._row(job), self._row(lease)
                if lease.state in {"active", "busy"} and not confirmed_absent:
                    return "stale", self._row(job), self._row(lease)
                if self._lease_has_active_origin(ldata, lease.state) and not (confirmed_absent or from_active):
                    return "stale", self._row(job), self._row(lease)
                ldata.update(
                    state="deleted",
                    current_job_id=None,
                    remote_deleted=True,
                    remote_deleted_at=now,
                    remote_delete_source=source,
                    last_seen=now,
                )
                if from_active or self._lease_has_active_origin(ldata, lease.state):
                    ldata["from_active_missing_handle"] = True
                    ldata["was_active_missing_handle"] = True
                    ldata["ever_active"] = True
                lease.state = "deleted"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                bound = (
                    job.state == "allocating" and (job.data or {}).get("lease_id") == lease.id
                    and (job.instance_id or (job.data or {}).get("instance_id")) == (lease.instance_id or ldata.get("instance_id"))
                )
                if bound:
                    inst_id = lease.instance_id or ldata.get("instance_id")
                    inst = session.scalar(select(GpuInstanceRow).where(
                        GpuInstanceRow.id == inst_id, GpuInstanceRow.owner_id == owner_id).with_for_update()) if inst_id else None
                    if inst is not None and (inst.data or {}).get("lease_id") == lease.id and (
                        inst.current_job_id in {None, job.id} and (inst.data or {}).get("current_job_id") in {None, job.id}
                    ):
                        inst.data = {**(inst.data or {}), "state": "deleted", "current_job_id": None,
                                     "runtime_deleted": True, "remote_deleted": True,
                                     "deletion_confirmed_at": now, "deleted_at": now, "retention_reserved": 0.0}
                        inst.state = "deleted"
                        inst.current_job_id = None
                        inst.version += 1
                        inst.updated_at = utcnow()
                    self._apply_job_transition(session, job, owner_id, "cleaning_up", "GPU runtime disappeared before execution",
                        {"requested_outcome": "failed", "cleanup_final_state": "failed", "error_code": "gpu_runtime_missing"}, now)
                session.flush()
                return ("retired" if bound else "retired_detached"), self._row(job), self._row(lease)

    def _can_locally_delete_orphan_placeholder_in_session(
        self,
        session,
        owner_id: str,
        lease: GpuLeaseRow | None,
        instance_id: str | None = None,
        *,
        allow_deleting_state: bool = False,
        cleanup_job_id: str | None = None,
        allow_deleted_placeholder: bool = False,
    ) -> tuple[bool, str, GpuInstanceRow | None]:
        allowed_states = {"creating", "deleting"} if allow_deleting_state else {"creating"}
        if allow_deleted_placeholder:
            allowed_states.add("deleted")
        if lease is None or lease.owner_id != owner_id or lease.state not in allowed_states:
            return False, "lease_not_creating", None
        ldata = lease.data or {}
        eff_inst_id = instance_id or lease.instance_id or ldata.get("instance_id")
        if self._lease_has_active_origin(ldata, lease.state):
            return False, "active_origin", None
        if ldata.get("create_rpc_in_flight"):
            return False, "create_rpc_in_flight", None
        if (
            ldata.get("ever_active")
            or ldata.get("last_active_at") is not None
            or ldata.get("active_execution") is not None
            or ldata.get("warm_proof") is not None
        ):
            return False, "ever_active", None
        if (
            (isinstance(ldata.get("handle"), Mapping) and bool(ldata.get("handle")))
            or bool(ldata.get("runtime_name"))
            or extract_tunnel_endpoint(lease)
            or ldata.get("tunnel_nbh") is not None
            or ldata.get("operation_name")
            or ldata.get("proxy_url")
            or ldata.get("encrypted_proxy_token")
        ):
            return False, "remote_identity", None
        if (
            int(ldata.get("create_attempts") or 0) > 0
            or ldata.get("create_started")
            or ldata.get("create_rpc_started_at")
            or ldata.get("selected_spec")
            or ldata.get("create_intent_id")
            or ldata.get("create_intent") is not None
        ):
            return False, "create_intent", None
        executions = session.scalars(select(GpuExecutionRow).where(GpuExecutionRow.owner_id == owner_id).with_for_update()).all()
        inst = None
        if eff_inst_id:
            inst = session.scalar(
                select(GpuInstanceRow)
                .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                .with_for_update()
            )
        inst_data_for_scope = (inst.data or {}) if inst is not None else None
        assoc_jobs = self._collect_associated_jobs_for_scope(
            session,
            owner_id,
            lease_id=lease.id,
            instance_id=eff_inst_id,
            lease_data=ldata,
            inst_data=inst_data_for_scope,
        )
        assoc_job_ids = {jrow.id for jrow in assoc_jobs}
        if isinstance(ldata.get("owner_job_id"), str) and ldata["owner_job_id"]:
            assoc_job_ids.add(ldata["owner_job_id"])
        if isinstance(ldata.get("current_job_id"), str) and ldata["current_job_id"]:
            assoc_job_ids.add(ldata["current_job_id"])
        if any(
            (ex.data or {}).get("lease_id") == lease.id
            or (eff_inst_id and (ex.instance_id or (ex.data or {}).get("instance_id")) == eff_inst_id)
            or ((ex.data or {}).get("job_id") in assoc_job_ids)
            for ex in executions
        ):
            return False, "execution_exists", inst
        for jrow in assoc_jobs:
            jdata = jrow.data or {}
            job_intent = jdata.get("start_intent")
            if isinstance(job_intent, Mapping) and job_intent.get("lease_id") == lease.id:
                return False, "job_start_intent", inst
            if jdata.get("execution_id") or jdata.get("active_execution") is not None:
                return False, "job_execution_evidence", inst
            j_inst = jrow.instance_id or jdata.get("instance_id")
            if jrow.state in {"loading", "running", "uploading", "cleaning_up"} and jrow.id != cleanup_job_id and (
                jdata.get("lease_id") == lease.id or (eff_inst_id and j_inst == eff_inst_id)
            ):
                return False, "active_job_reference", inst
        if self._has_positive_usage_or_segments(
            session,
            owner_id,
            lease_id=lease.id,
            instance_id=eff_inst_id,
            job_ids=assoc_job_ids,
        ):
            return False, "positive_usage_exists", inst
        if eff_inst_id and inst is not None:
            idata = inst.data or {}
            if (
                inst.state in {"ready", "busy", "retained_idle", "unknown"}
                or idata.get("ever_active")
                or idata.get("runtime_created")
                or idata.get("active_execution") is not None
                or idata.get("warm_proof") is not None
                or idata.get("handle")
                or idata.get("runtime_name")
                or extract_tunnel_endpoint(inst)
                or idata.get("tunnel_nbh") is not None
                or idata.get("operation_name")
            ):
                return False, "instance_history", inst
        return True, "pure", inst

    def can_locally_delete_orphan_placeholder(
        self,
        owner_id: str,
        lease_id: str,
        instance_id: str | None = None,
    ) -> tuple[bool, str]:
        """Verify inside DB tx that a creating placeholder has zero history before local deletion."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False, "locked"
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(
                    select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update()
                )
                ok, reason, _ = self._can_locally_delete_orphan_placeholder_in_session(
                    session, owner_id, lease, instance_id, allow_deleting_state=False
                )
                return ok, reason

    def claim_create_intent(
        self,
        job_id: str,
        owner_id: str,
        lease_id: str,
        *,
        spec: Mapping[str, object],
        max_retries: int,
        now: float,
        require_instance: bool = False,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically claim the 0 -> 1 create attempt with create_intent_id before provider.create."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if not job or not lease or lease.owner_id != owner_id:
                    raise KeyError(job_id)
                jdata = copy.deepcopy(job.data or {})
                ldata = copy.deepcopy(lease.data or {})
                fenced, has_handle = self._is_create_fenced_in_session(
                    session, job, lease, owner_id, job_id, require_instance=require_instance
                )
                if fenced:
                    return "fenced", self._row(job), self._row(lease)
                if has_handle:
                    return "has_handle", self._row(job), self._row(lease)
                attempts = int(ldata.get("create_attempts") or 0)
                from_active = self._lease_has_active_origin(ldata, lease.state)
                if from_active and (
                    not ldata.get("from_active_missing_handle")
                    or not ldata.get("was_active_missing_handle")
                    or not ldata.get("create_intent_origin")
                ):
                    ldata["from_active_missing_handle"] = True
                    ldata["was_active_missing_handle"] = True
                    ldata.setdefault("create_intent_origin", "once_active")
                    ldata["last_seen"] = now
                    lease.data = ldata
                    lease.version += 1
                    lease.updated_at = utcnow()
                    session.flush()
                if from_active or attempts > 0 or lease.state == "unknown":
                    return "recover", self._row(job), self._row(lease)
                if self._create_rpc_live_in_flight(ldata, now):
                    return "in_flight", self._row(job), self._row(lease)
                total_attempts = int(jdata.get("alloc_create_attempts") or 0) + attempts
                if total_attempts > max_retries:
                    return "exhausted", self._row(job), self._row(lease)
                intent_id = ldata.get("create_intent_id") or str(uuid.uuid4())
                intent_origin = str(ldata.get("create_intent_origin") or ldata.get("origin") or "new")
                if intent_origin.strip().lower() in _ACTIVE_ORIGINS:
                    return "recover", self._row(job), self._row(lease)
                rpc_token = str(uuid.uuid4())
                rpc_inst_id = lease.instance_id or ldata.get("instance_id") or (jdata.get("instance_id") if jdata.get("lease_id") in {None, lease.id} else None)
                ldata.update(
                    create_intent_id=intent_id,
                    create_intent_origin=intent_origin,
                    create_intent={
                        "intent_id": intent_id,
                        "origin": intent_origin,
                        "from_active": False,
                        "state": "claimed",
                        "claimed_at": now,
                    },
                    create_attempts=attempts + 1,
                    from_active_missing_handle=False,
                    create_started=now,
                    create_rpc_in_flight=True,
                    create_rpc_token=rpc_token,
                    create_rpc_attempt=attempts + 1,
                    create_rpc_instance_id=rpc_inst_id,
                    create_rpc_started_at=now,
                    create_rpc_finished_at=None,
                    selected_spec=copy.deepcopy(dict(spec)),
                    last_seen=now,
                )
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return "create", self._row(job), self._row(lease)

    def retry_absent_lease_create(
        self,
        job_id: str,
        owner_id: str,
        expected_lease_id: str,
        *,
        spec: Mapping[str, object],
        max_retries: int,
        now: float,
        expected_lease_version: int | None = None,
        expected_create_attempts: int | None = None,
        require_instance: bool = False,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically claim bounded R2 create retry on the SAME lease/request/runtime/instance/budget."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == expected_lease_id).with_for_update())
                if not job or not lease or lease.owner_id != owner_id:
                    raise KeyError(job_id)
                jdata = copy.deepcopy(job.data or {})
                ldata = copy.deepcopy(lease.data or {})
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", self._row(job), self._row(lease)
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(
                    expected_create_attempts
                ):
                    return "stale", self._row(job), self._row(lease)
                if self._create_rpc_live_in_flight(ldata, now):
                    return "in_flight", self._row(job), self._row(lease)
                fenced, has_handle = self._is_create_fenced_in_session(
                    session,
                    job,
                    lease,
                    owner_id,
                    job_id,
                    allowed_lease_states={"creating", "unknown"},
                    require_instance=require_instance,
                )
                if fenced or self._lease_has_active_origin(ldata, lease.state) or jdata.get("lease_id") != expected_lease_id:
                    return "fenced", self._row(job), self._row(lease)
                if has_handle:
                    return "has_handle", self._row(job), self._row(lease)
                attempts = int(ldata.get("create_attempts") or 0)
                if attempts > max_retries:
                    return "exhausted", self._row(job), self._row(lease)
                intent_id = ldata.get("create_intent_id") or str(uuid.uuid4())
                intent_origin = str(ldata.get("create_intent_origin") or "retry")
                if intent_origin.strip().lower() in _ACTIVE_ORIGINS:
                    return "fenced", self._row(job), self._row(lease)
                rpc_token = str(uuid.uuid4())
                rpc_inst_id = lease.instance_id or ldata.get("instance_id") or (jdata.get("instance_id") if jdata.get("lease_id") in {None, lease.id} else None)
                ldata.pop("last_error", None)
                ldata.update(
                    create_intent_id=intent_id,
                    create_intent_origin=intent_origin,
                    create_intent={
                        "intent_id": intent_id,
                        "origin": intent_origin,
                        "from_active": False,
                        "state": "retrying",
                        "claimed_at": now,
                    },
                    create_attempts=attempts + 1,
                    from_active_missing_handle=False,
                    create_started=now,
                    create_rpc_in_flight=True,
                    create_rpc_token=rpc_token,
                    create_rpc_attempt=attempts + 1,
                    create_rpc_instance_id=rpc_inst_id,
                    create_rpc_started_at=now,
                    create_rpc_finished_at=None,
                    selected_spec=copy.deepcopy(dict(spec)),
                    last_seen=now,
                )
                lease.state = "creating"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return "create", self._row(job), self._row(lease)

    def retire_absent_placeholder_for_retry(
        self,
        job_id: str,
        owner_id: str,
        *,
        expected_lease_id: str | None = None,
        expected_generation: int,
        total_attempts: int,
        now: float,
    ) -> tuple[bool, dict | None]:
        """Atomically retire an uncreated pure placeholder and bump alloc_generation for bounded R3."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return False, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                jdata = copy.deepcopy(job.data or {})
                if (
                    job.state != "allocating"
                    or jdata.get("cancel_requested")
                    or jdata.get("requested_outcome") == "cancelled"
                    or int(jdata.get("alloc_generation") or 0) != int(expected_generation)
                    or jdata.get("lease_id") != expected_lease_id
                ):
                    return False, self._row(job)
                lease = session.get(GpuLeaseRow, expected_lease_id) if expected_lease_id else None
                ldata: dict[str, object] = {}
                if lease is not None:
                    if lease.owner_id != owner_id or lease.state not in {"creating", "unknown"}:
                        return False, self._row(job)
                    ldata = copy.deepcopy(lease.data or {})
                    create_intent = ldata.get("create_intent")
                    non_pending_intent = create_intent is not None and not (
                        isinstance(create_intent, Mapping) and create_intent.get("state") == "pending"
                    )
                    if (
                        int(ldata.get("create_attempts") or 0) > 0
                        or ldata.get("handle")
                        or ldata.get("tunnel_endpoint")
                        or ldata.get("tunnel_nbh") is not None
                        or ldata.get("operation_name")
                        or ldata.get("proxy_url")
                        or self._lease_has_active_origin(ldata, lease.state)
                        or ldata.get("create_intent_id")
                        or ldata.get("create_started")
                        or ldata.get("selected_spec")
                        or non_pending_intent
                        or ldata.get("cleanup_claim_token")
                        or ldata.get("manual_stop_requested")
                        or ldata.get("runtime_deleted")
                        or ldata.get("remote_deleted")
                        or ldata.get("owner_job_id") != job_id
                        or ldata.get("current_job_id") not in {None, job_id}
                        or ldata.get("active_execution") is not None
                        or ldata.get("warm_proof") is not None
                    ):
                        return False, self._row(job)
                instance_id = jdata.get("instance_id") or ldata.get("instance_id") or (lease.instance_id if lease else None)
                if lease is not None and self._create_rpc_live_in_flight(ldata, now):
                    return False, self._row(job)
                if (jdata.get("execution_id") is not None) or (jdata.get("active_execution") is not None):
                    return False, self._row(job)
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=instance_id,
                        lease=lease,
                        job=job,
                        exclude_lease_id=expected_lease_id,
                        exclude_instance_id=instance_id,
                    )[0]
                ):
                    return False, self._row(job)
                assoc_jobs = self._collect_associated_jobs_for_scope(
                    session,
                    owner_id,
                    lease_id=expected_lease_id,
                    instance_id=instance_id,
                    lease_data=ldata,
                )
                assoc_job_ids = {job_id} | {jr.id for jr in assoc_jobs}
                for jr in assoc_jobs:
                    jr_data = jr.data or {}
                    if (
                        jr_data.get("execution_id") is not None
                        or jr_data.get("active_execution") is not None
                        or (jr.id != job_id and jr.state not in {"completed", "failed", "cancelled"})
                    ):
                        return False, self._row(job)
                all_execs = session.scalars(select(GpuExecutionRow)).all()
                if any(
                    (instance_id and ((e.data or {}).get("instance_id") or e.instance_id) == instance_id)
                    or (expected_lease_id and (e.data or {}).get("lease_id") == expected_lease_id)
                    or ((e.data or {}).get("job_id") in assoc_job_ids)
                    for e in all_execs
                ):
                    return False, self._row(job)
                if self._has_positive_usage_or_segments(
                    session,
                    owner_id,
                    lease_id=expected_lease_id,
                    instance_id=instance_id,
                    job_ids=assoc_job_ids,
                ):
                    return False, self._row(job)
                all_jobs = session.scalars(select(GpuJobRow)).all()
                if any(
                    jrow.id != job_id
                    and (
                        jrow.state not in {"completed", "failed", "cancelled"}
                        or (jrow.data or {}).get("execution_id") is not None
                        or (jrow.data or {}).get("active_execution") is not None
                    )
                    and (
                        (instance_id and (jrow.instance_id or (jrow.data or {}).get("instance_id")) == instance_id)
                        or (expected_lease_id and (jrow.data or {}).get("lease_id") == expected_lease_id)
                    )
                    for jrow in all_jobs
                ):
                    return False, self._row(job)
                if instance_id:
                    instance = session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == instance_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if instance is None or instance.state not in {"allocating", "creating", "requested"}:
                        return False, self._row(job)
                    idata = copy.deepcopy(instance.data or {})
                    if (
                        idata.get("lease_id") not in {None, expected_lease_id}
                        or idata.get("ever_active")
                        or idata.get("runtime_created")
                        or idata.get("active_execution") is not None
                        or idata.get("warm_proof") is not None
                        or idata.get("cleanup_claim_token")
                        or idata.get("manual_stop_requested")
                        or idata.get("runtime_deleted")
                        or idata.get("remote_deleted")
                        or idata.get("current_job_id") not in {None, job_id}
                    ):
                        return False, self._row(job)
                    idata.update(
                        current_job_id=None,
                        runtime_deleted=True,
                        remote_deleted=True,
                        retired_pure_placeholder=True,
                        retention_reserved=0.0,
                        deleted_at=now,
                        deletion_confirmed_at=now,
                        last_seen_at=now,
                    )
                    instance.state = "deleted"
                    instance.current_job_id = None
                    instance.data = idata
                    instance.version += 1
                    instance.updated_at = utcnow()
                    session.add(
                        GpuEventRow(
                            id=str(uuid.uuid4()),
                            owner_id=owner_id,
                            project_id=instance.project_id,
                            state="deleted",
                            version=0,
                            data={
                                "instance_id": instance_id,
                                "job_id": job_id,
                                "state": "deleted",
                                "message": "pure placeholder retired",
                            },
                            created_at=utcnow(),
                            updated_at=utcnow(),
                        )
                    )
                if lease is not None:
                    ldata.update(
                        state="deleted",
                        current_job_id=None,
                        remote_deleted=True,
                        remote_deleted_at=now,
                        remote_delete_source="recovery_probe_absent",
                        last_seen=now,
                    )
                    lease.state = "deleted"
                    lease.data = ldata
                    lease.version += 1
                    lease.updated_at = utcnow()
                jdata.update(
                    alloc_generation=int(expected_generation) + 1,
                    alloc_create_attempts=int(total_attempts),
                    lease_id=None,
                    instance_id=None,
                )
                job.data = jdata
                job.instance_id = None
                job.version += 1
                job.updated_at = utcnow()
                session.flush()
                return True, self._row(job)

    def commit_execution_terminal(
        self,
        job_id: str,
        owner_id: str,
        *,
        execution_id: str | None,
        terminal_state: str,
        message: str,
        patch: Mapping[str, object] | None,
        now: float,
        proof: Mapping[str, object] | None = None,
        expected_state: str | None = "running",
        expected_generation: int | None = None,
        require_instance: bool = False,
    ) -> dict:
        """Persist warm_proof (or dual-clear cold) under active_execution identity before exposing terminal state."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                raise RuntimeError("admission lock unavailable")
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                status, lease, instance, execution, active_gen = self._evaluate_execution_fence_in_session(
                    session,
                    job,
                    owner_id,
                    execution_id=execution_id,
                    expected_state=expected_state,
                    expected_generation=expected_generation,
                    require_instance=require_instance,
                )
                if status == "stale":
                    return self._row(job)
                if status == "fenced":
                    return self._row(job)

                verified_warm = self._validate_warm_proof(
                    job.data or {},
                    execution_id,
                    terminal_state,
                    proof,
                    generation=active_gen,
                    now=now,
                )
                timings = proof.get("timings") if isinstance(proof, Mapping) and isinstance(proof.get("timings"), Mapping) else {}
                restore_skipped = "warm_reuse" in timings

                self._persist_runtime_warm_proof(lease, instance, job.data or {}, verified_warm)

                next_patch = dict(patch or {})
                if terminal_state in {"uploading", "completed"}:
                    next_patch.setdefault("restore_skipped", restore_skipped)
                result_job = self._apply_job_transition(session, job, owner_id, terminal_state, message, next_patch, now)
                if execution is not None:
                    edata = copy.deepcopy(execution.data or {})
                    if instance is not None and execution.instance_id is None and edata.get("instance_id") == instance.id:
                        execution.instance_id = instance.id
                    if terminal_state in {"uploading", "completed"}:
                        edata["state"] = "completed"
                        edata["finished_at"] = now
                        if verified_warm is not None:
                            edata["worker_reported_execution_id"] = verified_warm["worker_reported_execution_id"]
                            edata["warm_saved"] = True
                            edata["warm_epoch_id"] = verified_warm["warm_epoch_id"]
                            edata["warm_fingerprint"] = verified_warm["warm_fingerprint"]
                            edata["warm_proof"] = copy.deepcopy(dict(verified_warm))
                            if verified_warm.get("runtime_meta") is not None:
                                edata["runtime_meta"] = copy.deepcopy(verified_warm["runtime_meta"])
                        else:
                            edata["warm_saved"] = False
                            edata["warm_epoch_id"] = None
                            edata["warm_fingerprint"] = None
                            edata["warm_proof"] = None
                        execution.state = "completed"
                    else:
                        edata["warm_saved"] = False
                        edata["warm_epoch_id"] = None
                        edata["warm_fingerprint"] = None
                        edata["warm_proof"] = None
                    execution.data = edata
                    execution.version += 1
                    execution.updated_at = utcnow()
                    session.flush()
                return result_job

    def finalize_retained_cleanup(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_id: str,
        now: float,
        final_state: str = "completed",
        instance_id: str | None = None,
        expected_instance_state: str | None = None,
        idle_deadline_at: float | None = None,
        hard_deadline_at: float | None = None,
        compatibility_signature: str | None = None,
        legacy_mock_idle: bool = False,
        require_instance: bool = True,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically transition lease -> idle, execution -> completed, and job -> terminal under admission_lock."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if not job:
                    raise KeyError(job_id)
                if job.state in {"completed", "failed", "cancelled"}:
                    return "already_terminal", self._row(job), None
                jdata = copy.deepcopy(job.data or {})
                if (
                    job.state not in {"cleaning_up", "cleanup_pending"}
                    or final_state != "completed"
                    or jdata.get("cancel_requested")
                    or jdata.get("requested_outcome") == "cancelled"
                    or self._has_stop_or_delete_flag(jdata)
                ):
                    return "aborted", self._row(job), None
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id:
                    return "aborted", self._row(job), None
                if (
                    jdata.get("lease_id") != lease.id
                    if not legacy_mock_idle
                    else jdata.get("lease_id") not in {None, lease.id}
                ):
                    return "aborted", self._row(job), self._row(lease)
                ldata = copy.deepcopy(lease.data or {})
                if (
                    lease.state not in {"active", "busy"}
                    or self._has_stop_or_delete_flag(ldata)
                    or ldata.get("owner_job_id") != job_id
                    or ldata.get("current_job_id") not in {None, job_id}
                    or not isinstance(ldata.get("handle"), Mapping)
                    or not ldata.get("handle")
                ):
                    return "aborted", self._row(job), self._row(lease)
                if job.instance_id and jdata.get("instance_id") and job.instance_id != jdata.get("instance_id"):
                    return "aborted", self._row(job), self._row(lease)
                if lease.instance_id and ldata.get("instance_id") and lease.instance_id != ldata.get("instance_id"):
                    return "aborted", self._row(job), self._row(lease)
                job_inst_id = jdata.get("instance_id") or job.instance_id
                lease_inst_id = ldata.get("instance_id") or lease.instance_id
                eff_inst_id = instance_id or job_inst_id or lease_inst_id
                if not legacy_mock_idle:
                    if not eff_inst_id or job_inst_id != eff_inst_id or lease_inst_id != eff_inst_id:
                        return "aborted", self._row(job), self._row(lease)
                    if instance_id is not None and instance_id != eff_inst_id:
                        return "aborted", self._row(job), self._row(lease)
                else:
                    inst_refs = {v for v in (instance_id, job_inst_id, lease_inst_id) if v}
                    if len(inst_refs) > 1:
                        return "aborted", self._row(job), self._row(lease)
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if eff_inst_id
                    else None
                )
                idata = copy.deepcopy(instance.data or {}) if instance is not None else {}
                if instance is not None:
                    inst_lease_mismatch = (
                        idata.get("lease_id") != lease.id
                        if not legacy_mock_idle
                        else idata.get("lease_id") not in {None, lease.id}
                    )
                    if (
                        instance.state in {"deleted", "deleting", "releasing", "cleanup_pending", "quarantined", "unknown"}
                        or self._has_stop_or_delete_flag(idata)
                        or idata.get("current_job_id") not in {None, job_id}
                        or inst_lease_mismatch
                    ):
                        return "aborted", self._row(job), self._row(lease)
                lease_rt = extract_runtime_id(lease)
                inst_rt = extract_runtime_id(instance) if instance is not None else None
                if lease_rt and inst_rt and lease_rt != inst_rt:
                    return "aborted", self._row(job), self._row(lease)
                lease_ep = extract_tunnel_endpoint(lease)
                inst_ep = extract_tunnel_endpoint(instance) if instance is not None else None
                if lease_ep and inst_ep and lease_ep != inst_ep:
                    return "aborted", self._row(job), self._row(lease)
                if not is_valid_runtime_handle(
                    ldata.get("handle"),
                    owner_id=owner_id,
                    lease_runtime_id=lease_rt,
                    instance_runtime_id=inst_rt,
                    provider=ldata.get("provider") or jdata.get("provider") or (idata.get("provider") if instance is not None else None),
                ):
                    return "aborted", self._row(job), self._row(lease)
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=eff_inst_id,
                        lease=lease,
                        instance=instance,
                        job=job,
                        exclude_lease_id=lease.id,
                        exclude_instance_id=eff_inst_id,
                    )[0]
                ):
                    return "aborted", self._row(job), self._row(lease)
                exec_id = jdata.get("execution_id")
                execution = session.get(GpuExecutionRow, exec_id) if exec_id else None
                exec_ok, active_gen = self._execution_matches_bindings_and_generation(
                    owner_id=owner_id,
                    job=job,
                    execution=execution,
                    lease=lease,
                    instance=instance,
                    lease_id=lease.id,
                    instance_id=eff_inst_id,
                )
                if not exec_ok:
                    return "aborted", self._row(job), self._row(lease)
                if not legacy_mock_idle:
                    if instance is None or instance.state != "retained_idle":
                        return "aborted", self._row(job), self._row(lease)
                eff_hard = min_deadlines(
                    idata.get("hard_deadline_at") if instance is not None else None,
                    ldata.get("hard_deadline_at"),
                    ldata.get("deadline_epoch"),
                    jdata.get("hard_deadline_at"),
                    jdata.get("deadline_epoch"),
                    hard_deadline_at,
                )
                eff_idle = min_deadlines(
                    idata.get("retain_until") if instance is not None else None,
                    idata.get("idle_deadline_at") if instance is not None else None,
                    ldata.get("retain_until"),
                    ldata.get("idle_deadline_at"),
                    jdata.get("retain_until"),
                    jdata.get("idle_deadline_at"),
                    idle_deadline_at,
                )
                retain_deadline = min_deadlines(eff_hard, eff_idle)
                if retain_deadline is not None and now >= float(retain_deadline):
                    return "expired", self._row(job), self._row(lease)
                ldata["state"] = "idle"
                ldata["current_job_id"] = None
                ldata["last_seen"] = now
                if legacy_mock_idle:
                    ldata["last_active_at"] = now
                if eff_idle is not None:
                    ldata["idle_deadline_at"] = eff_idle
                    if (
                        idata.get("retain_until") is not None
                        or ldata.get("retain_until") is not None
                        or jdata.get("retain_until") is not None
                    ):
                        ldata["retain_until"] = eff_idle
                if eff_hard is not None:
                    ldata["deadline_epoch"] = eff_hard
                if compatibility_signature is not None:
                    ldata["compatibility_signature"] = compatibility_signature
                lease.state = "idle"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                exec_id = jdata.get("execution_id")
                if exec_id:
                    execution = session.get(GpuExecutionRow, exec_id)
                    if execution is not None and execution.owner_id == owner_id:
                        edata = copy.deepcopy(execution.data or {})
                        edata.update(state="completed", finished_at=now)
                        if eff_inst_id and execution.instance_id is None and edata.get("instance_id") == eff_inst_id:
                            execution.instance_id = eff_inst_id
                        warm_proof = ldata.get("warm_proof")
                        if not isinstance(warm_proof, Mapping):
                            warm_proof = idata.get("warm_proof")
                        warm_gen = self._valid_generation_int(warm_proof.get("generation")) if isinstance(warm_proof, Mapping) else None
                        warm_gen_ok = (
                            (active_gen is not None and warm_gen == active_gen)
                            or (active_gen is None and warm_proof.get("generation") is None)
                        ) if isinstance(warm_proof, Mapping) else False
                        if (
                            isinstance(warm_proof, Mapping)
                            and warm_gen_ok
                            and warm_proof.get("warm_saved") is True
                            and warm_proof.get("warm_execution_id") == exec_id
                            and warm_proof.get("warm_epoch_id")
                            and warm_proof.get("warm_fingerprint")
                        ):
                            edata.setdefault("warm_saved", True)
                            edata.setdefault("warm_epoch_id", warm_proof["warm_epoch_id"])
                            edata.setdefault("warm_fingerprint", warm_proof["warm_fingerprint"])
                            if edata.get("runtime_meta") is None and isinstance(warm_proof.get("runtime_meta"), Mapping):
                                edata["runtime_meta"] = copy.deepcopy(dict(warm_proof["runtime_meta"]))
                        execution.state = "completed"
                        execution.data = edata
                        execution.version += 1
                        execution.updated_at = utcnow()
                next_patch: dict[str, object] = {"cleanup_next_epoch": 0}
                if jdata.get("error_code") in {"gpu_cleanup_pending", "gpu_instance_completion_pending"}:
                    next_patch.update(error_code=None, error_message=None)
                msg = (
                    "Mock outputs verified; idle lease retained"
                    if legacy_mock_idle
                    else "GPU cleanup completed; instance retained"
                )
                final_job = self._apply_job_transition(session, job, owner_id, final_state, msg, next_patch, now)
                return "ok", final_job, self._row(lease)

    @staticmethod
    def _derive_instance_budget_reserved(
        idata: Mapping[str, object],
        *,
        job_data: Mapping[str, object] | None = None,
        lease_data: Mapping[str, object] | None = None,
    ) -> float:
        candidates = [
            idata.get("budget_reserved"),
            idata.get("last_budget_reserved"),
            idata.get("reserved_units"),
        ]
        if isinstance(job_data, Mapping):
            candidates.extend(
                [job_data.get("budget_reserved"), job_data.get("budget_units"), job_data.get("estimated_units")]
            )
        if isinstance(lease_data, Mapping):
            candidates.append(lease_data.get("budget_reserved"))
        for raw in candidates:
            if raw is not None and not isinstance(raw, bool):
                try:
                    val = float(raw)
                    if 0 < val < float("inf"):
                        return val
                except (TypeError, ValueError):
                    pass
        spec = idata.get("spec") or (lease_data.get("selected_spec") if isinstance(lease_data, Mapping) else None)
        if isinstance(spec, Mapping):
            key = spec.get("key") if isinstance(spec.get("key"), Mapping) else spec
            accel = str(key.get("accelerator") or "").upper()
            shape = str(key.get("shape") or "").upper()
            if accel == "A100":
                return 6.0 if shape == "SHAPE_HIGHMEM" else 4.0
            if accel in {"V100", "L4"}:
                return 2.0 if accel == "L4" else 3.0
            if accel == "T4":
                return 1.5
        return 1.0

    @staticmethod
    def _create_attempt_matches(ldata, rpc_token, attempt, instance_id):
        """Compare the original RPC identity without accepting missing persisted fields as wildcards."""
        return (
            ldata.get("create_rpc_token") == rpc_token
            and ldata.get("create_attempts") == attempt
            and ldata.get("create_rpc_attempt", attempt) == attempt
            and ldata.get("create_rpc_instance_id") == instance_id
        )

    def _isolate_late_handle(self, session, lease, owner_id, handle, rpc_token, attempt, instance_id, now):
        """Keep a stale RPC's remote resource and reservation separate from the current attempt.

        Quarantine prevents ambiguous/same-runtime handles from being automatically deleted.
        Deterministic IDs make retries idempotent, including recovery after a process restart.
        """
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, repr((lease.id, rpc_token, attempt, instance_id, dict(handle)))))
        isolation_id = "late-" + key
        if session.get(GpuLeaseRow, isolation_id) is not None:
            return
        ldata = lease.data or {}
        inst_id = "late-i-" + key
        original = session.get(GpuInstanceRow, instance_id) if instance_id else None
        budget = self._derive_instance_budget_reserved(
            original.data if original is not None else {}, lease_data=ldata,
        )
        common = {"provider": ldata.get("provider"), "account_id": ldata.get("account_id"),
                  "runtime_id": extract_runtime_id(handle), "handle": copy.deepcopy(dict(handle)),
                  "state": "quarantined", "remote_deleted": False, "last_seen": now,
                  "late_create_source_lease_id": lease.id, "create_rpc_token": rpc_token,
                  "create_attempts": attempt, "create_rpc_instance_id": instance_id,
                  "owner_job_id": None, "current_job_id": None}
        session.add(GpuLeaseRow(id=isolation_id, owner_id=owner_id, project_id=lease.project_id,
            instance_id=inst_id, state="quarantined", version=0,
            data={**common, "instance_id": inst_id}, created_at=utcnow(), updated_at=utcnow()))
        session.add(GpuInstanceRow(id=inst_id, owner_id=owner_id, project_id=lease.project_id,
            account_id=ldata.get("account_id"), state="quarantined", version=0,
            data={**common, "lease_id": isolation_id, "budget_reserved": budget, "runtime_created": True},
            created_at=utcnow(), updated_at=utcnow()))
        session.flush()

    def abort_allocation_cas(self, job_snapshot, lease_snapshot, now):
        """Finish a pre-create fence only while its captured job and RPC binding still own allocation."""
        owner_id = job_snapshot["owner_id"]
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return self.get("jobs", job_snapshot["id"], owner_id)
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(select(GpuJobRow).where(
                    GpuJobRow.id == job_snapshot["id"], GpuJobRow.owner_id == owner_id).with_for_update())
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_snapshot["id"], GpuLeaseRow.owner_id == owner_id).with_for_update())
                if job is None:
                    return None
                if (lease is None or job.state != "allocating" or job.version != job_snapshot.get("version")
                    or lease.version != lease_snapshot.get("version")
                    or (job.data or {}).get("lease_id") != lease.id
                    or (job.instance_id or (job.data or {}).get("instance_id")) != job_snapshot.get("instance_id")
                    or not self._create_attempt_matches(lease.data or {}, lease_snapshot.get("create_rpc_token"),
                        lease_snapshot.get("create_attempts"), lease_snapshot.get("create_rpc_instance_id"))):
                    return self._row(job)
                cancelled = (job.data or {}).get("cancel_requested") or (job.data or {}).get("requested_outcome") == "cancelled"
                return self._apply_job_transition(session, job, owner_id, "cleaning_up", "GPU allocation fenced",
                    {"requested_outcome": "cancelled" if cancelled else "failed",
                     "cleanup_final_state": "cancelled" if cancelled else "failed", "error_code": "gpu_allocation_aborted"}, now)

    def finish_create_rpc_failure(self, job_id, owner_id, lease_id, *, rpc_token, expected_attempt,
                                  expected_instance_id, now, code=None, message=None):
        """CAS a failed RPC and its job transition; a rebound job or newer attempt receives zero writes."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(select(GpuJobRow).where(
                    GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update())
                lease = session.scalar(select(GpuLeaseRow).where(
                    GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update())
                if job is None or lease is None:
                    return "missing", self._row(job) if job else None, self._row(lease) if lease else None
                ldata = copy.deepcopy(lease.data or {})
                if (
                    not self._create_attempt_matches(ldata, rpc_token, expected_attempt, expected_instance_id)
                    or job.state != "allocating" or (job.data or {}).get("lease_id") != lease.id
                    or (job.instance_id or (job.data or {}).get("instance_id")) != expected_instance_id
                    or ldata.get("owner_job_id") != job.id
                    or ldata.get("current_job_id") not in {None, job.id}
                ):
                    return "stale", self._row(job), self._row(lease)
                ldata.update(create_rpc_in_flight=False, create_rpc_finished_at=now, last_seen=now)
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                if code is not None:
                    self._apply_job_transition(session, job, owner_id, "cleaning_up", message,
                        {"requested_outcome": "failed", "cleanup_final_state": "failed",
                         "error_code": code, "error_message": message}, now)
                session.flush()
                return "ok", self._row(job), self._row(lease)

    def _recover_late_handle_in_session(
        self,
        session,
        job: GpuJobRow | None,
        lease: GpuLeaseRow,
        instance: GpuInstanceRow | None,
        owner_id: str,
        handle: Mapping[str, object],
        *,
        now: float,
    ) -> None:
        handle_copy = copy.deepcopy(dict(handle))
        rt_id = extract_runtime_id(handle_copy) or extract_runtime_id(lease)
        rt_name = handle_copy.get("runtime_name")
        ep = extract_tunnel_endpoint(handle_copy) or extract_tunnel_endpoint(lease)
        ldata = copy.deepcopy(lease.data or {})
        existing_handle = ldata.get("handle")
        if (
            lease.state in {"active", "busy", "idle"}
            and isinstance(existing_handle, Mapping)
            and existing_handle
            and dict(existing_handle) != handle_copy
        ):
            # Lease has already moved to a distinct live runtime handle; never clobber it with a stale attempt.
            session.flush()
            return
        ldata.update(
            state="cleanup_pending",
            handle=handle_copy,
            remote_deleted=False,
            remote_delete_source=None,
            runtime_deleted=False,
            deleted=False,
            cleanup_claim_token=None,
            cleanup_claimed_at=None,
            cleanup_next_epoch=0,
            create_rpc_in_flight=False,
            create_rpc_finished_at=now,
            last_seen=now,
        )
        if rt_id:
            ldata["runtime_id"] = rt_id
        if isinstance(rt_name, str) and rt_name.strip():
            ldata["runtime_name"] = rt_name.strip()
        if ep:
            ldata["tunnel_endpoint"] = handle_copy.get("tunnel_endpoint") or ep
        lease.state = "cleanup_pending"
        lease.data = ldata
        lease.version += 1
        lease.updated_at = utcnow()
        job_belongs_to_lease = (
            job is not None
            and ((job.data or {}).get("lease_id") == lease.id or ((job.data or {}).get("lease_id") is None and ldata.get("owner_job_id") == job.id))
        )
        scoped_job_data = job.data if (job is not None and job_belongs_to_lease) else None
        lease_inst_ref = ldata.get("create_rpc_instance_id") or ldata.get("instance_id") or lease.instance_id
        if (
            instance is not None
            and instance.owner_id == owner_id
            and lease_inst_ref in {None, instance.id}
            and (instance.data or {}).get("lease_id") in {None, lease.id}
            and (instance.data or {}).get("current_job_id") in {None, ldata.get("owner_job_id")}
        ):
            idata = copy.deepcopy(instance.data or {})
            was_deleted = instance.state == "deleted"
            idata.update(
                state="cleanup_pending",
                current_job_id=None,
                runtime_created=True,
                runtime_deleted=False,
                remote_deleted=False,
                deleted=False,
                cleanup_claim_token=None,
                cleanup_claimed_at=None,
                retention_reserved=0.0,
                budget_reserved=self._derive_instance_budget_reserved(
                    idata,
                    job_data=scoped_job_data,
                    lease_data=ldata,
                ),
                handle=copy.deepcopy(handle_copy),
                last_seen_at=now,
            )
            if rt_id:
                idata["runtime_id"] = rt_id
            if isinstance(rt_name, str) and rt_name.strip():
                idata["runtime_name"] = rt_name.strip()
            if ep:
                idata["tunnel_endpoint"] = handle_copy.get("tunnel_endpoint") or ep
            instance.state = "cleanup_pending"
            instance.current_job_id = None
            instance.data = idata
            instance.version += 1
            instance.updated_at = utcnow()
            if was_deleted:
                session.add(
                    GpuEventRow(
                        id=str(uuid.uuid4()),
                        owner_id=owner_id,
                        project_id=instance.project_id,
                        state="cleanup_pending",
                        version=0,
                        data={
                            "instance_id": instance.id,
                            "job_id": job.id if job_belongs_to_lease and job is not None else ldata.get("owner_job_id"),
                            "state": "cleanup_pending",
                            "message": "late runtime handle recovered for cleanup and quota retention",
                        },
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
        if job_belongs_to_lease and job.state == "allocating" and (
            (job.instance_id or (job.data or {}).get("instance_id")) == lease_inst_ref
        ):
            cancelled = (job.data or {}).get("cancel_requested") or (job.data or {}).get("requested_outcome") == "cancelled"
            self._apply_job_transition(session, job, owner_id, "cleaning_up", "Late GPU allocation requires cleanup",
                {"requested_outcome": "cancelled" if cancelled else "failed",
                 "cleanup_final_state": "cancelled" if cancelled else "failed",
                 "error_code": "gpu_allocation_aborted"}, now)
        session.flush()

    def clear_create_rpc_in_flight(
        self,
        lease_id: str,
        owner_id: str,
        *,
        rpc_token: str | None = None,
        expected_attempt: int | None = None,
        expected_instance_id: str | None = None,
        now: float,
    ) -> dict | None:
        """Clear create_rpc_in_flight once the RPC finishes (only if rpc_token still matches)."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return None
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id:
                    return None
                ldata = copy.deepcopy(lease.data or {})
                if not self._create_attempt_matches(ldata, rpc_token, expected_attempt, expected_instance_id):
                    return self._row(lease)
                if not ldata.get("create_rpc_in_flight"):
                    return self._row(lease)
                ldata["create_rpc_in_flight"] = False
                ldata["create_rpc_finished_at"] = now
                ldata["last_seen"] = now
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return self._row(lease)

    def commit_create_handle_success(
        self,
        job_id: str,
        owner_id: str,
        lease_id: str,
        *,
        handle: Mapping[str, object],
        rpc_token: str | None = None,
        expected_attempt: int | None = None,
        expected_instance_id: str | None = None,
        started_at: float,
        deadline_epoch: float | None = None,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically activate lease+job with handle, or recover into cleanup_pending (with quota retained) if fenced/deleted."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id:
                    return "missing", (self._row(job) if job else None), None
                jdata = copy.deepcopy(job.data or {}) if job is not None else {}
                ldata = copy.deepcopy(lease.data or {})
                if (ldata.get("owner_job_id") not in {None, job_id}
                    or ldata.get("current_job_id") not in {None, job_id}):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, expected_instance_id, started_at)
                    return "stale", (self._row(job) if job else None), self._row(lease)

                if job is not None and (job.data or {}).get("lease_id") == lease.id and (
                    (job.instance_id or (job.data or {}).get("instance_id")) != expected_instance_id
                ):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, expected_instance_id, started_at)
                    return "stale", self._row(job), self._row(lease)
                lease_inst_id = ldata.get("create_rpc_instance_id") or ldata.get("instance_id") or lease.instance_id
                job_belongs = (
                    job is not None
                    and (
                        jdata.get("lease_id") == lease.id
                        or (jdata.get("lease_id") is None and ldata.get("owner_job_id") == job.id)
                    )
                )
                inst_id = lease_inst_id or expected_instance_id or (jdata.get("instance_id") if job_belongs else None)
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if inst_id
                    else None
                )
                idata = (instance.data or {}) if instance is not None else {}
                if not self._create_attempt_matches(ldata, rpc_token, expected_attempt, expected_instance_id):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, expected_instance_id, started_at)
                    return "stale", (self._row(job) if job else None), self._row(lease)
                if job_belongs and job.state != "allocating" and lease.state in {"active", "busy", "idle"}:
                    # A read-only restart reconciliation may have committed this very RPC already.
                    if dict(ldata.get("handle") or {}) != dict(handle):
                        self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                                  expected_attempt, expected_instance_id, started_at)
                    return "stale", self._row(job), self._row(lease)
                conflict, _ = has_conflicting_active_runtime(
                    session,
                    instance_id=inst_id,
                    handle=handle,
                    lease=lease,
                    instance=instance,
                    job=job,
                    exclude_lease_id=lease.id,
                    exclude_instance_id=inst_id,
                )
                fenced = (
                    conflict
                    or job is None
                    or job.state != "allocating"
                    or bool(jdata.get("cancel_requested"))
                    or jdata.get("requested_outcome") == "cancelled"
                    or self._has_stop_or_delete_flag(jdata)
                    or jdata.get("lease_id") not in {None, lease.id}
                    or lease.state in {"deleted", "deleting", "cleanup_pending", "releasing", "quarantined", "idle"}
                    or self._has_stop_or_delete_flag(ldata)
                    or ldata.get("owner_job_id") != job_id
                    or ldata.get("current_job_id") not in {None, job_id}
                    or (
                        instance is not None
                        and (
                            instance.state in {"deleted", "deleting", "releasing", "cleanup_pending", "quarantined", "unknown", "retained_idle"}
                            or self._has_stop_or_delete_flag(idata)
                            or idata.get("current_job_id") not in {None, job_id}
                            or idata.get("lease_id") not in {None, lease.id}
                        )
                    )
                )
                if fenced:
                    self._recover_late_handle_in_session(session, job, lease, instance, owner_id, handle, now=started_at)
                    return "fenced", (self._row(job) if job else None), self._row(lease)
                handle_copy = copy.deepcopy(dict(handle))
                ldata.update(
                    state="active",
                    handle=handle_copy,
                    runtime_id=extract_runtime_id(handle_copy) or ldata.get("runtime_id"),
                    create_rpc_in_flight=False,
                    create_rpc_finished_at=started_at,
                    last_seen=started_at,
                    from_active_missing_handle=False,
                    was_active_missing_handle=False,
                    ever_active=True,
                )
                if isinstance(handle_copy.get("runtime_name"), str):
                    ldata["runtime_name"] = handle_copy["runtime_name"]
                if handle_copy.get("tunnel_endpoint"):
                    ldata["tunnel_endpoint"] = handle_copy["tunnel_endpoint"]
                lease.state = "active"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                if instance is not None:
                    next_idata = copy.deepcopy(idata)
                    next_idata.update(runtime_created=True, last_seen_at=started_at)
                    if ldata.get("runtime_id"):
                        next_idata["runtime_id"] = ldata["runtime_id"]
                    if ldata.get("tunnel_endpoint"):
                        next_idata["tunnel_endpoint"] = ldata["tunnel_endpoint"]
                    instance.data = next_idata
                    instance.version += 1
                    instance.updated_at = utcnow()
                job_patch = {
                    "request_id": ldata.get("request_id"),
                    "runtime_id": ldata.get("runtime_id"),
                    "deadline_epoch": deadline_epoch or ldata.get("deadline_epoch") or jdata.get("deadline_epoch"),
                    "started_epoch": started_at,
                    "started_at": started_at,
                }
                updated_job = self._apply_job_transition(
                    session, job, owner_id, "preparing", "GPU runtime allocated", job_patch, started_at
                )
                return "ok", updated_job, self._row(lease)

    def recover_late_create_handle(
        self,
        job_id: str | None,
        owner_id: str,
        lease_id: str,
        *,
        handle: Mapping[str, object],
        instance_id: str | None = None,
        rpc_token: str | None = None,
        expected_attempt: int | None = None,
        now: float,
    ) -> tuple[str, dict | None, dict | None]:
        """Recover a late create handle into cleanup_pending with instance quota preserved even after deletion."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = (
                    session.scalar(
                        select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                    )
                    if job_id
                    else None
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id:
                    return "missing", (self._row(job) if job else None), None
                ldata = lease.data or {}
                if (ldata.get("owner_job_id") not in {None, job_id}
                    or ldata.get("current_job_id") not in {None, job_id}):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, instance_id, now)
                    return "stale", (self._row(job) if job else None), self._row(lease)

                jdata = (job.data or {}) if job is not None else {}
                if not self._create_attempt_matches(ldata, rpc_token, expected_attempt, instance_id):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, instance_id, now)
                    return "stale", (self._row(job) if job else None), self._row(lease)
                if job is not None and (job.data or {}).get("lease_id") == lease.id and (
                    (job.instance_id or (job.data or {}).get("instance_id")) != instance_id
                ):
                    self._isolate_late_handle(session, lease, owner_id, handle, rpc_token,
                                              expected_attempt, instance_id, now)
                    return "stale", self._row(job), self._row(lease)
                lease_inst_id = ldata.get("create_rpc_instance_id") or ldata.get("instance_id") or lease.instance_id
                if instance_id and lease_inst_id in {None, instance_id}:
                    eff_inst_id = instance_id
                else:
                    eff_inst_id = lease_inst_id
                    if (
                        not eff_inst_id
                        and job is not None
                        and (jdata.get("lease_id") == lease.id or (jdata.get("lease_id") is None and ldata.get("owner_job_id") == job.id))
                    ):
                        eff_inst_id = jdata.get("instance_id") or job.instance_id
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if eff_inst_id
                    else None
                )
                self._recover_late_handle_in_session(session, job, lease, instance, owner_id, handle, now=now)
                return "ok", (self._row(job) if job else None), self._row(lease)

    def recover_create_present_cas(
        self,
        job_id: str,
        owner_id: str,
        lease_id: str,
        *,
        handle: Mapping[str, object],
        expected_lease_version: int | None = None,
        expected_create_attempts: int | None = None,
        expected_runtime_id: str | None = None,
        expected_tunnel_endpoint: str | None = None,
        expected_operation_name: str | None = None,
        operation_name: str | None = None,
        deadline_epoch: float | None = None,
        check_snapshot: bool = True,
        connection_metadata: Mapping[str, object] | None = None,
        expected_rpc_token: str | None = None,
        expected_rpc_instance_id: str | None = None,
        now: float,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically promote a probed-present runtime from allocating -> preparing only if pre-probe state matches."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if lease is None or lease.owner_id != owner_id or job is None:
                    return "missing", (self._row(job) if job else None), (self._row(lease) if lease and lease.owner_id == owner_id else None)
                jdata = copy.deepcopy(job.data or {})
                ldata = copy.deepcopy(lease.data or {})
                if job.state != "allocating" or jdata.get("cancel_requested") or jdata.get("requested_outcome") == "cancelled" or self._has_stop_or_delete_flag(jdata):
                    return "stale", self._row(job), self._row(lease)
                j_lease = jdata.get("lease_id")
                if j_lease is not None and j_lease != lease.id:
                    return "stale", self._row(job), self._row(lease)
                if j_lease is None and ldata.get("owner_job_id") != job_id:
                    return "stale", self._row(job), self._row(lease)
                if ldata.get("owner_job_id") not in {None, job_id} or ldata.get("current_job_id") not in {None, job_id}:
                    return "stale", self._row(job), self._row(lease)
                if lease.state in {"deleted", "deleting", "cleanup_pending", "releasing", "quarantined", "idle"} or self._has_stop_or_delete_flag(ldata):
                    return "stale", self._row(job), self._row(lease)
                if self._create_rpc_live_in_flight(ldata, now) and (
                    expected_rpc_token is None or ldata.get("create_rpc_token") != expected_rpc_token
                    or ldata.get("create_rpc_instance_id") != expected_rpc_instance_id
                ):
                    return "stale", self._row(job), self._row(lease)
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", self._row(job), self._row(lease)
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(expected_create_attempts):
                    return "stale", self._row(job), self._row(lease)
                if check_snapshot:
                    cur_op = ldata.get("operation_name") or None
                    exp_op = expected_operation_name or None
                    if cur_op != exp_op:
                        return "stale", self._row(job), self._row(lease)
                    cur_rt = extract_runtime_id(lease)
                    if cur_rt != (expected_runtime_id or None):
                        return "stale", self._row(job), self._row(lease)
                    cur_ep = extract_tunnel_endpoint(lease)
                    norm_exp_ep = normalize_tunnel_endpoint(expected_tunnel_endpoint)
                    if cur_ep != norm_exp_ep:
                        return "stale", self._row(job), self._row(lease)
                inst_id = ldata.get("instance_id") or lease.instance_id or jdata.get("instance_id") or job.instance_id
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if inst_id
                    else None
                )
                idata = copy.deepcopy(instance.data or {}) if instance is not None else {}
                if instance is not None and (
                    instance.state in {"deleted", "deleting", "releasing", "cleanup_pending", "quarantined", "unknown", "retained_idle"}
                    or self._has_stop_or_delete_flag(idata)
                    or idata.get("current_job_id") not in {None, job_id}
                    or idata.get("lease_id") not in {None, lease.id}
                ):
                    return "stale", self._row(job), self._row(lease)
                conflict, _ = has_conflicting_active_runtime(
                    session,
                    instance_id=inst_id,
                    handle=handle,
                    lease=lease,
                    instance=instance,
                    job=job,
                    exclude_lease_id=lease.id,
                    exclude_instance_id=inst_id,
                )
                if conflict:
                    return "conflict", self._row(job), self._row(lease)
                handle_copy = copy.deepcopy(dict(handle))
                if isinstance(handle_copy, dict) and not handle_copy.get("job_id"):
                    handle_copy["job_id"] = job.id
                if isinstance(handle_copy, dict) and not handle_copy.get("owner_job_id"):
                    handle_copy["owner_job_id"] = job.id
                ldata.update(
                    state="active",
                    handle=handle_copy,
                    runtime_id=extract_runtime_id(handle_copy) or ldata.get("runtime_id"),
                    create_rpc_in_flight=False,
                    create_rpc_finished_at=now,
                    last_seen=now,
                    from_active_missing_handle=False,
                    was_active_missing_handle=False,
                    ever_active=True,
                )
                if isinstance(handle_copy.get("runtime_name"), str) and handle_copy["runtime_name"].strip():
                    ldata["runtime_name"] = handle_copy["runtime_name"].strip()
                if handle_copy.get("tunnel_endpoint"):
                    ldata["tunnel_endpoint"] = handle_copy["tunnel_endpoint"]
                if operation_name and not ldata.get("operation_name"):
                    ldata["operation_name"] = operation_name
                if isinstance(connection_metadata, Mapping):
                    for key in ("proxy_url", "encrypted_proxy_token", "proxy_token_expires_at"):
                        if key in connection_metadata:
                            ldata[key] = copy.deepcopy(connection_metadata[key])
                lease.state = "active"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                if instance is not None:
                    idata.update(runtime_created=True, last_seen_at=now)
                    if ldata.get("runtime_id"):
                        idata["runtime_id"] = ldata["runtime_id"]
                    if ldata.get("tunnel_endpoint"):
                        idata["tunnel_endpoint"] = ldata["tunnel_endpoint"]
                    instance.data = idata
                    instance.version += 1
                    instance.updated_at = utcnow()
                job_patch: dict[str, object] = {
                    "lease_id": lease.id,
                    "runtime_id": ldata.get("runtime_id"),
                    "started_epoch": jdata.get("started_epoch") or now,
                    "started_at": jdata.get("started_at") or now,
                }
                eff_deadline = deadline_epoch or ldata.get("deadline_epoch") or jdata.get("deadline_epoch")
                if eff_deadline is not None:
                    job_patch["deadline_epoch"] = eff_deadline
                updated_job = self._apply_job_transition(
                    session, job, owner_id, "preparing", "GPU runtime recovered", job_patch, now
                )
                return "ok", updated_job, self._row(lease)

    def acquire_failed_create_cleanup_cas(
        self,
        job_id: str,
        owner_id: str,
        lease_id: str,
        *,
        handle: Mapping[str, object],
        expected_lease_version: int | None,
        expected_create_attempts: int | None,
        expected_runtime_id: str | None = None,
        expected_tunnel_endpoint: str | None = None,
        expected_operation_name: str | None = None,
        from_active: bool = False,
        last_error: str | None = None,
        now: float,
    ) -> tuple[str, dict | None, str | None]:
        """Claim cleanup_pending ownership via CAS before remote provider.delete on a failed operation."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                if not lease or lease.owner_id != owner_id:
                    return "missing", None, None
                ldata = copy.deepcopy(lease.data or {})
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", self._row(lease), None
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(
                    expected_create_attempts
                ):
                    return "stale", self._row(lease), None
                if lease.state in {"deleted", "idle", "active", "busy"} or self._create_rpc_live_in_flight(ldata, now):
                    return "stale", self._row(lease), None
                if ldata.get("cleanup_claim_token") and self._claim_is_live(ldata, now):
                    return "stale", self._row(lease), None
                cur_rt = extract_runtime_id(lease)
                if expected_runtime_id and cur_rt and cur_rt != expected_runtime_id:
                    return "stale", self._row(lease), None
                cur_ep = extract_tunnel_endpoint(lease)
                norm_exp_ep = normalize_tunnel_endpoint(expected_tunnel_endpoint)
                if norm_exp_ep and cur_ep and cur_ep != norm_exp_ep:
                    return "stale", self._row(lease), None
                cur_op = ldata.get("operation_name")
                if expected_operation_name and cur_op and cur_op != expected_operation_name:
                    return "stale", self._row(lease), None
                inst_id = (job.data.get("instance_id") if job else None) or ldata.get("instance_id") or lease.instance_id
                instance = (
                    session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if inst_id
                    else None
                )
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=inst_id,
                        handle=handle,
                        lease=lease,
                        instance=instance,
                        exclude_lease_id=lease.id,
                        exclude_instance_id=inst_id,
                    )[0]
                ):
                    return "conflict", self._row(lease), None
                claim_token = str(uuid.uuid4())
                active_hist = bool(from_active or self._lease_has_active_origin(ldata, lease.state))
                ldata.update(
                    handle=copy.deepcopy(dict(handle)),
                    state="cleanup_pending",
                    remote_deleted=False,
                    from_active_missing_handle=active_hist,
                    was_active_missing_handle=active_hist,
                    cleanup_claim_token=claim_token,
                    cleanup_claimed_at=now,
                    last_seen=now,
                )
                if last_error:
                    ldata["last_error"] = last_error
                lease.state = "cleanup_pending"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                if instance is not None and instance.state == "deleted":
                    idata = copy.deepcopy(instance.data or {})
                    idata.update(
                        state="cleanup_pending",
                        cleanup_claim_token=claim_token,
                        cleanup_claimed_at=now,
                        runtime_deleted=False,
                        remote_deleted=False,
                        deleted=False,
                        retention_reserved=0.0,
                        budget_reserved=self._derive_instance_budget_reserved(
                            idata,
                            job_data=(job.data if job is not None else None),
                            lease_data=ldata,
                        ),
                        last_seen_at=now,
                    )
                    instance.state = "cleanup_pending"
                    instance.data = idata
                    instance.version += 1
                    instance.updated_at = utcnow()
                session.flush()
                return "ok", self._row(lease), claim_token

    @staticmethod
    def _resolve_lease_provider_in_session(
        session,
        ldata: Mapping[str, object],
        job: GpuJobRow | None = None,
    ) -> str | None:
        """Resolve provider from lease data or associated job/instance so Colab handles cannot bypass strict checks."""
        if isinstance(ldata.get("provider"), str) and ldata["provider"]:
            return ldata["provider"]
        if job is not None:
            j_provider = (job.data or {}).get("provider")
            if isinstance(j_provider, str) and j_provider:
                return j_provider
        owner_job_id = ldata.get("owner_job_id") or ldata.get("current_job_id")
        if isinstance(owner_job_id, str) and owner_job_id:
            owner_job = session.get(GpuJobRow, owner_job_id)
            if owner_job is not None:
                j_provider = (owner_job.data or {}).get("provider")
                if isinstance(j_provider, str) and j_provider:
                    return j_provider
        inst_id = ldata.get("instance_id")
        if isinstance(inst_id, str) and inst_id:
            inst = session.get(GpuInstanceRow, inst_id)
            if inst is not None:
                i_provider = (inst.data or {}).get("provider")
                if isinstance(i_provider, str) and i_provider:
                    return i_provider
        return None

    def quarantine_lease_cas(
        self,
        owner_id: str,
        lease_id: str,
        *,
        job_id: str | None = None,
        instance_id: str | None = None,
        expected_lease_version: int | None = None,
        expected_create_attempts: int | None = None,
        from_active: bool = False,
        reason: str = "missing_identity",
        error_code: str = "gpu_identity_unverifiable",
        message: str = "GPU runtime quarantined",
        now: float,
    ) -> tuple[str, dict | None, dict | None]:
        """Atomically quarantine a lease (and job/instance) only if pre-probe CAS matches."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None, None
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id).with_for_update())
                job = (
                    session.scalar(
                        select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                    )
                    if job_id
                    else None
                )
                if lease is None or lease.owner_id != owner_id:
                    return "missing", (self._row(job) if job else None), None
                ldata = copy.deepcopy(lease.data or {})
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", (self._row(job) if job else None), self._row(lease)
                if expected_create_attempts is not None and int(ldata.get("create_attempts") or 0) != int(
                    expected_create_attempts
                ):
                    return "stale", (self._row(job) if job else None), self._row(lease)
                if lease.state in {"deleted", "idle"} or (
                    lease.state in {"active", "busy"}
                    and is_valid_runtime_handle(
                        ldata.get("handle"),
                        owner_id=owner_id,
                        provider=self._resolve_lease_provider_in_session(session, ldata, job),
                    )
                ) or self._create_rpc_live_in_flight(ldata, now):
                    return "stale", (self._row(job) if job else None), self._row(lease)
                eff_from_active = bool(from_active or self._lease_has_active_origin(ldata, lease.state))
                ldata.update(
                    state="quarantined",
                    current_job_id=None,
                    quarantine_reason=reason,
                    quarantined_at=now,
                    cleanup_claim_token=None,
                    cleanup_claimed_at=None,
                    last_seen=now,
                )
                if eff_from_active:
                    ldata["from_active_missing_handle"] = True
                    ldata["was_active_missing_handle"] = True
                    ldata["ever_active"] = True
                lease.state = "quarantined"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                job_bound_to_lease = job is not None and (job.data or {}).get("lease_id") == lease.id
                eff_inst_id = (
                    instance_id
                    or ((job.data or {}).get("instance_id") if job_bound_to_lease else None)
                    or lease.instance_id
                    or ldata.get("instance_id")
                )
                if eff_inst_id:
                    inst = session.scalar(
                        select(GpuInstanceRow)
                        .where(GpuInstanceRow.id == eff_inst_id, GpuInstanceRow.owner_id == owner_id)
                        .with_for_update()
                    )
                    if (
                        inst is not None
                        and inst.state not in {"deleted", "quarantined"}
                        and (inst.data or {}).get("lease_id") in {None, lease.id}
                    ):
                        idata = copy.deepcopy(inst.data or {})
                        idata.update(
                            state="quarantined",
                            current_job_id=None,
                            quarantine_reason=reason,
                            quarantined_at=now,
                            cleanup_claim_token=None,
                            cleanup_claimed_at=None,
                            last_seen_at=now,
                        )
                        inst.state = "quarantined"
                        inst.current_job_id = None
                        inst.data = idata
                        inst.version += 1
                        inst.updated_at = utcnow()
                        session.add(
                            GpuEventRow(
                                id=str(uuid.uuid4()),
                                owner_id=owner_id,
                                project_id=inst.project_id,
                                state="quarantined",
                                version=0,
                                data={
                                    "instance_id": inst.id,
                                    "job_id": job.id if job_bound_to_lease and job is not None else ldata.get("owner_job_id"),
                                    "state": "quarantined",
                                    "message": reason,
                                },
                                created_at=utcnow(),
                                updated_at=utcnow(),
                            )
                        )
                updated_job = self._row(job) if job is not None else None
                if job_bound_to_lease and job is not None:
                    if job.state not in {"completed", "failed", "cancelled"}:
                        updated_job = self._apply_job_transition(
                            session,
                            job,
                            owner_id,
                            "failed",
                            message,
                            {
                                "error_code": error_code,
                                "error_message": message,
                                "finished_at": now,
                            },
                            now,
                        )
                    else:
                        updated_job = self._row(job)
                session.flush()
                return "ok", updated_job, self._row(lease)

    def retire_pure_orphan_placeholder_cas(
        self,
        owner_id: str,
        lease_id: str,
        *,
        instance_id: str | None = None,
        cleanup_job_id: str | None = None,
        expected_lease_version: int | None = None,
        expected_instance_version: int | None = None,
        allow_deleted_placeholder: bool = False,
        now: float,
    ) -> tuple[str, str]:
        """Verify pure placeholder AND delete lease + instance inside ONE atomic admission+DB tx."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", "locked"
            with self.store.db_lock(), self.store.Session.begin() as session:
                lease = session.scalar(
                    select(GpuLeaseRow).where(GpuLeaseRow.id == lease_id, GpuLeaseRow.owner_id == owner_id).with_for_update()
                )
                if lease is None:
                    return "missing", "missing"
                if expected_lease_version is not None and int(lease.version) != int(expected_lease_version):
                    return "stale", "version_mismatch"
                if cleanup_job_id is not None:
                    job = session.scalar(select(GpuJobRow).where(
                        GpuJobRow.id == cleanup_job_id, GpuJobRow.owner_id == owner_id
                    ).with_for_update())
                    if (
                        job is None or job.state not in {"cleaning_up", "cleanup_pending"}
                        or (job.data or {}).get("lease_id") != lease.id
                        or (lease.data or {}).get("owner_job_id") != job.id
                        or (lease.data or {}).get("current_job_id") not in {None, job.id}
                        or (job.instance_id or (job.data or {}).get("instance_id")) != instance_id
                    ):
                        return "stale", "job_binding_changed"
                ok, reason, inst = self._can_locally_delete_orphan_placeholder_in_session(
                    session, owner_id, lease, instance_id, allow_deleting_state=False,
                    cleanup_job_id=cleanup_job_id, allow_deleted_placeholder=allow_deleted_placeholder,
                )
                if not ok:
                    return "not_pure", reason
                ldata = copy.deepcopy(lease.data or {})
                eff_inst_id = instance_id or lease.instance_id or ldata.get("instance_id")
                if inst is not None and expected_instance_version is not None and int(inst.version) != int(expected_instance_version):
                    return "stale", "instance_version_mismatch"
                if (
                    has_conflicting_active_runtime(
                        session,
                        instance_id=eff_inst_id,
                        lease=lease,
                        instance=inst,
                        exclude_lease_id=lease_id,
                        exclude_instance_id=eff_inst_id,
                    )[0]
                ):
                    return "conflict", "runtime_conflict"
                if inst is not None and inst.state != "deleted":
                    idata = copy.deepcopy(inst.data or {})
                    idata.update(
                        current_job_id=None,
                        runtime_deleted=True,
                        remote_deleted=True,
                        retired_pure_placeholder=True,
                        retention_reserved=0.0,
                        deleted_at=now,
                        deletion_confirmed_at=now,
                        last_seen_at=now,
                    )
                    inst.state = "deleted"
                    inst.current_job_id = None
                    inst.data = idata
                    inst.version += 1
                    inst.updated_at = utcnow()
                    session.add(
                        GpuEventRow(
                            id=str(uuid.uuid4()),
                            owner_id=owner_id,
                            project_id=inst.project_id,
                            state="deleted",
                            version=0,
                            data={
                                "instance_id": inst.id,
                                "state": "deleted",
                                "message": "pure orphan placeholder deleted atomically",
                            },
                            created_at=utcnow(),
                            updated_at=utcnow(),
                        )
                    )
                ldata.update(
                    state="deleted",
                    current_job_id=None,
                    cleanup_claim_token=None,
                    remote_deleted=True,
                    remote_deleted_at=now,
                    remote_delete_source="orphan_placeholder_local_retire",
                    last_seen=now,
                )
                lease.state = "deleted"
                lease.data = ldata
                lease.version += 1
                lease.updated_at = utcnow()
                session.flush()
                return "retired", "ok"

    def record_dispatch_poll_backoff(
        self,
        job_id: str,
        owner_id: str,
        *,
        execution_id: str,
        next_epoch: float,
        attempts: int,
        now: float,
        deadline_at: float | None = None,
    ) -> tuple[str, dict | None]:
        """Persist backoff metadata on a loading job stuck in `dispatching` so other jobs are not starved."""
        with self.store.admission_lock(timeout=1.0) as acquired:
            if not acquired:
                return "locked", None
            with self.store.db_lock(), self.store.Session.begin() as session:
                job = session.scalar(
                    select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
                )
                if job is None:
                    return "missing", None
                jdata = copy.deepcopy(job.data or {})
                intent = jdata.get("start_intent")
                if (
                    job.state != "loading"
                    or jdata.get("execution_id") != execution_id
                    or not isinstance(intent, Mapping)
                    or intent.get("state") != "dispatching"
                ):
                    return "stale", self._row(job)
                patch: dict[str, object] = {
                    "dispatch_next_epoch": float(next_epoch),
                    "dispatch_poll_attempts": int(attempts),
                    "dispatch_last_poll_at": float(now),
                }
                if deadline_at is not None:
                    patch["dispatch_recovery_deadline_at"] = float(deadline_at)
                jdata.update(patch)
                job.data = jdata
                job.version += 1
                job.updated_at = utcnow()
                session.flush()
                return "ok", self._row(job)

    def find_runtime_conflict_lease(
        self,
        *,
        instance_id: str | None = None,
        runtime_id: str | None = None,
        runtime_name: str | None = None,
        tunnel_endpoint: str | None = None,
        handle: object = None,
        lease: object = None,
        instance: object = None,
        job: object = None,
        exclude_lease_id: str | None = None,
        exclude_instance_id: str | None = None,
    ) -> tuple[bool, dict | None]:
        """Authoritatively check DB for any other non-deleted lease or active instance sharing runtime/endpoint."""
        with self.store.db_lock(), self.store.Session.begin() as session:
            conflict_lease = find_conflicting_non_deleted_lease(
                session,
                instance_id=instance_id,
                runtime_id=runtime_id,
                runtime_name=runtime_name,
                tunnel_endpoint=tunnel_endpoint,
                handle=handle,
                lease=lease,
                instance=instance,
                job=job,
                exclude_lease_id=exclude_lease_id,
            )
            if conflict_lease is not None:
                return True, self._row(conflict_lease)
            conflict, _ = has_conflicting_active_runtime(
                session,
                instance_id=instance_id,
                runtime_id=runtime_id,
                runtime_name=runtime_name,
                tunnel_endpoint=tunnel_endpoint,
                handle=handle,
                lease=lease,
                instance=instance,
                job=job,
                exclude_lease_id=exclude_lease_id,
                exclude_instance_id=exclude_instance_id,
            )
            return conflict, None

    @staticmethod
    def _owned_execution_id(data: Mapping[str, object] | None) -> str | None:
        if not isinstance(data, Mapping):
            return None
        active = data.get("active_execution")
        if isinstance(active, Mapping):
            value = active.get("active_execution_id") or active.get("execution_id")
            return value if isinstance(value, str) and value else None
        value = data.get("active_execution_id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _persist_runtime_warm_proof(
        lease: GpuLeaseRow | None,
        instance: GpuInstanceRow | None,
        job_data: Mapping[str, object],
        verified_warm: Mapping[str, object] | None,
    ) -> None:
        if lease is not None and lease.state != "deleted":
            lease_data = copy.deepcopy(lease.data or {})
            lease_data["active_execution"] = None
            if verified_warm is not None:
                lease_data["warm_proof"] = copy.deepcopy(dict(verified_warm))
                if job_data.get("compatibility_signature"):
                    lease_data["restored_signature"] = job_data["compatibility_signature"]
            else:
                lease_data["warm_proof"] = None
                lease_data["restored_signature"] = None
            lease.data = lease_data
            lease.version += 1
            lease.updated_at = utcnow()
        if instance is not None and instance.state != "deleted":
            inst_data = copy.deepcopy(instance.data or {})
            inst_data["active_execution"] = None
            inst_data["warm_proof"] = copy.deepcopy(dict(verified_warm)) if verified_warm is not None else None
            instance.data = inst_data
            instance.version += 1
            instance.updated_at = utcnow()

    def clear_owned_execution_and_warm_proof(
        self,
        owner_id: str,
        job_id: str,
        execution_id: str | None,
    ) -> None:
        """Dual-clear warm_proof/active_execution only if execution_id still owns the runtime."""
        if not execution_id:
            return
        with self.store.db_lock(), self.store.Session.begin() as session:
            job = session.scalar(
                select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update()
            )
            if job is None or job.data.get("execution_id") != execution_id:
                return
            lease_id = job.data.get("lease_id")
            lease = session.get(GpuLeaseRow, lease_id) if lease_id else None
            if lease is not None and lease.owner_id != owner_id:
                lease = None
            instance_id = (
                job.data.get("instance_id")
                or (lease.data.get("instance_id") if lease is not None else None)
                or (lease.instance_id if lease is not None else None)
            )
            instance = session.get(GpuInstanceRow, instance_id) if instance_id else None
            if instance is not None and instance.owner_id != owner_id:
                instance = None

            execution = session.get(GpuExecutionRow, execution_id)
            active = (lease.data or {}).get("active_execution") if lease is not None else None
            exec_ok, _ = self._execution_matches_bindings_and_generation(
                owner_id=owner_id, job=job, execution=execution, lease=lease, instance=instance,
                active=active if isinstance(active, Mapping) else None,
            )
            if not exec_ok:
                return

            target_lease = lease if (lease is not None and self._owned_execution_id(lease.data) == execution_id) else None
            target_instance = (
                instance
                if (instance is not None and self._owned_execution_id(instance.data) == execution_id)
                else None
            )
            if target_lease is not None or target_instance is not None:
                self._persist_runtime_warm_proof(target_lease, target_instance, job.data or {}, None)

    def transition_job(self, job_id, owner_id, state, message, patch, now, expected_state=None, expected_execution_id=...):
        """Commit job, execution, batch aggregate and audit event together."""
        terminal = {'completed', 'failed', 'cancelled'}
        progress = patch.get('progress')
        if progress is not None and (isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1):
            raise ValueError('progress必须在0到1之间')
        with self.store.db_lock(), self.store.Session.begin() as session:
            job = session.scalar(select(GpuJobRow).where(GpuJobRow.id == job_id, GpuJobRow.owner_id == owner_id).with_for_update())
            if not job:
                raise KeyError(job_id)
            if expected_state is not None and job.state != expected_state:
                return self._row(job)
            if expected_execution_id is not ... and job.data.get('execution_id') != expected_execution_id:
                return self._row(job)
            if job.state in terminal:
                return self._row(job)
            return self._apply_job_transition(session, job, owner_id, state, message, patch, now)

    def event(self, job_id, owner_id, state, message, progress=None):
        job = self.get("jobs", job_id, owner_id)
        if job is None:
            raise KeyError(job_id)
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1
        ):
            raise ValueError("progress必须在0到1之间")
        data = {"job_id": job_id, "message": message, "state": state}
        if progress is not None:
            data["progress"] = progress
        return self.create("events", owner_id, data, project_id=job.get("project_id"))
