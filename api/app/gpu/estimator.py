"""Calibrated pure-sampling reuse estimator and lifetime window guard.

Replaces the coarse 65%-fixed `rates.length_factor` heuristic for runtime reuse
decisions with per-workflow/version/GPU/mode calibrated sampling baselines and
strictly admitted historical P90 sampling timings.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

COMFY_START_S = 30.0
TAIL_S = 60.0
SAFETY_MARGIN_S = 60.0
COLD_SAMPLING_SAFETY_FACTOR = 1.25
HIST_P90_SAFETY_FACTOR = 1.20
MIN_HIST_SAMPLES = 3
RECENT_HIST_WINDOW = 20

VALID_GUARD_MODES = frozenset({"off", "observe", "enforce"})

GPU_TIME_FACTOR: dict[str, float] = {
    "A100": 0.55,
    "L4": 1.0,
    "T4": 2.2,
}

_GPU_TIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("A100", re.compile(r"(?:^|[^A-Z0-9])A100(?:[^A-Z0-9]|$)", re.IGNORECASE)),
    ("L4", re.compile(r"(?:^|[^A-Z0-9])L4(?:[^A-Z0-9]|$)", re.IGNORECASE)),
    ("T4", re.compile(r"(?:^|[^A-Z0-9])T4(?:[^A-Z0-9]|$)", re.IGNORECASE)),
)


class UncalibratedWorkloadError(ValueError):
    """Raised when a workflow, version, mode, GPU tier, or workload parameter is uncalibrated."""


@dataclass(frozen=True)
class BaselineSpec:
    workflow_key: str
    workflow_version: str
    ref_gpu: str
    modality: str
    def_w: int
    def_h: int
    mode_base_s: Mapping[str, float]
    def_len: int | None = None
    def_steps: int | None = None
    inter_seg_s: float = 0.0
    len_exponent: float = 1.7
    pixel_exponent: float = 1.35
    is_relay: bool = False
    is_dual_sampler: bool = False
    def_krea_steps: int = 8
    def_moss_steps: int = 25
    def_moss_denoise: float = 0.36


@dataclass(frozen=True)
class ReuseAssessment:
    can_reuse: bool
    need_seconds: float | None
    remaining_seconds: float | None
    warm_verified: bool
    evidence_source: str
    sampling_source: str
    reason: str


COLD_SAMPLING_BASELINES: dict[tuple[str, str], BaselineSpec] = {
    ("minimax-h3-ref2v", "1.4.0"): BaselineSpec(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        ref_gpu="L4",
        modality="video",
        def_w=544,
        def_h=960,
        def_len=124,
        len_exponent=1.7,
        pixel_exponent=1.35,
        mode_base_s={
            "fast": 150.0,
            "fast8": 240.0,
            "refine": 270.0,
            "refine8": 360.0,
            "hq": 540.0,
        },
    ),
    ("minimax-h3-i2v", "1.1.0"): BaselineSpec(
        workflow_key="minimax-h3-i2v",
        workflow_version="1.1.0",
        ref_gpu="L4",
        modality="video",
        def_w=480,
        def_h=864,
        def_len=124,
        def_steps=8,
        len_exponent=1.7,
        pixel_exponent=1.35,
        mode_base_s={
            "default": 210.0,
            "fast8": 210.0,
        },
    ),
    ("minimax-h3-relay", "1.0.0"): BaselineSpec(
        workflow_key="minimax-h3-relay",
        workflow_version="1.0.0",
        ref_gpu="L4",
        modality="video",
        def_w=544,
        def_h=960,
        def_len=124,
        inter_seg_s=15.0,
        len_exponent=1.7,
        pixel_exponent=1.35,
        is_relay=True,
        mode_base_s={
            "fast": 150.0,
            "fast8": 240.0,
            "hq": 540.0,
        },
    ),
    ("krea2-moss-dual-t2i", "1.0.0"): BaselineSpec(
        workflow_key="krea2-moss-dual-t2i",
        workflow_version="1.0.0",
        ref_gpu="T4",
        modality="image",
        def_w=832,
        def_h=1248,
        def_steps=33,
        pixel_exponent=1.25,
        is_dual_sampler=True,
        mode_base_s={"default": 45.0},
    ),
    ("krea2-moss-dual-t2i", "1.1.1"): BaselineSpec(
        workflow_key="krea2-moss-dual-t2i",
        workflow_version="1.1.1",
        ref_gpu="T4",
        modality="image",
        def_w=832,
        def_h=1248,
        def_steps=33,
        pixel_exponent=1.25,
        is_dual_sampler=True,
        mode_base_s={"default": 45.0},
    ),
    ("moss-image-sft-t2i", "1.0.0"): BaselineSpec(
        workflow_key="moss-image-sft-t2i",
        workflow_version="1.0.0",
        ref_gpu="T4",
        modality="image",
        def_w=832,
        def_h=1248,
        def_steps=35,
        pixel_exponent=1.25,
        mode_base_s={"default": 40.0},
    ),
}


def min_deadlines(*vals: float | int | None) -> float | None:
    """Return the minimum finite deadline, ignoring None (never empty min() or 0 coercion)."""
    finite: list[float] = []
    for value in vals:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"deadline must be numeric or None, got {value!r}")
        num = float(value)
        if not math.isfinite(num):
            raise ValueError(f"deadline must be finite, got {value!r}")
        finite.append(num)
    return min(finite) if finite else None


def env_weights_restore_seconds(warm_verified: bool, modality: str | None) -> float:
    """Penalty seconds for restoring environment and model weights when not verified warm."""
    if warm_verified:
        return 0.0
    if str(modality or "").strip().lower() == "image":
        return 180.0
    return 330.0


def normalize_gpu_tier(value: Any) -> str | None:
    """Normalize an accelerator label or NVIDIA device name to A100/L4/T4, else None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    upper = text.upper()
    if upper in GPU_TIME_FACTOR:
        return upper
    for tier, pattern in _GPU_TIER_PATTERNS:
        if pattern.search(text):
            return tier
    return None


def _require_pos_num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UncalibratedWorkloadError(f"workload parameter {name!r} must be a positive number")
    num = float(value)
    if not math.isfinite(num) or num <= 0.0:
        raise UncalibratedWorkloadError(f"workload parameter {name!r} must be positive and finite")
    return num


def _resolve_baseline(
    workflow_key: str | None,
    workflow_version: str | int | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
) -> tuple[BaselineSpec, str]:
    if not isinstance(workflow_key, str) or not workflow_key.strip():
        raise UncalibratedWorkloadError("workflow_key is required")
    if workflow_version is None or isinstance(workflow_version, bool):
        raise UncalibratedWorkloadError("workflow_version is required")
    ver_str = str(workflow_version).strip()
    if not ver_str:
        raise UncalibratedWorkloadError("workflow_version is required")
    spec = COLD_SAMPLING_BASELINES.get((workflow_key.strip(), ver_str))
    if spec is None:
        raise UncalibratedWorkloadError(f"uncalibrated workflow {(workflow_key, ver_str)!r}")
    if not isinstance(params, Mapping):
        raise UncalibratedWorkloadError("workload params must be a mapping")

    raw_mode = mode if mode is not None else params.get("mode")
    norm_mode = "default" if raw_mode is None or (isinstance(raw_mode, str) and not raw_mode.strip()) else str(raw_mode).strip()
    if norm_mode not in spec.mode_base_s:
        raise UncalibratedWorkloadError(
            f"uncalibrated mode {raw_mode!r} for {(spec.workflow_key, spec.workflow_version)!r}"
        )
    return spec, norm_mode


def workload_ratio(
    workflow_key: str | None,
    workflow_version: str | int | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
    *,
    require_all_params: bool = False,
) -> float:
    """Compute structural workload ratio relative to the baseline reference workload."""
    spec, norm_mode = _resolve_baseline(workflow_key, workflow_version, mode, params)
    assert isinstance(params, Mapping)

    width = _require_pos_num(params.get("width"), "width")
    height = _require_pos_num(params.get("height"), "height")
    px_ratio = ((width * height) / float(spec.def_w * spec.def_h)) ** spec.pixel_exponent

    if spec.def_len is not None:
        length = _require_pos_num(params.get("length"), "length")
        len_ratio = (length / float(spec.def_len)) ** spec.len_exponent
    else:
        len_ratio = 1.0

    if spec.is_relay:
        if "segments" not in params or params.get("segments") is None:
            raise UncalibratedWorkloadError("relay workload requires positive 'segments'")
        segments = _require_pos_num(params.get("segments"), "segments")
        mode_base = spec.mode_base_s[norm_mode]
        raw_cost = mode_base * len_ratio * px_ratio * segments + max(0.0, segments - 1.0) * spec.inter_seg_s
        return raw_cost / mode_base

    if spec.is_dual_sampler:
        if require_all_params:
            for key in ("krea_steps", "moss_steps", "moss_denoise"):
                if key not in params or params.get(key) is None:
                    raise UncalibratedWorkloadError(f"dual-sampler history requires {key!r}")
        krea_steps = (
            _require_pos_num(params["krea_steps"], "krea_steps")
            if "krea_steps" in params
            else float(spec.def_krea_steps)
        )
        moss_steps = (
            _require_pos_num(params["moss_steps"], "moss_steps")
            if "moss_steps" in params
            else float(spec.def_moss_steps)
        )
        moss_denoise = (
            _require_pos_num(params["moss_denoise"], "moss_denoise")
            if "moss_denoise" in params
            else float(spec.def_moss_denoise)
        )
        eff_steps = krea_steps + moss_steps * max(1.0, moss_denoise / spec.def_moss_denoise)
        return px_ratio * (eff_steps / float(spec.def_steps or 33))

    if spec.def_steps is not None:
        if require_all_params and ("steps" not in params or params.get("steps") is None):
            raise UncalibratedWorkloadError("history requires explicit 'steps'")
        steps = _require_pos_num(params["steps"], "steps") if "steps" in params else float(spec.def_steps)
        return (steps / float(spec.def_steps)) * len_ratio * px_ratio

    return len_ratio * px_ratio


def _job_gpu_tier(job: Mapping[str, Any]) -> str | None:
    diag = job.get("gpu_diagnostics")
    if isinstance(diag, Mapping):
        devices = diag.get("devices")
        if isinstance(devices, list):
            for dev in devices:
                if isinstance(dev, Mapping):
                    tier = normalize_gpu_tier(dev.get("name"))
                    if tier:
                        return tier
            return None
    for candidate in (
        job.get("gpu_reported"),
        job.get("gpu_tier"),
        (((job.get("snapshot") or {}).get("manifest") or {}).get("compute_spec") or {}).get("accelerator"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return normalize_gpu_tier(candidate)
    return None


def _job_mode(job: Mapping[str, Any]) -> str | None:
    resolved = ((job.get("snapshot") or {}).get("resolved_params")) or {}
    if isinstance(resolved, Mapping) and resolved.get("mode") is not None:
        return str(resolved["mode"]).strip() or None
    manifest = ((job.get("snapshot") or {}).get("manifest")) or {}
    params_spec = manifest.get("parameters") or {}
    if isinstance(params_spec, Mapping):
        for spec in params_spec.values():
            if isinstance(spec, Mapping) and "variants" in spec and spec.get("default") is not None:
                return str(spec["default"]).strip() or None
    return None


def _is_mock_record(
    job: Mapping[str, Any],
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    if job.get("mock") or job.get("provider") == "mock":
        return True
    policy = job.get("execution_policy") or {}
    if isinstance(policy, Mapping) and policy.get("provider") == "mock":
        return True
    manifest = ((job.get("snapshot") or {}).get("manifest")) or {}
    if isinstance(manifest, Mapping) and manifest.get("mock"):
        return True
    job_id = job.get("id")
    if usage_by_job and job_id in usage_by_job and usage_by_job[job_id].get("mock"):
        return True
    return False


def _actual_job_gpu_tier(job: Mapping[str, Any]) -> str | None:
    """Return actual hardware GPU tier from runtime evidence only (never requested tier)."""
    diag = job.get("gpu_diagnostics")
    if isinstance(diag, Mapping):
        devices = diag.get("devices")
        if isinstance(devices, list):
            for dev in devices:
                if isinstance(dev, Mapping):
                    tier = normalize_gpu_tier(dev.get("name"))
                    if tier:
                        return tier
            return None
    reported = job.get("gpu_reported")
    if isinstance(reported, str) and reported.strip():
        return normalize_gpu_tier(reported)
    return None


def is_legacy_mock_job(job: Mapping[str, Any]) -> bool:
    """True only for legacy unit-test mock/bare-handle fixtures without workflow specs."""
    if not isinstance(job, Mapping):
        return False
    policy = job.get("execution_policy") or {}
    provider = job.get("provider") or (policy.get("provider") if isinstance(policy, Mapping) else None)
    if provider != "mock":
        return False
    snapshot = job.get("snapshot") or {}
    manifest = snapshot.get("manifest") if isinstance(snapshot, Mapping) else None
    resolved = snapshot.get("resolved_params") if isinstance(snapshot, Mapping) else None
    workflow_key = job.get("workflow_key") or (manifest.get("workflow_key") if isinstance(manifest, Mapping) else None)
    workflow_version = job.get("workflow_version") or (
        manifest.get("version") or manifest.get("workflow_version") if isinstance(manifest, Mapping) else None
    )
    if (
        not workflow_key
        and workflow_version is None
        and not resolved
        and (manifest is None or (isinstance(manifest, Mapping) and len(manifest) == 0))
    ):
        return True
    if isinstance(manifest, Mapping) and manifest.get("mock") is True:
        return True
    return not workflow_key or workflow_key == "demo-mock"


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = 0.9 * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def qualified_history_samples(
    *,
    workflow_key: str,
    workflow_version: str,
    gpu_tier: str,
    mode: str,
    history_jobs: Iterable[Mapping[str, Any]] = (),
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[float]:
    """Return normalized completed non-mock sampling seconds matching (wf, ver, gpu, mode)."""
    spec, norm_mode = _resolve_baseline(workflow_key, workflow_version, mode, {"width": 1, "height": 1, "length": 1, "segments": 1})
    candidates: list[tuple[float, int, float]] = []
    for idx, job in enumerate(history_jobs):
        if not isinstance(job, Mapping):
            continue
        if job.get("state") != "completed" or job.get("prep"):
            continue
        if job.get("error_code") or job.get("cancel_requested"):
            continue
        if job.get("requested_outcome") not in {None, "completed"}:
            continue
        if job.get("cleanup_final_state") not in {None, "completed"}:
            continue
        if _is_mock_record(job, usage_by_job):
            continue
        exec_id = job.get("execution_id")
        if executions_by_id and exec_id and exec_id in executions_by_id:
            if executions_by_id[exec_id].get("state") != "completed":
                continue

        if str(job.get("workflow_key") or "").strip() != spec.workflow_key:
            continue
        if str(job.get("workflow_version") or "").strip() != spec.workflow_version:
            continue
        if _actual_job_gpu_tier(job) != gpu_tier:
            continue

        resolved = (job.get("snapshot") or {}).get("resolved_params")
        if not isinstance(resolved, Mapping):
            continue
        raw_mode = resolved.get("mode")
        if len(spec.mode_base_s) > 1 or "default" not in spec.mode_base_s:
            if not isinstance(raw_mode, str) or not raw_mode.strip():
                continue
            hist_mode = raw_mode.strip()
        else:
            hist_mode = (
                "default"
                if raw_mode is None or (isinstance(raw_mode, str) and not raw_mode.strip())
                else str(raw_mode).strip()
            )
        if hist_mode != norm_mode:
            continue

        timings = job.get("worker_timings")
        if not isinstance(timings, Mapping):
            continue
        raw_sampling = timings.get("sampling")
        if isinstance(raw_sampling, bool) or not isinstance(raw_sampling, (int, float)):
            continue
        sampling_s = float(raw_sampling)
        if not math.isfinite(sampling_s) or sampling_s <= 0.0:
            continue

        try:
            ratio = workload_ratio(
                spec.workflow_key,
                spec.workflow_version,
                hist_mode,
                resolved,
                require_all_params=True,
            )
        except UncalibratedWorkloadError:
            continue

        ts = 0.0
        for ts_key in ("finished_at", "stopped_at", "started_epoch", "created_epoch"):
            raw_ts = job.get(ts_key)
            if isinstance(raw_ts, (int, float)) and not isinstance(raw_ts, bool) and math.isfinite(float(raw_ts)):
                ts = float(raw_ts)
                break
        candidates.append((ts, idx, sampling_s / ratio))

    candidates.sort(key=lambda item: (item[0], item[1]))
    recent = candidates[-RECENT_HIST_WINDOW:]
    return [item[2] for item in recent]


def _estimate_sampling_with_source(
    workflow_key: str | None,
    workflow_version: str | int | None,
    gpu_tier: str | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
    *,
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[float, str, BaselineSpec]:
    tier = normalize_gpu_tier(gpu_tier)
    if tier is None:
        raise UncalibratedWorkloadError(f"uncalibrated gpu_tier {gpu_tier!r}")
    spec, norm_mode = _resolve_baseline(workflow_key, workflow_version, mode, params)
    req_ratio = workload_ratio(spec.workflow_key, spec.workflow_version, norm_mode, params, require_all_params=False)

    if history_jobs is None and repo is not None and owner_id:
        history_jobs = repo.list("jobs", owner_id)
        if executions_by_id is None:
            executions_by_id = {row["id"]: row for row in repo.list("executions", owner_id)}
        if usage_by_job is None:
            usage_by_job = {row["job_id"]: row for row in repo.list("usage", owner_id) if row.get("job_id")}

    samples = qualified_history_samples(
        workflow_key=spec.workflow_key,
        workflow_version=spec.workflow_version,
        gpu_tier=tier,
        mode=norm_mode,
        history_jobs=history_jobs or (),
        executions_by_id=executions_by_id,
        usage_by_job=usage_by_job,
    )
    if len(samples) >= MIN_HIST_SAMPLES:
        return _p90(samples) * req_ratio * HIST_P90_SAFETY_FACTOR, "history_p90", spec

    gpu_scale = GPU_TIME_FACTOR[tier] / GPU_TIME_FACTOR[spec.ref_gpu]
    cold_s = spec.mode_base_s[norm_mode] * gpu_scale * req_ratio * COLD_SAMPLING_SAFETY_FACTOR
    return cold_s, "cold_baseline", spec


def estimate_sampling_seconds(
    workflow_key: str | None,
    workflow_version: str | int | None,
    gpu_tier: str | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
    *,
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> float:
    """Return calibrated conservative pure sampling seconds, else raise UncalibratedWorkloadError."""
    sampling_s, _, _ = _estimate_sampling_with_source(
        workflow_key,
        workflow_version,
        gpu_tier,
        mode,
        params,
        history_jobs=history_jobs,
        repo=repo,
        owner_id=owner_id,
        executions_by_id=executions_by_id,
        usage_by_job=usage_by_job,
    )
    return sampling_s


def assess_reuse_window(
    *,
    workflow_key: str | None,
    workflow_version: str | int | None,
    gpu_tier: str | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
    modality: str | None,
    warm_verified: bool,
    now: float,
    deadlines: Iterable[float | int | None],
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> ReuseAssessment:
    """Assess whether remaining runtime lifetime safely covers calibrated restart/reuse + sampling."""
    try:
        sampling_s, sampling_source, spec = _estimate_sampling_with_source(
            workflow_key,
            workflow_version,
            gpu_tier,
            mode,
            params,
            history_jobs=history_jobs,
            repo=repo,
            owner_id=owner_id,
            executions_by_id=executions_by_id,
            usage_by_job=usage_by_job,
        )
    except UncalibratedWorkloadError as exc:
        return ReuseAssessment(
            can_reuse=False,
            need_seconds=None,
            remaining_seconds=None,
            warm_verified=bool(warm_verified),
            evidence_source="uncalibrated",
            sampling_source="none",
            reason=str(exc),
        )

    eff_modality = modality or spec.modality
    restore_s = env_weights_restore_seconds(bool(warm_verified), eff_modality)
    need_s = COMFY_START_S + restore_s + sampling_s + TAIL_S + SAFETY_MARGIN_S

    try:
        effective_deadline = min_deadlines(*tuple(deadlines))
    except ValueError as exc:
        return ReuseAssessment(
            can_reuse=False,
            need_seconds=need_s,
            remaining_seconds=None,
            warm_verified=bool(warm_verified),
            evidence_source="warm_proof" if warm_verified else "cold",
            sampling_source=sampling_source,
            reason=str(exc),
        )

    if effective_deadline is None:
        return ReuseAssessment(
            can_reuse=True,
            need_seconds=need_s,
            remaining_seconds=None,
            warm_verified=bool(warm_verified),
            evidence_source="warm_proof" if warm_verified else "cold",
            sampling_source=sampling_source,
            reason="no_finite_deadline",
        )

    remaining_s = float(effective_deadline) - float(now)
    can_reuse = remaining_s >= need_s
    return ReuseAssessment(
        can_reuse=can_reuse,
        need_seconds=need_s,
        remaining_seconds=remaining_s,
        warm_verified=bool(warm_verified),
        evidence_source="warm_proof" if warm_verified else "cold",
        sampling_source=sampling_source,
        reason="ok" if can_reuse else "insufficient_remaining_lifetime",
    )


def evaluate_reuse_window(
    *,
    workflow_key: str | None,
    workflow_version: str | int | None,
    gpu_tier: str | None,
    mode: str | None,
    params: Mapping[str, Any] | None,
    modality: str | None,
    warm_verified: bool,
    now: float,
    deadlines: Iterable[float | int | None],
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[bool, float | None]:
    """Return (can_reuse, need_seconds). Fails closed with (False, None) on uncalibrated inputs."""
    assessment = assess_reuse_window(
        workflow_key=workflow_key,
        workflow_version=workflow_version,
        gpu_tier=gpu_tier,
        mode=mode,
        params=params,
        modality=modality,
        warm_verified=warm_verified,
        now=now,
        deadlines=deadlines,
        history_jobs=history_jobs,
        repo=repo,
        owner_id=owner_id,
        executions_by_id=executions_by_id,
        usage_by_job=usage_by_job,
    )
    return (assessment.can_reuse, assessment.need_seconds)


def _valid_generation(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _select_warm_proof(
    instance_data: Mapping[str, Any] | None,
    lease_data: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    has_instance = isinstance(instance_data, Mapping)
    has_lease = isinstance(lease_data, Mapping)
    if has_instance and instance_data.get("active_execution") is not None:
        return None
    if has_lease and lease_data.get("active_execution") is not None:
        return None
    inst_proof = instance_data.get("warm_proof") if has_instance else None
    lease_proof = lease_data.get("warm_proof") if has_lease else None

    if has_instance and has_lease:
        # When both instance and lease exist, both must hold identical, current-generation proof.
        if not (isinstance(inst_proof, Mapping) and isinstance(lease_proof, Mapping)):
            return None
        if inst_proof.get("warm_saved") is not True or lease_proof.get("warm_saved") is not True:
            return None
        inst_id = instance_data.get("id")
        lease_inst_id = lease_data.get("instance_id")
        if inst_id is not None and lease_inst_id is not None and inst_id != lease_inst_id:
            return None
        lease_id = lease_data.get("id")
        inst_lease_id = instance_data.get("lease_id")
        if lease_id is not None and inst_lease_id is not None and lease_id != inst_lease_id:
            return None
        if (
            inst_proof.get("worker_reported_execution_id") != lease_proof.get("worker_reported_execution_id")
            or inst_proof.get("warm_saved") != lease_proof.get("warm_saved")
            or inst_proof.get("warm_execution_id") != lease_proof.get("warm_execution_id")
            or inst_proof.get("warm_epoch_id") != lease_proof.get("warm_epoch_id")
            or inst_proof.get("warm_fingerprint") != lease_proof.get("warm_fingerprint")
            or inst_proof.get("generation") != lease_proof.get("generation")
            or inst_proof.get("compatibility_signature") != lease_proof.get("compatibility_signature")
            or inst_proof.get("runtime_meta") != lease_proof.get("runtime_meta")
        ):
            return None
        inst_gen = _valid_generation(instance_data.get("generation"))
        lease_gen = _valid_generation(lease_data.get("generation"))
        proof_gen = _valid_generation(inst_proof.get("generation"))
        lease_proof_gen = _valid_generation(lease_proof.get("generation"))
        if (
            inst_gen is None
            or lease_gen is None
            or proof_gen is None
            or lease_proof_gen is None
            or not (inst_gen == lease_gen == proof_gen == lease_proof_gen)
        ):
            return None
        inst_sig = instance_data.get("compatibility_signature")
        lease_sig = lease_data.get("compatibility_signature")
        if inst_sig is not None and lease_sig is not None and inst_sig != lease_sig:
            return None
        return inst_proof

    if has_instance:
        if instance_data.get("lease_id"):
            return None
        if not isinstance(inst_proof, Mapping) or inst_proof.get("warm_saved") is not True:
            return None
        inst_gen = _valid_generation(instance_data.get("generation"))
        proof_gen = _valid_generation(inst_proof.get("generation"))
        if inst_gen is None or proof_gen is None or inst_gen != proof_gen:
            return None
        return inst_proof

    if has_lease:
        # Separate true no-instance legacy branch; reject leases bound to an omitted instance.
        if lease_data.get("instance_id"):
            return None
        if not isinstance(lease_proof, Mapping) or lease_proof.get("warm_saved") is not True:
            return None
        lease_gen = _valid_generation(lease_data.get("generation"))
        proof_gen = _valid_generation(lease_proof.get("generation"))
        if lease_gen is None or proof_gen is None or lease_gen != proof_gen:
            return None
        return lease_proof

    return None


def is_warm_verified_for_job(
    job_data: Mapping[str, Any],
    *,
    instance_data: Mapping[str, Any] | None = None,
    lease_data: Mapping[str, Any] | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Return True only if warm_proof matches job's manifest, generation, epoch, execution, and python_mm."""
    proof = _select_warm_proof(instance_data, lease_data)
    if not proof or proof.get("warm_saved") is not True:
        return False
    exec_id = proof.get("warm_execution_id")
    if proof.get("worker_reported_execution_id") != exec_id:
        return False
    epoch_id = proof.get("warm_epoch_id")
    if not isinstance(exec_id, str) or not exec_id.strip() or not isinstance(epoch_id, str) or not epoch_id.strip():
        return False
    warm_fp = proof.get("warm_fingerprint")
    if not isinstance(warm_fp, str) or not warm_fp.strip():
        return False
    proof_gen = _valid_generation(proof.get("generation"))
    if proof_gen is None:
        return False
    job_sig = job_data.get("compatibility_signature")
    proof_sig = proof.get("compatibility_signature")
    if proof_sig is not None and job_sig is not None and proof_sig != job_sig:
        return False
    for src in (instance_data, lease_data):
        if isinstance(src, Mapping):
            if _valid_generation(src.get("generation")) != proof_gen:
                return False
            src_epoch = src.get("warm_epoch_id")
            if isinstance(src_epoch, str) and src_epoch and src_epoch != epoch_id:
                return False
            src_exec = src.get("warm_execution_id")
            if isinstance(src_exec, str) and src_exec and src_exec != exec_id:
                return False
            src_sig = src.get("compatibility_signature")
            if proof_sig is not None and src_sig is not None and proof_sig != src_sig:
                return False
            restored_sig = src.get("restored_signature")
            if isinstance(restored_sig, str) and restored_sig and job_sig is not None and restored_sig != job_sig:
                return False
    if not isinstance(executions_by_id, Mapping) or exec_id not in executions_by_id:
        return False
    exec_row = executions_by_id[exec_id]
    if not isinstance(exec_row, Mapping) or exec_row.get("state") != "completed":
        return False
    if "warm_saved" in exec_row and exec_row.get("warm_saved") is not True:
        return False
    exec_gen = _valid_generation(exec_row.get("generation"))
    if exec_gen is None or exec_gen != proof_gen:
        return False
    exec_epoch = exec_row.get("warm_epoch_id")
    if not isinstance(exec_epoch, str) or not exec_epoch.strip() or exec_epoch != epoch_id:
        return False
    exec_fp = exec_row.get("warm_fingerprint")
    if not isinstance(exec_fp, str) or not exec_fp.strip() or exec_fp != warm_fp:
        return False
    exec_inst_id = exec_row.get("instance_id")
    exec_lease_id = exec_row.get("lease_id")
    has_exec_inst = isinstance(exec_inst_id, str) and bool(exec_inst_id.strip())
    has_exec_lease = isinstance(exec_lease_id, str) and bool(exec_lease_id.strip())
    if not (has_exec_inst and has_exec_lease):
        return False
    if not isinstance(instance_data, Mapping) or not isinstance(lease_data, Mapping):
        return False
    inst_id = instance_data.get("id")
    if not isinstance(inst_id, str) or not inst_id.strip() or exec_inst_id != inst_id:
        return False
    lease_id = lease_data.get("id")
    if not isinstance(lease_id, str) or not lease_id.strip() or exec_lease_id != lease_id:
        return False
    meta = proof.get("runtime_meta")
    python_mm = meta.get("python_mm") if isinstance(meta, Mapping) else None
    if not isinstance(python_mm, str) or not python_mm:
        return False
    manifest = (job_data.get("snapshot") or {}).get("manifest")
    if not isinstance(manifest, Mapping):
        return False
    from .runtime_worker import compute_manifest_warm_fingerprint

    expected_fp = compute_manifest_warm_fingerprint(copy.deepcopy(dict(manifest)), python_mm=python_mm)
    return warm_fp == expected_fp and exec_fp == expected_fp


def assess_job_reuse_window(
    job_data: Mapping[str, Any],
    *,
    now: float,
    instance_data: Mapping[str, Any] | None = None,
    lease_data: Mapping[str, Any] | None = None,
    deadlines: Iterable[float | int | None] | None = None,
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
) -> ReuseAssessment:
    """Extract job + runtime metadata and assess reuse lifetime window."""
    snapshot = job_data.get("snapshot") or {}
    manifest = snapshot.get("manifest") or {}
    params = snapshot.get("resolved_params")
    workflow_key = job_data.get("workflow_key") or manifest.get("workflow_key")
    workflow_version = job_data.get("workflow_version") or manifest.get("version") or manifest.get("workflow_version")
    modality = manifest.get("modality")

    gpu_tier = None
    for src in (instance_data, lease_data, job_data):
        if isinstance(src, Mapping):
            gpu_tier = _job_gpu_tier(src)
            if gpu_tier is not None:
                break
    if gpu_tier is None and isinstance(instance_data, Mapping):
        req_spec = instance_data.get("runtime_spec_requested") or {}
        if isinstance(req_spec, Mapping):
            gpu_tier = normalize_gpu_tier(req_spec.get("accelerator"))

    mode = _job_mode(job_data)
    eff_owner = owner_id or job_data.get("owner_id")
    if executions_by_id is None and repo is not None and eff_owner and hasattr(repo, "list"):
        try:
            loaded_execs = repo.list("executions", eff_owner)
            executions_by_id = {
                row["id"]: row
                for row in loaded_execs
                if isinstance(row, Mapping) and isinstance(row.get("id"), str) and row.get("id")
            }
        except Exception:
            executions_by_id = {}
    warm_verified = is_warm_verified_for_job(
        job_data,
        instance_data=instance_data,
        lease_data=lease_data,
        executions_by_id=executions_by_id,
    )
    if deadlines is None:
        deadlines = (
            instance_data.get("hard_deadline_at") if isinstance(instance_data, Mapping) else None,
            lease_data.get("hard_deadline_at") if isinstance(lease_data, Mapping) else None,
            lease_data.get("deadline_epoch") if isinstance(lease_data, Mapping) else None,
            job_data.get("hard_deadline_at"),
            job_data.get("deadline_epoch"),
        )
    return assess_reuse_window(
        workflow_key=workflow_key,
        workflow_version=workflow_version,
        gpu_tier=gpu_tier,
        mode=mode,
        params=params,
        modality=modality,
        warm_verified=warm_verified,
        now=now,
        deadlines=deadlines,
        history_jobs=history_jobs,
        repo=repo,
        owner_id=owner_id or job_data.get("owner_id"),
        executions_by_id=executions_by_id,
        usage_by_job=usage_by_job,
    )


def load_session_history(
    session,
    owner_id: str,
    job_model,
    exec_model=None,
    usage_model=None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load owner history rows inside an active DB transaction without leaving the fence."""
    job_rows = session.scalars(select(job_model).where(job_model.owner_id == owner_id)).all()
    history_jobs = [
        {**copy.deepcopy(row.data or {}), "id": row.id, "owner_id": row.owner_id, "state": row.state}
        for row in job_rows
    ]
    executions_by_id: dict[str, dict[str, Any]] = {}
    if exec_model is not None:
        for row in session.scalars(select(exec_model).where(exec_model.owner_id == owner_id)).all():
            executions_by_id[row.id] = {
                **copy.deepcopy(row.data or {}),
                "id": row.id,
                "owner_id": row.owner_id,
                "state": row.state,
            }
    usage_by_job: dict[str, dict[str, Any]] = {}
    if usage_model is not None:
        for row in session.scalars(select(usage_model).where(usage_model.owner_id == owner_id)).all():
            data = {**copy.deepcopy(row.data or {}), "id": row.id, "owner_id": row.owner_id, "state": row.state}
            if data.get("job_id"):
                usage_by_job[data["job_id"]] = data
    return history_jobs, executions_by_id, usage_by_job


def check_job_reuse_guard(
    job_data: Mapping[str, Any],
    *,
    guard_mode: str = "enforce",
    now: float,
    instance_data: Mapping[str, Any] | None = None,
    lease_data: Mapping[str, Any] | None = None,
    deadlines: Iterable[float | int | None] | None = None,
    history_jobs: Iterable[Mapping[str, Any]] | None = None,
    repo: Any | None = None,
    owner_id: str | None = None,
    executions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    usage_by_job: Mapping[str, Mapping[str, Any]] | None = None,
    observe_cb: Callable[[ReuseAssessment], None] | None = None,
) -> tuple[bool, ReuseAssessment | None]:
    """Shared reuse lifetime guard policy for allocate, handoff, attach_idle, and reuse."""
    mode = str(guard_mode or "enforce").strip().lower()
    if mode not in VALID_GUARD_MODES:
        raise ValueError(f"invalid reuse lifetime guard mode: {guard_mode!r}")
    if mode == "off" or is_legacy_mock_job(job_data):
        return (True, None)

    assessment = assess_job_reuse_window(
        job_data,
        now=now,
        instance_data=instance_data,
        lease_data=lease_data,
        deadlines=deadlines,
        history_jobs=history_jobs,
        repo=repo,
        owner_id=owner_id,
        executions_by_id=executions_by_id,
        usage_by_job=usage_by_job,
    )
    if mode == "observe":
        if observe_cb is not None:
            observe_cb(assessment)
        return (True, assessment)
    return (assessment.can_reuse, assessment)

