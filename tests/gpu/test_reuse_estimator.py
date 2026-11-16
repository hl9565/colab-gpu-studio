"""Tests for PR4 §5.2 Item 4: pure-sampling reuse estimator, min_deadlines & lifetime guard."""

from __future__ import annotations

import math
import threading

import pytest
from app.db import Store
from app.gpu.compute import GpuConfig
from app.gpu.estimator import (
    COLD_SAMPLING_BASELINES,
    COMFY_START_S,
    GPU_TIME_FACTOR,
    HIST_P90_SAFETY_FACTOR,
    MIN_HIST_SAMPLES,
    SAFETY_MARGIN_S,
    TAIL_S,
    UncalibratedWorkloadError,
    env_weights_restore_seconds,
    estimate_sampling_seconds,
    evaluate_reuse_window,
    min_deadlines,
)
from app.gpu.instances import InstanceService
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler

from tests.gpu.test_pr4_signoff_regressions import seed_execution_binding


def _manifest(workflow_key: str, version: str, accelerator: str, modality: str = "video") -> dict:
    commit = "423cbd20511becb1f16122ab518e92e58bfa03d9"
    deps = {
        "comfyui_commit": commit,
        "custom_nodes": [],
        "weight_files": [
            {
                "repo_id": "Comfy-Org/Test",
                "filename": "model.safetensors",
                "revision": "54472221d27c695fca9621b0c3ed52f06c4af707",
                "sha256": "a" * 64,
                "size": 1024,
                "target": "diffusion_models/model.safetensors",
            }
        ],
    }
    return {
        "workflow_key": workflow_key,
        "version": version,
        "modality": modality,
        "comfyui_commit": commit,
        "comfy_commit": commit,
        "custom_nodes": [],
        "weight_files": deps["weight_files"],
        "dependencies": deps,
        "compute_spec": {"variant": "VARIANT_GPU", "accelerator": accelerator, "shape": "SHAPE_STANDARD"},
        "parameters": {},
    }


def _calibrated_job(
    *,
    job_id: str = "job-1",
    owner_id: str = "owner-1",
    workflow_key: str = "minimax-h3-ref2v",
    workflow_version: str = "1.4.0",
    accelerator: str = "L4",
    modality: str = "video",
    mode: str | None = "fast8",
    params: dict | None = None,
    provider: str = "colab",
    state: str = "queued",
    compatibility_signature: str = "sig-ref2v",
    deadline_epoch: float | None = None,
    **extra,
) -> dict:
    resolved = dict(params or {"width": 544, "height": 960, "length": 124})
    if mode is not None:
        resolved["mode"] = mode
    manifest = _manifest(workflow_key, workflow_version, accelerator, modality)
    if modality == "video":
        manifest["dependencies"] = {"comfyui_commit": "c1", "weight_files": [{"sha256": "w1", "target": "m.safetensors"}]}
    payload = {
        "id": job_id,
        "owner_id": owner_id,
        "project_id": "proj-1",
        "workflow_key": workflow_key,
        "workflow_version": workflow_version,
        "gpu_tier": accelerator,
        "state": state,
        "provider": provider,
        "execution_policy": {"provider": provider, "lifecycle": "retain", "mode": "batch"},
        "compatibility_signature": compatibility_signature,
        "attempt_number": 1,
        "lifecycle": "retain",
        "snapshot": {
            "manifest": manifest,
            "resolved_params": resolved,
        },
        **extra,
    }
    if deadline_epoch is not None:
        payload["deadline_epoch"] = deadline_epoch
    return payload


def test_min_deadlines_tightens_and_handles_all_none():
    assert min_deadlines() is None
    assert min_deadlines(None, None, None) is None
    assert min_deadlines(None, 500.0, None, 350.0) == 350.0
    assert min_deadlines(0.0, 500.0) == 0.0
    assert min_deadlines(600, 450.5, None) == 450.5


def test_gpu_config_reuse_lifetime_guard_mode_from_env(monkeypatch):
    monkeypatch.delenv("GPU_REUSE_LIFETIME_GUARD_MODE", raising=False)
    assert GpuConfig.from_env().reuse_lifetime_guard_mode == "enforce"

    for mode in ("off", "observe", "enforce", " OBSERVE "):
        monkeypatch.setenv("GPU_REUSE_LIFETIME_GUARD_MODE", mode)
        assert GpuConfig.from_env().reuse_lifetime_guard_mode == mode.strip().lower()

    monkeypatch.setenv("GPU_REUSE_LIFETIME_GUARD_MODE", "bad-mode")
    with pytest.raises(ValueError, match="GPU_REUSE_LIFETIME_GUARD_MODE"):
        GpuConfig.from_env()


@pytest.mark.parametrize(
    ("mode", "base_s"),
    [("fast", 150.0), ("fast8", 240.0), ("refine", 270.0), ("refine8", 360.0), ("hq", 540.0)],
)
@pytest.mark.parametrize("gpu_tier", ["T4", "L4", "A100"])
def test_ref2v_cold_baseline_matrix_all_modes_and_gpus(mode, base_s, gpu_tier):
    expected = base_s * (GPU_TIME_FACTOR[gpu_tier] / GPU_TIME_FACTOR["L4"]) * 1.25
    actual = estimate_sampling_seconds(
        "minimax-h3-ref2v",
        "1.4.0",
        gpu_tier,
        mode,
        {"width": 544, "height": 960, "length": 124, "mode": mode},
    )
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_l4_fast8_verified_warm_boundary_and_cold_restore_penalties():
    assert COMFY_START_S == 30.0
    assert TAIL_S == 60.0
    assert SAFETY_MARGIN_S == 60.0
    assert MIN_HIST_SAMPLES == 3
    assert HIST_P90_SAFETY_FACTOR == 1.20
    assert env_weights_restore_seconds(True, "video") == 0.0
    assert env_weights_restore_seconds(False, "video") == 330.0
    assert env_weights_restore_seconds(False, None) == 330.0
    assert env_weights_restore_seconds(False, "image") == 180.0

    params = {"width": 544, "height": 960, "length": 124, "mode": "fast8"}
    assert estimate_sampling_seconds("minimax-h3-ref2v", "1.4.0", "L4", "fast8", params) == pytest.approx(300.0)

    # Verified warm need = 30 + 0 + 300 + 60 + 60 = 450.0s
    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=params,
        modality="video",
        warm_verified=True,
        now=1000.0,
        deadlines=(1450.0,),
    ) == (True, pytest.approx(450.0))

    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=params,
        modality="video",
        warm_verified=True,
        now=1000.0,
        deadlines=(1449.0, 2000.0),
    ) == (False, pytest.approx(450.0))

    # Cold video need = 30 + 330 + 300 + 60 + 60 = 780.0s
    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=params,
        modality="video",
        warm_verified=False,
        now=1000.0,
        deadlines=(1780.0,),
    ) == (True, pytest.approx(780.0))

    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=params,
        modality="video",
        warm_verified=False,
        now=1000.0,
        deadlines=(1779.0,),
    ) == (False, pytest.approx(780.0))

    # All-None deadlines on calibrated workload -> does not turn None into 0
    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=params,
        modality="video",
        warm_verified=True,
        now=1000.0,
        deadlines=(None, None),
    ) == (True, pytest.approx(450.0))


def test_scaled_golden_formulas_across_all_five_builtin_workflows():
    # ref2v 362f / 1088x1088 on L4 fast8
    ref2v_expected = 240.0 * 1.0 * ((362 / 124) ** 1.7) * (((1088 * 1088) / (544 * 960)) ** 1.35) * 1.25
    assert estimate_sampling_seconds(
        "minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 1088, "height": 1088, "length": 362, "mode": "fast8"}
    ) == pytest.approx(ref2v_expected)

    # relay 362f x 8 on L4 fast8 (inter_seg_s = 15.0)
    relay_expected = (
        240.0 * 1.0 * ((362 / 124) ** 1.7) * (((544 * 960) / (544 * 960)) ** 1.35) * 8 + 7 * 15.0 * 1.0
    ) * 1.25
    assert estimate_sampling_seconds(
        "minimax-h3-relay",
        "1.0.0",
        "L4",
        "fast8",
        {"width": 544, "height": 960, "length": 362, "segments": 8, "mode": "fast8"},
    ) == pytest.approx(relay_expected)

    # i2v 1.1.0 default and fast8
    i2v_expected = 210.0 * 1.0 * (12 / 8) * ((200 / 124) ** 1.7) * (((480 * 864) / (480 * 864)) ** 1.35) * 1.25
    assert estimate_sampling_seconds(
        "minimax-h3-i2v", "1.1.0", "L4", None, {"width": 480, "height": 864, "length": 200, "steps": 12}
    ) == pytest.approx(i2v_expected)
    assert estimate_sampling_seconds(
        "minimax-h3-i2v", "1.1.0", "L4", "fast8", {"width": 480, "height": 864, "length": 200, "steps": 12}
    ) == pytest.approx(i2v_expected)

    # krea2-moss-dual-t2i 1.0.0 & 1.1.1 1536^2 / (krea=20, moss=50, denoise=0.36)
    krea_expected = 45.0 * 1.0 * (((1536 * 1536) / (832 * 1248)) ** 1.25) * ((20.0 + 50.0) / 33.0) * 1.25
    for ver in ("1.0.0", "1.1.1"):
        assert estimate_sampling_seconds(
            "krea2-moss-dual-t2i",
            ver,
            "T4",
            None,
            {"width": 1536, "height": 1536, "krea_steps": 20, "moss_steps": 50, "moss_denoise": 0.36},
        ) == pytest.approx(krea_expected)
        # Default fallback when dual-sampler step params omitted
        krea_default = 45.0 * 1.0 * (((832 * 1248) / (832 * 1248)) ** 1.25) * (33.0 / 33.0) * 1.25
        assert estimate_sampling_seconds(
            "krea2-moss-dual-t2i", ver, "T4", None, {"width": 832, "height": 1248}
        ) == pytest.approx(krea_default)

    # krea2-moss with higher moss_denoise=0.72 doubles effective moss steps weight
    krea_heavy_denoise = 45.0 * 1.0 * 1.0 * ((20.0 + 50.0 * (0.72 / 0.36)) / 33.0) * 1.25
    assert estimate_sampling_seconds(
        "krea2-moss-dual-t2i",
        "1.1.1",
        "T4",
        None,
        {"width": 832, "height": 1248, "krea_steps": 20, "moss_steps": 50, "moss_denoise": 0.72},
    ) == pytest.approx(krea_heavy_denoise)

    # moss-image-sft-t2i 1.0.0 1536^2 / steps=60 on T4
    moss_expected = 40.0 * 1.0 * (((1536 * 1536) / (832 * 1248)) ** 1.25) * (60.0 / 35.0) * 1.25
    assert estimate_sampling_seconds(
        "moss-image-sft-t2i", "1.0.0", "T4", None, {"width": 1536, "height": 1536, "steps": 60}
    ) == pytest.approx(moss_expected)

    assert len(COLD_SAMPLING_BASELINES) == 6


@pytest.mark.parametrize(
    ("workflow_key", "workflow_version", "gpu_tier", "mode", "params"),
    [
        ("unknown-wf", "1.4.0", "L4", "fast8", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "9.9.9", "L4", "fast8", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "H100", "fast8", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", None, "fast8", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "ultra", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", None, {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 544, "height": 960}),
        ("minimax-h3-relay", "1.0.0", "L4", "fast8", {"width": 544, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 0, "height": 960, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 544, "height": -32, "length": 124}),
        ("minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": True, "height": 960, "length": 124}),
        ("moss-image-sft-t2i", "1.0.0", "T4", None, {"width": 832, "height": 1248, "steps": 0}),
        ("krea2-moss-dual-t2i", "1.1.1", "T4", None, {"width": 832, "height": 1248, "moss_denoise": 0.0}),
    ],
)
def test_uncalibrated_or_invalid_inputs_fail_closed_without_fake_cap(
    workflow_key, workflow_version, gpu_tier, mode, params
):
    with pytest.raises(UncalibratedWorkloadError):
        estimate_sampling_seconds(workflow_key, workflow_version, gpu_tier, mode, params)

    assert evaluate_reuse_window(
        workflow_key=workflow_key,
        workflow_version=workflow_version,
        gpu_tier=gpu_tier,
        mode=mode,
        params=params,
        modality="video",
        warm_verified=True,
        now=0.0,
        deadlines=(999_999.0,),
    ) == (False, None)


def test_history_admission_rejects_poisoned_and_fewer_than_three_then_uses_p90():
    base_params = {"width": 544, "height": 960, "length": 124, "mode": "fast8"}

    def hist_job(jid, *, state="completed", provider="colab", timings=None, params=None, accelerator="L4", reported="L4", include_mode=True, **extra):
        effective_params = dict(params or base_params)
        if not include_mode:
            effective_params.pop("mode", None)
        job = _calibrated_job(
            job_id=jid,
            state=state,
            provider=provider,
            accelerator=accelerator,
            mode=effective_params.get("mode") if include_mode else None,
            params=effective_params,
            worker_timings=timings if timings is not None else {"sampling": 200.0},
            **({"gpu_reported": reported} if reported is not None else {}),
            **extra,
        )
        return job

    poisoned = [
        # Failed partial sampling (runtime_worker:1413 + scheduler:1045)
        hist_job("poison-failed", state="failed", timings={"sampling": 2.0}, error_code="gpu_worker_failed"),
        # Canceled / aborted
        hist_job("poison-cancelled", state="cancelled", timings={"sampling": 2.0}),
        hist_job("poison-aborted", state="aborted", timings={"sampling": 2.0}),
        # Mock run
        hist_job("poison-mock", provider="mock", timings={"sampling": 2.0}),
        # Prep run
        hist_job("poison-prep", prep=True, timings={"sampling": 2.0}),
        # Different version / mode / GPU
        hist_job("poison-ver", workflow_version="1.3.0", timings={"sampling": 2.0}),
        hist_job("poison-mode", params={**base_params, "mode": "fast"}, timings={"sampling": 2.0}),
        hist_job("poison-gpu", accelerator="A100", reported="A100", timings={"sampling": 2.0}),
        # Requested L4 only (no actual gpu_diagnostics / gpu_reported hardware evidence) -> MUST NOT enter history
        hist_job("poison-requested-only", reported=None, timings={"sampling": 2.0}),
        # Requested L4, actually ran on A100 via gpu_reported -> MUST NOT be counted as L4
        hist_job("poison-mismatched-reported", accelerator="L4", reported="A100", timings={"sampling": 2.0}),
        # Missing explicit resolved["mode"] on multi-mode workflow even if manifest parameter default says fast8
        {
            **hist_job("poison-implicit-mode", include_mode=False, timings={"sampling": 2.0}),
            "snapshot": {
                **hist_job("poison-implicit-mode", include_mode=False, timings={"sampling": 2.0})["snapshot"],
                "manifest": {
                    **_manifest("minimax-h3-ref2v", "1.4.0", "L4", "video"),
                    "parameters": {"mode": {"variants": ["fast8", "hq"], "default": "fast8"}},
                },
            },
        },
        # Missing sampling timing (wall clock elapsed only) or zero/negative sampling
        hist_job("poison-nosampling", timings={"comfy_start": 30.0}, elapsed_minutes=0.5),
        hist_job("poison-zero", timings={"sampling": 0.0}),
        # Missing required resolved_params
        hist_job("poison-noparams", params={"width": 544, "height": 960}, timings={"sampling": 2.0}),
    ]

    # Add 2 valid samples (< MIN_HIST_SAMPLES=3): poisoned records must NOT lower estimate or unlock P90
    two_good = poisoned + [
        hist_job("good-1", timings={"sampling": 100.0}, finished_at=10.0),
        hist_job("good-2", timings={"sampling": 110.0}, finished_at=20.0),
    ]
    assert evaluate_reuse_window(
        workflow_key="minimax-h3-ref2v",
        workflow_version="1.4.0",
        gpu_tier="L4",
        mode="fast8",
        params=base_params,
        modality="video",
        warm_verified=True,
        now=1000.0,
        deadlines=(1450.0,),
        history_jobs=two_good,
    ) == (True, pytest.approx(450.0))

    # Add 3rd qualified sample (2.0x workload -> sampling 400s => normalized 200s? wait: use same params [100, 110, 200])
    # Wait: test normalized scaling explicitly: 362f = (362/124)**1.7 ratio!
    ratio_362 = (362 / 124) ** 1.7
    three_good = two_good + [
        hist_job(
            "good-3",
            params={"width": 544, "height": 960, "length": 362, "mode": "fast8"},
            timings={"sampling": 200.0 * ratio_362},
            finished_at=30.0,
        ),
    ]
    # Normalized samples: [100.0, 110.0, 200.0]; linear P90 at 0.9*(3-1)=1.8 -> 110 + 0.8*(200-110) = 182.0
    sampling_p90 = estimate_sampling_seconds(
        "minimax-h3-ref2v", "1.4.0", "L4", "fast8", base_params, history_jobs=three_good
    )
    assert sampling_p90 == pytest.approx(182.0 * HIST_P90_SAFETY_FACTOR)


def test_monotonicity_across_dimensions_steps_and_segments():
    ref_small = estimate_sampling_seconds(
        "minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 544, "height": 960, "length": 124, "mode": "fast8"}
    )
    ref_wider = estimate_sampling_seconds(
        "minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 768, "height": 960, "length": 124, "mode": "fast8"}
    )
    ref_longer = estimate_sampling_seconds(
        "minimax-h3-ref2v", "1.4.0", "L4", "fast8", {"width": 768, "height": 960, "length": 240, "mode": "fast8"}
    )
    assert ref_small <= ref_wider <= ref_longer

    seg1 = estimate_sampling_seconds(
        "minimax-h3-relay", "1.0.0", "L4", "fast8", {"width": 544, "height": 960, "length": 124, "segments": 1, "mode": "fast8"}
    )
    seg4 = estimate_sampling_seconds(
        "minimax-h3-relay", "1.0.0", "L4", "fast8", {"width": 544, "height": 960, "length": 124, "segments": 4, "mode": "fast8"}
    )
    assert seg1 < seg4

    krea_low = estimate_sampling_seconds(
        "krea2-moss-dual-t2i", "1.1.1", "T4", None, {"width": 832, "height": 1248, "krea_steps": 8, "moss_steps": 20, "moss_denoise": 0.36}
    )
    krea_high = estimate_sampling_seconds(
        "krea2-moss-dual-t2i", "1.1.1", "T4", None, {"width": 832, "height": 1248, "krea_steps": 16, "moss_steps": 40, "moss_denoise": 0.60}
    )
    assert krea_low < krea_high


@pytest.mark.parametrize(("remaining", "expected_handoff"), [(450.0, True), (449.0, False)])
def test_batch_handoff_evaluates_following_job_and_warm_proof(remaining, expected_handoff):
    repo = Repository(Store("sqlite:///:memory:"))
    now = 1000.0
    job1_payload = _calibrated_job(job_id="j1", compatibility_signature="sig-same")
    job2_payload = _calibrated_job(job_id="j2", compatibility_signature="sig-same")
    # Remove id before create_batch so create_batch generates ids deterministically
    for p in (job1_payload, job2_payload):
        p.pop("id")
    repo.create_batch("owner-1", [job1_payload, job2_payload], "batch-key", {"mode": "batch"})
    first, following = sorted(repo.list("jobs", "owner-1"), key=lambda row: row["queue_order"])

    # Compute real manifest fingerprint via runtime_worker helper
    from app.gpu.runtime_worker import compute_manifest_warm_fingerprint

    fp = compute_manifest_warm_fingerprint(following["snapshot"]["manifest"], python_mm="3.12")
    assert fp is not None
    warm_proof = {
        "warm_execution_id": "exec-1", "worker_reported_execution_id": "exec-1",
        "warm_saved": True,
        "warm_epoch_id": "epoch-1",
        "warm_fingerprint": fp,
        "generation": 1,
        "runtime_meta": {"python_mm": "3.12"},
    }
    instance = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "account_id": "acct-1",
            "current_job_id": first["id"],
            "budget_reserved": 1.0,
            "hard_deadline_at": now + remaining,
            "generation": 1,
            "warm_proof": warm_proof,
            "compatibility_signature": "sig-same",
        },
        record_id="inst-1",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": first["id"],
            "instance_id": instance["id"],
            "account_id": "acct-1",
            "deadline_epoch": now + 2000.0,  # looser than instance hard_deadline_at; min_deadlines must tighten!
            "generation": 1,
            "warm_proof": warm_proof,
            "compatibility_signature": "sig-same",
            "handle": {"runtime_id": "rt-1", "job_id": first["id"], "owner_job_id": first["id"]},
        },
        record_id="lease-1",
    )
    repo.create(
        "executions",
        "owner-1",
        {
            "state": "completed",
            "job_id": first["id"],
            "instance_id": instance["id"],
            "lease_id": lease["id"],
            "generation": 1,
            "warm_epoch_id": "epoch-1",
            "warm_fingerprint": fp,
        },
        record_id="exec-1",
    )
    repo.update(
        "jobs",
        first["id"],
        {"state": "cleaning_up", "instance_id": instance["id"], "lease_id": lease["id"], "execution_id": "exec-1"},
        "owner-1",
    )
    repo.update("jobs", following["id"], {"state": "waiting_for_capacity"}, "owner-1")

    assert repo.handoff_lease("owner-1", first["id"], following["id"], lease["id"], now) is expected_handoff


def test_instance_service_allocate_respects_guard_modes_and_refuses_short_retained_instance():
    repo = Repository(Store("sqlite:///:memory:"))
    repo.create(
        "accounts",
        "owner-1",
        {
            "account_id": "acct-1",
            "provider": "colab",
            "connected": True,
            "enabled": True,
            "allow_allocation": True,
            "projects": ["proj-1"],
            "budget": {"daily": 100.0, "monthly": 100.0},
        },
        record_id="acct-1",
    )
    now = 1000.0
    job = repo.create(
        "jobs",
        "owner-1",
        _calibrated_job(job_id="job-new", state="waiting_for_capacity", budget_units=1.0),
        record_id="job-new",
    )
    from app.gpu.runtime_worker import compute_manifest_warm_fingerprint

    fp = compute_manifest_warm_fingerprint(job["snapshot"]["manifest"], python_mm="3.12")
    assert fp is not None
    warm_proof = {
        "warm_execution_id": "exec-old", "worker_reported_execution_id": "exec-old",
        "warm_saved": True,
        "warm_epoch_id": "epoch-old",
        "warm_fingerprint": fp,
        "generation": 1,
        "runtime_meta": {"python_mm": "3.12"},
    }
    repo.create(
        "leases",
        "owner-1",
        {
            "account_id": "acct-1",
            "provider": "colab",
            "state": "idle",
            "owner_job_id": "job-prev",
            "instance_id": "inst-short",
            "compatibility_signature": "sig-ref2v",
            "idle_deadline_at": now + 449.0,
            "hard_deadline_at": now + 449.0,
            "deadline_epoch": now + 449.0,
            "generation": 1,
            "warm_proof": warm_proof,
            "handle": {"runtime_name": "runtimes/rt-short", "runtime_id": "rt-short"},
        },
        record_id="lease-short",
    )
    short_idle = repo.create(
        "instances",
        "owner-1",
        {
            "account_id": "acct-1",
            "provider": "colab",
            "state": "retained_idle",
            "lease_id": "lease-short",
            "current_job_id": None,
            "compatibility_signature": "sig-ref2v",
            "hard_deadline_at": now + 449.0,
            "idle_deadline_at": now + 449.0,
            "generation": 1,
            "warm_proof": warm_proof,
        },
        record_id="inst-short",
    )
    repo.create(
        "executions",
        "owner-1",
        {
            "state": "completed",
            "job_id": "job-prev",
            "instance_id": short_idle["id"],
            "lease_id": "lease-short",
            "generation": 1,
            "warm_epoch_id": "epoch-old",
            "warm_fingerprint": fp,
        },
        record_id="exec-old",
    )

    # enforce mode rejects short_idle and creates a fresh instance
    enforcing = InstanceService(
        repo,
        clock=lambda: now,
        config={"unit_rate": 0.1, "daily_units": 100.0, "monthly_units": 100.0, "reuse_lifetime_guard_mode": "enforce"},
    )
    allocated = enforcing.allocate(job)
    assert allocated["id"] != short_idle["id"]
    assert allocated["selection_reason"] != "reuse_idle"

    # observe mode records telemetry without blocking reuse
    observing = InstanceService(
        repo,
        clock=lambda: now,
        config={"unit_rate": 0.1, "daily_units": 100.0, "monthly_units": 100.0, "reuse_lifetime_guard_mode": "observe"},
    )
    job_obs = repo.create(
        "jobs",
        "owner-1",
        _calibrated_job(job_id="job-obs", state="waiting_for_capacity", budget_units=1.0),
        record_id="job-obs",
    )
    reused_obs = observing.allocate(job_obs)
    assert reused_obs["id"] == short_idle["id"]
    events = repo.list("events", "owner-1")
    assert any(
        ev.get("event") == "reuse_lifetime_observed"
        and ev.get("instance_id") == short_idle["id"]
        and ev.get("need_s") == pytest.approx(450.0)
        and math.isclose(ev.get("remaining_s"), 449.0)
        and ev.get("warm_verified") is True
        for ev in events
    )


def test_legacy_attach_idle_lease_handles_all_none_idle_deadline_and_racing_schedulers(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'race.db'}")
    repo = Repository(store)
    now = 1000.0
    job_a = repo.create("jobs", "owner-1", _calibrated_job(job_id="job-a", state="waiting_for_capacity", params={"width": 544, "height": 960, "length": 124, "mode": "fast8"}), record_id="job-a")
    job_b = repo.create("jobs", "owner-1", _calibrated_job(job_id="job-b", state="waiting_for_capacity", params={"width": 544, "height": 960, "length": 124, "mode": "fast8"}), record_id="job-b")

    from app.gpu.runtime_worker import compute_manifest_warm_fingerprint

    fp = compute_manifest_warm_fingerprint(job_a["snapshot"]["manifest"], python_mm="3.12")
    assert fp is not None
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "idle",
            "owner_job_id": "prev",
            "compatibility_signature": "sig-ref2v",
            "idle_deadline_at": None,  # Must NOT coerce None -> 0!
            "deadline_epoch": now + 900.0,
            "generation": 1,
            "warm_proof": {
                "warm_execution_id": "exec-prev", "worker_reported_execution_id": "exec-prev",
                "warm_saved": True,
                "warm_epoch_id": "ep-prev",
                "warm_fingerprint": fp,
                "generation": 1,
                "runtime_meta": {"python_mm": "3.12"},
            },
            "handle": {"runtime_id": "rt-shared", "job_id": "prev", "owner_job_id": "prev"},
        },
        record_id="lease-idle",
    )
    repo.create(
        "executions",
        "owner-1",
        {
            "state": "completed",
            "job_id": "prev",
            "lease_id": lease["id"],
            "generation": 1,
            "warm_epoch_id": "ep-prev",
            "warm_fingerprint": fp,
        },
        record_id="exec-prev",
    )

    results = []
    barrier = threading.Barrier(2)

    def try_attach(jid):
        barrier.wait()
        with store.dispatch_lock():
            results.append((jid, repo.attach_idle_lease("owner-1", jid, lease["id"], now)))

    t1 = threading.Thread(target=try_attach, args=(job_a["id"],))
    t2 = threading.Thread(target=try_attach, args=(job_b["id"],))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    winners = [jid for jid, ok in results if ok]
    assert len(winners) == 1
    assert repo.get("leases", lease["id"], "owner-1")["owner_job_id"] == winners[0]


def test_begin_failure_dual_clears_owned_warm_proof_without_disturbing_successor():
    repo = Repository(Store("sqlite:///:memory:"))
    inst = repo.create(
        "instances",
        "owner-1",
        {
            "state": "busy",
            "account_id": "acct-1",
            "current_job_id": "job-b",
            "active_execution_id": "exec-b",
            "warm_proof": {"warm_execution_id": "exec-a", "worker_reported_execution_id": "exec-a", "warm_saved": True, "warm_epoch_id": "ep-a", "warm_fingerprint": "fp-a"},
        },
        record_id="inst-1",
    )
    lease = repo.create(
        "leases",
        "owner-1",
        {
            "state": "active",
            "owner_job_id": "job-b",
            "instance_id": inst["id"],
            "active_execution_id": "exec-b",
            "warm_proof": {"warm_execution_id": "exec-a", "worker_reported_execution_id": "exec-a", "warm_saved": True, "warm_epoch_id": "ep-a", "warm_fingerprint": "fp-a"},
            "handle": {"runtime_id": "rt-1", "job_id": "job-b", "owner_job_id": "job-b"},
        },
        record_id="lease-1",
    )
    stale_job = repo.create(
        "jobs",
        "owner-1",
        {"state": "running", "execution_id": "exec-a", "lease_id": lease["id"], "instance_id": inst["id"]},
        record_id="job-a",
    )

    scheduler = Scheduler(repo)
    scheduler._begin_failure(stale_job, "gpu_timeout", "timed out")

    # Successor B's ownership must remain completely intact
    assert repo.get("leases", lease["id"], "owner-1")["active_execution_id"] == "exec-b"
    assert repo.get("instances", inst["id"], "owner-1")["active_execution_id"] == "exec-b"

    # Now fail job-b which DOES own active_execution_id: warm_proof must dual-clear to cold!
    active_job_b = repo.create(
        "jobs",
        "owner-1",
        {"state": "uploading", "execution_id": "exec-b", "lease_id": lease["id"], "instance_id": inst["id"]},
        record_id="job-b",
    )
    active_job_b = seed_execution_binding(repo, active_job_b, lease, instance=inst)
    scheduler._begin_failure(active_job_b, "gpu_upload_failed", "upload error")
    assert repo.get("leases", lease["id"], "owner-1").get("warm_proof") is None
    assert repo.get("instances", inst["id"], "owner-1").get("warm_proof") is None


def test_bare_colab_or_unattributed_job_never_bypasses_reuse_guard():
    from app.gpu.estimator import check_job_reuse_guard, is_legacy_mock_job

    bare_colab = {"id": "j-colab", "provider": "colab", "execution_policy": {"provider": "colab"}}
    bare_unknown_provider = {"id": "j-none"}
    assert is_legacy_mock_job(bare_colab) is False
    assert is_legacy_mock_job(bare_unknown_provider) is False

    allowed, assessment = check_job_reuse_guard(
        bare_colab,
        guard_mode="enforce",
        now=1000.0,
        deadlines=(999_999.0,),
    )
    assert allowed is False
    assert assessment is not None
    assert assessment.can_reuse is False
