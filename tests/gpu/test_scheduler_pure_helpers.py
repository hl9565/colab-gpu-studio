import hashlib

import pytest

from app.gpu.compute import ComputeError, has_gpu_spec, select_compute_spec, spec_key
from app.gpu.output_protocol import build_result_metadata

T4 = {"key": {"variant": "VARIANT_GPU", "accelerator": "T4"}, "eligible": True}
L4 = {"key": {"variant": "GPU", "accelerator": "L4"}, "eligible": True}
CPU = {"key": {"variant": "VARIANT_CPU", "accelerator": "NONE"}, "eligible": True}
NO_MATCH = "没有符合工作流runtime的GPU规格"
UNBOUND_SMOKE = "system smoke必须绑定已选择的GPU规格"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"key": {"variant": "GPU", "accelerator": "测试"}}, '{"accelerator":"测试","variant":"GPU"}'),
        ({"key": {}}, "{}"),
        ({"key": "raw-key"}, "raw-key"),
        ({"key": ""}, None),
        ({"key": None}, None),
        ({}, None),
    ],
)
def test_spec_key(spec, expected):
    assert spec_key(spec) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(T4, True), (L4, True), (CPU, False), ({"key": "GPU"}, False), ({"key": {}}, False), ({}, False)],
)
def test_has_gpu_spec(spec, expected):
    assert has_gpu_spec(spec) is expected


def test_regular_job_matches_reviewed_spec():
    job = {"snapshot": {"manifest": {"compute_spec": {"accelerator": "T4", "variant": "VARIANT_GPU"}}}}
    assert select_compute_spec(job, iter([L4, T4])) is T4
    with pytest.raises(ComputeError) as error:
        select_compute_spec(job, [L4])
    assert str(error.value) == NO_MATCH


def test_regular_job_requires_reviewed_spec_even_with_request():
    with pytest.raises(ComputeError) as error:
        select_compute_spec({"requested_spec": T4}, [T4])
    assert str(error.value) == NO_MATCH


@pytest.mark.parametrize("workflow_smoke", [False, True])
def test_system_smoke_rejects_unbound_request(workflow_smoke):
    job = {"snapshot": {"system_smoke": True, "workflow_smoke": workflow_smoke}}
    with pytest.raises(ComputeError) as error:
        select_compute_spec(job, [T4])
    assert str(error.value) == UNBOUND_SMOKE


def test_system_smoke_request_takes_priority_over_policy_and_review():
    job = {
        "requested_spec": spec_key(L4),
        "execution_policy": {"requested_spec": T4},
        "snapshot": {"system_smoke": True, "manifest": {"compute_spec": T4["key"]}},
    }
    assert select_compute_spec(job, [T4, L4]) is L4


def test_system_smoke_uses_policy_then_review_as_fallbacks():
    snapshot = {"system_smoke": True, "manifest": {"compute_spec": T4["key"]}}
    job = {"snapshot": snapshot, "execution_policy": {"requested_spec": L4}}
    assert select_compute_spec(job, [T4, L4]) is L4
    assert select_compute_spec({"snapshot": snapshot}, [T4, L4]) is T4


@pytest.mark.parametrize("eligible", [False, None, 1, "true"])
def test_workflow_smoke_uses_first_eligible_gpu_without_review(eligible):
    job = {"snapshot": {"workflow_smoke": True}}
    assert select_compute_spec(job, [CPU, {**L4, "eligible": eligible}, T4, L4]) is T4


def test_prep_overrides_request_and_review_with_fixed_cpu_spec():
    job = {"requested_spec": T4, "snapshot": {"prep": True, "manifest": {"compute_spec": L4["key"]}}}
    other_cpu = {**CPU, "key": {**CPU["key"], "shape": "OTHER"}}
    assert select_compute_spec(job, [T4, other_cpu, CPU], prep_spec_key=CPU["key"]) is CPU
    with pytest.raises(ComputeError) as error:
        select_compute_spec(job, [T4], prep_spec_key=T4["key"])
    assert str(error.value) == NO_MATCH


def test_prep_does_not_fall_back_to_arbitrary_cpu_without_prep_key():
    with pytest.raises(ComputeError) as error:
        select_compute_spec({"snapshot": {"prep": True, "workflow_smoke": True}}, [CPU])
    assert str(error.value) == NO_MATCH


def test_result_metadata_reuses_workflow_sha256_and_keeps_all_fields():
    job = {
        "id": "job-1",
        "workflow_key": "workflow",
        "workflow_version": 3,
        "model_id": "model",
        "gpu_reported": "T4",
        "gpu_diagnostics": {"memory": 16},
        "snapshot": {
            "workflow_sha256": "existing-hash",
            "model_revision": "revision",
            "resolved_params": {"seed": 42},
            "system_smoke": True,
        },
    }
    assert build_result_metadata(job) == {
        "job_id": "job-1",
        "snapshot_sha256": "existing-hash",
        "workflow_key": "workflow",
        "workflow_version": 3,
        "model_id": "model",
        "model_revision": "revision",
        "resolved_params": {"seed": 42},
        "system_smoke": True,
        "gpu_reported": "T4",
        "gpu_diagnostics": {"memory": 16},
    }


def test_result_metadata_hashes_canonical_snapshot():
    snapshot = {"text": "测试", "system_smoke": 1, "custom": complex(1, 2)}
    expected = hashlib.sha256('{"custom":"(1+2j)","system_smoke":1,"text":"测试"}'.encode()).hexdigest()
    metadata = build_result_metadata({"id": "job-1", "snapshot": snapshot})
    assert metadata["snapshot_sha256"] == expected
    assert metadata["system_smoke"] is False
    assert build_result_metadata({"id": "job-1", "snapshot": dict(reversed(snapshot.items()))}) == metadata


@pytest.mark.parametrize("snapshot", [None, "not-a-dict", ["not-a-dict"]])
def test_result_metadata_defaults_for_non_dict_snapshot(snapshot):
    metadata = build_result_metadata({"id": "job-1", "snapshot": snapshot})
    assert metadata == build_result_metadata({"id": "job-1"})
    assert metadata["snapshot_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert metadata["resolved_params"] == {}
    assert metadata["model_revision"] is None
    assert metadata["system_smoke"] is False
