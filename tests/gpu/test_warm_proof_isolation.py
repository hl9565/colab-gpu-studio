"""PR4 §5.1 regression tests: warm_proof isolation, single-atomic terminal worker status, and MCP passthrough."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import threading
import uuid
from pathlib import Path

from app.config import settings
from app.db import Store
from app.gpu import runtime_worker as worker
from app.gpu.artifacts import LocalArtifactStore
from app.gpu.mcp import MCPGPUExecutor, build_poll_code
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from app.storage import asset_path

from tests.gpu.test_lease_handle_ownership import StrictMock

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
SHA_W = "c" * 64


def _manifest(commit: str = COMMIT_A, *, use_comfy_version: bool = False) -> dict:
    key = "comfy_version" if use_comfy_version else "comfy_commit"
    return {
        "state": "published",
        "reviewed": True,
        key: commit,
        "node_allowlist": ["KSampler"],
        "weight_files": [
            {
                "source": "huggingface",
                "repo_id": "org/model",
                "filename": "model.safetensors",
                "revision": COMMIT_A,
                "target": "checkpoints/model.safetensors",
                "sha256": SHA_W,
                "size": 8,
            }
        ],
    }


def _snapshot(commit: str = COMMIT_A, *, use_comfy_version: bool = False) -> dict:
    return {"manifest": _manifest(commit, use_comfy_version=use_comfy_version), "bound_graph": {}}


def _local_mcp_executor(remote_root: Path, store=None, owner_id: str = "owner-1") -> MCPGPUExecutor:
    executor = object.__new__(MCPGPUExecutor)
    executor.store = store
    executor.owner_id = owner_id
    executor.artifact_store = LocalArtifactStore(settings.data_dir / "assets" / "artifacts")
    executor._session = lambda *args, **kwargs: "session-local"

    def _tool(name: str, args: dict) -> dict:
        code = args["code"].replace("/content/ai-studio", str(remote_root))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"mcp-{name}", "exec"), {})
        return json.loads(out.getvalue())

    executor._tool = _tool
    return executor


def _setup_job_on_instance_lease(
    repo: Repository,
    *,
    owner_id: str = "owner-1",
    commit: str = COMMIT_A,
    use_comfy_version: bool = False,
    provider: str = "colab",
    lifecycle: str = "retain",
) -> tuple[dict, dict, dict]:
    snap = _snapshot(commit, use_comfy_version=use_comfy_version)
    job, _ = repo.create_job(
        owner_id,
        {
            "project_id": "proj-1",
            "provider": provider,
            "snapshot": snap,
            "compatibility_signature": f"sig-{commit[:8]}",
            "execution_policy": {"provider": provider, "lifecycle": lifecycle},
        },
        f"idem-{uuid.uuid4()}",
    )
    instance = repo.create(
        "instances",
        owner_id,
        {
            "account_id": "acct-1",
            "provider": provider,
            "state": "busy",
            "current_job_id": job["id"],
            "compatibility_signature": job["compatibility_signature"],
            "generation": 0,
            "retention_mode": lifecycle,
            "hard_deadline_at": 10**10,
        },
    )
    lease = repo.create(
        "leases",
        owner_id,
        {
            "owner_job_id": job["id"],
            "current_job_id": job["id"],
            "provider": provider,
            "state": "active",
            "account_id": "acct-1",
            "instance_id": instance["id"],
            "deadline_epoch": 10**10,
            "compatibility_signature": job["compatibility_signature"],
            "handle": {
                "runtime_name": f"runtimes/rt-{instance['id']}",
                "runtime_id": f"rt-{instance['id']}",
                "request_id": "req-1",
                "owner_id": owner_id,
                "job_id": job["id"],
                "owner_job_id": job["id"],
            },
        },
    )
    repo.update("instances", instance["id"], {"lease_id": lease["id"]}, owner_id)
    job = repo.update(
        "jobs",
        job["id"],
        {
            "state": "loading",
            "snapshot": snap,
            "instance_id": instance["id"],
            "lease_id": lease["id"],
            "account_id": "acct-1",
        },
        owner_id,
    )
    job = repo.prepare_execution(job["id"], owner_id, lease["id"], 100.0)
    job = repo.update("jobs", job["id"], {"state": "running"}, owner_id)
    return (
        repo.get("jobs", job["id"], owner_id),
        repo.get("instances", instance["id"], owner_id),
        repo.get("leases", lease["id"], owner_id),
    )


def _stub_successful_comfy_env(monkeypatch, *, make_output_size: int = 16, fail_after_restore: bool = False):
    def fake_env(root: Path, *args, **kwargs):
        (root / "ComfyUI" / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
        return {"source": "drive"}

    def fake_weights(comfy: Path, manifest: dict, *args, **kwargs):
        for w in manifest.get("weight_files", []):
            p = comfy / "models" / w["target"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"w" * (w.get("size") or 4))
        return [{"target": w["target"], "sha256": w["sha256"], "source": "drive"} for w in manifest.get("weight_files", [])]

    monkeypatch.setattr(worker, "_probe_gpu", lambda: {"source": "runtime_nvidia_smi", "captured_at": 1.0, "devices": [{"name": "NVIDIA L4", "memory_mib": 24576}]})
    monkeypatch.setattr(worker, "_restore_env", fake_env)
    monkeypatch.setattr(worker, "_restore_weights", fake_weights)

    if fail_after_restore:
        def boom(*args, **kwargs):
            raise worker.WorkerError("comfy sampling failed")

        monkeypatch.setattr(worker.subprocess, "Popen", boom)
        return

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: FakeProc())

    def fake_http(url: str, method: str = "GET", body: dict | None = None, timeout: float = 30):
        if url.endswith("/system_stats"):
            return {"system": {"os": "linux"}}
        if url.endswith("/prompt"):
            return {"prompt_id": "p-1"}
        if "/history/p-1" in url:
            return {
                "p-1": {
                    "outputs": {
                        "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
                    }
                }
            }
        return {}

    monkeypatch.setattr(worker, "_http_json", fake_http)

    class FakeResponse:
        def __init__(self, size: int):
            self._remaining = size

        def read(self, n: int = -1):
            if self._remaining <= 0:
                return b""
            chunk = b"x" * min(n if n > 0 else self._remaining, self._remaining)
            self._remaining -= len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(worker, "urlopen", lambda *a, **k: FakeResponse(make_output_size))


PROOF_KEYS = ("worker_reported_execution_id", "warm_saved", "warm_epoch_id", "warm_fingerprint", "runtime_meta")


def test_e2e_worker_to_mcp_to_scheduler_all_5_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    remote = tmp_path / "remote"
    remote.mkdir()

    # --- Scenario 1: artifacts (mcp-chunks-v1) ---
    exec1 = str(uuid.uuid4())
    root1 = remote / exec1
    root1.mkdir()
    _stub_successful_comfy_env(monkeypatch, make_output_size=32)
    payload1 = root1 / "payload.json"
    snap1 = _snapshot(COMMIT_A)
    payload1.write_text(json.dumps({
        "execution_dir": str(root1),
        "execution_id": exec1,
        "owner_id": "owner-1",
        "output_transport": "mcp-chunks-v1",
        "snapshot": snap1,
    }))
    assert worker.main([str(payload1)]) == 0
    st1 = json.loads((root1 / "status.json").read_text())
    warm_state = json.loads((remote / ".warm" / "state.json").read_text())
    assert set(PROOF_KEYS) <= set(st1)
    assert st1["state"] == "completed"
    assert st1["worker_reported_execution_id"] == exec1
    assert st1["warm_saved"] is True
    assert st1["warm_epoch_id"] == warm_state["warm_epoch_id"] and isinstance(st1["warm_epoch_id"], str)
    assert st1["runtime_meta"]["python_mm"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    expected_fp = worker.compute_manifest_warm_fingerprint(snap1["manifest"], python_mm=st1["runtime_meta"]["python_mm"])
    assert st1["warm_fingerprint"] == expected_fp

    mcp_ex = _local_mcp_executor(remote)
    polled1 = mcp_ex.poll({}, {"id": "j1", "execution_id": exec1})
    for k in PROOF_KEYS:
        assert polled1[k] == st1[k]

    # --- Scenario 2: outputs inline -> Scheduler full pipeline & DB warm_proof ---
    repo = Repository(Store(f"sqlite:///{tmp_path / 'studio.db'}"))
    job2, inst2, lease2 = _setup_job_on_instance_lease(repo, commit=COMMIT_A)
    exec2 = job2["execution_id"]
    root2 = remote / exec2
    root2.mkdir()
    payload2 = root2 / "payload.json"
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    _stub_successful_comfy_env(monkeypatch, make_output_size=len(png_bytes))
    monkeypatch.setattr(
        worker,
        "urlopen",
        lambda *a, **k: io.BytesIO(png_bytes) if False else type("R", (), {
            "_buf": io.BytesIO(png_bytes),
            "read": lambda self, n=-1: self._buf.read(n),
            "__enter__": lambda self: self,
            "__exit__": lambda *ar: False,
        })(),
    )
    payload2.write_text(json.dumps({
        "execution_dir": str(root2),
        "execution_id": exec2,
        "owner_id": "owner-1",
        "snapshot": job2["snapshot"],
    }))
    assert worker.main([str(payload2)]) == 0
    st2 = json.loads((root2 / "status.json").read_text())
    assert st2["worker_reported_execution_id"] == exec2
    assert st2["warm_saved"] is True

    # Pass real repo to MCP executor
    mcp_ex.store = repo
    provider = StrictMock(repo)
    scheduler = Scheduler(repo, provider_factory=lambda _: provider, executor_factory=lambda _: mcp_ex)
    after_run = scheduler._run(job2)
    assert after_run["state"] == "uploading"
    # warm_proof MUST already be committed on BOTH instance & lease BEFORE releasing/completing!
    inst_db = repo.get("instances", inst2["id"], "owner-1")
    lease_db = repo.get("leases", lease2["id"], "owner-1")
    assert inst_db["warm_proof"]["warm_execution_id"] == exec2
    assert inst_db["warm_proof"]["warm_epoch_id"] == st2["warm_epoch_id"]
    assert inst_db["warm_proof"]["warm_fingerprint"] == st2["warm_fingerprint"]
    assert lease_db["warm_proof"]["warm_epoch_id"] == st2["warm_epoch_id"]
    # Restarted scheduler reading internal store (not _PUBLIC) sees warm_proof
    assert "warm_proof" not in scheduler.instances.get("owner-1", inst2["id"])
    restarted = Scheduler(Repository(repo.store), provider_factory=lambda _: provider, executor_factory=lambda _: mcp_ex)
    assert restarted._service_instance_for_job(after_run)["warm_proof"]["warm_epoch_id"] == st2["warm_epoch_id"]
    uploaded = scheduler._upload(after_run)
    assert uploaded["state"] == "cleaning_up"
    assert asset_path(repo.store.get("assets", uploaded["output_asset_ids"][0], "owner-1")).read_bytes() == png_bytes

    # --- Scenario 3: failed worker run -> B failing dual-clears warm_proof to cold ---
    job3, _ = repo.create_job(
        "owner-1",
        {
            "project_id": "proj-1",
            "provider": "colab",
            "snapshot": job2["snapshot"],
            "compatibility_signature": job2["compatibility_signature"],
            "execution_policy": {"provider": "colab", "lifecycle": "retain"},
        },
        f"idem-{uuid.uuid4()}",
    )
    # Hand lease2 & inst2 to job3 via _claim_existing/update and prepare_execution
    repo.update("instances", inst2["id"], {"state": "busy", "current_job_id": job3["id"]}, "owner-1")
    repo.update("leases", lease2["id"], {"state": "active", "owner_job_id": job3["id"], "current_job_id": job3["id"]}, "owner-1")
    job3 = repo.update("jobs", job3["id"], {"state": "loading", "instance_id": inst2["id"], "lease_id": lease2["id"], "account_id": "acct-1"}, "owner-1")
    job3 = repo.prepare_execution(job3["id"], "owner-1", lease2["id"], 200.0)
    # B taking stage immediately revoked A's warm_proof!
    assert repo.get("instances", inst2["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease2["id"], "owner-1").get("warm_proof") is None
    job3 = repo.update("jobs", job3["id"], {"state": "running"}, "owner-1")

    exec3 = job3["execution_id"]
    root3 = remote / exec3
    root3.mkdir()
    _stub_successful_comfy_env(monkeypatch, fail_after_restore=True)
    payload3 = root3 / "payload.json"
    payload3.write_text(json.dumps({
        "execution_dir": str(root3),
        "execution_id": exec3,
        "owner_id": "owner-1",
        "snapshot": job3["snapshot"],
    }))
    assert worker.main([str(payload3)]) == 0
    st3 = json.loads((root3 / "status.json").read_text())
    assert st3["state"] == "failed"
    assert st3["worker_reported_execution_id"] == exec3
    assert set(PROOF_KEYS) <= set(st3)
    polled3 = mcp_ex.poll({}, {"id": job3["id"], "execution_id": exec3})
    assert polled3["state"] == "failed"
    for k in PROOF_KEYS:
        assert polled3[k] == st3[k]
    after3 = scheduler._run(job3)
    assert after3["state"] == "cleaning_up"
    assert repo.get("instances", inst2["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease2["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease2["id"], "owner-1").get("restored_signature") is None

    # --- Scenario 4: >100MB gpu_output_too_large preserves 5 proof fields + gpu_diagnostics + timings ---
    exec4 = str(uuid.uuid4())
    root4 = remote / exec4
    root4.mkdir()
    _stub_successful_comfy_env(monkeypatch, make_output_size=32)
    payload4 = root4 / "payload.json"
    payload4.write_text(json.dumps({
        "execution_dir": str(root4),
        "execution_id": exec4,
        "owner_id": "owner-1",
        "snapshot": snap1,
    }))
    assert worker.main([str(payload4)]) == 0
    # Replace generated output file with sparse >100MB file (takes microseconds, no RAM bloat)
    meta4 = json.loads((root4 / "meta.json").read_text())
    big_rel = meta4["outputs"][0]["path"]
    with (root4 / big_rel).open("wb") as f:
        f.truncate(100 * 1024 * 1024 + 1024)
    st4 = json.loads((root4 / "status.json").read_text())
    assert st4["state"] == "completed" and st4["warm_saved"] is True
    assert "gpu_diagnostics" in st4 and st4["timings"]
    polled4 = mcp_ex.poll({}, {"id": "j4", "execution_id": exec4})
    assert polled4["state"] == "failed"
    assert polled4["error_code"] == "gpu_output_too_large"
    assert polled4["gpu_diagnostics"] == st4["gpu_diagnostics"]
    assert polled4["timings"] == st4["timings"]
    for k in PROOF_KEYS:
        assert polled4[k] == st4[k]

    # --- Scenario 5: main() exception at L1457-1460 drive_cache rewrite -> gpu_worker_bootstrap_failed ---
    # Seed stale .warm/state.json from earlier execution on disk!
    warm_dir = remote / ".warm"
    warm_dir.mkdir(exist_ok=True)
    (warm_dir / "state.json").write_text(json.dumps({
        "fingerprint": "stale-fp-from-a",
        "warm_epoch_id": "stale-epoch-from-a",
    }))
    exec5 = str(uuid.uuid4())
    root5 = remote / exec5
    root5.mkdir()
    pre_diag = {"source": "runtime_nvidia_smi", "captured_at": 1.0, "devices": [{"name": "NVIDIA T4", "memory_mib": 15360}]}
    worker._atomic_json(root5 / "status.json", {"state": "starting", "gpu_diagnostics": pre_diag})
    payload5 = root5 / "payload.json"
    payload5.write_text(json.dumps({
        "execution_dir": str(root5),
        "execution_id": exec5,
        "owner_id": "owner-1",
        "drive_cache": {"access_token": "ya29.fake-secret-token-value"},
        "snapshot": snap1,
    }))
    orig_atomic = worker._atomic_json

    def fail_on_payload_rewrite(path: Path, value):
        if Path(path) == payload5.resolve():
            raise OSError("simulated drive_cache strip write failure")
        return orig_atomic(path, value)

    monkeypatch.setattr(worker, "_atomic_json", fail_on_payload_rewrite)
    assert worker.main([str(payload5)]) == 1
    st5 = json.loads((root5 / "status.json").read_text())
    assert st5["state"] == "failed"
    assert st5["error_code"] == "gpu_worker_bootstrap_failed"
    assert st5["gpu_diagnostics"] == pre_diag
    assert set(PROOF_KEYS) <= set(st5)
    assert st5["worker_reported_execution_id"] == exec5
    assert st5["warm_saved"] is False
    assert st5["warm_epoch_id"] is None
    assert st5["warm_fingerprint"] is None
    assert st5["runtime_meta"]["python_mm"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    # Must NOT be polluted by stale .warm/state.json
    assert st5["warm_epoch_id"] != "stale-epoch-from-a"
    assert st5["warm_fingerprint"] != "stale-fp-from-a"

    polled5 = mcp_ex.poll({}, {"id": "j5", "execution_id": exec5})
    assert polled5["state"] == "failed"
    assert polled5["error_code"] == "gpu_worker_bootstrap_failed"
    assert polled5["worker_reported_execution_id"] == exec5
    assert polled5["warm_saved"] is False
    assert polled5["warm_epoch_id"] is None
    assert polled5["warm_fingerprint"] is None
    assert polled5["runtime_meta"] == st5["runtime_meta"]


def test_mcp_poll_never_forges_worker_reported_execution_id(tmp_path):
    remote = tmp_path / "remote"
    execution_id = str(uuid.uuid4())
    root = remote / execution_id
    root.mkdir(parents=True)
    # Simulate legacy or unparseable-payload bootstrap failure where worker_reported_execution_id is None
    worker._atomic_json(
        root / "status.json",
        {
            "state": "failed",
            "error_code": "gpu_worker_bootstrap_failed",
            "error_message": "bad payload",
            "worker_reported_execution_id": None,
            "warm_saved": False,
            "warm_epoch_id": None,
            "warm_fingerprint": None,
            "runtime_meta": {"python_mm": "3.14"},
        },
    )
    mcp_ex = _local_mcp_executor(remote)
    assert "worker_reported_execution_id" in build_poll_code(execution_id)
    res = mcp_ex.poll({}, {"id": "j", "execution_id": execution_id})
    assert res["execution_id"] == execution_id
    assert res["worker_reported_execution_id"] is None
    assert res["warm_saved"] is False

    # Corrupt payload JSON -> worker.main writes terminal status with all 5 fields & worker_reported_execution_id=None
    exec_bad = str(uuid.uuid4())
    bad_root = remote / exec_bad
    bad_root.mkdir()
    bad_payload = bad_root / "payload.json"
    bad_payload.write_text("{not-json")
    assert worker.main([str(bad_payload)]) == 1
    bad_st = json.loads((bad_root / "status.json").read_text())
    assert bad_st["state"] == "failed"
    assert bad_st["error_code"] == "gpu_worker_bootstrap_failed"
    assert set(PROOF_KEYS) <= set(bad_st)
    assert bad_st["worker_reported_execution_id"] is None
    assert bad_st["warm_saved"] is False
    bad_polled = mcp_ex.poll({}, {"id": "j-bad", "execution_id": exec_bad})
    assert bad_polled["worker_reported_execution_id"] is None
    assert bad_polled["warm_saved"] is False


def test_concurrent_status_reader_never_observes_terminal_missing_5_proof_fields(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _stub_successful_comfy_env(monkeypatch, make_output_size=8)
    errors: list[str] = []

    for idx in range(5):
        exec_id = str(uuid.uuid4())
        root = remote / exec_id
        root.mkdir()
        payload = root / "payload.json"
        payload.write_text(json.dumps({
            "execution_dir": str(root),
            "execution_id": exec_id,
            "owner_id": "owner-1",
            "snapshot": _snapshot(COMMIT_A),
        }))
        stop = threading.Event()
        status_path = root / "status.json"

        def reader():
            while not stop.is_set():
                if status_path.exists():
                    try:
                        data = json.loads(status_path.read_text())
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"corrupt json: {exc}")
                        return
                    if data.get("state") in {"completed", "failed"}:
                        missing = [k for k in PROOF_KEYS if k not in data]
                        if missing:
                            errors.append(f"terminal missing {missing} on iter {idx}: {data}")
                            return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        worker.main([str(payload)])
        stop.set()
        t.join(timeout=2)
    assert not errors, errors


def test_late_a_does_not_perturb_b_and_warm_saved_false_clears_cold():
    repo = Repository(Store("sqlite://"))
    job_a, inst, lease = _setup_job_on_instance_lease(repo, commit=COMMIT_A, use_comfy_version=True)
    exec_a = job_a["execution_id"]
    exec_row_a = repo.get("executions", exec_a, "owner-1")
    gen_a = exec_row_a["generation"]
    assert gen_a >= 1
    meta_ok = {"python_mm": "3.14", "warm_version": 1}
    fp_a = worker.compute_manifest_warm_fingerprint(job_a["snapshot"]["manifest"], python_mm="3.14")

    # Complete A properly first
    res_a = repo.commit_execution_terminal(
        job_a["id"],
        "owner-1",
        execution_id=exec_a,
        terminal_state="uploading",
        message="A done",
        patch={},
        now=200.0,
        proof={
            "worker_reported_execution_id": exec_a,
            "warm_saved": True,
            "warm_epoch_id": "epoch-a",
            "warm_fingerprint": fp_a,
            "runtime_meta": meta_ok,
            "timings": {"restore_weights": 1.2},
        },
        expected_generation=gen_a,
    )
    assert res_a["state"] == "uploading"
    assert repo.get("instances", inst["id"], "owner-1")["warm_proof"]["warm_epoch_id"] == "epoch-a"

    # Now B takes the stage -> A's warm_proof immediately revoked!
    job_b, _ = repo.create_job(
        "owner-1",
        {
            "project_id": "proj-1",
            "provider": "colab",
            "snapshot": _snapshot(COMMIT_A, use_comfy_version=True),
            "compatibility_signature": job_a["compatibility_signature"],
            "execution_policy": {"provider": "colab", "lifecycle": "retain"},
        },
        f"idem-{uuid.uuid4()}",
    )
    repo.update("instances", inst["id"], {"state": "busy", "current_job_id": job_b["id"]}, "owner-1")
    repo.update("leases", lease["id"], {"state": "active", "owner_job_id": job_b["id"], "current_job_id": job_b["id"]}, "owner-1")
    job_b = repo.update("jobs", job_b["id"], {"state": "loading", "instance_id": inst["id"], "lease_id": lease["id"], "account_id": "acct-1"}, "owner-1")
    job_b = repo.prepare_execution(job_b["id"], "owner-1", lease["id"], 210.0)
    exec_b = job_b["execution_id"]
    gen_b = repo.get("executions", exec_b, "owner-1")["generation"]
    assert gen_b > gen_a
    assert repo.get("instances", inst["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease["id"], "owner-1").get("warm_proof") is None
    assert repo.get("instances", inst["id"], "owner-1")["active_execution"]["active_execution_id"] == exec_b
    job_b = repo.update("jobs", job_b["id"], {"state": "running"}, "owner-1")

    # Late A arrives while B is active -> MUST NOT disturb B's active_execution or jobs or warm_proof!
    late_applied = repo.commit_execution_terminal(
        job_a["id"],
        "owner-1",
        execution_id=exec_a,
        terminal_state="uploading",
        message="late A",
        patch={},
        now=215.0,
        proof={
            "worker_reported_execution_id": exec_a,
            "warm_saved": True,
            "warm_epoch_id": "epoch-late-a",
            "warm_fingerprint": fp_a,
            "runtime_meta": meta_ok,
            "timings": {},
        },
        expected_generation=gen_a,
    )
    assert late_applied is None or late_applied.get("state") != "uploading" or repo.get("instances", inst["id"], "owner-1")["active_execution"]["active_execution_id"] == exec_b
    assert repo.get("instances", inst["id"], "owner-1")["active_execution"]["active_execution_id"] == exec_b
    assert repo.get("instances", inst["id"], "owner-1").get("warm_proof") is None
    assert repo.get("jobs", job_b["id"], "owner-1")["state"] == "running"

    # Now B completes media BUT warm_saved is False -> dual-cleared to cold!
    res_b = repo.commit_execution_terminal(
        job_b["id"],
        "owner-1",
        execution_id=exec_b,
        terminal_state="uploading",
        message="B media done",
        patch={},
        now=220.0,
        proof={
            "worker_reported_execution_id": exec_b,
            "warm_saved": False,
            "warm_epoch_id": None,
            "warm_fingerprint": None,
            "runtime_meta": meta_ok,
            "timings": {"warm_reuse": 0.1},
        },
        expected_generation=gen_b,
    )
    assert res_b["state"] == "uploading"
    assert repo.get("instances", inst["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease["id"], "owner-1").get("warm_proof") is None
    assert repo.get("leases", lease["id"], "owner-1").get("restored_signature") is None
    # Fingerprint alignment check: wrong python_mm rejects warm_proof -> stays cold!
    job_c, _, _ = _setup_job_on_instance_lease(repo, commit=COMMIT_A, use_comfy_version=True)
    exec_c = job_c["execution_id"]
    gen_c = repo.get("executions", exec_c, "owner-1")["generation"]
    fp_wrong_py = worker.compute_manifest_warm_fingerprint(job_c["snapshot"]["manifest"], python_mm="3.10")
    repo.commit_execution_terminal(
        job_c["id"],
        "owner-1",
        execution_id=exec_c,
        terminal_state="uploading",
        message="wrong python fp",
        patch={},
        now=230.0,
        proof={
            "worker_reported_execution_id": exec_c,
            "warm_saved": True,
            "warm_epoch_id": "epoch-c",
            "warm_fingerprint": fp_wrong_py,
            "runtime_meta": {"python_mm": "3.14", "warm_version": 1},
            "timings": {},
        },
        expected_generation=gen_c,
    )
    assert repo.get("instances", job_c["instance_id"], "owner-1").get("warm_proof") is None
