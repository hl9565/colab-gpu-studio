import json
import shutil
import subprocess
from pathlib import Path

import pytest
from app.gpu import builtin
from app.gpu.manifest import bind_parameters, resolve_parameters


def _run_node_plugin_tests() -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node binary is not available on PATH")
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tests" / "test_video_canvas_plugin.mjs"
    proc = subprocess.run(
        [node, str(script), "--emit-payloads"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _load_bundle(filename: str) -> dict:
    root = Path(builtin.__file__).with_name("templates") / "builtin"
    return json.loads((root / filename).read_text(encoding="utf-8"))


def test_video_canvas_plugin_binds_ref2v_and_differs_only_by_vae_node_13():
    payloads = _run_node_plugin_tests()
    ref2v = _load_bundle("minimax-h3-ref2v-1.4.0.json")
    manifest = ref2v["manifest"]
    api_graph = ref2v["api_graph"]

    dully_body = payloads["dully"]
    official_body = payloads["official"]
    assert dully_body["workflow_key"] == official_body["workflow_key"] == "minimax-h3-ref2v"
    assert dully_body["workflow_version"] == official_body["workflow_version"] == manifest["version"]

    resolved_dully = resolve_parameters(manifest, dully_body["params"])
    resolved_official = resolve_parameters(manifest, official_body["params"])
    assert resolved_dully["seed"] == 3916315536
    assert resolved_official["seed"] == 3916315536

    graph_dully = bind_parameters(manifest, api_graph, resolved_dully)
    graph_official = bind_parameters(manifest, api_graph, resolved_official)

    assert graph_dully["23"]["inputs"]["noise_seed"] == 3916315536
    assert graph_official["23"]["inputs"]["noise_seed"] == 3916315536
    assert graph_dully["13"]["inputs"]["vae_name"] == (
        "minimax_h3_video_vae_dully_seamless_v1_int8convrot.safetensors"
    )
    assert graph_official["13"]["inputs"]["vae_name"] == (
        "minimax_h3_video_vae_int8_convrot.safetensors"
    )

    changed_nodes = {
        node_id
        for node_id in set(graph_dully) | set(graph_official)
        if graph_dully.get(node_id) != graph_official.get(node_id)
    }
    assert changed_nodes == {"13"}


def test_video_canvas_plugin_handles_2s_random_seed_landscape_and_i2v():
    payloads = _run_node_plugin_tests()
    ref2v = _load_bundle("minimax-h3-ref2v-1.4.0.json")
    i2v = _load_bundle("minimax-h3-i2v-1.1.0.json")

    resolved_2s = resolve_parameters(ref2v["manifest"], payloads["ref2s"]["params"])
    assert resolved_2s["length"] == 56
    assert 0 <= resolved_2s["seed"] <= 4294967295
    graph_2s = bind_parameters(ref2v["manifest"], ref2v["api_graph"], resolved_2s)
    assert graph_2s["20"]["inputs"]["length"] == 56

    resolved_169 = resolve_parameters(ref2v["manifest"], payloads["ref169"]["params"])
    graph_169 = bind_parameters(ref2v["manifest"], ref2v["api_graph"], resolved_169)
    assert graph_169["20"]["inputs"]["width"] == 960
    assert graph_169["20"]["inputs"]["height"] == 544
    assert resolved_169["seed"] == 0

    resolved_i2v = resolve_parameters(i2v["manifest"], payloads["i2v"]["params"])
    graph_i2v = bind_parameters(i2v["manifest"], i2v["api_graph"], resolved_i2v)
    assert resolved_i2v["seed"] == 42
    assert graph_i2v["22"]["inputs"]["steps"] == 8
    assert graph_i2v["23"]["inputs"]["noise_seed"] == 42
