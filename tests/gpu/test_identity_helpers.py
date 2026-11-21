from types import SimpleNamespace

from app.gpu import identity, repository


def test_repository_reexports_identity_helpers():
    assert repository.retarget_handle is identity.retarget_handle
    assert repository.extract_tunnel_endpoint is identity.extract_tunnel_endpoint
    assert repository.extract_runtime_id is identity.extract_runtime_id
    assert repository.collect_runtime_identities is identity.collect_runtime_identities
    assert repository._collect_runtime_identities is identity.collect_runtime_identities


def test_conflicting_mapping_keeps_three_scan_rules_distinct():
    payload = {
        "runtime_name": "runtimes/top-name",
        "tunnel_endpoint": "https://top.example/?invalid=1",
        "handle": {"runtime_name": "runtimes/top-handle", "endpoint": "invalid path"},
        "data": {
            "runtime_id": " data-id ",
            "endpoint": "invalid/path",
            "handle": {
                "runtime_id": "data-handle",
                "tunnel_endpoint": "HTTPS://Data-Handle.Example:443/",
                "endpoint": "Data-Handle-Alias",
            },
            "data": {"runtime_id": "deep-id", "endpoint": "Deep-Endpoint"},
        },
    }

    assert identity.extract_runtime_id(payload) == "data-id"
    assert identity.extract_tunnel_endpoint(payload) == "https://data-handle.example"

    runtime_ids = {"previous-id"}
    endpoints = {"previous-endpoint"}
    identity.collect_runtime_identities(payload, runtime_ids, endpoints)
    assert runtime_ids == {"previous-id", "top-name", "top-handle", "data-id", "data-handle", "deep-id"}
    assert endpoints == {"previous-endpoint", "https://data-handle.example", "data-handle-alias", "deep-endpoint"}

    without_data_id = {**payload, "data": {**payload["data"], "runtime_id": None}}
    assert identity.extract_runtime_id(without_data_id) == "top-name"
    assert identity.extract_runtime_id({**without_data_id, "handle": None}) == "data-handle"
    with_valid_top_handle = {**payload, "handle": {**payload["handle"], "endpoint": "Top-Handle"}}
    assert identity.extract_tunnel_endpoint(with_valid_top_handle) == "top-handle"


def test_row_reads_columns_and_data_but_ignores_top_level_handle():
    row = SimpleNamespace(
        runtime_id=" column-id ",
        runtime_name="runtimes/column-name",
        endpoint="Column-Endpoint",
        handle={"runtime_id": "ignored-id", "endpoint": "Ignored-Endpoint"},
        data={"handle": {"runtime_id": "data-id", "endpoint": "Data-Endpoint"}},
    )

    assert identity.extract_runtime_id(row) == "column-id"
    assert identity.extract_tunnel_endpoint(row) == "column-endpoint"

    runtime_ids, endpoints = set(), set()
    identity.collect_runtime_identities(row, runtime_ids, endpoints)
    assert runtime_ids == {"column-id", "column-name", "data-id"}
    assert endpoints == {"column-endpoint", "data-endpoint"}

    row.runtime_id = row.runtime_name = row.endpoint = None
    assert identity.extract_runtime_id(row) == "data-id"
    assert identity.extract_tunnel_endpoint(row) == "data-endpoint"

    row.data = {}
    assert identity.extract_runtime_id(row) is None
    assert identity.extract_tunnel_endpoint(row) is None


def test_retarget_handle_shallow_copies_without_mutating_original():
    metadata = {"runtime_id": "shared-runtime"}
    original = {"job_id": "old-job", "owner_job_id": "old-owner", "metadata": metadata}

    retargeted = identity.retarget_handle(original, "next-job")

    assert retargeted is not original
    assert retargeted == {"job_id": "next-job", "owner_job_id": "next-job", "metadata": metadata}
    assert retargeted["metadata"] is metadata
    assert original == {"job_id": "old-job", "owner_job_id": "old-owner", "metadata": metadata}
