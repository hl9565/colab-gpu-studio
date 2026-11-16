import json
import uuid

import httpx
import pytest
from app.db import Store
from app.gpu.colab import ColabError, GoogleColabProvider
from app.gpu.repository import Repository


class Google:
    def access_token(self, purpose):
        return "never-return-this-token"


class Repo(Repository):
    def __init__(self):
        super().__init__(Store("sqlite:///:memory:"))
        self.create("leases", "owner", {"owner_job_id": "job", "runtime_id": "studio-test",
            "request_id": str(uuid.uuid4()), "state": "creating"}, record_id="lease")

    @property
    def rows(self):
        return self.list("leases", "owner")


def provider(handler):
    repo = Repo()
    p = GoogleColabProvider(
        repo, "owner", google=Google(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    handle = {
        **repo.rows[0],
        "runtime_name": "runtimes/studio-test",
        "operation_name": "operations/create-test",
    }
    return p, repo, handle


def test_pending_operation_is_not_mistaken_for_deleted_gpu():
    def handle(req):
        assert req.url.path == "/v1/operations/create-test"
        return httpx.Response(200, json={"name": "operations/create-test", "done": False})

    p, repo, h = provider(handle)
    assert p.inspect(h) == {"ready": False, "deleted": False, "operation_pending": True}
    assert p.delete(h) is False


def test_operation_error_is_read_before_running_and_spec_is_real_key():
    def handle(req):
        if req.method == "POST":
            assert json.loads(req.content) == {
                "runtimeSpec": {
                    "variant": "VARIANT_GPU",
                    "accelerator": "T4",
                    "shape": "SHAPE_STANDARD",
                }
            }
            return httpx.Response(
                200,
                json={
                    "name": "operations/create-test",
                    "done": True,
                    "error": {"code": 9, "message": "upstream details never leak"},
                },
            )
        raise AssertionError("no runtime get after failed Operation")

    p, repo, h = provider(handle)
    with pytest.raises(ColabError, match="9"):
        p.create(
            {"key": {"variant": "GPU", "accelerator": "T4", "shape": "STANDARD"}, "eligible": True},
            repo.rows[0]["request_id"],
            "job",
            "studio-test",
        )


def test_cannot_delete_unrecorded_user_runtime_even_with_owner_field():
    p, _, h = provider(lambda req: (_ for _ in ()).throw(AssertionError("must not call Google")))
    h["runtime_name"] = "runtimes/manual-user-runtime"
    h["runtime_id"] = "manual-user-runtime"
    with pytest.raises(ColabError):
        p.delete(h)
