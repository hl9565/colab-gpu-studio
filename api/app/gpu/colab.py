"""Colab v1beta runtime lifecycle + v1 long-running Operations (official discoveries)."""

from __future__ import annotations

import json
import re
import time
import uuid

import httpx

from app.security import decrypt_secret, encrypt_secret

from .google import GoogleError, GoogleService
from .identity import (
    _RUNTIME_ID,
    canonicalize_tunnel_endpoint,
    parse_strict_runtime_name,
)


class ColabError(GoogleError):
    pass


class ColabAllocationError(ColabError):
    def __init__(self, code, reason):
        self.reason = reason
        # Only the numeric code is shown; Google's own text may carry private data.
        if reason == "NO_RUNTIMES":
            message = "Colab 暂时无法分配运行时（NO_RUNTIMES）；请稍后重试"
        else:
            message = f"Colab 分配 Operation 失败（code {code}）"
        super().__init__(message)


_CLI_PUBLIC_CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
_TUNNEL_BASE_URL = "https://colab.research.google.com"
_XSSI_PREFIX = ")]}'\n"
_HIGH_MEM_ONLY_ACCELERATORS = frozenset({"L4", "V5E1", "V6E1"})


def _uuid_to_web_safe_base64(uuid_val: uuid.UUID) -> str:
    uuid_str = str(uuid_val)
    return uuid_str.replace("-", "_") + ("." * (44 - len(uuid_str)))


_OPERATION = re.compile(r"^operations/[A-Za-z0-9._~-]+$")
_VARIANTS = {"VARIANT_CPU", "VARIANT_GPU", "VARIANT_TPU"}
_VARIANT_ALIASES = {"CPU": "VARIANT_CPU", "GPU": "VARIANT_GPU", "TPU": "VARIANT_TPU"}
_SHAPES = {"SHAPE_STANDARD", "SHAPE_HIGHMEM"}
_SHAPE_ALIASES = {"STANDARD": "SHAPE_STANDARD", "HIGHMEM": "SHAPE_HIGHMEM"}


class GoogleColabProvider:
    is_mock = False

    def __init__(
        self,
        store,
        owner_id,
        *,
        account_id=None,
        client=None,
        google=None,
        base_url="https://colaboratory.googleapis.com",
        timeout=10.0,
    ):
        self.store, self.owner_id, self.account_id = store, owner_id, account_id
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=False)
        self._owns_client = client is None
        self.google = google or GoogleService(store, owner_id, account_id=account_id, client=self.client)
        self.base_url = base_url.rstrip("/")

    def _request(self, method, path, *, params=None, json=None):
        token = self.google.access_token("colab")
        try:
            response = self.client.request(
                method,
                self.base_url + path,
                params=params,
                json=json,
                headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ColabError("Colab API 网络异常，远端结果可能未知，将保留租约进行协调") from exc
        if 300 <= response.status_code < 400:
            raise ColabError(f"Colab API 重定向异常（HTTP {response.status_code}）")
        if response.status_code >= 400 and response.status_code != 404:
            raise ColabError(f"Colab API 请求失败（HTTP {response.status_code}）")
        return response

    @staticmethod
    def _json(response):
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except ValueError as exc:
            raise ColabError("Colab API 返回无效 JSON") from exc

    def _is_cli_tunnel(self, lease: dict | None = None) -> bool:
        if isinstance(lease, dict) and lease.get("tunnel_endpoint"):
            return True
        try:
            cred = self.google._credential("colab", account_id=self.account_id)
        except Exception:
            cred = None
        if isinstance(cred, dict):
            cid = str(cred.get("client_id") or "").strip()
            if cid == _CLI_PUBLIC_CLIENT_ID or cid.startswith("764086051850-"):
                return True
        return False

    def _tunnel_request(self, method: str, path: str, *, params: dict | None = None, headers: dict | None = None):
        token = self.google.access_token("colab")
        req_params = {"authuser": "0"}
        if params:
            req_params.update(params)
        req_headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "X-Colab-Client-Agent": "colab-cli",
        }
        if headers:
            req_headers.update(headers)
        try:
            response = self.client.request(
                method,
                _TUNNEL_BASE_URL + path,
                params=req_params,
                headers=req_headers,
            )
        except httpx.HTTPError as exc:
            raise ColabError("Colab 隧道网络异常，远端结果可能未知，将保留租约进行协调") from exc
        if 300 <= response.status_code < 400:
            raise ColabError(f"Colab 隧道重定向异常（HTTP {response.status_code}）")
        if response.status_code >= 400 and response.status_code != 404:
            raise ColabError(f"Colab 隧道请求失败（HTTP {response.status_code}）")
        return response

    @staticmethod
    def _tunnel_json(response) -> dict:
        text = response.text or ""
        if text.startswith(_XSSI_PREFIX):
            text = text[len(_XSSI_PREFIX) :]
        if not text.strip():
            return {}
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except ValueError as exc:
            raise ColabError("Colab 隧道返回无效 JSON") from exc

    @staticmethod
    def _normalize_tunnel_endpoint(raw: object) -> str:
        """Normalize an unambiguous tunnel endpoint or fail closed via ColabError."""
        try:
            return canonicalize_tunnel_endpoint(raw)
        except ValueError as exc:
            raise ColabError(str(exc)) from exc

    @staticmethod
    def _norm_meta_key(key: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).casefold())

    _PAGINATION_CURSOR_KEYS = frozenset(
        {
            "nextpagetoken",
            "pagetoken",
            "prevpagetoken",
            "previouspagetoken",
            "next",
            "prev",
            "previous",
            "nextpage",
            "prevpage",
            "previouspage",
            "nexturl",
            "prevurl",
            "previousurl",
            "nextcursor",
            "cursor",
            "prevcursor",
            "previouscursor",
            "continuationtoken",
            "nexttoken",
        }
    )
    _PAGINATION_BOOL_FLAG_KEYS = frozenset(
        {
            "hasmore",
            "hasnextpage",
            "hasnext",
            "hasmorepages",
            "haspreviouspage",
            "hasprevpage",
            "hasprevious",
            "hasprev",
            "moreavailable",
            "more",
            "truncated",
            "istruncated",
            "incomplete",
            "partial",
            "ispartial",
        }
    )
    _LINK_PAGINATION_RE = re.compile(r"\b(?:next|prev|previous)\b|page=|cursor=|pagetoken=")

    @staticmethod
    def _parse_non_bool_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    @staticmethod
    def _is_present_cursor_value(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Any non-empty cursor string (including "0") is treated as a live cursor.
            return bool(value)
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return bool(value)

    @staticmethod
    def _is_truthy_pagination_value(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            stripped = value.strip()
            return bool(stripped) and stripped.casefold() not in {"false", "0", "no", "null", "none"}
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return bool(value)

    @classmethod
    def _has_incomplete_pagination_headers(cls, headers: object, item_count: int | None) -> bool:
        if not headers:
            return False
        items = headers.items() if hasattr(headers, "items") else ()
        cursor_headers = {prefix + key for prefix in ("", "x") for key in cls._PAGINATION_CURSOR_KEYS} | {
            "xnextpagetoken",
            "nextpagetoken",
            "xpagetoken",
            "pagetoken",
            "xprevpagetoken",
            "prevpagetoken",
            "xpreviouspagetoken",
            "previouspagetoken",
            "xcontinuationtoken",
            "continuationtoken",
            "xnextcursor",
            "nextcursor",
            "xprevcursor",
            "prevcursor",
            "xcursor",
            "cursor",
        }
        bool_headers = {
            "xhasmore",
            "hasmore",
            "xhasnextpage",
            "hasnextpage",
            "xhaspreviouspage",
            "haspreviouspage",
            "xhasprevpage",
            "hasprevpage",
            "xtruncated",
            "truncated",
            "xincomplete",
            "incomplete",
            "xpartial",
            "partial",
        }
        page_headers = {"xpage", "page", "xcurrentpage", "currentpage", "xpagenumber", "pagenumber"}
        total_page_headers = {"xtotalpages", "totalpages", "xpagecount", "pagecount", "xpages", "pages"}
        total_headers = {"xtotalcount", "totalcount", "xtotalitems", "totalitems", "xtotal", "total"}
        for raw_key, raw_val in items:
            norm_key = cls._norm_meta_key(raw_key)
            if norm_key == "contentrange":
                return True
            if norm_key == "link":
                link_val = str(raw_val or "").casefold()
                if cls._LINK_PAGINATION_RE.search(link_val):
                    return True
            if norm_key in cursor_headers and cls._is_present_cursor_value(raw_val):
                return True
            if norm_key in bool_headers and cls._is_truthy_pagination_value(raw_val):
                return True
            if norm_key in page_headers:
                p_num = cls._parse_non_bool_int(raw_val)
                if p_num is not None and p_num > 1:
                    return True
            if norm_key in total_page_headers:
                tp_num = cls._parse_non_bool_int(raw_val)
                if tp_num is not None and tp_num > 1:
                    return True
            if norm_key in total_headers and item_count is not None:
                parsed = cls._parse_non_bool_int(raw_val)
                if parsed is None or parsed > item_count:
                    return True
        return False

    @classmethod
    def _has_incomplete_pagination_payload(cls, node: object, item_count: int, *, skip_root_list_key: str | None = None) -> bool:
        count_keys = {"totalcount", "totalelements", "totalitems", "totalresults", "total"}
        remaining_keys = {"remaining", "remainingcount"}

        def _walk(value: object, *, is_root: bool = False) -> bool:
            if isinstance(value, dict):
                page_num: int | None = None
                total_pages: int | None = None
                for k, v in value.items():
                    if is_root and skip_root_list_key and k == skip_root_list_key:
                        continue
                    nk = cls._norm_meta_key(k)
                    if nk in cls._PAGINATION_CURSOR_KEYS and cls._is_present_cursor_value(v):
                        return True
                    if nk in cls._PAGINATION_BOOL_FLAG_KEYS and cls._is_truthy_pagination_value(v):
                        return True
                    if nk in remaining_keys and not isinstance(v, bool):
                        parsed_rem = cls._parse_non_bool_int(v)
                        if parsed_rem is not None and parsed_rem > 0:
                            return True
                    if nk in count_keys and not isinstance(v, bool):
                        parsed_cnt = cls._parse_non_bool_int(v)
                        if parsed_cnt is not None and parsed_cnt > item_count:
                            return True
                    if nk in {"page", "currentpage", "pagenumber", "pageindex"} and not isinstance(v, bool):
                        parsed_page = cls._parse_non_bool_int(v)
                        if parsed_page is not None:
                            if parsed_page > 1:
                                return True
                            page_num = parsed_page
                    if nk in {"totalpages", "pagecount", "pages"} and not isinstance(v, bool):
                        parsed_tp = cls._parse_non_bool_int(v)
                        if parsed_tp is not None:
                            if parsed_tp > 1:
                                return True
                            total_pages = parsed_tp
                    if _walk(v, is_root=False):
                        return True
                if page_num is not None and total_pages is not None and page_num < total_pages:
                    return True
            elif isinstance(value, list):
                for item in value:
                    if _walk(item, is_root=False):
                        return True
            return False

        return _walk(node, is_root=True)

    @classmethod
    def _extract_complete_list_from_payload(
        cls,
        headers: object,
        payload: object,
        *,
        list_key: str,
        context: str,
        allow_missing_list_key: bool = False,
    ) -> list[dict]:
        if cls._has_incomplete_pagination_headers(headers, None):
            raise ColabError(f"{context}响应包含不完整分页响应头，无法证明列表完整")
        if not isinstance(payload, dict):
            raise ColabError(f"{context}响应格式无效")
        raw_items = payload.get(list_key)
        if raw_items is None and allow_missing_list_key:
            raw_items = []
        if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
            raise ColabError(f"{context}响应缺少有效的 {list_key} 列表")
        item_count = len(raw_items)
        if cls._has_incomplete_pagination_headers(headers, item_count):
            raise ColabError(f"{context}响应包含不完整分页响应头，无法证明列表完整")
        if cls._has_incomplete_pagination_payload(payload, item_count, skip_root_list_key=list_key):
            raise ColabError(f"{context}响应未完整分页，无法证明列表完整")
        return list(raw_items)

    def _tunnel_assignments(self) -> list[dict]:
        response = self._tunnel_request("GET", "/tun/m/assignments")
        if response.status_code != 200:
            raise ColabError(f"Colab 分配列表响应不完整或状态异常（HTTP {response.status_code}）")
        headers = getattr(response, "headers", None) or {}
        if self._has_incomplete_pagination_headers(headers, None):
            raise ColabError("Colab 分配列表响应包含不完整分页响应头，无法证明运行时不存在")
        payload = self._tunnel_json(response)
        raw_entries = self._extract_complete_list_from_payload(
            headers,
            payload,
            list_key="assignments",
            context="Colab 分配列表",
        )
        rows: list[dict] = []
        seen: set[str] = set()
        for entry in raw_entries:
            normalized = self._normalize_tunnel_endpoint(entry.get("endpoint"))
            if normalized in seen:
                raise ColabError("Colab 分配列表包含重复的归一化 endpoint 身份")
            seen.add(normalized)
            rows.append(entry)
        return rows

    def list_specs(self):
        if self._is_cli_tunnel():
            info = self._tunnel_json(self._tunnel_request("GET", "/tun/m/ccu-info"))
            specs = [
                {
                    "key": {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"},
                    "eligible": True,
                }
            ]
            for gpu in info.get("eligibleGpus") or []:
                if isinstance(gpu, str) and gpu:
                    specs.append(
                        {
                            "key": {"variant": "VARIANT_GPU", "accelerator": gpu.upper(), "shape": "SHAPE_STANDARD"},
                            "eligible": True,
                        }
                    )
                    if gpu.upper() not in _HIGH_MEM_ONLY_ACCELERATORS:
                        specs.append(
                            {
                                "key": {"variant": "VARIANT_GPU", "accelerator": gpu.upper(), "shape": "SHAPE_HIGHMEM"},
                                "eligible": True,
                            }
                        )
            for tpu in info.get("eligibleTpus") or []:
                if isinstance(tpu, str) and tpu:
                    specs.append(
                        {
                            "key": {"variant": "VARIANT_TPU", "accelerator": tpu.upper(), "shape": "SHAPE_STANDARD"},
                            "eligible": True,
                        }
                    )
            return specs
        return self._json(self._request("GET", "/v1beta/runtimespecs")).get("runtimeSpecs", [])

    def subscription(self):
        if self._is_cli_tunnel():
            info = self._tunnel_json(self._tunnel_request("GET", "/tun/m/ccu-info"))
            return {
                "name": "subscription",
                "tier": "SUBSCRIPTION_TIER_PRO" if float(info.get("currentBalance") or 0) > 0 else "SUBSCRIPTION_TIER_NONE",
                "paidComputeUnitsBalance": info.get("currentBalance", 0),
                "consumptionRateHourly": info.get("consumptionRateHourly", 0),
                "assignmentsCount": info.get("assignmentsCount", 0),
                "eligibleGpus": info.get("eligibleGpus", []),
            }
        return self._json(self._request("GET", "/v1beta/subscription"))

    def list_runtimes(self):
        if self._is_cli_tunnel():
            out = []
            for a in self._tunnel_assignments():
                endpoint = a.get("endpoint")
                if not endpoint:
                    continue
                variant_num = a.get("variant")
                variant = "VARIANT_GPU" if variant_num in (1, "GPU", "VARIANT_GPU") else ("VARIANT_TPU" if variant_num in (2, "TPU", "VARIANT_TPU") else "VARIANT_CPU")
                shape_num = a.get("machineShape")
                shape = "SHAPE_HIGHMEM" if shape_num in (1, "HIGH_RAM", "SHAPE_HIGHMEM") else "SHAPE_STANDARD"
                out.append(
                    {
                        "name": f"runtimes/{endpoint}",
                        "runtimeSpec": {
                            "variant": variant,
                            "accelerator": str(a.get("accelerator") or "NONE"),
                            "shape": shape,
                        },
                        "version": "colab-cli-tunnel",
                    }
                )
            return out
        # Never return Google's raw connectionInfo to routes.
        response = self._request("GET", "/v1beta/runtimes")
        if response.status_code != 200:
            raise ColabError(f"Colab 运行时列表响应不完整（HTTP {response.status_code}）")
        headers = getattr(response, "headers", None) or {}
        payload = self._json(response)
        rows = self._extract_complete_list_from_payload(
            headers,
            payload,
            list_key="runtimes",
            context="Colab 运行时列表",
            allow_missing_list_key=True,
        )
        return [
            {"name": r.get("name"), "runtimeSpec": r.get("runtimeSpec"), "version": r.get("version")}
            for r in rows
        ]

    def _lease(self, runtime_id, request_id=None, job_id=None):
        for lease in self.store.list("leases", self.owner_id):
            if (
                lease.get("runtime_id") != runtime_id
                or not lease.get("owner_job_id")
                or not lease.get("request_id")
            ):
                continue
            if request_id and request_id != lease.get("request_id"):
                continue
            if job_id and job_id != lease.get("owner_job_id"):
                continue
            if lease.get("provider", "colab") != "colab":
                continue
            return lease
        raise ColabError("拒绝操作未登记归属的 runtime；不会触碰用户手动运行时")

    def _owned(self, handle):
        if not isinstance(handle, dict) or handle.get("owner_id") != self.owner_id:
            raise ColabError("Colab 租约归属无效")
        runtime = handle.get("runtime_name", "")
        parsed_runtime_id = parse_strict_runtime_name(runtime)
        if not parsed_runtime_id or handle.get("runtime_id") != parsed_runtime_id:
            raise ColabError("Colab runtime 句柄无效")
        return self._lease(
            parsed_runtime_id, handle.get("request_id"), handle.get("owner_job_id") or handle.get("job_id")
        )

    @staticmethod
    def _spec(spec):
        if not isinstance(spec, dict):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        if "key" in spec:
            if spec.get("eligible") is not True:
                raise ColabError("runtimeSpec 必须明确标记 eligible=true")
            key = spec["key"]
        else:
            key = spec
        if not isinstance(key, dict):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        key = dict(key)
        key["variant"] = _VARIANT_ALIASES.get(key.get("variant"), key.get("variant"))
        key["shape"] = _SHAPE_ALIASES.get(key.get("shape"), key.get("shape"))
        if (
            set(key) != {"variant", "accelerator", "shape"}
            or key.get("variant") not in _VARIANTS
            or key.get("shape") not in _SHAPES
        ):
            raise ColabError("runtimeSpec 必须来自当前账户真实可用规格的 key")
        if not isinstance(key.get("accelerator"), str) or not key["accelerator"]:
            raise ColabError("runtimeSpec accelerator 无效")
        return dict(key)

    @staticmethod
    def _operation_failure(operation):
        """Summarise a failed allocation Operation by its code and ErrorInfo reason only.

        Google's message and metadata may contain credentials or notebook data.
        """
        error = operation.get("error")
        if not error:
            return {}
        reason = next((d.get("reason") for d in error.get("details") or []
                       if d.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"), None)
        return {"allocation_failed": True,
                "allocation_error_code": error.get("code", "unknown"),
                "allocation_failure_reason": reason}

    @staticmethod
    def _raise_operation_error(operation):
        failure = GoogleColabProvider._operation_failure(operation)
        if failure:
            raise ColabAllocationError(failure["allocation_error_code"], failure["allocation_failure_reason"])

    def _persist_create_metadata(self, lease: dict, patch: dict) -> None:
        """Only the original persisted create attempt may accept response metadata."""
        self.store.update_lease_cas(
            lease["id"], self.owner_id, patch,
            expected_version=lease.get("version"),
            expected_create_attempts=int(lease.get("create_attempts") or 0),
            expected_create_identity=(lease.get("create_rpc_token"), lease.get("create_attempts"),
                                      lease.get("create_rpc_instance_id")),
        )

    def _create_tunnel(self, lease: dict, key: dict, request_uuid: uuid.UUID, request_id: str, runtime_id: str) -> dict:
        nbh = _uuid_to_web_safe_base64(request_uuid)
        params: dict[str, str] = {"nbh": nbh}
        variant = key.get("variant")
        accelerator = str(key.get("accelerator") or "NONE").upper()
        shape = key.get("shape")
        if variant == "VARIANT_GPU":
            params["variant"] = "GPU"
            if accelerator and accelerator != "NONE":
                params["accelerator"] = accelerator
        elif variant == "VARIANT_TPU":
            params["variant"] = "TPU"
            if accelerator and accelerator != "NONE":
                params["accelerator"] = accelerator
        if shape == "SHAPE_HIGHMEM" and accelerator not in _HIGH_MEM_ONLY_ACCELERATORS:
            params["shape"] = "hm"

        first = self._tunnel_json(self._tunnel_request("GET", "/tun/m/assign", params=params))
        if "endpoint" in first and isinstance(first.get("runtimeProxyInfo"), dict):
            assigned = first
        else:
            xsrf = first.get("token")
            if not isinstance(xsrf, str) or not xsrf:
                raise ColabError("Colab 隧道分配未返回 XSRF 令牌")
            assigned = self._tunnel_json(
                self._tunnel_request(
                    "POST",
                    "/tun/m/assign",
                    params=params,
                    headers={"X-Goog-Colab-Token": xsrf},
                )
            )
        endpoint = assigned.get("endpoint")
        proxy = assigned.get("runtimeProxyInfo") or {}
        proxy_url = proxy.get("url")
        proxy_token = proxy.get("token")
        if not (isinstance(endpoint, str) and endpoint and isinstance(proxy_url, str) and proxy_url and isinstance(proxy_token, str) and proxy_token):
            raise ColabError("Colab 隧道分配响应缺少 endpoint 或代理连接信息")
        ttl = int(proxy.get("tokenExpiresInSeconds") or 3600)
        op_name = f"operations/tunnel-{runtime_id}"
        metadata = {
            "operation_name": op_name,
            "operation_done": True,
            "requested_spec": key,
            "tunnel_endpoint": endpoint,
            "tunnel_nbh": nbh,
            "proxy_url": proxy_url.rstrip("/"),
            "encrypted_proxy_token": encrypt_secret(proxy_token),
            "proxy_token_expires_at": time.time() + max(60, ttl - 60),
        }
        self._persist_create_metadata(lease, metadata)
        return {
            "runtime_name": "runtimes/" + runtime_id,
            "operation_name": op_name,
            "tunnel_endpoint": endpoint,
            "tunnel_nbh": nbh,
            "connection_metadata": metadata,
            "ready": True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "owner_id": self.owner_id,
        }

    def create(self, spec, request_id, job_id, runtime_id):
        if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id):
            raise ColabError("runtimeId 必须是字母开头的 RFC1034 标识")
        try:
            request_uuid = uuid.UUID(request_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ColabError("Colab requestId 必须是 UUID4") from exc
        if request_uuid.version != 4:
            raise ColabError("Colab requestId 必须是 UUID4")
        lease = self._lease(runtime_id, request_id, job_id)
        key = self._spec(spec)
        if self._is_cli_tunnel(lease):
            return self._create_tunnel(lease, key, request_uuid, request_id, runtime_id)
        response = self._request(
            "POST",
            "/v1beta/runtimes",
            params={"requestId": request_id, "runtimeId": runtime_id},
            json={"runtimeSpec": key},
        )
        operation = self._json(response)
        name = operation.get("name")
        if not isinstance(name, str) or not _OPERATION.fullmatch(name):
            raise ColabError("创建响应缺少合法 Operation，保留既定 runtimeId 待协调")
        self._persist_create_metadata(
            lease,
            {"operation_name": name, "operation_done": operation.get("done") is True, "requested_spec": key},
        )
        self._raise_operation_error(operation)
        return {
            "runtime_name": "runtimes/" + runtime_id,
            "operation_name": name,
            "ready": operation.get("done") is True,
            "runtime_id": runtime_id,
            "request_id": request_id,
            "owner_id": self.owner_id,
        }

    def _operation(self, handle, lease):
        name = handle.get("operation_name") or lease.get("operation_name")
        if not name:
            return None
        if not _OPERATION.fullmatch(name):
            raise ColabError("Operation 名称无效")
        response = self._request("GET", "/v1/" + name)
        if response.status_code == 404:
            # Missing operation is not proof of runtime deletion; inspect runtime below.
            return None
        return self._json(response)

    def _sync_tunnel_lease(self, lease: dict) -> dict | None:
        local_norm = self._normalize_tunnel_endpoint(lease.get("tunnel_endpoint"))
        assignments = self._tunnel_assignments()
        matched = next(
            (a for a in assignments if self._normalize_tunnel_endpoint(a.get("endpoint")) == local_norm),
            None,
        )
        if matched is None:
            return None
        proxy = matched.get("runtimeProxyInfo") or {}
        patch: dict = {}
        if isinstance(proxy.get("url"), str) and proxy["url"]:
            patch["proxy_url"] = proxy["url"].rstrip("/")
        if isinstance(proxy.get("token"), str) and proxy["token"]:
            patch["encrypted_proxy_token"] = encrypt_secret(proxy["token"])
            ttl = int(proxy.get("tokenExpiresInSeconds") or 3600)
            patch["proxy_token_expires_at"] = time.time() + max(60, ttl - 60)
        return {**matched, "connection_metadata": patch}

    def inspect(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            endpoint = lease.get("tunnel_endpoint")
            if not endpoint:
                unknown = int(lease.get("create_attempts", 0)) > 0 and not lease.get("operation_done")
                return {
                    "ready": False,
                    "deleted": False,
                    "allocation_unknown": unknown,
                    "identity_unverifiable": True,
                    "remote_verified": False,
                }
            try:
                matched = self._sync_tunnel_lease(lease)
            except ColabError:
                return {
                    "ready": False,
                    "deleted": False,
                    "allocation_unknown": True,
                    "remote_verified": False,
                }
            if matched is None:
                return {"ready": False, "deleted": True, "remote_verified": True}
            return {
                "ready": True,
                "deleted": False,
                "runtime_name": handle["runtime_name"],
                "runtime_spec": lease.get("requested_spec") or {},
                "version": "colab-cli-tunnel",
                "connection_metadata": matched.get("connection_metadata") or {},
            }
        operation = self._operation(handle, lease)
        if operation is not None and operation.get("done") is not True:
            return {"ready": False, "deleted": False, "operation_pending": True}
        # A failed Operation still needs the runtime GET: a runtime that exists must be cleaned up.
        failure = self._operation_failure(operation) if operation is not None else {}
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if response.status_code == 404:
            # No Operation + a previously attempted create whose response was lost is ambiguous.
            op_name = handle.get("operation_name") or lease.get("operation_name")
            unknown = operation is None and (
                bool(op_name)
                or (int(lease.get("create_attempts", 0)) > 0 and not lease.get("operation_done"))
            )
            return {
                "ready": False,
                "deleted": not unknown,
                "allocation_unknown": unknown,
                "remote_verified": not unknown,
                **failure,
            }
        if failure:
            return {"ready": False, "deleted": False, **failure}
        runtime = self._json(response)
        return {
            "ready": bool(runtime.get("connectionInfo")),
            "deleted": False,
            "runtime_name": handle["runtime_name"],
            "runtime_spec": runtime.get("runtimeSpec"),
            "version": runtime.get("version"),
        }

    def connection_info(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            matched = self._sync_tunnel_lease(lease)
            fresh = {**lease, **((matched or {}).get("connection_metadata") or {})}
            url = fresh.get("proxy_url")
            enc_tok = fresh.get("encrypted_proxy_token")
            if not url or not enc_tok:
                raise ColabError("Colab connectionInfo 未就绪")
            return {
                "url": url,
                "token": decrypt_secret(enc_tok),
                "expire_time": fresh.get("proxy_token_expires_at"),
            }
        response = self._request("GET", "/v1beta/" + handle["runtime_name"])
        info = self._json(response).get("connectionInfo", {})
        if not info.get("url") or not info.get("token"):
            raise ColabError("Colab connectionInfo 未就绪")
        return {"url": info["url"], "token": info["token"], "expire_time": info.get("expireTime")}

    def delete(self, handle):
        lease = self._owned(handle)
        if self._is_cli_tunnel(lease):
            endpoint = lease.get("tunnel_endpoint")
            if not endpoint:
                return False
            matched = self._sync_tunnel_lease(lease)
            if matched is None:
                return True
            raw_target = matched.get("endpoint") or endpoint
            unassign_target = (
                self._normalize_tunnel_endpoint(raw_target)
                if "://" in str(raw_target)
                else str(raw_target).strip().rstrip("/")
            )
            unassign_get = self._tunnel_request("GET", f"/tun/m/unassign/{unassign_target}")
            if unassign_get.status_code == 404:
                return self._sync_tunnel_lease(lease) is None
            xsrf = self._tunnel_json(unassign_get).get("token")
            if not isinstance(xsrf, str) or not xsrf:
                raise ColabError("Colab 隧道释放缺少 XSRF 令牌")
            self._tunnel_request(
                "POST",
                f"/tun/m/unassign/{unassign_target}",
                headers={"X-Goog-Colab-Token": xsrf},
            )
            return self._sync_tunnel_lease(lease) is None
        operation = self._operation(handle, lease)
        if operation is not None and operation.get("done") is not True:
            # Deleting a missing runtime during a still-pending create is not release confirmation.
            return False
        # A failed completed Operation may still have allocated transient state; GET/DELETE regardless.
        current = self._request("GET", "/v1beta/" + handle["runtime_name"])
        if current.status_code == 404:
            op_name = handle.get("operation_name") or lease.get("operation_name")
            if (
                operation is None
                and not lease.get("operation_done")
                and (bool(op_name) or int(lease.get("create_attempts", 0)) > 0)
            ):
                # Reconcile lost create response by looking for only our deterministic runtime metadata.
                response = self._request("GET", "/v1/operations")
                if response.status_code == 404:
                    raw_ops = []
                else:
                    if response.status_code != 200:
                        return False
                    ops_headers = getattr(response, "headers", None) or {}
                    try:
                        ops_payload = self._json(response)
                        raw_ops = self._extract_complete_list_from_payload(
                            ops_headers,
                            ops_payload,
                            list_key="operations",
                            context="Colab 操作列表",
                        )
                    except ColabError:
                        return False
                matches = [
                    o
                    for o in raw_ops
                    if (o.get("metadata") or {}).get("runtime") == handle["runtime_name"]
                ]
                if not matches or any(o.get("done") is not True for o in matches):
                    return False
            return True
        self._request("DELETE", "/v1beta/" + handle["runtime_name"], params={"allowMissing": "true"})
        return self._request("GET", "/v1beta/" + handle["runtime_name"]).status_code == 404

    def close(self):
        if self._owns_client:
            self.client.close()
