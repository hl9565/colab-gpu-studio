"""Shared zero-DB identity canonicalization and handle validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit

_RUNTIME_ID = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUNTIME = re.compile(r"^runtimes/([a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)$")
_MOCK_RUNTIME_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")

_ALIAS_TUNNEL_SCHEMES = frozenset({"colab", "tunnel", "endpoint", "colab-tunnel"})
_HTTP_TUNNEL_SCHEMES = frozenset({"http", "https"})


def canonicalize_tunnel_endpoint(raw: object) -> str:
    """Canonicalize a Colab tunnel endpoint URI or bare token, raising ValueError on ambiguity."""
    if not isinstance(raw, str):
        raise ValueError("Colab 隧道 endpoint 身份无效")
    cleaned = raw.strip()
    if not cleaned or any(ch.isspace() for ch in cleaned) or "?" in cleaned or "#" in cleaned:
        raise ValueError("Colab 隧道 endpoint 格式异常，无法安全比对")
    cleaned = cleaned.rstrip("/")
    if not cleaned:
        raise ValueError("Colab 隧道 endpoint 格式异常，无法安全比对")
    if "://" in cleaned:
        parsed = urlsplit(cleaned)
        scheme = parsed.scheme.lower()
        if parsed.query or parsed.fragment or "@" in parsed.netloc:
            raise ValueError("Colab 隧道 endpoint 包含未规范化的 URL/scheme，无法安全比对")
        if scheme in _ALIAS_TUNNEL_SCHEMES:
            if parsed.netloc and not parsed.path.strip("/") and ":" not in parsed.netloc:
                cleaned = parsed.netloc
            else:
                raise ValueError("Colab 隧道 endpoint 包含未规范化的 URL/scheme，无法安全比对")
        elif scheme in _HTTP_TUNNEL_SCHEMES:
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError("Colab 隧道 endpoint 端口无效，无法安全比对") from exc
            if not parsed.hostname or parsed.path.strip("/"):
                raise ValueError("Colab 隧道 endpoint 包含未规范化的 URL/scheme，无法安全比对")
            host = parsed.hostname.casefold()
            if (scheme == "https" and port in (None, 443)) or (scheme == "http" and port in (None, 80)):
                return f"{scheme}://{host}"
            return f"{scheme}://{host}:{port}"
        else:
            raise ValueError("Colab 隧道 endpoint 包含未规范化的 URL/scheme，无法安全比对")
    if "/" in cleaned or ":" in cleaned:
        raise ValueError("Colab 隧道 endpoint 路径不规范，无法安全比对")
    return cleaned.casefold()


def normalize_tunnel_endpoint(raw: object) -> str | None:
    """Normalize a tunnel endpoint using the canonical rule, returning None when invalid."""
    try:
        return canonicalize_tunnel_endpoint(raw)
    except ValueError:
        return None


def parse_strict_runtime_name(raw_name: object) -> str | None:
    """Strictly parse `runtimes/<id>` matching Google Colab provider rules without trimming."""
    if not isinstance(raw_name, str):
        return None
    match = _RUNTIME.fullmatch(raw_name)
    return match.group(1) if match else None


def extract_runtime_name_id(raw_name: object) -> str | None:
    """Extract canonical runtime ID from `runtimes/<id>` or resource path ending in `/runtimes/<id>`."""
    if not isinstance(raw_name, str):
        return None
    val = raw_name.strip()
    if not val:
        return None
    if val.startswith("runtimes/"):
        tail = val[len("runtimes/"):].strip("/")
    elif "/runtimes/" in val:
        _, tail = val.rsplit("/runtimes/", 1)
        tail = tail.strip("/")
    else:
        return None
    if not tail or "/" in tail or any(ch.isspace() for ch in tail):
        return None
    return tail


def is_valid_runtime_handle(
    handle: object,
    *,
    job_id: str | None = None,
    owner_id: str | None = None,
    lease_runtime_id: str | None = None,
    instance_runtime_id: str | None = None,
    provider: str | None = None,
) -> bool:
    """Validate handle identity, strict Google runtime naming, runtime_id consistency, owner/job, and endpoints."""
    if not isinstance(handle, Mapping) or not handle:
        return False

    raw_runtime_name = handle.get("runtime_name")
    raw_runtime_id = handle.get("runtime_id")
    raw_tunnel_ep = handle.get("tunnel_endpoint")
    raw_ep = handle.get("endpoint")
    # A mock flag in a handle cannot relax a Colab account's identity contract.
    if provider == "colab" or raw_tunnel_ep is not None or raw_ep is not None:
        parsed = parse_strict_runtime_name(raw_runtime_name)
        if parsed is None or raw_runtime_id != parsed:
            return False

    has_name = raw_runtime_name is not None
    has_rt_id = raw_runtime_id is not None

    name_rt_id: str | None = None
    is_mock_name = False
    if has_name:
        if (
            not isinstance(raw_runtime_name, str)
            or not raw_runtime_name
            or raw_runtime_name != raw_runtime_name.strip()
            or any(ch.isspace() for ch in raw_runtime_name)
        ):
            return False
        if "/" in raw_runtime_name or raw_runtime_name.lower().startswith("runtimes"):
            name_rt_id = parse_strict_runtime_name(raw_runtime_name)
            if name_rt_id is None:
                return False
        elif raw_runtime_name.startswith("mock-runtime-"):
            is_mock_name = True
            suffix = raw_runtime_name[len("mock-runtime-"):]
            if not suffix or not _MOCK_RUNTIME_ID.fullmatch(suffix):
                return False
            name_rt_id = suffix
        else:
            if not _RUNTIME_ID.fullmatch(raw_runtime_name):
                return False
            name_rt_id = raw_runtime_name

    rt_id_clean: str | None = None
    if has_rt_id:
        if (
            not isinstance(raw_runtime_id, str)
            or not raw_runtime_id
            or raw_runtime_id != raw_runtime_id.strip()
            or any(ch.isspace() for ch in raw_runtime_id)
            or "/" in raw_runtime_id
        ):
            return False
        if provider == "colab" and not is_mock_name and not handle.get("mock"):
            if not _RUNTIME_ID.fullmatch(raw_runtime_id):
                return False
        elif not _MOCK_RUNTIME_ID.fullmatch(raw_runtime_id):
            return False
        rt_id_clean = raw_runtime_id
        if name_rt_id is not None and rt_id_clean != name_rt_id:
            return False

    norm_tunnel_ep: str | None = None
    if raw_tunnel_ep is not None:
        norm_tunnel_ep = normalize_tunnel_endpoint(raw_tunnel_ep)
        if norm_tunnel_ep is None:
            return False
    norm_ep: str | None = None
    if raw_ep is not None:
        norm_ep = normalize_tunnel_endpoint(raw_ep)
        if norm_ep is None:
            return False
    if norm_tunnel_ep is not None and norm_ep is not None and norm_tunnel_ep != norm_ep:
        return False
    has_endpoint = norm_tunnel_ep is not None or norm_ep is not None

    if not (has_name or has_rt_id or has_endpoint):
        return False

    if owner_id is not None:
        h_owner = handle.get("owner_id")
        if h_owner not in (None, "") and h_owner != owner_id:
            return False
    if job_id is not None:
        h_job = handle.get("job_id")
        h_owner_job = handle.get("owner_job_id")
        if h_job not in (None, "") and h_job != job_id:
            return False
        if h_owner_job not in (None, "") and h_owner_job != job_id:
            return False

    lease_rt_clean = lease_runtime_id.strip() if isinstance(lease_runtime_id, str) and lease_runtime_id.strip() else None
    inst_rt_clean = (
        instance_runtime_id.strip() if isinstance(instance_runtime_id, str) and instance_runtime_id.strip() else None
    )
    if lease_rt_clean and inst_rt_clean and lease_rt_clean != inst_rt_clean:
        return False

    effective_rt = rt_id_clean or name_rt_id
    if effective_rt is not None:
        if lease_rt_clean and effective_rt != lease_rt_clean:
            return False
        if inst_rt_clean and effective_rt != inst_rt_clean:
            return False
    return True


def row_has_positive_usage(row: object) -> bool:
    """Return True if a usage or instance_usage_segment row records positive billing/duration evidence."""
    data = getattr(row, "data", None) if not isinstance(row, Mapping) else row
    if not isinstance(data, Mapping):
        data = {}
    for attr_or_key in (
        "actual_units",
        "estimated_units",
        "units",
        "units_actual",
        "units_estimate",
        "billing_units",
        "billed_units",
        "ccu_used",
        "ccu",
        "duration_seconds",
        "duration_minutes",
        "total_duration_seconds",
        "total_duration_minutes",
        "runtime_seconds",
        "runtime_minutes",
        "elapsed_seconds",
        "elapsed_minutes",
        "seconds",
        "minutes",
    ):
        raw = getattr(row, attr_or_key, None) if not isinstance(row, Mapping) else row.get(attr_or_key)
        if raw is None and isinstance(data, Mapping):
            raw = data.get(attr_or_key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            if float(raw) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False
