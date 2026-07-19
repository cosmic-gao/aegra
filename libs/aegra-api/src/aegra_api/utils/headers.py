"""Extract allowlisted request headers to inject into a run's config.

Mirrors LangGraph Platform's ``configurable_headers``: an ``http`` config
allowlist (fnmatch includes/excludes) decides which incoming headers ride into
``config['configurable']`` so graphs can read per-request context (tenant, auth
hints) without the server hardcoding header names.
"""

import contextvars
import fnmatch
from collections.abc import Iterable, Mapping

from aegra_api.config import load_http_config

_configurable_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "configurable_headers", default=None
)


def get_configurable_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return the subset of *headers* the http config allows into run config."""
    if not headers:
        return {}
    http_config = load_http_config()
    patterns = http_config.get("configurable_headers") if http_config else None
    if not patterns:
        return {}
    includes = patterns.get("includes", [])
    excludes = patterns.get("excludes", [])
    if not includes:
        return {}

    configurable: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if any(fnmatch.fnmatch(lowered, p.lower()) for p in excludes):
            continue
        if any(fnmatch.fnmatch(lowered, p.lower()) for p in includes):
            configurable[lowered] = value
    return configurable


def set_request_configurable_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> None:
    """Store the allowlisted headers for the current request (called by middleware)."""
    decoded = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}
    _configurable_headers.set(get_configurable_headers(decoded))


def current_configurable_headers() -> dict[str, str]:
    """Return a copy of the allowlisted headers captured for the current request."""
    return dict(_configurable_headers.get() or {})
