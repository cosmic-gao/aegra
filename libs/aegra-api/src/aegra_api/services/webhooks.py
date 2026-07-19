"""Outbound webhook delivery for run completion.

Fires a single POST at each run's terminal state (success/error/interrupted),
mirroring LangGraph Platform. Adds bounded retry with full-jitter backoff, a
per-attempt timeout, optional Standard-Webhooks-style HMAC-SHA256 signing, and
SSRF hardening (private/loopback/link-local/reserved IPs blocked unless
explicitly allowed).
"""

import hashlib
import hmac
import ipaddress
import json
import socket
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from aegra_api.settings import settings
from aegra_api.utils.url import redact_url

logger = structlog.getLogger(__name__)


class _WebhookAttemptFailed(Exception):
    """Internal signal that one delivery attempt failed and should be retried."""


class WebhookValidationError(ValueError):
    """Raised when a webhook URL fails scheme/host or SSRF validation."""


def validate_webhook_url(value: str | None) -> str | None:
    """Validate a webhook URL, returning it unchanged (or None).

    Requires an http(s) scheme and a host, and — unless
    ``WEBHOOK_ALLOW_PRIVATE_IPS`` — a host that does not resolve to a private,
    loopback, link-local, or reserved address (SSRF guard).
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise WebhookValidationError("webhook must use http or https scheme")
    if not parsed.hostname:
        raise WebhookValidationError("webhook must include a host")
    if not settings.webhook.WEBHOOK_ALLOW_PRIVATE_IPS and _resolves_to_private(parsed.hostname):
        raise WebhookValidationError("webhook host resolves to a private or reserved address")
    return value


def _resolves_to_private(host: str) -> bool:
    """True when *host* is or resolves to a private/reserved IP.

    Guards SSRF to internal services and the cloud metadata endpoint. Fails
    closed: a resolution error is treated as private (blocked).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def _sign(secret: str, timestamp: str, body: bytes) -> str:
    """Standard-Webhooks-style HMAC-SHA256 over ``{timestamp}.{body}``."""
    signed = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


async def _post_once(client: httpx.AsyncClient, url: str, body: bytes, safe_url: str) -> None:
    """POST once; raise ``_WebhookAttemptFailed`` on transport error or non-2xx.

    Signs per-attempt with a fresh timestamp so retries carry a valid signature.
    """
    headers = {"Content-Type": "application/json"}
    if settings.webhook.WEBHOOK_SIGNING_SECRET:
        ts = str(int(time.time()))
        headers["Webhook-Signature"] = _sign(settings.webhook.WEBHOOK_SIGNING_SECRET, ts, body)
    try:
        resp = await client.post(url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("Webhook attempt failed", url=safe_url, error=str(exc))
        raise _WebhookAttemptFailed from exc
    if 200 <= resp.status_code < 300:
        return
    logger.warning("Webhook non-2xx", url=safe_url, status=resp.status_code)
    raise _WebhookAttemptFailed


async def deliver_webhook(url: str, payload: dict[str, Any]) -> bool:
    """POST *payload* to *url* with bounded retry + full-jitter backoff (tenacity).

    Returns True on a 2xx, False when attempts are exhausted. Never raises — a
    failing webhook must not affect run completion.
    """
    cfg = settings.webhook
    body = json.dumps(payload, default=str).encode()
    safe_url = redact_url(url)

    async with httpx.AsyncClient(timeout=cfg.WEBHOOK_TIMEOUT_SECONDS) as client:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max(1, cfg.WEBHOOK_MAX_ATTEMPTS)),
                wait=wait_random_exponential(multiplier=cfg.WEBHOOK_BACKOFF_BASE_SECONDS),
                retry=retry_if_exception_type(_WebhookAttemptFailed),
                reraise=True,
            ):
                with attempt:
                    await _post_once(client, url, body, safe_url)
            return True
        except _WebhookAttemptFailed:
            logger.error("Webhook delivery exhausted retries", url=safe_url, attempts=cfg.WEBHOOK_MAX_ATTEMPTS)
            return False
