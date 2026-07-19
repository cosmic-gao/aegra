"""URL helpers shared across webhook delivery and cron storage."""

from urllib.parse import urlparse


def redact_url(url: str) -> str:
    """Strip userinfo from a URL so credentials never reach logs or clients."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if parsed.username is None and parsed.password is None:
        return url
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return parsed._replace(netloc=host).geturl()
