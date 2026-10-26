from __future__ import annotations

import hmac
import logging
import os
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


auth_logger = logging.getLogger("ohamlab.auth")
request_logger = logging.getLogger("ohamlab.request")

# These routes and their descendants require X-OhamLab-Key.
PROTECTED_PATHS = ("/api/agents", "/api/tasks", "/mcp")


def _expected_api_key() -> str:
    """Read the shared secret from the Space secret/environment."""
    return os.getenv("OHAM_MCP_API_KEY", "").strip()


def _authorized(request: Request) -> bool:
    """Require X-OhamLab-Key to match the configured MCP secret."""
    expected = _expected_api_key()
    supplied = request.headers.get("X-OhamLab-Key", "").strip()

    if not supplied:
        auth_logger.warning(
            "AUTH DENIED reason=missing_x_ohamlab_key method=%s path=%s",
            request.method, request.url.path,
        )
        return False

    if not expected:
        auth_logger.error(
            "AUTH DENIED reason=mcp_key_not_configured method=%s path=%s",
            request.method, request.url.path,
        )
        return False

    if not hmac.compare_digest(supplied, expected):
        auth_logger.warning(
            "AUTH DENIED reason=x_ohamlab_key_mismatch method=%s path=%s",
            request.method, request.url.path,
        )
        return False

    auth_logger.info(
        "AUTH ALLOWED method=%s path=%s scheme=custom-header",
        request.method, request.url.path,
    )
    return True


def _is_protected(path: str) -> bool:
    path = path.rstrip("/") or "/"
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_PATHS)


def install_auth_middleware(app: FastAPI) -> None:
    """Install request logging and X-OhamLab-Key protection on a FastAPI app."""
    @app.middleware("http")
    async def protect_api(request: Request, call_next):
        started = time.monotonic()
        path = request.url.path.rstrip("/") or "/"
        host = request.headers.get("host", "")
        client_ip = request.client.host if request.client else "unknown"
        protected = _is_protected(path)

        request_logger.info(
            "REQUEST START method=%s path=%s host=%r client_ip=%s "
            "forwarded_for=%r forwarded_proto=%r user_agent=%r",
            request.method, path, host, client_ip,
            request.headers.get("x-forwarded-for", ""),
            request.headers.get("x-forwarded-proto", ""),
            request.headers.get("user-agent", ""),
        )

        if protected and not _authorized(request):
            configured = bool(_expected_api_key())
            status_code = 401 if configured else 503
            detail = (
                "Missing or invalid X-OhamLab-Key header" if configured
                else "MCP authentication is not configured. Set OHAM_MCP_API_KEY in Space Secrets."
            )
            response = JSONResponse(status_code=status_code, content={"detail": detail})
        else:
            try:
                response = await call_next(request)
            except Exception:
                request_logger.exception(
                    "REQUEST ERROR method=%s path=%s host=%r client_ip=%s duration_ms=%.2f",
                    request.method, path, host, client_ip,
                    (time.monotonic() - started) * 1000,
                )
                raise

        request_logger.info(
            "REQUEST END method=%s path=%s host=%r client_ip=%s status=%d "
            "auth=%s duration_ms=%.2f",
            request.method, path, host, client_ip, response.status_code,
            "denied" if protected and response.status_code in (401, 503) else ("passed" if protected else "not_required"),
            (time.monotonic() - started) * 1000,
        )
        return response
