"""No-proxy, no-redirect transport for the opt-in local-model application."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Request | None:
        raise ValueError("local_model_redirect_forbidden")


def loopback_transport(request: Request, timeout: float) -> bytes:
    url = urlsplit(request.full_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"} or url.username or url.password:
        raise ValueError("numeric_loopback_endpoint_required")
    client = build_opener(ProxyHandler({}), NoRedirect())
    with client.open(request, timeout=timeout) as response:
        return bytes(response.read())
