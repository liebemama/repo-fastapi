# app/core/errors.py
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_405_METHOD_NOT_ALLOWED,
    HTTP_408_REQUEST_TIMEOUT,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_501_NOT_IMPLEMENTED,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from app.core.config import get_settings


log = logging.getLogger("errors")


def _wants_html(request: Request) -> bool:
    """
    Determine whether the client prefers HTML over JSON.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        bool: True if HTML is preferred, False otherwise.
    """
    fmt = request.query_params.get("format", "").lower()
    if fmt == "html":
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def _request_id(request: Request) -> str | None:
    """
    Extract request ID from request state if available.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        Optional[str]: Request ID or None.
    """
    return getattr(request.state, "request_id", None)


def _render(
    request: Request,
    status_code: int,
    message: str,
    *,
    code: int | None = None,
    details: Any = None,
    template_name: str = "error.html",
) -> HTMLResponse | JSONResponse:
    """
    Return an HTML or JSON response based on client preference.

    Args:
        request (Request): The incoming HTTP request.
        status_code (int): HTTP status code.
        message (str): Human-readable error message.
        code (Optional[int], optional): Custom error code. Defaults to None.
        details (Any, optional): Additional error details. Defaults to None.
        template_name (str, optional): Template to use for HTML. Defaults to "error.html".

    Returns:
        HTMLResponse | JSONResponse: Rendered error response.
    """
    settings = get_settings()
    payload: dict[str, Any] = {
        "code": code or status_code,
        "message": message,
        "details": details,
        "path": str(request.url.path),
        "method": request.method,
        "request_id": _request_id(request),
        "app_name": settings.APP_NAME,
        "env": settings.ENV,
    }

    if _wants_html(request):
        try:
            templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
            # New signature: TemplateResponse(request, name, context, ...)
            return templates.TemplateResponse(
                request,
                template_name,
                {
                    **payload,  # no "request" key needed anymore
                    "title": f"{status_code} – {message}",
                },
                status_code=status_code,
            )
        except TemplateNotFound:
            html = (
                f"<h1>{status_code} – {message}</h1>"
                f"<p><strong>Path:</strong> {payload['path']}</p>"
                f"<p><strong>Method:</strong> {payload['method']}</p>" + (f"<pre>{details}</pre>" if details else "")
            )
            return HTMLResponse(content=html, status_code=status_code)

    return JSONResponse(
        status_code=status_code,
        content={
            "code": payload["code"],
            "message": payload["message"],
            "details": payload["details"],
            "path": payload["path"],
            "method": payload["method"],
            "request_id": payload["request_id"],
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all custom exception handlers with the FastAPI app.

    Args:
        app (FastAPI): The FastAPI application instance.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exc(request: Request, exc: StarletteHTTPException):
        if exc.status_code == HTTP_404_NOT_FOUND:
            log.info("404 Not Found: %s (%s)", request.url.path, _request_id(request))
            return _render(request, HTTP_404_NOT_FOUND, "Not Found", code=404)

        if exc.status_code == HTTP_405_METHOD_NOT_ALLOWED:
            log.info("405 Method Not Allowed: %s %s", request.method, request.url.path)
            return _render(
                request,
                HTTP_405_METHOD_NOT_ALLOWED,
                "Method Not Allowed",
                code=405,
                details={"method": request.method},
            )

        if exc.status_code == HTTP_401_UNAUTHORIZED:
            log.warning("401 Unauthorized: %s", request.url.path)
            return _render(request, HTTP_401_UNAUTHORIZED, "Unauthorized", code=401)

        if exc.status_code == HTTP_403_FORBIDDEN:
            log.warning("403 Forbidden: %s", request.url.path)
            return _render(request, HTTP_403_FORBIDDEN, "Forbidden", code=403)

        if exc.status_code == HTTP_413_REQUEST_ENTITY_TOO_LARGE:
            log.warning("413 Payload Too Large on %s", request.url.path)
            return _render(
                request,
                HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "Payload Too Large",
                code=413,
            )

        if exc.status_code == HTTP_429_TOO_MANY_REQUESTS:
            log.warning("429 Too Many Requests on %s", request.url.path)
            return _render(
                request,
                HTTP_429_TOO_MANY_REQUESTS,
                "Too Many Requests",
                code=429,
            )

        log.error("HTTP %s: %s", exc.status_code, exc.detail)
        return _render(
            request,
            exc.status_code,
            str(exc.detail) if exc.detail else "HTTP Error",
            code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exc(request: Request, exc: RequestValidationError):
        log.debug("422 RequestValidationError on %s", request.url.path, exc_info=exc)
        return _render(
            request,
            HTTP_422_UNPROCESSABLE_ENTITY,
            "Validation error",
            code=422,
            details=exc.errors(),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_exc(request: Request, exc: ValidationError):
        log.debug("422 Pydantic ValidationError on %s", request.url.path, exc_info=exc)
        return _render(
            request,
            HTTP_422_UNPROCESSABLE_ENTITY,
            "Validation error",
            code=422,
            details=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def global_exc(request: Request, exc: Exception):
        """
        Catch-all handler for unexpected exceptions. Hides traceback in production.

        Args:
            request (Request): The incoming HTTP request.
            exc (Exception): The raised exception.

        Returns:
            HTMLResponse | JSONResponse: Rendered error response.
        """
        settings = get_settings()
        details = str(exc) if settings.ENV.lower() == "development" else None
        log.exception("500 Internal Server Error on %s", request.url.path)
        return _render(
            request,
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            code=500,
            details=details,
        )

    _ = (
        HTTP_400_BAD_REQUEST,
        HTTP_401_UNAUTHORIZED,
        HTTP_403_FORBIDDEN,
        HTTP_404_NOT_FOUND,
        HTTP_405_METHOD_NOT_ALLOWED,
        HTTP_408_REQUEST_TIMEOUT,
        HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        HTTP_422_UNPROCESSABLE_ENTITY,
        HTTP_429_TOO_MANY_REQUESTS,
        HTTP_500_INTERNAL_SERVER_ERROR,
        HTTP_501_NOT_IMPLEMENTED,
        HTTP_503_SERVICE_UNAVAILABLE,
    )
