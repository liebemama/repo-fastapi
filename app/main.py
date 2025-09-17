import asyncio
import contextlib
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging_ import setup_logging
from app.routes import plugins as plugins_routes
from app.routes.workflow import router as workflow_router
from app.runtime.model_pool import get_model_pool


# Import the plugin registry helper from your loader.
# If it's not present for any reason, we fall back to an empty list.
try:
    from app.plugins.loader import list_plugins  # returns dict{name: manifest} or list[str]
except Exception:  # pragma: no cover - safety net

    def list_plugins():  # type: ignore
        return {}


logger = logging.getLogger(__name__)

# ========================
# Lifespan handler
# ========================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start a lightweight background sweeper for the model pool during app lifetime.
    """
    pool = get_model_pool()

    async def sweeper():
        while True:
            pool.sweep_idle()
            await asyncio.sleep(60)

    task = asyncio.create_task(sweeper())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ========================
# Application init
# ========================

settings = get_settings()
setup_logging()
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Static files
# app/main.py
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(settings.SAMPLES_DIR)), name="samples")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestIDMiddleware)

# Register error handlers
register_exception_handlers(app)

# ========================
# Routes
# ========================


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": settings.APP_NAME})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/env")
def env():
    return settings.summary()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(str(settings.STATIC_DIR / "favicon.ico"))


# Routers
app.include_router(plugins_routes.router, tags=["plugins"])
app.include_router(workflow_router, prefix="/workflow", tags=["workflow"])

# ========================
# OpenAPI enrichment
# ========================


def _collect_plugins_and_tasks():
    """
    Returns (plugin_names, all_tasks) from the loader's registry.
    - plugin_names: sorted list of plugin ids
    - all_tasks: sorted list of distinct task names (if manifests declare "tasks")
    """
    try:
        registry = list_plugins()
    except Exception as e:
        logger.warning("Could not fetch plugin list for OpenAPI: %s", e)
        return [], []

    plugin_names = []
    all_tasks_set = set()

    if isinstance(registry, dict):
        plugin_names = list(registry.keys())
        for manifest in registry.values():
            tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
            if isinstance(tasks, (list | tuple | set)):
                all_tasks_set.update(tasks)
    elif isinstance(registry, (list | tuple | set)):
        plugin_names = list(registry)
    else:
        # Unknown type; ignore gracefully
        pass

    plugin_names = sorted(set(plugin_names))
    all_tasks = sorted(all_tasks_set)
    return plugin_names, all_tasks


def custom_openapi():
    """
    Generate the OpenAPI schema once, then cache it.
    Additionally, inject enum lists for `plugin` (and optionally `task`)
    properties across all schemas so Swagger shows real options.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=getattr(settings, "APP_NAME", "App"),
        version=str(getattr(settings, "VERSION", "0.1.0")),
        routes=app.routes,
    )

    try:
        components = schema.get("components", {}).get("schemas", {})
        plugin_names, all_tasks = _collect_plugins_and_tasks()

        if components and plugin_names:
            for s in components.values():
                props = s.get("properties", {})
                # Inject enum for "plugin"
                p = props.get("plugin")
                if isinstance(p, dict) and p.get("type") == "string":
                    p["enum"] = plugin_names

                # (Optional) Inject enum for "task" if you have a global list
                t = props.get("task")
                if isinstance(t, dict) and t.get("type") == "string" and all_tasks:
                    t["enum"] = all_tasks

    except Exception:
        logger.exception("Failed to inject plugin/task enums into OpenAPI")

    app.openapi_schema = schema
    return app.openapi_schema


# Replace FastAPI's default OpenAPI generator with our custom one
app.openapi = custom_openapi
