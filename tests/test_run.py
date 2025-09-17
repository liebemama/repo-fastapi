# tests/test_run.py
import socket

import uvicorn

from app.core.config import get_settings
from app.core.logging_ import setup_logging


def find_free_port(start=8000, tries=50):
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    setup_logging()
    s = get_settings()
    port = find_free_port(getattr(s, "PORT", 8000))
    uvicorn.run(
        "app.main:app",  # لكي يعمل reload بدون تحذير
        host="127.0.0.1",  # ← بدل 0.0.0.0
        port=port,
        reload=getattr(s, "RELOAD", False),
        workers=getattr(s, "WORKERS", 1),  # عامل واحد على GPU واحد
        log_level=getattr(s, "LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
