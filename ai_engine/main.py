"""CLI & FastAPI ASGI entrypoint for StoreGuard headless AI engine."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import threading
from typing import Optional

from fastapi import FastAPI
import uvicorn

from .api.routes import build_router
from .api.ws_events import EventBroadcaster
from .config import AppConfig, load_config, mask_rtsp_url
from .inference.gpu_scheduler import EngineOrchestrator

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")


def create_app(
    *,
    config_path: str | Path,
    base_dir: Optional[str | Path] = None,
    config_override: Optional[AppConfig] = None,
    solution_path: Optional[str | Path] = None,
) -> FastAPI:
    cfg_path = Path(config_path)
    sol_path = Path(solution_path) if solution_path is not None else (cfg_path.parent / "storeguard.apex")
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    broadcaster = EventBroadcaster()
    state: dict = {"orch": None, "server": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        broadcaster.bind_loop(asyncio.get_running_loop())
        cfg = config_override if config_override is not None else load_config(cfg_path)
        for cam in cfg.cameras:
            logging.getLogger("storeguard").info("Configured camera %s (%s) url=%s zones=%d", cam.camera_id, cam.name, mask_rtsp_url(cam.rtsp_url), len(cam.zones))
        orch = EngineOrchestrator(cfg, root_dir, broadcaster, solution_path=sol_path)
        state["orch"] = orch
        orch.start()
        try:
            yield
        finally:
            orch.stop()

    def trigger_shutdown() -> None:
        srv = state.get("server")
        if srv is not None:
            threading.Timer(0.1, lambda: setattr(srv, "should_exit", True)).start()

    app = FastAPI(title="StoreGuard AI Engine", version="1.0.0", lifespan=lifespan)
    app.state.storeguard = state
    app.include_router(
        build_router(
            get_orchestrator=lambda: state["orch"],
            get_broadcaster=lambda: broadcaster,
            get_config_path=lambda: cfg_path,
            trigger_shutdown=trigger_shutdown,
        )
    )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="StoreGuard Multi-Stream AI Engine (Apex Vision Runtime)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "runtime_data" / "cameras.json"))
    parser.add_argument("--solution", default=None, help="Path to multi-flow storeguard.apex solution file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = create_app(config_path=args.config, solution_path=args.solution)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    app.state.storeguard["server"] = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
