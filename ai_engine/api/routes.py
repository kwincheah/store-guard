"""FastAPI REST & WebSocket API router for StoreGuard AI Engine."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import AppConfig, ZoneConfig, load_config
from ..inference.gpu_scheduler import EngineOrchestrator
from .ws_events import EventBroadcaster


class ZonesUpdateRequest(BaseModel):
    zones: List[ZoneConfig]


class DeviceUpdateRequest(BaseModel):
    device: str


class SolutionLoadRequest(BaseModel):
    path: Optional[str] = None
    xml: Optional[str] = None


class SolutionSaveRequest(BaseModel):
    path: Optional[str] = None


class ParamUpdateRequest(BaseModel):
    value: Any


def build_router(
    *,
    get_orchestrator: Callable[[], EngineOrchestrator],
    get_broadcaster: Callable[[], EventBroadcaster],
    get_config_path: Callable[[], Path],
    trigger_shutdown: Callable[[], None],
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health():
        return get_orchestrator().get_health()

    @router.get("/cameras")
    def list_cameras():
        orch = get_orchestrator()
        return {"cameras": [c.model_dump() for c in orch.cfg.cameras]}

    @router.get("/cameras/{camera_id}/snapshot")
    def camera_snapshot(
        camera_id: str,
        overlays: bool = False,
        layers: Optional[str] = None,
        max_width: int = 0,
        quality: int = 85,
    ):
        mw = 0 if max_width <= 0 else max(160, min(3840, int(max_width)))
        q = max(40, min(95, int(quality)))
        orch = get_orchestrator()
        try:
            jpg, exists = orch.get_snapshot(
                camera_id,
                overlays=overlays,
                max_width=mw,
                quality=q,
                overlay_layers=layers,
            )
        except TypeError:
            jpg, exists = orch.get_snapshot(
                camera_id,
                overlays=overlays,
                max_width=mw,
                quality=q,
            )
        if not exists:
            raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_id}'")
        if jpg is None:
            raise HTTPException(status_code=503, detail=f"Camera '{camera_id}' has no decoded frame yet")
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/cameras/{camera_id}/mjpeg")
    async def camera_mjpeg(
        camera_id: str,
        overlays: bool = False,
        layers: Optional[str] = None,
        max_width: int = 640,
        quality: int = 75,
        fps: float = 8.0,
        max_frames: int = 0,
    ):
        orch = get_orchestrator()
        if camera_id not in getattr(orch, "_workers", {}):
            raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_id}'")
        mw = 0 if max_width <= 0 else max(160, min(3840, int(max_width)))
        q = max(40, min(95, int(quality)))
        period = 1.0 / max(1.0, min(15.0, float(fps)))
        limit = max(0, int(max_frames))

        async def _stream():
            sent = 0
            while True:
                t0 = time.monotonic()
                try:
                    jpg, exists = orch.get_snapshot(
                        camera_id,
                        overlays=overlays,
                        max_width=mw,
                        quality=q,
                        overlay_layers=layers,
                    )
                except TypeError:
                    jpg, exists = orch.get_snapshot(camera_id, overlays=overlays, max_width=mw, quality=q)
                if not exists:
                    break
                if jpg:
                    header = (
                        f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                    ).encode("ascii")
                    yield header + jpg + b"\r\n"
                    sent += 1
                    if limit > 0 and sent >= limit:
                        break
                rem = period - (time.monotonic() - t0)
                await asyncio.sleep(max(0.01, rem))

        return StreamingResponse(
            _stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @router.put("/cameras/{camera_id}/zones")
    def update_zones(camera_id: str, body: ZonesUpdateRequest):
        ok = get_orchestrator().update_camera_zones(camera_id, body.zones)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_id}'")
        return {"ok": True, "camera_id": camera_id, "zone_count": len(body.zones)}

    @router.post("/cameras/{camera_id}/rerun")
    def rerun_camera(camera_id: str):
        ok = get_orchestrator().rerun_camera_detection(camera_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Unknown camera '{camera_id}'")
        return {"ok": True, "camera_id": camera_id}

    @router.post("/cameras/reload")
    def reload_cameras(cfg: AppConfig | None = None):
        orch = get_orchestrator()
        new_cfg = cfg if cfg is not None else load_config(get_config_path())
        active = orch.reload_config(new_cfg)
        return {"ok": True, "active_streams": active, "total_cameras": len(new_cfg.cameras)}

    @router.put("/engine/device")
    def set_engine_device(body: DeviceUpdateRequest):
        return get_orchestrator().set_device(body.device)

    @router.post("/shutdown")
    def shutdown():
        trigger_shutdown()
        return {"ok": True}

    # --- Hikrobot VisionMaster-style Apex SDK Endpoints ---

    @router.get("/sdk/solution")
    def sdk_get_solution():
        return get_orchestrator().get_solution_info()

    @router.get("/sdk/solution/xml")
    def sdk_get_solution_xml():
        orch = get_orchestrator()
        return {
            "name": orch.solution.manager.name,
            "solution_path": str(orch.solution_path),
            "xml": orch.get_solution_xml(),
        }

    @router.post("/sdk/solution/load")
    def sdk_load_solution(body: SolutionLoadRequest):
        try:
            return get_orchestrator().load_solution(path=body.path, xml_text=body.xml)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load Apex solution: {exc}")

    @router.post("/sdk/solution/save")
    def sdk_save_solution(body: Optional[SolutionSaveRequest] = None):
        orch = get_orchestrator()
        saved = orch.save_solution(body.path if body else None)
        return {"ok": True, "solution_path": saved}

    @router.get("/sdk/procedures")
    def sdk_list_procedures():
        return {"procedures": get_orchestrator().list_procedures()}

    @router.get("/sdk/procedures/{name}")
    def sdk_get_procedure(name: str):
        try:
            return get_orchestrator().get_procedure_summary(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown procedure '{name}'")

    @router.post("/sdk/procedures/{name}/run")
    def sdk_run_procedure(name: str):
        try:
            return get_orchestrator().run_procedure_once(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown procedure '{name}'")

    @router.get("/sdk/procedures/{name}/params")
    def sdk_get_procedure_params(name: str):
        try:
            return get_orchestrator().get_procedure_summary(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown procedure '{name}'")

    @router.put("/sdk/procedures/{name}/params/{node_id}/{param}")
    def sdk_set_procedure_param(name: str, node_id: str, param: str, body: ParamUpdateRequest):
        try:
            return get_orchestrator().set_procedure_param(name, node_id, param, body.value)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        broadcaster = get_broadcaster()
        await broadcaster.register(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.unregister(websocket)
        except Exception:
            broadcaster.unregister(websocket)

    return router
