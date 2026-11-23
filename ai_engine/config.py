"""Canonical configuration, atomic persistence, RTSP masking, and .apex importer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field, field_validator


def clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def clamp_rect_norm(rect: List[float]) -> List[float]:
    if len(rect) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x = clamp01(rect[0])
    y = clamp01(rect[1])
    w = clamp01(rect[2])
    h = clamp01(rect[3])
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]


class ZoneConfig(BaseModel):
    name: str
    role: Literal["shelf", "aisle"] = "shelf"
    rect_norm: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        s = str(v or "").strip()
        return s or "zone"

    @field_validator("rect_norm")
    @classmethod
    def _norm_bounds(cls, v: List[float]) -> List[float]:
        return clamp_rect_norm([float(x) for x in (v or [])])

    def to_pixel_rect(self, w_img: int, h_img: int) -> Tuple[int, int, int, int] | None:
        nx, ny, nw, nh = self.rect_norm
        if nw <= 0.0 or nh <= 0.0 or w_img <= 0 or h_img <= 0:
            return None
        x1 = max(0, min(w_img - 1, int(round(nx * w_img))))
        y1 = max(0, min(h_img - 1, int(round(ny * h_img))))
        rw = max(1, min(w_img - x1, int(round(nw * w_img))))
        rh = max(1, min(h_img - y1, int(round(nh * h_img))))
        return (x1, y1, rw, rh)


class CameraConfig(BaseModel):
    camera_id: str
    name: str = ""
    rtsp_url: str = ""
    sub_stream_url: str = ""
    enabled: bool = True
    dwell_sec: float = 3.0
    release_wait_sec: float = 3.5
    cooldown_sec: float = 12.0
    zones: List[ZoneConfig] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _default_name(cls, v: str, info) -> str:
        s = str(v or "").strip()
        return s or str(info.data.get("camera_id", "Camera"))


class VlmConfig(BaseModel):
    provider: str = "gemini"
    base_url: str = ""
    model_name: str = "gemini-2.5-flash"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_sec: float = 25.0
    max_concurrent: int = 3
    alert_threshold_pct: float = 65.0
    dry_run_confidence: float = 0.85
    min_request_interval_sec: Optional[float] = None
    use_native_video: bool = True
    video_fps: float = 2.0
    max_sequence_frames: int = 14


class EngineSettings(BaseModel):
    device: str = "auto"
    target_fps: int = 8
    worker_threads: int = 0
    idle_fps: float = 3.0
    batch_max_size: int = 8
    batch_max_wait_ms: float = 8.0
    use_tensorrt: bool = True
    decode_backend: str = "auto"
    ring_max_width: int = 1280
    clip_pre_sec: float = 6.0
    clip_post_sec: float = 4.0
    retention_days: int = 30
    max_evidence_gb: float = 20.0
    pose_conf: float = 0.25
    pose_iou: float = 0.50
    bag_conf: float = 0.25
    bag_iou: float = 0.45
    bag_infer_stride: int = 3
    context_classes: List[str] = Field(
        default_factory=lambda: [
            "backpack",
            "handbag",
            "suitcase",
            "tote bag",
            "shopping basket",
            "shopping cart",
            "cell phone",
            "smartphone",
            "bottle",
        ]
    )
    evidence_dir: str = "runtime_data/evidence"
    models_dir: str = "models"
    pose_model: str = "yolo11n-pose.onnx"
    bag_model: str = "yolo11n.onnx"
    vlm: VlmConfig = Field(default_factory=VlmConfig)


class AppConfig(BaseModel):
    engine: EngineSettings = Field(default_factory=EngineSettings)
    cameras: List[CameraConfig] = Field(default_factory=list)


def mask_rtsp_url(url: str) -> str:
    """Mask rtsp://user:password@host passwords before logging."""
    s = str(url or "").strip()
    return re.sub(r"(rtsp[s]?://[^:/@\s]+:)([^@/\s]+)(@)", r"\1***\3", s, flags=re.IGNORECASE)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.is_file():
        return AppConfig()
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return AppConfig.model_validate(raw)


def save_config_atomic(path: str | Path, cfg: AppConfig) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cfg.model_dump(), fh, indent=2)
    os.replace(tmp, p)
    return p


def parse_apex_xml(
    apex_path: str | Path,
    *,
    camera_id: str = "cam_01",
    ref_w: int = 1280,
    ref_h: int = 720,
) -> CameraConfig:
    """Extract RTSP stream URL and normalized shelf/aisle ROIs from an Apex Studio .apex file."""
    tree = ET.parse(str(apex_path))
    root = tree.getroot()
    rtsp_url = ""
    name = root.attrib.get("name", camera_id) or camera_id
    zones: List[ZoneConfig] = []

    for node in root.findall(".//Node"):
        tool_name = (node.attrib.get("tool_name") or "").strip()
        params: Dict[str, Any] = {}
        for p_el in node.findall("./Parameters/Parameter"):
            pname = p_el.attrib.get("name", "")
            ptype = p_el.attrib.get("type", "str")
            ptext = (p_el.text or "").strip()
            if ptype == "json" and ptext:
                try:
                    params[pname] = json.loads(ptext)
                except json.JSONDecodeError:
                    params[pname] = ptext
            else:
                params[pname] = ptext

        if tool_name == "Image Source" and not rtsp_url:
            rtsp_url = str(params.get("stream_url") or "")
        elif tool_name == "ROI Zone Manager":
            raw_zones = params.get("zones")
            if isinstance(raw_zones, list):
                rw_ref = max(1, int(ref_w))
                rh_ref = max(1, int(ref_h))
                for idx, z in enumerate(raw_zones):
                    if not isinstance(z, dict):
                        continue
                    zname = str(z.get("name") or f"zone_{idx}")
                    rect = z.get("rect")
                    if not isinstance(rect, list) or len(rect) < 4:
                        continue
                    x, y, w, h = [float(v) for v in rect[:4]]
                    if w <= 0 or h <= 0:
                        continue
                    role: Literal["shelf", "aisle"] = "shelf" if "shelf" in zname.lower() else "aisle"
                    zones.append(
                        ZoneConfig(
                            name=zname,
                            role=role,
                            rect_norm=[x / rw_ref, y / rh_ref, w / rw_ref, h / rh_ref],
                        )
                    )

    return CameraConfig(camera_id=camera_id, name=name, rtsp_url=rtsp_url, zones=zones)
