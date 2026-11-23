"""StoreGuard AI engine orchestrator powered natively by Apex Vision SDK flows."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..apex_bootstrap import ensure_apex_vision_loaded

ensure_apex_vision_loaded()

from apex_vision.sdk import ApexRuntimeEngine  # noqa: E402
from ..api.ws_events import EventBroadcaster
from ..config import AppConfig, ZoneConfig


def compute_aisle_crop(zones: List[ZoneConfig], w_img: int, h_img: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute union crop across aisle and shelf zones when they occupy <75% of full frame."""
    rects_all: List[Tuple[int, int, int, int]] = []
    has_aisle = False
    for z in zones:
        r = z.to_pixel_rect(w_img, h_img)
        if r is None:
            continue
        if z.role == "aisle":
            has_aisle = True
        rects_all.append(r)
    if not rects_all:
        return None

    if has_aisle:
        x1 = min(r[0] for r in rects_all)
        y1 = min(r[1] for r in rects_all)
        x2 = max(r[0] + r[2] for r in rects_all)
        y2 = max(r[1] + r[3] for r in rects_all)
        mw = max(12, int((x2 - x1) * 0.10))
        mh = max(12, int((y2 - y1) * 0.10))
        cx1, cy1 = max(0, x1 - mw), max(0, y1 - mh)
        cx2, cy2 = min(w_img, x2 + mw), min(h_img, y2 + mh)
    else:
        exp_x1: List[int] = []
        exp_y1: List[int] = []
        exp_x2: List[int] = []
        exp_y2: List[int] = []
        for rx, ry, rw, rh in rects_all:
            pad_x = max(120, int(rw * 0.90))
            pad_up = max(160, int(rh * 1.15))
            pad_down = max(80, int(rh * 0.45))
            exp_x1.append(max(0, rx - pad_x))
            exp_y1.append(max(0, ry - pad_up))
            exp_x2.append(min(w_img, rx + rw + pad_x))
            exp_y2.append(min(h_img, ry + rh + pad_down))
        cx1, cy1 = min(exp_x1), min(exp_y1)
        cx2, cy2 = max(exp_x2), max(exp_y2)

    cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
    if (cw * ch) >= 0.75 * float(max(1, w_img * h_img)):
        return None
    return (cx1, cy1, cw, ch)


class EngineOrchestrator(ApexRuntimeEngine):
    """StoreGuard multi-camera orchestrator executing every camera via Apex Vision `<ApexFlow>` pipelines."""

    def __init__(
        self,
        cfg: AppConfig,
        base_dir: str | Path,
        broadcaster: EventBroadcaster,
        *,
        solution_path: Optional[str | Path] = None,
    ):
        super().__init__(cfg, base_dir, broadcaster, solution_path=solution_path)
