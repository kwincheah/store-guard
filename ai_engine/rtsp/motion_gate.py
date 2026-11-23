"""Lightweight CPU motion pre-gate to skip empty aisles."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import ZoneConfig


class MotionGate:
    """Downscaled gray-diff motion detector using ~0.35s reference window + tail hysteresis."""

    def __init__(
        self,
        *,
        diff_thresh: int = 16,
        ratio_thresh: float = 0.005,
        tail_sec: float = 3.0,
        ref_interval_sec: float = 0.35,
        sample_size: tuple[int, int] = (160, 90),
    ):
        self.diff_thresh = int(diff_thresh)
        self.ratio_thresh = float(ratio_thresh)
        self.tail_sec = float(tail_sec)
        self.ref_interval_sec = float(ref_interval_sec)
        self.sample_size = sample_size
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_ts: float = 0.0
        self._motion_until: float = 0.0
        self._last_motion_norm_box: Optional[tuple[float, float, float, float]] = None

    def reset(self) -> None:
        self._ref_gray = None
        self._ref_ts = 0.0
        self._motion_until = 0.0
        self._last_motion_norm_box = None

    def get_motion_crop(self, w_img: int, h_img: int, *, min_side: int = 300) -> Optional[tuple[int, int, int, int]]:
        """Return a zoomed square pixel crop around the latest motion region when beneficial."""
        if self._last_motion_norm_box is None or w_img <= 0 or h_img <= 0:
            return None
        nx, ny, nw, nh = self._last_motion_norm_box
        x1 = int(round(nx * w_img))
        y1 = int(round(ny * h_img))
        x2 = int(round((nx + nw) * w_img))
        y2 = int(round((ny + nh) * h_img))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        side = max(int(min_side), (x2 - x1) + 140, (y2 - y1) + 140)
        rx1 = max(0, min(max(0, w_img - side), cx - side // 2))
        ry1 = max(0, min(max(0, h_img - side), cy - side // 2))
        rw = min(w_img - rx1, side)
        rh = min(h_img - ry1, side)
        if rw <= 0 or rh <= 0 or (rw * rh) >= 0.65 * float(w_img * h_img):
            return None
        return (rx1, ry1, rw, rh)

    def check(
        self,
        frame_bgr: np.ndarray,
        now_ts: float,
        aisle_zones: Sequence[ZoneConfig] = (),
    ) -> bool:
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        sw, sh = self.sample_size
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._ref_gray is None or self._ref_gray.shape != gray.shape:
            self._ref_gray = gray
            self._ref_ts = now_ts
            self._motion_until = max(self._motion_until, now_ts + self.tail_sec)
            return True

        diff = cv2.absdiff(gray, self._ref_gray)
        if now_ts < self._ref_ts or (now_ts - self._ref_ts) >= self.ref_interval_sec:
            self._ref_gray = gray
            self._ref_ts = now_ts

        aisles = [z for z in aisle_zones if z.role == "aisle" and z.rect_norm[2] > 0 and z.rect_norm[3] > 0]
        motion_mask: Optional[np.ndarray] = None
        if aisles:
            mask = np.zeros((sh, sw), dtype=np.uint8)
            for z in aisles:
                rect = z.to_pixel_rect(sw, sh)
                if rect is not None:
                    x, y, rw, rh = rect
                    mask[y:y + rh, x:x + rw] = 1
            area = int(mask.sum())
            if area > 0:
                changed = int(np.count_nonzero((diff > self.diff_thresh) & (mask > 0)))
                ratio = changed / float(area)
                motion_mask = (diff > self.diff_thresh) & (mask > 0)
            else:
                motion_mask = diff > self.diff_thresh
                ratio = float(np.count_nonzero(motion_mask)) / float(sw * sh)
        else:
            motion_mask = diff > self.diff_thresh
            ratio = float(np.count_nonzero(motion_mask)) / float(sw * sh)

        if ratio >= self.ratio_thresh:
            self._motion_until = max(self._motion_until, now_ts + self.tail_sec)
            ys, xs = np.where(motion_mask)
            if len(xs) >= 12:
                self._last_motion_norm_box = (
                    float(xs.min()) / float(sw),
                    float(ys.min()) / float(sh),
                    float(xs.max() - xs.min() + 1) / float(sw),
                    float(ys.max() - ys.min() + 1) / float(sh),
                )

        return now_ts <= self._motion_until
