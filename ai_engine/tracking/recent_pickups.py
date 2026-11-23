"""Cross-camera rolling pool of recent store shelf pickups (zero ReID).

Maintains a lightweight 2-minute TTL pool of clean person/merchandise crops
from any camera when a shopper reaches a shelf or removes merchandise. When a
downstream camera detects a shelf-free concealment motion ("Path B"), this pool
builds a compact labeled contact sheet so the VLM can visually compare clothing,
bags, and carried items across camera views in a single inference call.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def compute_upper_color_hist(crop_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Compute a normalized upper-torso HSV histogram for rush-hour tie-breaking.

    Returns ``None`` if the crop is missing or too small to isolate the torso.
    Never used as a hard identity filter — only as a soft ranking tie-breaker
    when more than ``max_tiles`` pickups are active simultaneously.
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return None
    if crop_bgr.ndim < 3 or crop_bgr.shape[2] < 3:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return None

    y1 = max(0, int(round(h * 0.12)))
    y2 = min(h, max(y1 + 4, int(round(h * 0.55))))
    x1 = max(0, int(round(w * 0.15)))
    x2 = min(w, max(x1 + 4, int(round(w * 0.85))))
    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).astype(np.float32).flatten()
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-6:
        return None
    return hist / norm


def _hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity in [0.0, 1.0] between two normalized color histograms."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    sim = float(np.dot(a, b))
    return max(0.0, min(1.0, sim))


def _encode_crop_jpeg(crop_bgr: np.ndarray, max_dim: int = 288, quality: int = 80) -> Optional[bytes]:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    resized = (
        cv2.resize(crop_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else crop_bgr
    )
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else None


@dataclass
class PickupEntry:
    camera_id: str
    track_id: int
    ts: float
    crop_jpeg: bytes
    shelf_zone: str = ""
    delta_score: float = 0.0
    upper_color_hist: Optional[np.ndarray] = None
    resolution: str = "pending"
    extra: Dict[str, Any] = field(default_factory=dict)


class RecentPickupsPool:
    """Thread-safe rolling store of recent shelf pickups across all cameras."""

    def __init__(self, ttl_sec: float = 120.0, max_entries: int = 24) -> None:
        self.ttl_sec = max(5.0, float(ttl_sec))
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._entries: List[PickupEntry] = []

    def prune(self, now_ts: float) -> None:
        with self._lock:
            self._prune_unlocked(float(now_ts))

    def _prune_unlocked(self, now_ts: float) -> None:
        self._entries = [
            e for e in self._entries
            if 0.0 <= (now_ts - e.ts) <= self.ttl_sec or (now_ts < e.ts and (e.ts - now_ts) <= 2.0)
        ]
        # Keep only up to max_entries newest
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.ts, reverse=True)
            self._entries = self._entries[: self.max_entries]

    def record_pickup(
        self,
        *,
        camera_id: str,
        track_id: int,
        ts: float,
        person_crop_bgr: Optional[np.ndarray] = None,
        crop_jpeg: Optional[bytes] = None,
        shelf_zone: str = "",
        delta_score: float = 0.0,
        upper_color_hist: Optional[np.ndarray] = None,
        resolution: str = "pending",
        dedup_sec: float = 6.0,
    ) -> Optional[PickupEntry]:
        now = float(ts)
        jpeg_bytes = crop_jpeg if crop_jpeg else (
            _encode_crop_jpeg(person_crop_bgr) if person_crop_bgr is not None else None
        )
        if not jpeg_bytes:
            return None

        hist = upper_color_hist if upper_color_hist is not None else compute_upper_color_hist(person_crop_bgr)
        entry = PickupEntry(
            camera_id=str(camera_id),
            track_id=int(track_id),
            ts=now,
            crop_jpeg=jpeg_bytes,
            shelf_zone=str(shelf_zone or "shelf"),
            delta_score=float(delta_score),
            upper_color_hist=hist,
            resolution=str(resolution or "pending"),
        )

        with self._lock:
            self._prune_unlocked(now)
            # Deduplicate rapid updates for the exact same (camera_id, track_id)
            replaced = False
            for idx, existing in enumerate(self._entries):
                if (
                    existing.camera_id == entry.camera_id
                    and existing.track_id == entry.track_id
                    and abs(now - existing.ts) <= dedup_sec
                ):
                    if existing.resolution == "carried_openly" and entry.resolution == "pending":
                        entry.resolution = "carried_openly"
                    if entry.upper_color_hist is None:
                        entry.upper_color_hist = existing.upper_color_hist
                    self._entries[idx] = entry
                    replaced = True
                    break
            if not replaced:
                self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries.sort(key=lambda e: e.ts, reverse=True)
                self._entries = self._entries[: self.max_entries]
        return entry

    def mark_resolution(
        self,
        camera_id: str,
        track_id: int,
        resolution: str,
        now_ts: Optional[float] = None,
    ) -> bool:
        """Update resolution of matching recent pickup (`carried_openly`, `returned`, `concealed`)."""
        cam_s = str(camera_id)
        tid = int(track_id)
        res = str(resolution or "").strip().lower()
        updated = False
        with self._lock:
            if now_ts is not None:
                self._prune_unlocked(float(now_ts))
            for idx in range(len(self._entries) - 1, -1, -1):
                if self._entries[idx].camera_id == cam_s and self._entries[idx].track_id == tid:
                    if res == "returned":
                        self._entries.pop(idx)
                    else:
                        self._entries[idx].resolution = res
                    updated = True
                    break
        return updated

    def get_active(
        self,
        now_ts: float,
        exclude_camera_track: Optional[Tuple[str, int]] = None,
    ) -> List[PickupEntry]:
        now = float(now_ts)
        with self._lock:
            self._prune_unlocked(now)
            items = [
                e for e in self._entries
                if e.resolution != "returned"
                and (exclude_camera_track is None or (e.camera_id, e.track_id) != exclude_camera_track)
            ]
            items.sort(key=lambda e: e.ts, reverse=True)
            return list(items)

    def build_reference_sheet(
        self,
        now_ts: float,
        query_hist: Optional[np.ndarray] = None,
        max_tiles: int = 6,
        tile_px: int = 240,
        exclude_camera_track: Optional[Tuple[str, int]] = None,
    ) -> Tuple[Optional[bytes], List[Dict[str, Any]]]:
        """Build a 1x2 / 2x2 / 2x3 JPEG contact sheet of recent store pickups.

        Returns ``(sheet_jpeg_bytes, selected_metadata_list)`` or ``(None, [])``
        when no active pickups exist inside the 2-minute window.
        """
        now = float(now_ts)
        cap = max(1, min(6, int(max_tiles)))
        candidates = self.get_active(now, exclude_camera_track=exclude_camera_track)
        if not candidates:
            return None, []

        scored: List[Tuple[float, PickupEntry, float]] = []
        if len(candidates) <= cap:
            for e in candidates:
                sim = _hist_similarity(query_hist, e.upper_color_hist)
                scored.append((e.ts, e, sim))
            scored.sort(key=lambda item: item[0], reverse=True)
        else:
            # Rush-hour soft tie-breaker: recency + upper-torso color similarity
            for e in candidates:
                age = max(0.0, now - e.ts)
                recency = max(0.0, 1.0 - (age / self.ttl_sec))
                sim = _hist_similarity(query_hist, e.upper_color_hist)
                rank = 0.60 * recency + 0.40 * sim
                scored.append((rank, e, sim))
            scored.sort(key=lambda item: (item[0], item[1].ts), reverse=True)

        selected = scored[:cap]
        n = len(selected)
        cols = 2 if n <= 4 else 3
        rows = 1 if n <= 2 else 2

        banner_h = 28
        label_h = 22
        t_w = max(160, int(tile_px))
        t_h = max(160, int(tile_px))
        canvas = np.full((banner_h + rows * t_h, cols * t_w, 3), (20, 24, 32), dtype=np.uint8)

        # Header banner
        cv2.rectangle(canvas, (0, 0), (cols * t_w, banner_h), (29, 36, 48), -1)
        cv2.line(canvas, (0, banner_h - 1), (cols * t_w, banner_h - 1), (56, 72, 96), 1)
        cv2.putText(
            canvas,
            f"RECENT STORE PICKUPS (LAST {int(self.ttl_sec)}s) - REF SHEET",
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 244, 248),
            1,
            cv2.LINE_AA,
        )

        meta: List[Dict[str, Any]] = []
        for idx, (_score, entry, sim) in enumerate(selected):
            r = idx // cols
            c = idx % cols
            x0 = c * t_w
            y0 = banner_h + r * t_h

            arr = np.frombuffer(entry.crop_jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            inner_h = t_h - label_h - 4
            inner_w = t_w - 4
            if img is not None and img.size > 0:
                ih, iw = img.shape[:2]
                scale = min(float(inner_w) / float(max(1, iw)), float(inner_h) / float(max(1, ih)))
                nw = max(1, int(round(iw * scale)))
                nh = max(1, int(round(ih * scale)))
                resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                px = x0 + 2 + (inner_w - nw) // 2
                py = y0 + label_h + 2 + (inner_h - nh) // 2
                canvas[py : py + nh, px : px + nw] = resized

            # Top label bar per tile
            is_open = entry.resolution == "carried_openly"
            strip_bg = (22, 85, 58) if is_open else (34, 42, 56)
            cv2.rectangle(canvas, (x0, y0), (x0 + t_w - 1, y0 + label_h), strip_bg, -1)

            age_s = max(0, int(round(now - entry.ts)))
            zone_short = (entry.shelf_zone or "shelf")[:14]
            tag = " OPEN" if is_open else ""
            label_text = f"#{idx + 1} {entry.camera_id} | {age_s}s ago | {zone_short}{tag}"
            cv2.putText(
                canvas,
                label_text[:40],
                (x0 + 6, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (245, 248, 252),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (x0, y0), (x0 + t_w - 1, y0 + t_h - 1), (56, 68, 88), 1)

            meta.append({
                "tile": idx + 1,
                "camera_id": entry.camera_id,
                "track_id": entry.track_id,
                "age_sec": round(max(0.0, now - entry.ts), 2),
                "shelf_zone": entry.shelf_zone,
                "delta_score": round(entry.delta_score, 3),
                "resolution": entry.resolution,
                "color_sim": round(sim, 3),
            })

        ok, out_buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None, []
        return bytes(out_buf), meta
