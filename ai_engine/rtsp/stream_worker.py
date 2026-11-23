"""Per-camera RTSP/video grabber worker with latest-frame slot and 10s JPEG ring buffer."""
from __future__ import annotations

from collections import deque
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import CameraConfig, ZoneConfig, mask_rtsp_url


def _clean_url(url: str) -> str:
    clean = str(url or "").strip().strip('"').strip("'")
    if clean.lower().startswith("file:///"):
        from urllib.parse import unquote
        clean = unquote(clean[8:])
    return clean


class CameraStreamWorker:
    """Dedicated background capture thread keeping latest frame + rolling 10s compressed ring."""

    def __init__(
        self,
        camera_cfg: CameraConfig,
        *,
        ring_buffer_sec: float = 10.0,
        ring_fps: float = 8.0,
        on_health_change: Optional[Callable[[Dict[str, object]], None]] = None,
    ):
        self._cfg = camera_cfg
        self._ring_fps = max(1.0, float(ring_fps))
        self._ring_min_dt = 1.0 / self._ring_fps
        maxlen = max(16, int(round(max(2.0, float(ring_buffer_sec)) * self._ring_fps)))
        self._ring: Deque[Tuple[float, bytes]] = deque(maxlen=maxlen)
        self._last_ring_ts = -1e9

        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._seq: int = 0
        self._loop_count: int = 0
        self._last_capture_wall: float = 0.0

        self._is_local_file: bool = os.path.isfile(_clean_url(camera_cfg.rtsp_url))
        self._clip_detection_done: bool = False
        self._restart_req_token: int = 0
        self._restart_ack_token: int = 0
        self._eof_last_frame: Optional[np.ndarray] = None
        self._eof_ring_snapshot: List[Tuple[float, bytes]] = []
        self._snap_cache: Dict[Tuple[int, int, bool], Tuple[int, Any, bytes]] = {}

        self._state: str = "OFFLINE"
        self._last_error: str = ""
        self._fps_est: float = 0.0
        self._motion_active: bool = False
        self._on_health_change = on_health_change

    @property
    def camera_id(self) -> str:
        return self._cfg.camera_id

    @property
    def loop_count(self) -> int:
        with self._lock:
            return self._loop_count

    @property
    def is_local_file(self) -> bool:
        with self._lock:
            return self._is_local_file

    @property
    def clip_detection_done(self) -> bool:
        with self._lock:
            return self._clip_detection_done

    def set_clip_detection_done(self, done: bool) -> None:
        with self._lock:
            self._clip_detection_done = bool(done)

    @property
    def restart_pending(self) -> bool:
        with self._lock:
            return self._restart_req_token != self._restart_ack_token

    def restart_from_start(self) -> None:
        """Schedule a clean rewind to frame 0 and re-enable Pass 0 detection for local videos."""
        with self._lock:
            self._restart_req_token += 1
            self._clip_detection_done = False
            self._loop_count = 0
            self._eof_last_frame = None
            self._eof_ring_snapshot = []
            self._snap_cache.clear()
            self._ring.clear()
            self._last_ring_ts = -1e9

    def pop_eof_context(self) -> Tuple[Optional[np.ndarray], List[Tuple[float, bytes]]]:
        """Return and clear the final frame + ring buffer snapshot saved right before clip rewind."""
        with self._lock:
            frame = self._eof_last_frame
            ring = self._eof_ring_snapshot
            self._eof_last_frame = None
            self._eof_ring_snapshot = []
            return frame, ring

    @property
    def config(self) -> CameraConfig:
        with self._lock:
            return self._cfg

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"rtsp-{self._cfg.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_sec: float = 3.0) -> None:
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)

    def update_config(self, new_cfg: CameraConfig) -> bool:
        """Update config; returns True if stream URL/enabled changed and thread should restart."""
        clean_new = _clean_url(new_cfg.rtsp_url)
        with self._lock:
            need_restart = (
                clean_new != _clean_url(self._cfg.rtsp_url)
                or bool(new_cfg.enabled) != bool(self._cfg.enabled)
            )
            self._cfg = new_cfg
            self._is_local_file = os.path.isfile(clean_new)
            if need_restart:
                self._clip_detection_done = False
                self._restart_ack_token = self._restart_req_token
                self._loop_count = 0
                self._eof_last_frame = None
                self._eof_ring_snapshot = []
                self._snap_cache.clear()
                self._ring.clear()
                self._last_ring_ts = -1e9
        if need_restart:
            self.stop()
            if new_cfg.enabled and clean_new:
                self.start()
        return need_restart

    def set_motion_active(self, active: bool) -> None:
        with self._lock:
            self._motion_active = bool(active)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0, self._seq
            return self._latest_frame.copy(), self._latest_ts, self._seq

    def _get_latest_frame_ref(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._latest_frame, self._latest_ts, self._seq

    def snapshot_ring_buffer(self) -> List[Tuple[float, bytes]]:
        with self._lock:
            return list(self._ring)

    def get_snapshot_jpeg(
        self,
        zones: Optional[List[ZoneConfig]] = None,
        *,
        overlays: bool = False,
        overlay_layers: Any = None,
        overlay_tracks: Optional[Sequence[Dict[str, Any]]] = None,
        overlay_poses: Optional[Sequence[Dict[str, Any]]] = None,
        overlay_bags: Optional[Sequence[Dict[str, Any]]] = None,
        overlay_motion_roi: Optional[Sequence[Any]] = None,
        preview_only: bool = False,
        quality: int = 85,
        max_width: int = 0,
        cache_key: Any = None,
    ) -> Optional[bytes]:
        from apex_vision.sdk.stream_worker import _draw_label_pill, _parse_overlay_layers

        q = max(40, min(95, int(quality)))
        max_w = max(0, min(3840, int(max_width)))
        active_layers = _parse_overlay_layers(bool(overlays), overlay_layers)
        has_overlays = bool(overlays and active_layers)
        cache_slot = (max_w, q, has_overlays, active_layers)

        raw_frame, _, seq = self._get_latest_frame_ref()
        if raw_frame is None:
            return None

        if cache_key is not None:
            with self._lock:
                cached = self._snap_cache.get(cache_slot)
                if cached is not None and cached[0] == seq and cached[1] == cache_key:
                    return cached[2]

        orig_h, orig_w = raw_frame.shape[:2]
        if max_w > 0 and orig_w > max_w:
            scale = max_w / float(orig_w)
            nw = max(2, int(round(orig_w * scale)))
            nh = max(2, int(round(orig_h * scale)))
            frame = cv2.resize(raw_frame, (nw, nh), interpolation=cv2.INTER_AREA)
            sx = nw / float(orig_w)
            sy = nh / float(orig_h)
        else:
            frame = raw_frame.copy() if has_overlays else raw_frame
            sx = 1.0
            sy = 1.0

        if has_overlays:
            layers_set = set(active_layers)
            h, w = frame.shape[:2]
            thick = max(2, int(round(w / 480.0)))
            font_px = max(11, min(17, int(round(w / 50.0))))
            if zones and ("shelf" in layers_set or "aisle" in layers_set):
                for z in zones:
                    role_norm = "aisle" if getattr(z, "role", "shelf") == "aisle" else "shelf"
                    if role_norm not in layers_set:
                        continue
                    rect = z.to_pixel_rect(w, h)
                    if rect is None:
                        continue
                    x, y, rw, rh = rect
                    color = (0, 140, 255) if role_norm == "shelf" else (25, 195, 255)
                    pill_bg = (14, 72, 165) if role_norm == "shelf" else (18, 95, 135)
                    cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)
                    _draw_label_pill(
                        frame,
                        f"{role_norm.upper()} · {z.name}",
                        x,
                        y,
                        pill_bg,
                        (255, 255, 255),
                        font_px=max(10, font_px - 1),
                        border_bgr=color,
                        above=True,
                    )

            if "botsort" in layers_set and overlay_tracks:
                for tr in overlay_tracks:
                    bb = tr.get("bbox")
                    if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                        continue
                    x = int(round(float(bb[0]) * sx))
                    y = int(round(float(bb[1]) * sy))
                    bw = int(round(float(bb[2]) * sx))
                    bh = int(round(float(bb[3]) * sy))
                    if bw <= 2 or bh <= 2:
                        continue
                    x1 = max(0, min(w - 2, x))
                    y1 = max(0, min(h - 2, y))
                    x2 = max(x1 + 2, min(w - 1, x + bw))
                    y2 = max(y1 + 2, min(h - 1, y + bh))

                    tid = int(tr.get("track_id", 0))
                    conf = float(tr.get("confidence") or 0.0)
                    phase = str(tr.get("phase") or "").upper()
                    armed = bool(tr.get("armed"))

                    if armed:
                        box_bgr = (68, 63, 244)
                        pill_bgr = (45, 22, 175)
                        txt_bgr = (255, 255, 255)
                        label = f"BOT-SORT #{tid} · ARMED"
                    elif phase in ("REACH", "TUCK", "BLADED", "CROUCH"):
                        box_bgr = (11, 158, 245)
                        pill_bgr = (10, 115, 195)
                        txt_bgr = (255, 255, 255)
                        label = f"BOT-SORT #{tid} · {phase}"
                    elif preview_only:
                        box_bgr = (248, 189, 56)
                        pill_bgr = (145, 98, 18)
                        txt_bgr = (255, 255, 255)
                        label = f"BOT-SORT #{tid}" if conf <= 0 else f"BOT-SORT #{tid} · {int(round(conf * 100))}%"
                    else:
                        box_bgr = (129, 185, 16)
                        pill_bgr = (72, 115, 12)
                        txt_bgr = (255, 255, 255)
                        label = f"BOT-SORT #{tid}" if conf <= 0 else f"BOT-SORT #{tid} · {int(round(conf * 100))}%"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_bgr, thick)
                    clen = max(10, min(x2 - x1, y2 - y1) // 4)
                    c_thick = thick + 1
                    for px, py, dx, dy in (
                        (x1, y1, 1, 1),
                        (x2, y1, -1, 1),
                        (x1, y2, 1, -1),
                        (x2, y2, -1, -1),
                    ):
                        cv2.line(frame, (px, py), (px + dx * clen, py), box_bgr, c_thick, cv2.LINE_AA)
                        cv2.line(frame, (px, py), (px, py + dy * clen), box_bgr, c_thick, cv2.LINE_AA)

                    _draw_label_pill(
                        frame,
                        label,
                        x1,
                        y1,
                        pill_bgr,
                        txt_bgr,
                        font_px=font_px,
                        border_bgr=box_bgr,
                        above=True,
                    )

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        jpg_bytes = buf.tobytes()
        if cache_key is not None:
            with self._lock:
                if self._seq == seq:
                    if len(self._snap_cache) >= 8 and cache_slot not in self._snap_cache:
                        self._snap_cache.pop(next(iter(self._snap_cache)), None)
                    self._snap_cache[cache_slot] = (seq, cache_key, jpg_bytes)
        return jpg_bytes

    def get_health(self) -> Dict[str, object]:
        with self._lock:
            age_ms = (
                int(max(0.0, (time.monotonic() - self._last_capture_wall) * 1000.0))
                if self._last_capture_wall > 0.0
                else -1
            )
            return {
                "camera_id": self._cfg.camera_id,
                "state": self._state,
                "fps": round(self._fps_est, 1),
                "motion_active": self._motion_active,
                "last_frame_age_ms": age_ms,
                "error": self._last_error,
                "is_local_file": self._is_local_file,
                "clip_detection_done": self._clip_detection_done,
                "loop_count": self._loop_count,
            }

    def _set_state(self, state: str, error: str = "") -> None:
        emit = False
        with self._lock:
            if state != self._state or (error and error != self._last_error):
                self._state = state
                self._last_error = error
                emit = True
            else:
                self._last_error = error
        if emit and self._on_health_change is not None:
            try:
                self._on_health_change(self.get_health())
            except Exception:
                pass

    def _run_loop(self) -> None:
        backoff_steps = (1.0, 2.0, 5.0, 10.0)
        fail_count = 0

        while not self._stop_evt.is_set():
            url = _clean_url(self._cfg.rtsp_url)
            if not self._cfg.enabled or not url:
                self._set_state("OFFLINE", "Camera disabled or RTSP URL empty")
                self._stop_evt.wait(1.0)
                continue

            is_file = os.path.isfile(url)
            if url.lower().startswith(("rtsp://", "rtsps://")):
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                fail_count += 1
                self._set_state("OFFLINE", f"Failed to open {mask_rtsp_url(url)}")
                wait_s = backoff_steps[min(fail_count - 1, len(backoff_steps) - 1)]
                self._stop_evt.wait(wait_s)
                continue

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            fail_count = 0
            with self._lock:
                self._is_local_file = is_file
            self._set_state("ONLINE", "")
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if is_file else 0.0
            frame_period = (1.0 / max(1.0, min(30.0, src_fps))) if src_fps > 0.0 else 0.0
            fps_window_t = time.monotonic()
            fps_frames = 0

            try:
                while not self._stop_evt.is_set():
                    t_loop = time.monotonic()
                    with self._lock:
                        req_tok = self._restart_req_token
                        restart_now = req_tok != self._restart_ack_token
                        if restart_now:
                            self._clip_detection_done = False
                            self._loop_count = 0
                            self._eof_last_frame = None
                            self._eof_ring_snapshot = []
                            self._ring.clear()
                            self._last_ring_ts = -1e9
                    if is_file and restart_now:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if is_file:
                            with self._lock:
                                self._eof_last_frame = self._latest_frame.copy() if self._latest_frame is not None else None
                                self._eof_ring_snapshot = list(self._ring)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                with self._lock:
                                    self._ring.clear()
                                    self._last_ring_ts = -1e9
                                    self._loop_count += 1
                        if not ret or frame is None:
                            self._set_state("RECONNECTING", f"Stream read error ({mask_rtsp_url(url)})")
                            break

                    wall_now = time.monotonic()
                    if is_file and src_fps > 0.0:
                        pos_frames = float(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                        ts = max(0.0, (pos_frames - 1.0) / src_fps) if pos_frames >= 1.0 else (pos_msec / 1000.0)
                    else:
                        ts = wall_now

                    fps_frames += 1
                    dt_fps = wall_now - fps_window_t
                    if dt_fps >= 2.0:
                        with self._lock:
                            self._fps_est = fps_frames / dt_fps
                        fps_frames = 0
                        fps_window_t = wall_now

                    with self._lock:
                        self._latest_frame = frame
                        self._latest_ts = ts
                        self._seq += 1
                        self._last_capture_wall = wall_now
                        if restart_now:
                            self._restart_ack_token = req_tok

                    if (wall_now - self._last_ring_ts) >= self._ring_min_dt:
                        ok_j, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        if ok_j:
                            with self._lock:
                                self._ring.append((ts, buf.tobytes()))
                            self._last_ring_ts = wall_now

                    if is_file and frame_period > 0.0:
                        elapsed = time.monotonic() - t_loop
                        if elapsed < frame_period:
                            self._stop_evt.wait(frame_period - elapsed)
            finally:
                cap.release()

            if not self._stop_evt.is_set():
                fail_count += 1
                wait_s = backoff_steps[min(fail_count - 1, len(backoff_steps) - 1)]
                self._stop_evt.wait(wait_s)
