"""Per-camera behavioral state machine with multi-person localized shelf delta + FIFO candidate queue."""
from __future__ import annotations

from collections import deque
import threading
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import ZoneConfig
from .grid_builder import build_2x3_grid, extract_sequence_frames
from .recent_pickups import compute_upper_color_hist
from .shelf_delta import LocalizedShelfDeltaTracker

PixelRect = Tuple[int, int, int, int]

_PERSONAL_BAG_CLASSES = {"backpack", "handbag", "suitcase", "tote bag", "tote", "purse", "bag"}
_BENIGN_VESSEL_CLASSES = {"shopping basket", "shopping cart", "basket", "cart"}
_PHONE_CLASSES = {"cell phone", "smartphone", "phone"}


def _pt_in_any(px: float, py: float, rects: Sequence[PixelRect], pad: int = 0) -> bool:
    for rx, ry, rw, rh in rects:
        if (rx - pad) <= px <= (rx + rw + pad) and (ry - pad) <= py <= (ry + rh + pad):
            return True
    return False


def _in_aisle(bbox: Tuple[int, int, int, int], aisle_rects: Sequence[PixelRect]) -> bool:
    if not aisle_rects:
        return True
    x, y, bw, bh = bbox
    cx = x + bw * 0.5
    return _pt_in_any(cx, y + bh * 0.55, aisle_rects) or _pt_in_any(cx, y + bh * 0.92, aisle_rects)


def _derive_hr_rects(aisle_rects: List[PixelRect], shelf_rects: List[PixelRect], w: int, h: int) -> List[PixelRect]:
    if aisle_rects:
        return aisle_rects
    derived: List[PixelRect] = []
    for sx, sy, sw, sh in shelf_rects:
        px, py = max(80, int(sw * 0.85)), max(60, int(sh * 0.45))
        x1, y1 = max(0, sx - px), max(0, sy - py // 2)
        x2, y2 = min(w, sx + sw + px), min(h, sy + sh + py)
        derived.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
    return derived


def _boxes_close(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], m: int = 36) -> bool:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (x1 + w1 + m < x2 or x2 + w2 + m < x1 or y1 + h1 + m < y2 or y2 + h2 + m < y1)


def _encode_jpg(arr: np.ndarray, max_dim: int = 640, quality: int = 85) -> bytes:
    if arr is None or arr.size == 0:
        return b""
    ah, aw = arr.shape[:2]
    if max(ah, aw) > max_dim:
        s = max_dim / float(max(ah, aw))
        arr = cv2.resize(arr, (max(1, int(aw * s)), max(1, int(ah * s))))
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _crop_person(frame_bgr: np.ndarray, x: int, y: int, bw: int, bh: int, w_img: int, h_img: int) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0 or bw <= 0 or bh <= 0:
        return None
    pad_x = max(16, int(round(bw * 0.14)))
    pad_y = max(12, int(round(bh * 0.08)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + bw + pad_x)
    y2 = min(h_img, y + bh + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def _pose_cues(
    kps: List[Any],
    p_bbox: PixelRect,
    shelf_rects: List[PixelRect],
    bag_boxes: List[PixelRect],
    basket_boxes: List[PixelRect],
    phone_boxes: List[PixelRect],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "reach_side": "",
        "touched_shelf_idx": None,
        "wrist_pts": [],
        "tucked": False,
        "occluded": False,
        "in_core_or_bag": False,
        "in_core": False,
        "tight_core": False,
        "in_personal_bag": False,
        "wrist_open": False,
        "two_hands_torso": False,
        "in_benign_basket": False,
        "holding_phone": False,
        "norm_sh": None,
    }
    if not isinstance(kps, list) or len(kps) < 13:
        return out

    def pt(idx: int) -> Optional[Tuple[float, float]]:
        p = kps[idx]
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        conf = float(p[2]) if len(p) > 2 else 0.9
        x, y = float(p[0]), float(p[1])
        if conf < 0.30 or (x <= 0 and y <= 0):
            return None
        return (x, y)

    ls, rs = pt(5), pt(6)
    le, re = pt(7), pt(8)
    lw, rw = pt(9), pt(10)
    lh, rh = pt(11), pt(12)
    nose = pt(0)
    px, py, pw, ph = p_bbox

    torso_h = max(24.0, ph * 0.42)
    if (ls or rs) and (lh or rh):
        sy = float(np.mean([p[1] for p in (ls, rs) if p is not None]))
        hy = float(np.mean([p[1] for p in (lh, rh) if p is not None]))
        torso_h = max(20.0, abs(hy - sy))
    if ls and rs:
        out["norm_sh"] = abs(ls[0] - rs[0]) / torso_h

    sh_cx = float(np.mean([p[0] for p in (ls, rs) if p is not None])) if (ls or rs) else (px + pw * 0.5)
    sh_cy = float(np.mean([p[1] for p in (ls, rs) if p is not None])) if (ls or rs) else (py + ph * 0.28)
    waist_y = float(np.mean([p[1] for p in (lh, rh) if p is not None])) if (lh or rh) else (py + ph * 0.55)

    reach_sides: List[str] = []
    for side, wp in (("left", lw), ("right", rw)):
        if wp is None:
            continue
        out["wrist_pts"].append((int(round(wp[0])), int(round(wp[1]))))
        stretch = ((wp[0] - sh_cx) ** 2 + (wp[1] - sh_cy) ** 2) ** 0.5
        if shelf_rects:
            for s_idx, s_rect in enumerate(shelf_rects):
                rx, ry, rw_s, rh_s = s_rect
                if rx <= wp[0] <= rx + rw_s and ry <= wp[1] <= ry + rh_s:
                    reach_sides.append(side)
                    if out["touched_shelf_idx"] is None:
                        out["touched_shelf_idx"] = s_idx
                    break
        elif (
            not _pt_in_any(wp[0], wp[1], bag_boxes)
            and not _pt_in_any(wp[0], wp[1], basket_boxes)
            and wp[1] <= sh_cy + 0.55 * torso_h
            and stretch > max(0.90 * torso_h, 0.50 * pw)
            and abs(wp[0] - sh_cx) > max(0.85 * pw, 0.90 * torso_h)
        ):
            reach_sides.append(side)

    if reach_sides:
        out["reach_side"] = "+".join(reach_sides)

    torso_arms = 0
    for side, wp in (("left", lw), ("right", rw)):
        if wp is None:
            continue
        in_basket = _pt_in_any(wp[0], wp[1], basket_boxes, pad=10)
        on_phone = _pt_in_any(wp[0], wp[1], phone_boxes, pad=14)
        in_bag = _pt_in_any(wp[0], wp[1], bag_boxes)
        in_core = abs(wp[0] - sh_cx) <= max(28.0, 0.36 * pw) and (sh_cy - 10.0) <= wp[1] <= (waist_y + 0.25 * torso_h)
        tight_core = abs(wp[0] - sh_cx) <= max(12.0, 0.22 * pw) and (sh_cy - 10.0) <= wp[1] <= (waist_y + 0.25 * torso_h)

        if in_core or in_bag:
            torso_arms += 1
        elif not in_basket and not on_phone:
            out["wrist_open"] = True

        if side in reach_sides:
            out["wrist_open"] = True
            continue

        if in_basket and not in_bag:
            out["in_benign_basket"] = True
            continue
        if on_phone and not in_bag:
            out["holding_phone"] = True
        if in_bag:
            out["in_personal_bag"] = True
        if in_core and not on_phone:
            out["in_core"] = True
        if tight_core and not on_phone:
            out["tight_core"] = True
        if in_core or in_bag:
            out["in_core_or_bag"] = True
            out["tucked"] = True

    if torso_arms >= 2 or (torso_arms >= 1 and len(out["wrist_pts"]) < 2):
        out["two_hands_torso"] = True

    if shelf_rects or nose is not None:
        for ep, wp in ((le, lw), (re, rw)):
            if ep is not None and wp is None:
                if _pt_in_any(ep[0], ep[1], basket_boxes, pad=14) and not _pt_in_any(ep[0], ep[1], bag_boxes):
                    out["in_benign_basket"] = True
                    continue
                if abs(ep[0] - sh_cx) <= max(35.0, 0.45 * pw) and (sh_cy - 10.0) <= ep[1] <= (waist_y + 0.35 * torso_h):
                    out["occluded"] = True
                    break
    return out


class BehaviorGate:
    """Isolated per-camera behavioral state machine + rolling crop buffer."""

    BUFFER_SEC = 12.0
    SAMPLE_SEC = 0.35
    ACTION_SAMPLE_SEC = 0.25
    TUCK_WIN_SEC = 7.0
    MERCH_TTL_SEC = 60.0
    BAG_RECENT_TTL_SEC = 3.0
    CARRY_ESCALATION_HOLD_SEC = 0.40
    BASKET_LATCH_FRAMES = 3
    BAG_STUFF_HOLD_SEC = 0.50
    GARMENT_HOLD_SEC = 0.70
    CROUCH_HOLD_SEC = 0.60
    BLADE_DROP = 0.40
    CROUCH_DROP = 0.28
    ID_LOSS_GRACE_SEC = 0.8
    MIN_FRAMES = 4
    TILE_PX = 384

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        *,
        dwell_sec: float = 3.0,
        release_wait_sec: float = 3.5,
        cooldown_sec: float = 12.0,
        zones: Sequence[ZoneConfig] = (),
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.dwell_sec = float(dwell_sec)
        self.release_wait_sec = float(release_wait_sec)
        self.cooldown_sec = float(cooldown_sec)

        self._lock = threading.Lock()
        self._zones: List[ZoneConfig] = list(zones)
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._dispatch_queue: Deque[Tuple[np.ndarray, Dict[str, Any]]] = deque(maxlen=16)
        self._pickup_events: Deque[Dict[str, Any]] = deque(maxlen=32)
        self._prev_raw_ts: Optional[float] = None
        self._ts_offset: float = 0.0
        self._shelf_delta = LocalizedShelfDeltaTracker()
        self._clean_scene_jpg: bytes = b""
        self._clean_scene_ts: float = -1e9

    def pop_pickup_events(self) -> List[Dict[str, Any]]:
        """Drain recent shelf-pickup crop events captured by this camera."""
        with self._lock:
            items = list(self._pickup_events)
            self._pickup_events.clear()
            return items

    def mark_carried_openly(self, track_id: int, ttl_sec: float = 60.0) -> None:
        """Keep Path A open-carry merchandise memory active when VLM reports carried_openly."""
        with self._lock:
            st = self._tracks.get(int(track_id))
            if st is None:
                return
            now_ref = float(st["last_seen"])
            until = now_ref + max(5.0, float(ttl_sec))
            st["has_merchandise_until"] = max(float(st.get("has_merchandise_until") or 0.0), until)
            st["carried_openly_confirmed_until"] = until
            st["carry_escalation_ready"] = True
            st["carry_escalation_since"] = None
            st["carry_escalation_not_before"] = float(st.get("last_dispatch_ts") or now_ref) + 2.0

    def update_zones(self, zones: Sequence[ZoneConfig]) -> None:
        with self._lock:
            self._zones = list(zones)
            self._shelf_delta.reset()

    def update_timing(self, *, camera_name: str, dwell_sec: float, release_wait_sec: float, cooldown_sec: float, zones: Sequence[ZoneConfig]) -> None:
        with self._lock:
            zones_changed = [z.model_dump() for z in zones] != [z.model_dump() for z in self._zones]
            self.camera_name = camera_name
            self.dwell_sec = float(dwell_sec)
            self.release_wait_sec = float(release_wait_sec)
            self.cooldown_sec = float(cooldown_sec)
            self._zones = list(zones)
            if zones_changed:
                self._shelf_delta.reset()

    def has_armed_tracks(self) -> bool:
        with self._lock:
            return any(st.get("armed_ts") is not None for st in self._tracks.values()) or len(self._dispatch_queue) > 0

    def get_track_states(self) -> Dict[int, Dict[str, Any]]:
        """Return lightweight status snapshot per active track for live camera overlays."""
        with self._lock:
            return {
                int(tid): {
                    "phase": str(st.get("last_phase") or ""),
                    "armed": st.get("armed_ts") is not None,
                    "dwell": round(float(st.get("dwell") or 0.0), 1),
                    "in_zone": bool(st.get("in_zone")),
                }
                for tid, st in self._tracks.items()
            }

    def flush_armed_on_clip_end(self, frame_bgr: Optional[np.ndarray]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Drain queued candidates and finalize any still-armed tracks at local clip EOF, then reset gate."""
        with self._lock:
            out: List[Tuple[np.ndarray, Dict[str, Any]]] = list(self._dispatch_queue)
            self._dispatch_queue.clear()
            if frame_bgr is not None and frame_bgr.size > 0:
                zone_names = [z.name for z in self._zones]
                for st in list(self._tracks.values()):
                    if (
                        st.get("armed_ts") is not None
                        and not self._should_suppress_active_browse(st)
                        and float(st.get("last_seen", 0.0)) >= float(st.get("cooldown", 0.0))
                    ):
                        out.append(self._build_candidate(st, frame_bgr, "clip_end", zone_names))
            self._tracks.clear()
            self._shelf_delta.reset()
            self._clean_scene_jpg = b""
            self._clean_scene_ts = -1e9
            return out

    def reset_all(self) -> None:
        """Completely clear all tracks, buffers, timestamps, and shelf baselines."""
        with self._lock:
            self._tracks.clear()
            self._dispatch_queue.clear()
            self._pickup_events.clear()
            self._prev_raw_ts = None
            self._ts_offset = 0.0
            self._shelf_delta.reset()
            self._clean_scene_jpg = b""
            self._clean_scene_ts = -1e9

    def on_video_loop_restart(self) -> None:
        """Reset loop cooldowns and shelf baselines when a local video file rewinds to frame 0."""
        with self._lock:
            self._shelf_delta.reset()
            self._clean_scene_jpg = b""
            self._clean_scene_ts = -1e9
            for tid, st in list(self._tracks.items()):
                if st.get("armed_ts") is not None:
                    st["cooldown"] = 0.0
                    st["carried_openly_confirmed_until"] = None
                    st["has_merchandise_until"] = None
                    st["carry_escalation_ready"] = False
                    st["carry_escalation_since"] = None
                    st["carry_escalation_not_before"] = 0.0
                    st["bag_seen_ts"].clear()
                    st["bags"].clear()
                    st["episode_bags"].clear()
                else:
                    self._tracks.pop(tid, None)

    def _monotonic_ts(self, raw_ts: float) -> float:
        cur = float(raw_ts)
        if self._prev_raw_ts is not None and cur + 0.05 < self._prev_raw_ts:
            self._ts_offset += self._prev_raw_ts + (1.0 / 24.0)
        self._prev_raw_ts = cur
        return self._ts_offset + cur

    def _build_candidate(
        self,
        st: Dict[str, Any],
        frame_bgr: np.ndarray,
        cause: str,
        zone_names: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if st["samples"]:
            ts_l, jpg_l, _ = st["samples"][-1]
            st["samples"][-1] = (ts_l, jpg_l, "RELEASE")
        grid = build_2x3_grid(st, frame_bgr, self.TILE_PX)
        seq_frames = extract_sequence_frames(st, max_frames=14)
        samples = st.get("samples") or []
        first_ts = st.get("first_sample_raw_ts")
        if first_ts is None and samples:
            first_ts = max(0.0, float(samples[0][0]) - self._ts_offset)
        tid = int(st["id"])
        delta_status = str(st.get("shelf_delta_status") or "none")
        delta_score = float(st.get("shelf_delta_score") or 0.0)
        delta_confirmed = bool(st.get("shelf_delta_confirmed"))
        trigger_mode = str(st.get("trigger_mode") or "shelf_reach_and_delta")
        cand = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "track_id": tid,
            "bbox": list(st["bbox"]),
            "dwell_sec": round(st["dwell"], 2),
            "reach_side": st["reach_side"],
            "behaviors": list(st["reasons"]),
            "nearby_bags": sorted(st["bags"] | set(st.get("episode_bags") or ())),
            "context_vessels": sorted(st.get("vessels") or ()),
            "has_store_vessel": bool(st.get("has_store_vessel")),
            "zone_names": zone_names or ["full_frame"],
            "dispatch_cause": cause,
            "trigger_mode": trigger_mode,
            "shelf_delta_status": delta_status,
            "shelf_delta_score": delta_score,
            "shelf_delta_confirmed": delta_confirmed,
            "sequence_frames": seq_frames,
            "first_sample_ts": first_ts,
            "pre_reach_jpg": st.get("pre_reach_jpg") or b"",
            "clean_scene_jpg": self._clean_scene_jpg or b"",
            "union_crop_rect": st.get("union_crop_rect"),
            "upper_color_hist": st.get("upper_color_hist"),
            "reason_hint": (
                f"Track ID #{tid} mode={trigger_mode} dwell={st['dwell']:.1f}s cause={cause} "
                f"shelf_delta={delta_status}({delta_score:.2f}) "
                f"vessels={','.join(sorted(st.get('vessels') or ())) or 'none'}"
            ),
        }
        return grid, cand

    @staticmethod
    def _should_suppress_active_browse(st: Dict[str, Any]) -> bool:
        return (
            st.get("trigger_mode") == "shelf_reach_and_delta"
            and not st["shelf_delta_confirmed"]
            and not st.get("armed_bag_contact")
            and int(st.get("post_arm_reach_frames") or 0) >= 2
        )

    @staticmethod
    def _reset_armed_cycle(st: Dict[str, Any]) -> None:
        st["armed_ts"] = st["reach_ts"] = st["reach_peak_ts"] = st["tuck_ts"] = st["bladed_ts"] = st["crouch_ts"] = None
        st["last_conceal_ts"] = None
        st["bag_stuff_since"] = st["garment_hide_since"] = st["crouch_hide_since"] = None
        st["trigger_mode"] = "shelf_reach_and_delta"
        st["pickup_delta_logged"] = False
        st["post_arm_reach_frames"] = 0
        st["armed_bag_contact"] = False
        st["reasons"].clear()
        st["episode_bags"].clear()
        st["shelf_delta_status"] = "none"
        st["shelf_delta_score"] = 0.0
        st["shelf_delta_confirmed"] = False

    def step(
        self,
        frame_bgr: np.ndarray,
        raw_ts: float,
        tracks: List[Dict[str, Any]],
        bag_dets: List[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, {"tracks": 0, "gate_state": "NO_IMAGE"}

        with self._lock:
            h_img, w_img = frame_bgr.shape[:2]
            now = self._monotonic_ts(raw_ts)

            shelf_rects: List[PixelRect] = []
            shelf_zone_names: List[str] = []
            aisle_rects: List[PixelRect] = []
            zone_names: List[str] = []
            for z in self._zones:
                rect = z.to_pixel_rect(w_img, h_img)
                if rect is None:
                    continue
                zone_names.append(z.name)
                if z.role == "shelf":
                    shelf_rects.append(rect)
                    shelf_zone_names.append(z.name)
                else:
                    aisle_rects.append(rect)
            hr_rects = _derive_hr_rects(aisle_rects, shelf_rects, w_img, h_img)

            bags = [b for b in (bag_dets or []) if isinstance(b, dict) and isinstance(b.get("Box") or b.get("bbox"), (list, tuple))]
            bag_boxes: List[PixelRect] = [tuple(int(round(float(v))) for v in (b.get("Box") or b.get("bbox"))[:4]) for b in bags]
            bag_labels: List[str] = [str(b.get("Class", b.get("label", "bag"))).strip().lower() for b in bags]

            # Pass 1: associate vessels and compute pose cues for all active tracks
            prepared: List[Tuple[int, Dict[str, Any], PixelRect, Dict[str, Any], List[PixelRect], bool]] = []
            delta_inputs: List[Dict[str, Any]] = []
            seen = set()

            for tr in tracks or []:
                if not isinstance(tr, dict):
                    continue
                tid = int(tr.get("track_id", -1))
                bb = tr.get("bbox") or []
                if tid < 0 or len(bb) < 4:
                    continue
                x, y, bw, bh = [int(v) for v in bb[:4]]
                seen.add(tid)
                st = self._tracks.setdefault(
                    tid,
                    {
                        "id": tid, "last_seen": now, "fc": 0, "in_zone": False, "enter_ts": None, "exit_ts": None,
                        "dwell": 0.0, "samples": deque(maxlen=80), "last_samp": -1e9, "sh": deque(maxlen=40),
                        "zone_sh": deque(maxlen=40), "heights": deque(maxlen=40), "last_y2": None,
                        "reach_ts": None, "reach_peak_ts": None, "reach_side": "", "tuck_ts": None,
                        "bladed_ts": None, "crouch_ts": None, "armed_ts": None, "reasons": [], "peak_ov": b"",
                        "last_conceal_ts": None, "last_dispatch_ts": None, "last_phase": "",
                        "cooldown": 0.0, "bags": set(), "bag_seen_ts": {}, "episode_bags": set(),
                        "vessels": set(), "bbox": [x, y, bw, bh],
                        "last_touched_shelf_idx": None, "shelf_delta_status": "none",
                        "shelf_delta_score": 0.0, "shelf_delta_confirmed": False,
                        "basket_hits": 0, "basket_miss": 0, "has_store_vessel": False,
                        "wrist_was_open": False, "bag_stuff_since": None,
                        "garment_hide_since": None, "crouch_hide_since": None,
                        "last_center": None, "norm_speed": 0.0,
                        "has_merchandise_until": None, "carried_openly_confirmed_until": None,
                        "carry_escalation_ready": False, "carry_escalation_since": None, "carry_escalation_not_before": 0.0,
                        "last_pickup_recorded_ts": -1e9, "pickup_delta_logged": False,
                        "trigger_mode": "shelf_reach_and_delta", "upper_color_hist": None,
                        "pre_reach_jpg": b"", "pre_reach_ts": None, "post_arm_reach_frames": 0,
                        "armed_bag_contact": False,
                        "first_sample_ts": None, "union_crop_rect": None,
                    },
                )
                cx_t, cy_t = x + bw * 0.5, y + bh * 0.5
                prev_c = st.get("last_center")
                if prev_c is not None and (now - prev_c[0]) > 1e-4:
                    dt_c = now - prev_c[0]
                    dist_c = ((cx_t - prev_c[1]) ** 2 + (cy_t - prev_c[2]) ** 2) ** 0.5
                    st["norm_speed"] = (dist_c / dt_c) / max(40.0, float(bh))
                else:
                    st["norm_speed"] = 0.0
                st["last_center"] = (now, cx_t, cy_t)
                st["last_seen"] = now
                st["fc"] += 1
                st["bbox"] = [x, y, bw, bh]

                margin = max(30, int(bw * 0.25))
                near_bags: List[PixelRect] = []
                near_baskets: List[PixelRect] = []
                near_phones: List[PixelRect] = []
                near_all_vessels: List[PixelRect] = []
                for idx_b, bb_i in enumerate(bag_boxes):
                    if not _boxes_close((x, y, bw, bh), bb_i, margin):
                        continue
                    lbl = bag_labels[idx_b]
                    st["vessels"].add(lbl)
                    if lbl in _BENIGN_VESSEL_CLASSES:
                        near_baskets.append(bb_i)
                        near_all_vessels.append(bb_i)
                    elif lbl in _PHONE_CLASSES:
                        near_phones.append(bb_i)
                    elif lbl in _PERSONAL_BAG_CLASSES or "bag" in lbl:
                        near_bags.append(bb_i)
                        near_all_vessels.append(bb_i)
                        st["bag_seen_ts"][lbl] = now

                stale_bags = [lbl_b for lbl_b, ts_b in st["bag_seen_ts"].items() if (now - float(ts_b)) > self.BAG_RECENT_TTL_SEC]
                for lbl_b in stale_bags:
                    st["bag_seen_ts"].pop(lbl_b, None)
                st["bags"] = set(st["bag_seen_ts"].keys())
                if st["reach_ts"] is not None or st["armed_ts"] is not None:
                    st["episode_bags"].update(st["bags"])

                if near_baskets:
                    st["basket_hits"] = min(12, int(st["basket_hits"]) + 1)
                    st["basket_miss"] = 0
                    if st["basket_hits"] >= self.BASKET_LATCH_FRAMES:
                        st["has_store_vessel"] = True
                else:
                    st["basket_miss"] = int(st["basket_miss"]) + 1
                    if st["basket_miss"] >= 15:
                        st["has_store_vessel"] = False
                        st["basket_hits"] = 0

                in_hr = _in_aisle((x, y, bw, bh), hr_rects)
                cues = _pose_cues(tr.get("keypoints") or [], (x, y, bw, bh), shelf_rects, near_bags, near_baskets, near_phones)
                if cues["wrist_open"]:
                    st["wrist_was_open"] = True
                if cues["touched_shelf_idx"] is not None:
                    st["last_touched_shelf_idx"] = cues["touched_shelf_idx"]

                prepared.append((tid, st, (x, y, bw, bh), cues, near_all_vessels, in_hr))
                delta_inputs.append({
                    "tid": tid,
                    "bbox": (x, y, bw, bh),
                    "wrists": cues["wrist_pts"],
                    "reach_active": bool(cues["reach_side"]),
                    "touched_shelf_idx": cues["touched_shelf_idx"],
                })

            if not self._clean_scene_jpg or (len(prepared) == 0 and (now - self._clean_scene_ts) >= 5.0):
                clean_enc = _encode_jpg(frame_bgr)
                if clean_enc:
                    self._clean_scene_jpg = clean_enc
                    self._clean_scene_ts = now

            delta_map = self._shelf_delta.step(frame_bgr, now, shelf_rects, delta_inputs)

            # Pass 2: update track state machines, multi-shelf crops, and dispatch queue
            for tid, st, (x, y, bw, bh), cues, near_all_vessels, in_hr in prepared:
                d_res = delta_map.get(tid) or {}
                d_status = str(d_res.get("status") or "none")
                d_score = float(d_res.get("score") or 0.0)
                d_conf = bool(d_res.get("confirmed"))

                if d_conf:
                    st["shelf_delta_status"] = "confirmed"
                    st["shelf_delta_score"] = max(float(st.get("shelf_delta_score") or 0.0), d_score)
                    st["shelf_delta_confirmed"] = True
                elif not st.get("shelf_delta_confirmed") and d_status != "none":
                    st["shelf_delta_status"] = d_status
                    st["shelf_delta_score"] = d_score

                if in_hr:
                    st["in_zone"] = True
                    st["exit_ts"] = None
                    if st["enter_ts"] is None:
                        st["enter_ts"] = now
                    st["dwell"] = now - st["enter_ts"]
                else:
                    if st["exit_ts"] is None:
                        st["exit_ts"] = now
                    if (now - st["exit_ts"]) > 0.5:
                        if st["armed_ts"] is not None and now >= st["cooldown"]:
                            self._dispatch_queue.append(self._build_candidate(st, frame_bgr, "zone_exit", zone_names))
                            st["last_dispatch_ts"] = now
                            st["cooldown"] = now + self.cooldown_sec
                        st["in_zone"] = False
                        st["enter_ts"] = None
                        st["dwell"] = 0.0
                        st["zone_sh"].clear()
                        self._reset_armed_cycle(st)
                        self._shelf_delta.clear_episode(tid)

                mature = st["in_zone"] and st["dwell"] >= self.dwell_sec and st["fc"] >= self.MIN_FRAMES
                fast_shelf_ready = (
                    st["in_zone"]
                    and st["dwell"] >= min(self.dwell_sec, 0.5)
                    and st["fc"] >= 3
                    and bool(st["shelf_delta_confirmed"])
                    and not cues["in_benign_basket"]
                )
                reach_now = bool(cues["reach_side"])
                pre_reach_path_b = (
                    st["armed_ts"] is not None
                    and st.get("trigger_mode") == "shelf_free_tuck"
                    and st["reach_ts"] is None
                    and "bag_stuff" not in st["reasons"]
                    and "crouch_hide" not in st["reasons"]
                )
                if pre_reach_path_b and (reach_now or d_conf):
                    self._reset_armed_cycle(st)
                if reach_now or d_conf:
                    if reach_now and st["armed_ts"] is None:
                        if st["reach_ts"] is None:
                            st["reach_ts"] = now
                        st["reach_peak_ts"] = now
                        st["reach_side"] = cues["reach_side"]
                    st["has_merchandise_until"] = max(float(st.get("has_merchandise_until") or 0.0), now + self.MERCH_TTL_SEC)
                    need_pickup_log = (now - float(st["last_pickup_recorded_ts"])) >= 4.0 or (
                        d_conf and not st.get("pickup_delta_logged")
                    )
                    if need_pickup_log:
                        p_crop = _crop_person(frame_bgr, x, y, bw, bh, w_img, h_img)
                        if p_crop is not None:
                            lsi = st.get("last_touched_shelf_idx")
                            z_lbl = (
                                shelf_zone_names[int(lsi)]
                                if lsi is not None and 0 <= int(lsi) < len(shelf_zone_names)
                                else (shelf_zone_names[0] if shelf_zone_names else (zone_names[0] if zone_names else "aisle"))
                            )
                            self._pickup_events.append({
                                "camera_id": self.camera_id,
                                "track_id": tid,
                                "ts": float(raw_ts),
                                "person_crop_bgr": p_crop,
                                "shelf_zone": z_lbl,
                                "delta_score": float(st.get("shelf_delta_score") or d_score),
                                "upper_color_hist": compute_upper_color_hist(p_crop),
                            })
                            st["last_pickup_recorded_ts"] = now
                            if d_conf:
                                st["pickup_delta_logged"] = True

                raw_tucked = cues["tucked"] or cues["occluded"]
                phone_suppress = cues["holding_phone"] and not st["shelf_delta_confirmed"]
                effective_tuck = raw_tucked and not cues["in_benign_basket"] and not phone_suppress
                conc = bool(effective_tuck or len(cues["wrist_pts"]) < 2)

                pool = st["zone_sh"] if len(st["zone_sh"]) >= 4 else st["sh"]
                bladed_frame = (
                    (not reach_now or conc)
                    and conc
                    and not cues["in_benign_basket"]
                    and cues["norm_sh"] is not None
                    and len(pool) >= 3
                    and cues["norm_sh"] <= float(np.percentile(list(pool), 85)) * (1.0 - self.BLADE_DROP)
                )
                crouch_frame = False
                if (not reach_now or conc) and not cues["in_benign_basket"] and len(st["heights"]) >= 3:
                    bh_base = float(np.percentile(list(st["heights"]), 90))
                    y2 = y + bh
                    steady = st["last_y2"] is None or abs(y2 - st["last_y2"]) <= max(28.0, bh_base * 0.12)
                    crouch_frame = bh_base > 30.0 and bh <= bh_base * (1.0 - self.CROUCH_DROP) and steady

                tuck_now = bladed_now = crouch_now = False
                bag_stuff_hit = garment_hit = crouch_hide_hit = False
                has_merchandise_active = st["has_merchandise_until"] is not None and now <= float(st["has_merchandise_until"])
                cd_ok = now >= st["cooldown"]
                if (
                    not cd_ok
                    and st.get("carry_escalation_ready")
                    and has_merchandise_active
                    and st["armed_ts"] is None
                    and now >= float(st.get("carry_escalation_not_before") or 0.0)
                ):
                    strong_carry_cue = (
                        not reach_now
                        and not cues["in_benign_basket"]
                        and not phone_suppress
                        and (
                            cues["in_personal_bag"]
                            or (cues["occluded"] and bladed_frame)
                            or (crouch_frame and (cues["in_core"] or cues["in_personal_bag"] or cues["occluded"]))
                        )
                    )
                    if strong_carry_cue:
                        if st["carry_escalation_since"] is None:
                            st["carry_escalation_since"] = now
                        elif (now - float(st["carry_escalation_since"])) >= self.CARRY_ESCALATION_HOLD_SEC:
                            st["cooldown"] = now
                            cd_ok = True
                            st["carry_escalation_ready"] = False
                            st["carry_escalation_since"] = None
                    else:
                        st["carry_escalation_since"] = None
                elif st["armed_ts"] is not None or not has_merchandise_active:
                    st["carry_escalation_since"] = None
                fast_reach_tuck = (
                    st["in_zone"]
                    and st["dwell"] >= min(self.dwell_sec, 1.0)
                    and st["fc"] >= self.MIN_FRAMES
                    and st["reach_ts"] is not None
                    and not cues["in_benign_basket"]
                    and not phone_suppress
                    and (cues["in_personal_bag"] or (cues["in_core"] and cues["two_hands_torso"]))
                )

                if (mature or fast_shelf_ready or fast_reach_tuck) and st["armed_ts"] is None:
                    if reach_now:
                        if st["reach_ts"] is None:
                            st["reach_ts"] = now
                        st["reach_peak_ts"] = now
                        st["reach_side"] = cues["reach_side"]
                    dt_r = (now - st["reach_ts"]) if st["reach_ts"] is not None else -1.0
                    rec_r = st["reach_ts"] is not None and 0.15 <= dt_r <= self.TUCK_WIN_SEC

                    if rec_r and effective_tuck:
                        tuck_now = True
                        st["tuck_ts"] = now
                    if rec_r and bladed_frame and (effective_tuck or st["shelf_delta_confirmed"] or cues["in_personal_bag"]):
                        bladed_now = True
                        st["bladed_ts"] = now
                    if rec_r and crouch_frame:
                        crouch_now = True
                        st["crouch_ts"] = now

                    # Path A delayed open-carry fallback within 60s of pickup/reach
                    if has_merchandise_active and not rec_r and not reach_now:
                        if effective_tuck or bladed_frame or crouch_frame:
                            if effective_tuck:
                                tuck_now = True
                                st["tuck_ts"] = now
                            if bladed_frame and (effective_tuck or cues["occluded"]):
                                bladed_now = True
                                st["bladed_ts"] = now
                            if crouch_frame:
                                crouch_now = True
                                st["crouch_ts"] = now
                            st["trigger_mode"] = "carry_delayed"

                if not bladed_frame and not bladed_now and cues["norm_sh"] is not None:
                    st["sh"].append(float(cues["norm_sh"]))
                    if in_hr and not reach_now:
                        st["zone_sh"].append(float(cues["norm_sh"]))
                if not crouch_frame and not crouch_now:
                    st["heights"].append(float(bh))
                st["last_y2"] = float(y + bh)

                # Path B: Smart Shelf-Free Tuck (catches cross-cam concealment with zero ReID)
                mature_b = st["in_zone"] and st["dwell"] >= min(self.dwell_sec, 0.8) and st["fc"] >= self.MIN_FRAMES
                vessel_suppress = st["has_store_vessel"] and not cues["in_personal_bag"]
                path_b_eligible = (
                    mature_b
                    and cd_ok
                    and st["armed_ts"] is None
                    and st["reach_ts"] is None
                    and not reach_now
                    and not st["shelf_delta_confirmed"]
                    and not has_merchandise_active
                    and st["wrist_was_open"]
                    and not cues["in_benign_basket"]
                    and not vessel_suppress
                    and not phone_suppress
                )
                if path_b_eligible:
                    bag_cond = cues["in_personal_bag"] and not cues["holding_phone"]
                    slowed_down = float(st.get("norm_speed") or 0.0) <= 0.85
                    garment_cond = (
                        (cues["in_core"] or (cues["occluded"] and bladed_frame))
                        and slowed_down
                        and (cues["two_hands_torso"] or bladed_frame)
                        and not cues["holding_phone"]
                    )
                    crouch_cond = (
                        crouch_frame
                        and (cues["in_core"] or cues["in_personal_bag"] or cues["occluded"])
                        and not cues["holding_phone"]
                    )

                    if bag_cond:
                        if st["bag_stuff_since"] is None:
                            st["bag_stuff_since"] = now
                        elif (now - st["bag_stuff_since"]) >= self.BAG_STUFF_HOLD_SEC:
                            bag_stuff_hit = True
                    else:
                        st["bag_stuff_since"] = None

                    if garment_cond:
                        if st["garment_hide_since"] is None:
                            st["garment_hide_since"] = now
                        elif (now - st["garment_hide_since"]) >= self.GARMENT_HOLD_SEC:
                            garment_hit = True
                    else:
                        st["garment_hide_since"] = None

                    if crouch_cond:
                        if st["crouch_hide_since"] is None:
                            st["crouch_hide_since"] = now
                        elif (now - st["crouch_hide_since"]) >= self.CROUCH_HOLD_SEC:
                            crouch_hide_hit = True
                    else:
                        st["crouch_hide_since"] = None
                else:
                    st["bag_stuff_since"] = st["garment_hide_since"] = st["crouch_hide_since"] = None

                path_b_hit = bag_stuff_hit or garment_hit or crouch_hide_hit

                phase = (
                    "TUCK" if (tuck_now or bag_stuff_hit or garment_hit) else
                    "BLADED" if bladed_now else
                    "CROUCH" if (crouch_now or crouch_hide_hit) else
                    "PRE" if not (mature or fast_shelf_ready or mature_b) else
                    "REACH" if reach_now else
                    "POST" if st["armed_ts"] is not None else
                    "BROWSE"
                )
                def _append_sample(phase_lbl: str) -> None:
                    ux1, uy1, ux2, uy2 = x, y, x + bw, y + bh
                    for bx, by, bbw, bbh in near_all_vessels:
                        ux1, uy1, ux2, uy2 = min(ux1, bx), min(uy1, by), max(ux2, bx + bbw), max(uy2, by + bbh)

                    target_shelf: Optional[PixelRect] = None
                    lsi = st.get("last_touched_shelf_idx")
                    if lsi is not None and 0 <= int(lsi) < len(shelf_rects):
                        target_shelf = shelf_rects[int(lsi)]
                    elif shelf_rects:
                        pcx = x + bw * 0.5
                        target_shelf = min(shelf_rects, key=lambda r: abs((r[0] + r[2] * 0.5) - pcx))

                    if target_shelf is not None:
                        sx_t, sy_t, sw_t, sh_t = target_shelf
                        if bh < 0.35 * float(h_img):
                            reach_r = max(120, int(bh * 1.5))
                            sx1 = max(sx_t, x - reach_r)
                            sy1 = max(sy_t, y - reach_r // 2)
                            sx2 = min(sx_t + sw_t, x + bw + reach_r)
                            sy2 = min(sy_t + sh_t, y + bh + reach_r // 2)
                            if sx2 > sx1 and sy2 > sy1:
                                ux1 = min(ux1, sx1)
                                uy1 = min(uy1, sy1)
                                ux2 = max(ux2, sx2)
                                uy2 = max(uy2, sy2)
                        elif (sw_t * sh_t) <= 0.45 * float(w_img * h_img):
                            ux1 = min(ux1, sx_t)
                            uy1 = min(uy1, sy_t)
                            ux2 = max(ux2, sx_t + sw_t)
                            uy2 = max(uy2, sy_t + sh_t)
                        else:
                            ux1 = min(ux1, sx_t + sw_t // 3)
                            ux2 = max(ux2, sx_t + (2 * sw_t) // 3)
                            uy2 = max(uy2, sy_t + sh_t)

                    pad_x, pad_y = max(35, int((ux2 - ux1) * 0.25)), max(25, int((uy2 - uy1) * 0.12))
                    cx1, cy1 = max(0, ux1 - pad_x), max(0, uy1 - pad_y)
                    cx2, cy2 = min(w_img, ux2 + pad_x), min(h_img, uy2 + pad_y)
                    prev_u = st.get("union_crop_rect")
                    if prev_u is None:
                        st["union_crop_rect"] = (cx1, cy1, max(1, cx2 - cx1), max(1, cy2 - cy1))
                    else:
                        ux_p, uy_p, uw_p, uh_p = prev_u
                        nx1 = min(ux_p, cx1)
                        ny1 = min(uy_p, cy1)
                        nx2 = max(ux_p + uw_p, cx2)
                        ny2 = max(uy_p + uh_p, cy2)
                        st["union_crop_rect"] = (nx1, ny1, max(1, nx2 - nx1), max(1, ny2 - ny1))

                    jpg = _encode_jpg(frame_bgr[cy1:cy2, cx1:cx2])
                    if jpg:
                        st["samples"].append((now, jpg, phase_lbl))
                        st["last_samp"] = now
                        if st["first_sample_ts"] is None:
                            st["first_sample_ts"] = now
                            st["first_sample_raw_ts"] = float(raw_ts)
                        if phase_lbl != "RELEASE" and (st["reach_ts"] is None or not st["pre_reach_jpg"]):
                            st["pre_reach_jpg"] = jpg
                            st["pre_reach_ts"] = now

                is_action_phase = phase in ("REACH", "TUCK", "BLADED", "CROUCH")
                phase_changed = is_action_phase and phase != st.get("last_phase")
                sample_interval = self.ACTION_SAMPLE_SEC if is_action_phase else self.SAMPLE_SEC
                if st["in_zone"] and (phase_changed or (now - st["last_samp"]) >= sample_interval or not st["samples"]):
                    _append_sample(phase)
                if st["in_zone"]:
                    st["last_phase"] = phase
                while st["samples"] and (now - st["samples"][0][0]) > self.BUFFER_SEC:
                    st["samples"].popleft()

                fast_delta_hit = fast_shelf_ready and (tuck_now or bladed_now or crouch_now or cues["in_core_or_bag"] or not reach_now)
                path_a_hit = (
                    cd_ok
                    and (((mature or fast_reach_tuck) and (tuck_now or bladed_now or crouch_now)) or fast_delta_hit)
                    and not cues["in_benign_basket"]
                )
                hit = path_a_hit or (path_b_hit and not cues["in_benign_basket"])

                if path_b_hit and not path_a_hit and st["armed_ts"] is None:
                    st["trigger_mode"] = "shelf_free_tuck"
                elif path_a_hit and st["armed_ts"] is None and st.get("trigger_mode") != "carry_delayed":
                    st["trigger_mode"] = "shelf_reach_and_delta"

                if cd_ok and (hit or st["armed_ts"] is not None):
                    for flag, lbl in (
                        (tuck_now, "reach_and_tuck"),
                        (bladed_now, "blading"),
                        (crouch_now, "crouch"),
                        (fast_delta_hit, "shelf_item_removed"),
                        (st.get("trigger_mode") == "carry_delayed" and (tuck_now or bladed_now or crouch_now), "carry_delayed"),
                        (bag_stuff_hit, "bag_stuff"),
                        (garment_hit, "garment_hide"),
                        (crouch_hide_hit, "crouch_hide"),
                    ):
                        if flag and lbl not in st["reasons"]:
                            st["reasons"].append(lbl)
                conceal_active_now = (
                    not reach_now
                    and not cues["in_benign_basket"]
                    and not phone_suppress
                    and (
                        cues["in_personal_bag"]
                        or cues["occluded"]
                        or bladed_frame
                        or crouch_frame
                        or cues["tight_core"]
                    )
                )
                if hit and st["armed_ts"] is None:
                    st["armed_ts"] = now
                    st["last_conceal_ts"] = now
                    st["armed_bag_contact"] = bool(cues["in_personal_bag"])
                    st["episode_bags"].update(st["bags"])
                    p_crop = _crop_person(frame_bgr, x, y, bw, bh, w_img, h_img)
                    st["upper_color_hist"] = compute_upper_color_hist(p_crop)
                    ov = frame_bgr.copy()
                    cv2.rectangle(ov, (x, y), (x + bw, y + bh), (0, 60, 255), 3)
                    st["peak_ov"] = _encode_jpg(ov, max_dim=768)
                elif st["armed_ts"] is not None:
                    if cues["in_personal_bag"]:
                        st["armed_bag_contact"] = True
                    if reach_now and now > st["armed_ts"]:
                        st["post_arm_reach_frames"] = int(st.get("post_arm_reach_frames") or 0) + 1
                    if conceal_active_now:
                        st["last_conceal_ts"] = now

                if st["armed_ts"] is not None:
                    min_arm_sec = min(self.release_wait_sec, max(0.35, self.release_wait_sec * 0.65))
                    post_quiet_sec = min(1.75, max(0.25, self.release_wait_sec * 0.50))
                    max_arm_sec = min(self.BUFFER_SEC - 2.0, self.release_wait_sec + min(4.0, self.release_wait_sec * 1.15))
                    last_c = float(st["last_conceal_ts"] if st.get("last_conceal_ts") is not None else st["armed_ts"])
                    armed_elapsed = now - float(st["armed_ts"])
                    quiet_elapsed = now - last_c
                    ready_to_release = (
                        (armed_elapsed >= min_arm_sec and quiet_elapsed >= post_quiet_sec)
                        or armed_elapsed >= max_arm_sec
                    )
                    if ready_to_release:
                        if self._should_suppress_active_browse(st):
                            self._reset_armed_cycle(st)
                            self._shelf_delta.clear_episode(tid)
                            confirmed_until = st.get("carried_openly_confirmed_until")
                            if confirmed_until is None or float(confirmed_until) < now:
                                st["has_merchandise_until"] = None
                        else:
                            if st["in_zone"] and (now - st["last_samp"]) >= 0.15:
                                _append_sample("RELEASE")
                            self._dispatch_queue.append(self._build_candidate(st, frame_bgr, "natural_release", zone_names))
                            st["last_dispatch_ts"] = now
                            self._reset_armed_cycle(st)
                            self._shelf_delta.clear_episode(tid)
                            st["cooldown"] = now + self.cooldown_sec

                if st["armed_ts"] is None and st["reach_ts"] is not None and (now - st["reach_ts"]) > self.TUCK_WIN_SEC:
                    st["reach_ts"] = None

            for tid, st in list(self._tracks.items()):
                if tid in seen:
                    continue
                miss = now - st["last_seen"]
                if st["armed_ts"] is not None and miss >= self.ID_LOSS_GRACE_SEC and now >= st["cooldown"]:
                    if not self._should_suppress_active_browse(st):
                        self._dispatch_queue.append(self._build_candidate(st, frame_bgr, "id_disappearance", zone_names))
                    self._tracks.pop(tid, None)
                    self._shelf_delta.clear_episode(tid)
                elif miss > 15.0:
                    self._tracks.pop(tid, None)
                    self._shelf_delta.clear_track(tid)

            if self._dispatch_queue:
                grid_out, cand_out = self._dispatch_queue.popleft()
                return grid_out, cand_out, {"tracks": len(seen), "queued": len(self._dispatch_queue), "gate_state": "DISPATCH"}

            return None, None, {
                "tracks": len(seen),
                "queued": 0,
                "gate_state": "ARMED" if any(s["armed_ts"] is not None for s in self._tracks.values()) else "MONITORING",
            }
