"""Chronological 2x3 visual sequence grid builder for VLM inspection and alerts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _decode_jpg(raw: bytes) -> Optional[np.ndarray]:
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def _encode_jpg_crop(
    frame_bgr: np.ndarray,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    max_dim: int = 640,
    quality: int = 85,
) -> bytes:
    if frame_bgr is None or frame_bgr.size == 0:
        return b""
    work = frame_bgr
    h, w = frame_bgr.shape[:2]
    if crop_rect is not None:
        cx, cy, cw, ch = [int(v) for v in crop_rect]
        if cw >= 32 and ch >= 32:
            x1 = max(0, min(w - 1, cx))
            y1 = max(0, min(h - 1, cy))
            x2 = max(x1 + 1, min(w, cx + cw))
            y2 = max(y1 + 1, min(h, cy + ch))
            if (x2 - x1) >= 32 and (y2 - y1) >= 32:
                work = frame_bgr[y1:y2, x1:x2]
    ah, aw = work.shape[:2]
    if max(ah, aw) > max_dim:
        s = max_dim / float(max(ah, aw))
        work = cv2.resize(work, (max(1, int(aw * s)), max(1, int(ah * s))))
    ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _fit_tile(arr: np.ndarray, size: int, title: str, color: Tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if arr is None or arr.size == 0:
        return canvas
    ah, aw = arr.shape[:2]
    top = 24
    scale = min(size / float(max(1, aw)), (size - top) / float(max(1, ah)))
    nw, nh = max(1, int(aw * scale)), max(1, int(ah * scale))
    rsz = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (size - nw) // 2
    y0 = top + ((size - top) - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rsz
    cv2.putText(canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def build_2x3_grid(track_state: Dict[str, Any], fallback_overview_bgr: np.ndarray, tile_px: int = 384) -> np.ndarray:
    """Assemble 1 overview tile + 5 chronological phase tiles into a 2x3 grid."""
    samples = list(track_state.get("samples") or [])
    if not samples:
        return fallback_overview_bgr.copy() if fallback_overview_bgr is not None else np.zeros((tile_px * 2, tile_px * 3, 3), dtype=np.uint8)

    def nearest(target: float | None, lo: int, hi: int, phases: Tuple[str, ...] = ()) -> int:
        cands = [i for i in range(lo, hi + 1) if samples[i][2] in phases] if phases else []
        if cands:
            return cands[0] if target is None else min(cands, key=lambda i: abs(samples[i][0] - target))
        return (lo + hi) // 2 if target is None else min(range(lo, hi + 1), key=lambda i: abs(samples[i][0] - target))

    n = len(samples)
    mid_ts = (
        track_state.get("last_conceal_ts")
        or track_state.get("tuck_ts")
        or track_state.get("bladed_ts")
        or track_state.get("crouch_ts")
        or track_state.get("armed_ts")
    )
    i_mid = nearest(mid_ts, 0, n - 1, ("TUCK", "BLADED", "CROUCH"))
    i_reach = nearest(track_state.get("reach_peak_ts") or track_state.get("reach_ts"), 0, i_mid, ("REACH",))
    i_pre = nearest(samples[i_reach][0] - 1.8, 0, i_reach, ("PRE", "BROWSE"))
    post_ts = samples[i_mid][0] + 1.5 if i_mid < n - 1 else None
    i_post = nearest(post_ts, min(n - 1, i_mid), n - 1, ("POST",))

    idxs = sorted({i_pre, i_reach, i_mid, i_post, n - 1})
    for idx in np.linspace(0, n - 1, 5, dtype=int).tolist():
        if len(idxs) >= 5:
            break
        if idx not in idxs:
            idxs.append(idx)
    idxs.sort()
    while len(idxs) < 5:
        idxs.append(idxs[-1])

    ov = _decode_jpg(track_state.get("peak_ov", b""))
    if ov is None:
        ov = fallback_overview_bgr

    tiles = [_fit_tile(ov, tile_px, f"1. AISLE OVERVIEW | ID:{track_state['id']}", (0, 220, 255))]
    t0 = samples[idxs[0]][0]
    for k, si in enumerate(idxs[:5], start=2):
        ts, jpg, phase = samples[si]
        crop = _decode_jpg(jpg)
        if crop is None:
            crop = ov
        color = (0, 140, 255) if phase in ("REACH", "TUCK", "BLADED", "CROUCH") else (120, 235, 160)
        tiles.append(_fit_tile(crop, tile_px, f"{k}. t+{max(0.0, ts - t0):.1f}s [{phase}]", color))

    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])


def extract_sequence_frames(track_state: Dict[str, Any], max_frames: int = 14) -> List[bytes]:
    """Select up to `max_frames` chronological JPEG crops prioritizing dense action transitions."""
    samples = list(track_state.get("samples") or [])
    if not samples:
        pre = track_state.get("pre_reach_jpg")
        return [pre] if pre else []
    limit = max(2, int(max_frames))
    pre_jpg = track_state.get("pre_reach_jpg") or b""

    if len(samples) <= limit:
        seq = [jpg for _, jpg, _ in samples if jpg]
    else:
        chosen: set[int] = {0, len(samples) - 1}
        mid_ts = (
            track_state.get("tuck_ts")
            or track_state.get("bladed_ts")
            or track_state.get("crouch_ts")
            or track_state.get("armed_ts")
        )
        reach_ts = track_state.get("reach_peak_ts") or track_state.get("reach_ts")
        last_conceal_ts = track_state.get("last_conceal_ts")

        # Prioritize dense consecutive samples around [reach - 0.5 .. last_conceal + 2.0] so
        # critical hand-to-bag/pocket transitions aren't lost in sparse uniform sampling.
        win_idxs: List[int] = []
        anchors = [float(t) for t in (reach_ts, mid_ts, last_conceal_ts) if t is not None]
        if anchors:
            win_lo = min(anchors) - 0.5
            win_hi = max(anchors) + 2.0
            win_idxs = [
                i
                for i, (ts, _, ph) in enumerate(samples)
                if (win_lo <= ts <= win_hi) or ph in ("REACH", "TUCK", "BLADED", "CROUCH", "POST", "RELEASE")
            ]
        else:
            win_idxs = [
                i
                for i, (_, _, ph) in enumerate(samples)
                if ph in ("REACH", "TUCK", "BLADED", "CROUCH", "POST", "RELEASE")
            ]

        action_budget = max(0, limit - len(chosen))
        if win_idxs and action_budget > 0:
            if len(win_idxs) <= action_budget:
                chosen.update(win_idxs)
            else:
                for pos in np.linspace(0, len(win_idxs) - 1, action_budget, dtype=int).tolist():
                    chosen.add(int(win_idxs[int(pos)]))

        if len(chosen) < limit:
            for idx in np.linspace(0, len(samples) - 1, limit, dtype=int).tolist():
                if len(chosen) >= limit:
                    break
                chosen.add(int(idx))

        seq = [samples[i][1] for i in sorted(chosen)[:limit] if samples[i][1]]

    if pre_jpg and seq and pre_jpg != seq[0]:
        if len(seq) < limit:
            seq = [pre_jpg] + seq
        elif len(seq) > 2:
            seq = [pre_jpg] + seq[1:]
        else:
            seq = [pre_jpg, seq[-1]]
    return seq[:limit]


def augment_sequence_with_ring_context(
    sequence_frames: List[bytes],
    ring_frames: List[Tuple[float, bytes]],
    *,
    first_sample_ts: Optional[float] = None,
    clean_scene_jpg: bytes = b"",
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    max_frames: int = 14,
) -> List[bytes]:
    """Prepend earlier chronological pre-reach/entry context frames from the rolling ring buffer."""
    seq = [f for f in (sequence_frames or []) if f]
    limit = max(2, int(max_frames))

    prepended: List[bytes] = []
    if ring_frames:
        if first_sample_ts is not None:
            earlier = [(ts, jpg) for ts, jpg in ring_frames if jpg and float(ts) < (float(first_sample_ts) - 0.5)]
        else:
            earlier = []

        if not earlier and len(seq) < limit:
            span_n = max(1, len(ring_frames) // 2)
            earlier = [(ts, jpg) for ts, jpg in ring_frames[:span_n] if jpg]

        if earlier:
            # Take up to 2..4 earliest/approach context frames spaced chronologically
            target_extra = min(4, max(2, limit - len(seq)))
            idxs = (
                list(range(len(earlier)))
                if len(earlier) <= target_extra
                else np.linspace(0, len(earlier) - 1, target_extra, dtype=int).tolist()
            )
            seen_pos = set()
            for idx in idxs:
                if idx in seen_pos:
                    continue
                seen_pos.add(idx)
                _, raw_jpg = earlier[idx]
                if crop_rect is not None:
                    decoded = _decode_jpg(raw_jpg)
                    enc = _encode_jpg_crop(decoded, crop_rect=crop_rect) if decoded is not None else raw_jpg
                    prepended.append(enc or raw_jpg)
                else:
                    prepended.append(raw_jpg)

    if not prepended and clean_scene_jpg and len(seq) < limit and (not seq or clean_scene_jpg != seq[0]):
        if crop_rect is not None:
            decoded = _decode_jpg(clean_scene_jpg)
            enc = _encode_jpg_crop(decoded, crop_rect=crop_rect) if decoded is not None else clean_scene_jpg
            prepended.append(enc or clean_scene_jpg)
        else:
            prepended.append(clean_scene_jpg)

    if not prepended:
        if len(seq) <= limit:
            return seq
        pick_idxs = sorted({int(i) for i in np.linspace(0, len(seq) - 1, limit, dtype=int).tolist()})
        return [seq[i] for i in pick_idxs]

    combined = prepended + seq
    if len(combined) <= limit:
        return combined

    # Preserve prepended early baseline frames + final departure frame + evenly thinned middle frames
    keep_head = min(len(prepended), 3)
    tail_count = limit - keep_head
    rest = combined[keep_head:]
    if len(rest) <= tail_count:
        return combined[:keep_head] + rest
    pick_idxs = sorted({int(i) for i in np.linspace(0, len(rest) - 1, tail_count, dtype=int).tolist()})
    return combined[:keep_head] + [rest[i] for i in pick_idxs]
