"""Per-camera BoT-SORT tracker pool with lightweight appearance ReID."""
from __future__ import annotations

from collections import deque
import pathlib
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..inference.device_utils import resolve_torch_device


class _DetSlice:
    """Minimal Ultralytics-compatible detection container for BOTSORT.update."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls_ids: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls_ids, dtype=np.float32).reshape(-1)
        if len(self.xyxy) > 0:
            x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
            self.xywh = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], axis=1)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, mask) -> "_DetSlice":
        return _DetSlice(self.xyxy[mask], self.conf[mask], self.cls[mask])


class _AppearanceReIDEncoder:
    """Device-aware 3-zone spatial appearance descriptor with zero-norm NaN guard."""

    def __init__(self, device: str = "auto"):
        self.device = resolve_torch_device(device)
        try:
            import torch
            self._torch = torch if self.device != "cpu" else None
        except Exception:
            self._torch = None

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> List[np.ndarray]:
        if dets is None or len(dets) == 0:
            return []
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            vec = np.zeros((36,), dtype=np.float32)
            vec[0] = 1.0
            return [vec.copy() for _ in range(len(dets))]

        h_img, w_img = img.shape[:2]
        crops: List[np.ndarray] = []
        for det in dets:
            cx, cy, bw, bh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            x1 = max(0, min(w_img - 1, int(round(cx - bw * 0.5))))
            y1 = max(0, min(h_img - 1, int(round(cy - bh * 0.5))))
            x2 = max(x1 + 1, min(w_img, int(round(cx + bw * 0.5))))
            y2 = max(y1 + 1, min(h_img, int(round(cy + bh * 0.5))))
            crop = cv2.resize(img[y1:y2, x1:x2], (16, 24), interpolation=cv2.INTER_AREA)
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            crops.append(crop)

        batch = np.asarray(crops, dtype=np.float32) / 255.0
        if self._torch is not None:
            t = self._torch.from_numpy(batch).to(self.device)
            zones = t.reshape(len(crops), 3, 8, 2, 8, 3)
            means = zones.mean(dim=(2, 4)).reshape(len(crops), -1)
            stds = zones.std(dim=(2, 4), unbiased=False).reshape(len(crops), -1)
            feats = self._torch.cat([means, stds], dim=1).detach().cpu().numpy().astype(np.float32)
        else:
            zones = batch.reshape(len(crops), 3, 8, 2, 8, 3)
            means = zones.mean(axis=(2, 4)).reshape(len(crops), -1)
            stds = zones.std(axis=(2, 4)).reshape(len(crops), -1)
            feats = np.concatenate([means, stds], axis=1).astype(np.float32)

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        zero_rows = norms[:, 0] < 1e-6
        if np.any(zero_rows):
            feats[zero_rows, 0] = 1.0
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = (feats / np.maximum(norms, 1e-6)).astype(np.float32)
        return [feats[i] for i in range(len(feats))]


class CameraTracker:
    """Single-camera BoT-SORT tracker instance."""

    def __init__(self, *, fps: int = 8, device: str = "auto", enable_reid: bool = True, gmc_method: Optional[str] = None):
        import yaml
        import ultralytics
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace

        cfg_path = pathlib.Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh) or {}
        cfg_dict["with_reid"] = bool(enable_reid)
        cfg_dict["gmc_method"] = gmc_method
        cfg_obj = IterableSimpleNamespace(**cfg_dict)
        self._tracker = BOTSORT(cfg_obj, frame_rate=max(1, int(fps)))
        self._tracker.encoder = _AppearanceReIDEncoder(device=device) if enable_reid else None
        self._trails: Dict[int, Deque[Tuple[int, int]]] = {}

    def update(self, frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR) if frame_bgr.ndim == 2 else frame_bgr
        h_img, w_img = bgr.shape[:2]

        boxes_xyxy: List[List[float]] = []
        confs: List[float] = []
        clss: List[float] = []
        labels: List[str] = []
        kps_list: List[Any] = []

        for item in detections or []:
            box = item.get("Box") or item.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x, y, bw, bh = [float(v) for v in box[:4]]
            if bw <= 0 or bh <= 0:
                continue
            boxes_xyxy.append([x, y, x + bw, y + bh])
            confs.append(float(item.get("Score", 0.5)))
            clss.append(float(item.get("Class_Id", 0)))
            labels.append(str(item.get("Class", "person")))
            kp = item.get("Keypoints")
            kps_list.append(kp if isinstance(kp, list) else [])

        det_slice = _DetSlice(
            np.asarray(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32),
            np.asarray(confs, dtype=np.float32),
            np.asarray(clss, dtype=np.float32),
        )
        tracked = self._tracker.update(det_slice, bgr)

        tracks_out: List[Dict[str, Any]] = []
        active_ids = set()
        if isinstance(tracked, np.ndarray) and tracked.ndim == 2 and len(tracked) > 0:
            for row in tracked:
                x1 = max(0, min(w_img - 1, int(round(float(row[0])))))
                y1 = max(0, min(h_img - 1, int(round(float(row[1])))))
                x2 = max(x1 + 1, min(w_img, int(round(float(row[2])))))
                y2 = max(y1 + 1, min(h_img, int(round(float(row[3])))))
                tid = int(row[4])
                score = round(float(row[5]), 4) if len(row) > 5 else 1.0
                det_idx = int(row[7]) if len(row) > 7 else -1
                bw, bh = x2 - x1, y2 - y1
                cx, cy = x1 + bw // 2, y1 + bh // 2

                trail = self._trails.setdefault(tid, deque(maxlen=32))
                trail.append((cx, cy))
                active_ids.add(tid)

                tracks_out.append(
                    {
                        "track_id": tid,
                        "label": labels[det_idx] if 0 <= det_idx < len(labels) else "person",
                        "confidence": score,
                        "bbox": [x1, y1, bw, bh],
                        "center": [cx, cy],
                        "keypoints": kps_list[det_idx] if 0 <= det_idx < len(kps_list) else [],
                        "trail": list(trail),
                    }
                )

        if len(self._trails) > 128:
            for stale_id in [k for k in self._trails if k not in active_ids]:
                self._trails.pop(stale_id, None)
        return tracks_out


class BoTSORTPool:
    """Thread-safe pool maintaining isolated BoT-SORT trackers per camera_id."""

    def __init__(self, *, fps: int = 8, device: str = "auto", enable_reid: bool = True, gmc_method: Optional[str] = None):
        self.fps = max(1, int(fps))
        self.device = device
        self.enable_reid = enable_reid
        self.gmc_method = gmc_method
        self._lock = threading.Lock()
        self._trackers: Dict[str, CameraTracker] = {}

    def update(self, camera_id: str, frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = CameraTracker(
                    fps=self.fps,
                    device=self.device,
                    enable_reid=self.enable_reid,
                    gmc_method=self.gmc_method,
                )
                self._trackers[camera_id] = tracker
        return tracker.update(frame_bgr, detections)

    def reset(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)

    def prune_except(self, active_ids: set[str]) -> None:
        with self._lock:
            for cid in [k for k in self._trackers if k not in active_ids]:
                self._trackers.pop(cid, None)
