"""Shared singleton ONNX Runtime session pool for YOLO11-Pose and bag detection."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .device_utils import resolve_onnx_providers

logger = logging.getLogger("storeguard.models")

_COCO_BAG_DEFAULTS: Dict[int, str] = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    67: "cell phone",
}


def letterbox_image(bgr: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, float, float]:
    """Resize with aspect preservation + gray padding into NCHW float32 [0..1] RGB tensor."""
    h, w = bgr.shape[:2]
    scale = min(size / float(max(1, w)), size / float(max(1, h)))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (size - nw) / 2.0
    pad_h = (size - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if canvas.shape[0] != size or canvas.shape[1] != size:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return blob[None, ...], scale, float(left), float(top)


def _normalize_predictions_2d(raw: np.ndarray, expected_cols: int | None = None) -> np.ndarray:
    """Normalize YOLO ONNX output [1, C, N] or [1, N, C] into shape (N, C)."""
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    if expected_cols is not None:
        if arr.shape[0] == expected_cols and arr.shape[1] != expected_cols:
            arr = arr.T
    elif arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.float32)


def _read_names_metadata(sess) -> Dict[int, str]:
    try:
        meta = sess.get_modelmeta().custom_metadata_map or {}
        if "names" in meta:
            parsed = ast.literal_eval(meta["names"])
            if isinstance(parsed, dict):
                return {int(k): str(v) for k, v in parsed.items()}
            if isinstance(parsed, (list, tuple)):
                return {i: str(v) for i, v in enumerate(parsed)}
    except Exception:
        pass
    return {}


class SharedModelPool:
    """Thread-safe shared ONNX Runtime sessions for all camera streams."""

    def __init__(self, models_dir: str | Path, pose_model: str, bag_model: str, device: str = "auto"):
        import onnxruntime as ort

        self.models_dir = Path(models_dir)
        self.providers, self.using_gpu, self.resolved_device = resolve_onnx_providers(device)
        self._lock = threading.Lock()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        pose_path = self._resolve_model(pose_model)
        bag_path = self._resolve_model(bag_model)

        self._pose_sess = ort.InferenceSession(str(pose_path), sess_options=opts, providers=self.providers)
        self._pose_input = self._pose_sess.get_inputs()[0].name
        self._pose_size = self._input_size(self._pose_sess, default=640)

        self._bag_sess = ort.InferenceSession(str(bag_path), sess_options=opts, providers=self.providers)
        self._bag_input = self._bag_sess.get_inputs()[0].name
        self._bag_size = self._input_size(self._bag_sess, default=640)
        self._bag_names = _read_names_metadata(self._bag_sess)

    def _resolve_model(self, name: str) -> Path:
        p = Path(name)
        if p.is_file():
            return p
        cand = self.models_dir / name
        if cand.is_file():
            return cand
        if "pose" in name.lower():
            fallback = self.models_dir / "yolo11n-pose.onnx"
            if fallback.is_file():
                logger.warning("Pose model '%s' not found; falling back to '%s'.", name, fallback)
                return fallback
        raise FileNotFoundError(f"ONNX model not found: {cand}")

    @staticmethod
    def _input_size(sess, default: int = 640) -> int:
        try:
            shape = sess.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                return int(shape[2])
        except Exception:
            pass
        return default

    def infer_pose(
        self,
        bgr: np.ndarray,
        *,
        conf: float = 0.35,
        iou: float = 0.50,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, Any]]:
        if bgr is None or bgr.size == 0:
            return []

        ox, oy = 0, 0
        work = bgr
        h_full, w_full = bgr.shape[:2]
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            ox = max(0, min(w_full - 1, cx))
            oy = max(0, min(h_full - 1, cy))
            work = bgr[oy:min(h_full, oy + ch), ox:min(w_full, ox + cw)]
            if work.size == 0:
                return []

        wh, ww = work.shape[:2]
        blob, scale, pad_x, pad_y = letterbox_image(work, self._pose_size)
        with self._lock:
            raw = self._pose_sess.run(None, {self._pose_input: blob})[0]

        preds = _normalize_predictions_2d(raw, expected_cols=56)
        if preds.size == 0 or preds.shape[1] < 5:
            return []

        scores = preds[:, 4]
        mask = scores >= conf
        if not np.any(mask):
            return []

        filtered = preds[mask]
        boxes_xywh_nms: List[List[float]] = []
        boxes_xywh_full: List[List[int]] = []
        scores_list: List[float] = []
        kps_all: List[List[List[float]]] = []

        has_kps = filtered.shape[1] >= 56
        for row in filtered:
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x1 = max(0.0, min(float(ww - 1), (cx - bw * 0.5 - pad_x) / scale))
            y1 = max(0.0, min(float(wh - 1), (cy - bh * 0.5 - pad_y) / scale))
            x2 = max(x1 + 1.0, min(float(ww), (cx + bw * 0.5 - pad_x) / scale))
            y2 = max(y1 + 1.0, min(float(wh), (cy + bh * 0.5 - pad_y) / scale))
            rw, rh = x2 - x1, y2 - y1
            boxes_xywh_nms.append([x1, y1, rw, rh])
            boxes_xywh_full.append([int(round(x1)) + ox, int(round(y1)) + oy, max(1, int(round(rw))), max(1, int(round(rh)))])
            scores_list.append(float(row[4]))

            kp_list: List[List[float]] = []
            if has_kps:
                kp_vec = row[5:56].reshape(17, 3)
                for kx, ky, kc in kp_vec:
                    fx = max(0.0, min(float(w_full - 1), ((float(kx) - pad_x) / scale) + ox))
                    fy = max(0.0, min(float(h_full - 1), ((float(ky) - pad_y) / scale) + oy))
                    kp_list.append([round(fx, 1), round(fy, 1), round(float(kc), 3)])
            kps_all.append(kp_list)

        keep = cv2.dnn.NMSBoxes(boxes_xywh_nms, scores_list, float(conf), float(iou))
        indices = keep.flatten().tolist() if len(keep) > 0 else []

        detections: List[Dict[str, Any]] = []
        for out_idx, idx in enumerate(indices):
            detections.append(
                {
                    "Index": out_idx,
                    "Class": "person",
                    "Class_Id": 0,
                    "Score": round(scores_list[idx], 4),
                    "Box": boxes_xywh_full[idx],
                    "Keypoints": kps_all[idx],
                }
            )
        return detections

    def infer_bags(
        self,
        bgr: np.ndarray,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        target_classes: Iterable[str] = (
            "backpack",
            "handbag",
            "suitcase",
            "tote bag",
            "shopping basket",
            "shopping cart",
            "cell phone",
            "smartphone",
            "bottle",
        ),
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, Any]]:
        if bgr is None or bgr.size == 0:
            return []

        ox, oy = 0, 0
        work = bgr
        h_full, w_full = bgr.shape[:2]
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            ox = max(0, min(w_full - 1, cx))
            oy = max(0, min(h_full - 1, cy))
            work = bgr[oy:min(h_full, oy + ch), ox:min(w_full, ox + cw)]
            if work.size == 0:
                return []

        wh, ww = work.shape[:2]
        blob, scale, pad_x, pad_y = letterbox_image(work, self._bag_size)
        with self._lock:
            raw = self._bag_sess.run(None, {self._bag_input: blob})[0]

        preds = _normalize_predictions_2d(raw)
        if preds.size == 0 or preds.shape[1] < 5:
            return []

        names = self._bag_names or _COCO_BAG_DEFAULTS
        wanted = {c.strip().lower() for c in target_classes if c.strip()}
        wanted_ids = [cid for cid, cname in names.items() if cname.lower() in wanted]
        if not wanted_ids:
            wanted_ids = [cid for cid in _COCO_BAG_DEFAULTS if _COCO_BAG_DEFAULTS[cid] in wanted]

        cls_scores_all = preds[:, 4:]
        valid_ids = [cid for cid in wanted_ids if cid < cls_scores_all.shape[1]]
        if not valid_ids:
            return []

        sub_scores = cls_scores_all[:, valid_ids]
        best_rel = np.argmax(sub_scores, axis=1)
        best_score = sub_scores[np.arange(len(sub_scores)), best_rel]
        mask = best_score >= conf
        if not np.any(mask):
            return []

        filtered_boxes = preds[mask, :4]
        filtered_scores = best_score[mask]
        filtered_cids = np.asarray(valid_ids)[best_rel[mask]]

        boxes_nms: List[List[float]] = []
        boxes_full: List[List[int]] = []
        scores_list: List[float] = []
        cids_list: List[int] = []

        for (cx, cy, bw, bh), sc, cid in zip(filtered_boxes, filtered_scores, filtered_cids):
            x1 = max(0.0, min(float(ww - 1), (float(cx) - float(bw) * 0.5 - pad_x) / scale))
            y1 = max(0.0, min(float(wh - 1), (float(cy) - float(bh) * 0.5 - pad_y) / scale))
            x2 = max(x1 + 1.0, min(float(ww), (float(cx) + float(bw) * 0.5 - pad_x) / scale))
            y2 = max(y1 + 1.0, min(float(wh), (float(cy) + float(bh) * 0.5 - pad_y) / scale))
            rw, rh = x2 - x1, y2 - y1
            boxes_nms.append([x1, y1, rw, rh])
            boxes_full.append([int(round(x1)) + ox, int(round(y1)) + oy, max(1, int(round(rw))), max(1, int(round(rh)))])
            scores_list.append(float(sc))
            cids_list.append(int(cid))

        keep = cv2.dnn.NMSBoxes(boxes_nms, scores_list, float(conf), float(iou))
        indices = keep.flatten().tolist() if len(keep) > 0 else []

        detections: List[Dict[str, Any]] = []
        for out_idx, idx in enumerate(indices):
            cid = cids_list[idx]
            detections.append(
                {
                    "Index": out_idx,
                    "Class": names.get(cid, str(cid)),
                    "Class_Id": cid,
                    "Score": round(scores_list[idx], 4),
                    "Box": boxes_full[idx],
                }
            )
        return detections
