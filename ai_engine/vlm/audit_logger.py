"""Structured JSONL audit logger + standard logger for every VLM call and outcome."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("storeguard.vlm")


class VlmAuditLogger:
    """Appends non-secret VLM request/verdict/error records to runtime_data/logs/vlm_calls.jsonl."""

    def __init__(self, logs_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.jsonl_path = self.logs_dir / "vlm_calls.jsonl"
        self._lock = threading.Lock()

    def log_verdict(
        self,
        *,
        candidate: Dict[str, Any],
        provider: str,
        model_name: str,
        decision: str,
        conf_pct: float,
        threshold_pct: float,
        suppression_reasons: Optional[List[str]] = None,
        verdict: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        latency_sec: Optional[float] = None,
    ) -> None:
        rec = self._base_record(
            event="VLM_VERDICT",
            candidate=candidate,
            provider=provider,
            model_name=model_name,
            decision=decision,
            latency_sec=latency_sec,
        )
        rec.update(
            {
                "event_id": event_id,
                "confidence_pct": round(float(conf_pct), 2),
                "threshold_pct": round(float(threshold_pct), 2),
                "suppression_reasons": list(suppression_reasons or []),
                "verdict": dict(verdict or {}),
            }
        )
        self._append(rec)
        logger.info(
            "VLM %s cam=%s track=#%s mode=%s cat=%s conf=%.1f%% dest=%s latency=%s reasons=%s",
            decision,
            rec["camera_id"],
            rec["track_id"],
            rec["trigger_mode"],
            (verdict or {}).get("behavior_category", ""),
            conf_pct,
            (verdict or {}).get("visible_destination", ""),
            f"{latency_sec:.2f}s" if latency_sec is not None else "n/a",
            ",".join(rec["suppression_reasons"]) or "none",
        )

    def log_error(
        self,
        *,
        candidate: Dict[str, Any],
        provider: str,
        model_name: str,
        error_msg: str,
        grid_path: str = "",
        latency_sec: Optional[float] = None,
    ) -> None:
        rec = self._base_record(
            event="VLM_ERROR",
            candidate=candidate,
            provider=provider,
            model_name=model_name,
            decision="VLM_ERROR",
            latency_sec=latency_sec,
        )
        rec.update(
            {
                "error": str(error_msg or ""),
                "grid_path": str(grid_path or ""),
            }
        )
        self._append(rec)
        logger.warning(
            "VLM_ERROR cam=%s track=#%s provider=%s model=%s latency=%s err=%s",
            rec["camera_id"],
            rec["track_id"],
            provider,
            model_name,
            f"{latency_sec:.2f}s" if latency_sec is not None else "n/a",
            rec["error"],
        )

    @staticmethod
    def _base_record(
        *,
        event: str,
        candidate: Dict[str, Any],
        provider: str,
        model_name: str,
        decision: str,
        latency_sec: Optional[float],
    ) -> Dict[str, Any]:
        lat = latency_sec if latency_sec is not None else candidate.get("_vlm_latency_sec")
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "camera_id": str(candidate.get("camera_id", "cam")),
            "camera_name": str(candidate.get("camera_name", "")),
            "track_id": int(candidate.get("track_id", 0)),
            "trigger_mode": str(candidate.get("trigger_mode") or "shelf_reach_and_delta"),
            "behaviors": list(candidate.get("behaviors") or []),
            "shelf_delta_status": str(candidate.get("shelf_delta_status") or "none"),
            "shelf_delta_score": float(candidate.get("shelf_delta_score") or 0.0),
            "frame_count": len(candidate.get("sequence_frames") or []),
            "media_mode": str(candidate.get("_vlm_media_mode") or ""),
            "provider": str(provider or ""),
            "model_name": str(model_name or ""),
            "latency_sec": round(float(lat), 3) if lat is not None else None,
            "decision": decision,
        }

    def _append(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass
