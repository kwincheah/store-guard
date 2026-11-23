"""Bounded concurrent Cloud VLM worker with ordered FIFO callbacks and multi-frame sequence support."""
from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import VlmConfig
from ..media.clip_writer import encode_ring_clip_mp4_bytes
from .prompt_builder import build_vlm_prompts

_PROVIDER_DEFAULTS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "openai": ("https://api.openai.com/v1", ("VLM_API_KEY", "OPENAI_API_KEY")),
    "deepseek": ("https://api.deepseek.com", ("VLM_API_KEY", "DEEPSEEK_API_KEY")),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", ("VLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "anthropic": ("https://api.anthropic.com", ("VLM_API_KEY", "ANTHROPIC_API_KEY")),
    "openrouter": ("https://openrouter.ai/api/v1", ("VLM_API_KEY", "OPENROUTER_API_KEY")),
    "ollama": ("http://127.0.0.1:11434/v1", ()),
    "lmstudio": ("http://127.0.0.1:1234/v1", ()),
    "custom": ("", ("VLM_API_KEY", "OPENAI_API_KEY")),
}


def parse_json_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            raw = brace.group(0).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def resolve_vlm_endpoint(cfg: VlmConfig) -> Tuple[str, str, str]:
    prov = (cfg.provider or "gemini").strip().lower()
    default_url, env_keys = _PROVIDER_DEFAULTS.get(prov, _PROVIDER_DEFAULTS["custom"])
    base_url = (cfg.base_url or "").strip() or os.environ.get("VLM_BASE_URL", "").strip() or default_url
    key = (cfg.api_key or "").strip()
    if not key:
        for ek in env_keys:
            val = os.environ.get(ek, "").strip()
            if val:
                key = val
                break
    return prov, base_url, key


class VlmRateLimitError(RuntimeError):
    """Raised when VLM endpoint rejects calls with HTTP 429 / 503 rate limits."""

    def __init__(self, message: str, *, retry_after_sec: float = 15.0, is_daily_quota: bool = False):
        super().__init__(message)
        self.retry_after_sec = max(0.0, float(retry_after_sec))
        self.is_daily_quota = bool(is_daily_quota)


def _parse_rate_limit_details(resp: Any) -> Tuple[float, bool, str]:
    wait_sec = 0.0
    headers = getattr(resp, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        ra_raw = headers.get("Retry-After") or headers.get("retry-after") or ""
        try:
            if ra_raw:
                wait_sec = max(0.0, float(str(ra_raw).strip()))
        except Exception:
            wait_sec = 0.0

    raw_text = ""
    try:
        j = resp.json()
        if isinstance(j, (dict, list)):
            raw_text = json.dumps(j)
    except Exception:
        pass
    if not raw_text:
        try:
            raw_text = str(getattr(resp, "text", "") or "")
        except Exception:
            raw_text = ""

    if wait_sec <= 0.0 and raw_text:
        for pat in (
            r"retryDelay[\"'\s:=]+([0-9]+(?:\.[0-9]+)?)s?",
            r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        ):
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                try:
                    wait_sec = max(0.0, float(m.group(1)))
                    break
                except Exception:
                    pass

    lower = raw_text.lower()
    is_daily = any(k in lower for k in ("perday", "requestsperday", "per_day", "requests per day"))
    if wait_sec <= 0.0:
        wait_sec = 60.0 if is_daily else 15.0
    detail = raw_text.strip()[:220] or "HTTP 429 Too Many Requests"
    return wait_sec, is_daily, detail


def _effective_min_interval(cfg: VlmConfig) -> float:
    val = getattr(cfg, "min_request_interval_sec", None)
    if val is not None:
        return max(0.0, float(val))
    return 6.0 if (getattr(cfg, "provider", "") or "").strip().lower() == "gemini" else 0.0


def _extract_choice_text(choice: Dict[str, Any]) -> Tuple[str, str]:
    msg_content = choice["message"]["content"]
    if isinstance(msg_content, list):
        text = "\n".join(p.get("text", "") for p in msg_content if isinstance(p, dict))
    else:
        text = str(msg_content)
    finish_reason = str(choice.get("finish_reason") or "").strip().lower()
    return text, finish_reason


def _extract_http_error_detail(resp: Any) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict):
            err = j.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:260]
            if j.get("message"):
                return str(j["message"])[:260]
            return json.dumps(j)[:260]
    except Exception:
        pass
    try:
        txt = str(getattr(resp, "text", "") or "").strip()
        return txt[:260] if txt else ""
    except Exception:
        return ""


def _post_with_400_fallback(
    requests_mod,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_sec: float,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
):
    rate_retried = False
    while True:
        resp = requests_mod.post(url, headers=headers, json=payload, timeout=timeout_sec)
        status = int(getattr(resp, "status_code", 200) or 200)
        if status == 400:
            err_detail = _extract_http_error_detail(resp)
            low = err_detail.lower()
            if "max_completion_tokens" in low and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                continue
            if "temperature" in low and "temperature" in payload:
                payload.pop("temperature", None)
                continue
            gen_cfg = payload.get("generationConfig")
            if isinstance(gen_cfg, dict):
                if ("thinkingconfig" in low or "thinkingbudget" in low or "thinking_budget" in low) and "thinkingConfig" in gen_cfg:
                    gen_cfg.pop("thinkingConfig", None)
                    continue
                if "temperature" in low and "temperature" in gen_cfg:
                    gen_cfg.pop("temperature", None)
                    continue
                if "videometadata" in low and isinstance(payload.get("contents"), list):
                    stripped = False
                    for c_item in payload["contents"]:
                        for part in (c_item.get("parts") or []):
                            if isinstance(part, dict) and "videoMetadata" in part:
                                part.pop("videoMetadata", None)
                                stripped = True
                    if stripped:
                        continue
                if "responseMimeType" in gen_cfg:
                    gen_cfg.pop("responseMimeType", None)
                    continue
            if "response_format" in payload:
                payload.pop("response_format", None)
                continue
            if "reasoning_effort" in payload:
                payload.pop("reasoning_effort", None)
                continue
            if err_detail:
                raise RuntimeError(f"HTTP 400 for {url}: {err_detail}")
        if status in (429, 503):
            wait_sec, is_daily, detail = _parse_rate_limit_details(resp)
            if not rate_retried and not is_daily and wait_sec <= 15.0:
                rate_retried = True
                if wait_sec > 0.0:
                    sleep_fn(wait_sec)
                continue
            if is_daily:
                msg = (
                    f"HTTP {status} Gemini daily free-tier RPD quota exhausted "
                    f"(enable billing or switch model/provider): {detail}"
                )
            else:
                msg = f"HTTP {status} Rate limit exceeded (cooldown {max(15.0, min(60.0, wait_sec)):.0f}s): {detail}"
            raise VlmRateLimitError(msg, retry_after_sec=wait_sec, is_daily_quota=is_daily)
        if status >= 400:
            err_detail = _extract_http_error_detail(resp)
            if err_detail:
                raise RuntimeError(f"HTTP {status} for {url}: {err_detail}")
        resp.raise_for_status()
        return resp


def _encode_sequence_b64(grid_bgr: np.ndarray, sequence_frames: Optional[List[bytes]] = None, max_frames: int = 16) -> List[str]:
    images_b64: List[str] = []
    if sequence_frames:
        valid = [bytes(raw) for raw in sequence_frames if isinstance(raw, (bytes, bytearray)) and raw]
        if len(valid) > max_frames and max_frames >= 2:
            pick_idxs = sorted({int(i) for i in np.linspace(0, len(valid) - 1, max_frames, dtype=int).tolist()})
            valid = [valid[i] for i in pick_idxs]
        else:
            valid = valid[:max_frames]
        for raw in valid:
            images_b64.append(base64.b64encode(raw).decode("ascii"))
    if not images_b64 and grid_bgr is not None and isinstance(grid_bgr, np.ndarray) and grid_bgr.size > 0:
        ok, buf = cv2.imencode(".jpg", grid_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            images_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return images_b64


def _resolve_gemini_native_url(base_url: str, model_name: str) -> Optional[str]:
    clean_url = (base_url or "").strip()
    if clean_url and "generativelanguage.googleapis.com" not in clean_url.lower():
        return None
    clean_model = (model_name or "gemini-2.5-flash").strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/"):]
    return f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"


def _extract_gemini_native_text(resp_json: Dict[str, Any]) -> Tuple[str, str]:
    candidates = resp_json.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return "", ""
    cand0 = candidates[0]
    parts = (cand0.get("content") or {}).get("parts") or []
    text = "\n".join(
        str(p.get("text") or "")
        for p in parts
        if isinstance(p, dict) and not p.get("thought") and p.get("text") is not None
    )
    finish_reason = str(cand0.get("finishReason") or "").strip().upper()
    return text, finish_reason


def _call_gemini_native_video(
    *,
    requests_mod,
    cfg: VlmConfig,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    mp4_bytes: bytes,
    reference_sheet_jpg: Optional[bytes] = None,
) -> Dict[str, Any]:
    url = _resolve_gemini_native_url(base_url, cfg.model_name)
    if not url:
        raise RuntimeError("Native Gemini endpoint unavailable for custom base_url")

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    parts: List[Dict[str, Any]] = []
    if reference_sheet_jpg:
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64.b64encode(bytes(reference_sheet_jpg)).decode("ascii"),
            }
        })
    video_fps = round(max(0.5, min(8.0, float(getattr(cfg, "video_fps", 2.0) or 2.0))), 2)
    parts.append({
        "inlineData": {
            "mimeType": "video/mp4",
            "data": base64.b64encode(bytes(mp4_bytes)).decode("ascii"),
        },
        "videoMetadata": {"fps": video_fps},
    })
    parts.append({"text": user_prompt})

    model_l = (cfg.model_name or "").strip().lower()
    req_tokens = int(getattr(cfg, "max_tokens", 2048) or 2048)
    max_out = max(2048, req_tokens)
    gen_cfg: Dict[str, Any] = {
        "responseMimeType": "application/json",
        "maxOutputTokens": max_out,
        "temperature": float(cfg.temperature),
    }
    if "2.5" in model_l and "pro" not in model_l:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 512}

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_cfg,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    resp = _post_with_400_fallback(requests_mod, url, headers, payload, cfg.timeout_sec)
    text, finish_reason = _extract_gemini_native_text(resp.json())
    parsed = parse_json_response(text)

    if not parsed and (finish_reason == "MAX_TOKENS" or text.lstrip().startswith(("{", "```"))):
        gen_cfg["maxOutputTokens"] = max(4096, max_out * 2)
        if "2.5" in model_l and "pro" not in model_l:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        resp = _post_with_400_fallback(requests_mod, url, headers, payload, cfg.timeout_sec)
        text, _ = _extract_gemini_native_text(resp.json())
        parsed = parse_json_response(text)

    if not parsed:
        raise RuntimeError(f"VLM response was not valid JSON: {text[:160]}")
    return parsed


def call_cloud_vlm(
    *,
    cfg: VlmConfig,
    system_prompt: str,
    user_prompt: str,
    grid_bgr: np.ndarray,
    sequence_frames: Optional[List[bytes]] = None,
    ring_frames: Optional[List[Tuple[float, bytes]]] = None,
    has_reference_sheet: bool = False,
) -> Dict[str, Any]:
    """Execute HTTP completion (or dry_run simulation) and return parsed verdict dict."""
    prov, base_url, api_key = resolve_vlm_endpoint(cfg)
    if prov == "dry_run":
        conf = max(0.0, min(1.0, float(cfg.dry_run_confidence)))
        cat = "high_risk_concealment_attempt" if conf >= 0.5 else "regular_browsing"
        return {
            "person_detected": True,
            "item_removed_from_shelf": "yes" if conf >= 0.5 else "no",
            "visible_destination": "personal_bag_or_clothing" if conf >= 0.5 else "returned_to_shelf",
            "likely_personal_object": False,
            "staff_or_byob_indicator": False,
            "behavior_category": cat,
            "confidence_score": conf,
            "sequence_verification": {
                "reach_detected": True,
                "hand_to_shelf_location": "shelf zone",
                "personal_bag_backpack_overlap": False,
                "occluded_movement": True,
            },
            "justification": "Dry-run VLM verification triggered from behavioral sequence.",
        }

    needs_key = prov not in ("ollama", "lmstudio")
    if not base_url:
        raise RuntimeError(f"Empty base URL for VLM provider '{prov}'")
    if needs_key and not api_key:
        raise RuntimeError(f"Missing API key for VLM provider '{prov}'")

    import requests

    if prov == "gemini" and bool(getattr(cfg, "use_native_video", True)) and ring_frames:
        native_url = _resolve_gemini_native_url(base_url, cfg.model_name)
        if native_url:
            mp4_bytes = encode_ring_clip_mp4_bytes(ring_frames, default_fps=8.0, max_dim=640)
            if mp4_bytes:
                ref_sheet = sequence_frames[0] if (has_reference_sheet and sequence_frames) else None
                return _call_gemini_native_video(
                    requests_mod=requests,
                    cfg=cfg,
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    mp4_bytes=mp4_bytes,
                    reference_sheet_jpg=ref_sheet,
                )

    seq_cap = max(6, min(16, int(getattr(cfg, "max_sequence_frames", 14) or 14)))
    images_b64 = _encode_sequence_b64(grid_bgr, sequence_frames, max_frames=6 if prov == "gemini" else seq_cap)
    if not images_b64:
        raise ValueError("No valid frames could be encoded for VLM verification.")

    if prov == "anthropic":
        url = base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        content: Any = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
            for b64 in images_b64
        ]
        content.append({"type": "text", "text": user_prompt})
        payload: Dict[str, Any] = {
            "model": cfg.model_name,
            "max_tokens": cfg.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        resp = _post_with_400_fallback(requests, url, headers, payload, cfg.timeout_sec)
        blocks = resp.json().get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        parsed = parse_json_response(text)
    else:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        user_content: Any = [{"type": "text", "text": user_prompt}]
        for b64 in images_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        model_l = (cfg.model_name or "").strip().lower()
        is_gemini = prov == "gemini" or "gemini" in model_l
        is_openai_reasoning = prov == "openai" and any(
            k in model_l for k in ("gpt-5", "gpt-6", "o1", "o3", "o4")
        )
        supports_reasoning_effort = is_gemini and any(v in model_l for v in ("2.5", "3", "4"))
        req_tokens = int(getattr(cfg, "max_tokens", 2048) or 2048)
        effective_max_tokens = max(req_tokens, 2048) if (is_gemini or is_openai_reasoning) else max(256, req_tokens)

        payload = {
            "model": cfg.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if is_openai_reasoning:
            payload["max_completion_tokens"] = effective_max_tokens
        else:
            payload["max_tokens"] = effective_max_tokens
            payload["temperature"] = cfg.temperature

        if supports_reasoning_effort:
            payload["reasoning_effort"] = "low"

        resp = _post_with_400_fallback(requests, url, headers, payload, cfg.timeout_sec)
        choice = resp.json()["choices"][0]
        text, finish_reason = _extract_choice_text(choice)
        parsed = parse_json_response(text)

        if not parsed and (finish_reason == "length" or text.lstrip().startswith(("{", "```"))):
            new_tokens = max(effective_max_tokens * 2, 4096)
            if is_openai_reasoning:
                payload["max_completion_tokens"] = new_tokens
            else:
                payload["max_tokens"] = new_tokens
            if supports_reasoning_effort and "2.5" in model_l and "pro" not in model_l:
                payload["reasoning_effort"] = "none"
            resp = _post_with_400_fallback(requests, url, headers, payload, cfg.timeout_sec)
            choice = resp.json()["choices"][0]
            text, _ = _extract_choice_text(choice)
            parsed = parse_json_response(text)

    if not parsed:
        raise RuntimeError(f"VLM response was not valid JSON: {text[:160]}")
    return parsed



class VlmWorker:
    """Bounded concurrent queue invoking Cloud VLM with per-track FIFO / non-blocking cross-track delivery."""

    def __init__(
        self,
        cfg: VlmConfig,
        *,
        on_verdict: Callable[[Dict[str, Any], Dict[str, Any], np.ndarray, List[Tuple[float, bytes]]], None],
        on_error: Callable[[Dict[str, Any], str, np.ndarray], None],
        max_queue: int = 16,
        max_concurrent: int = 3,
    ):
        self._cfg = cfg
        self._on_verdict = on_verdict
        self._on_error = on_error
        self._max_concurrent = max(1, min(8, int(getattr(cfg, "max_concurrent", max_concurrent) or max_concurrent)))
        self._queue: Deque[Tuple[Dict[str, Any], np.ndarray, List[Tuple[float, bytes]]]] = deque(maxlen=max(1, max_queue))
        self._inflight: Deque[Tuple[Future, Dict[str, Any], np.ndarray, List[Tuple[float, bytes]], float]] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._next_dispatch_ts = 0.0
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent, thread_name_prefix="storeguard-vlm-http")
        self._thread = threading.Thread(target=self._run, name="storeguard-vlm", daemon=True)
        self._thread.start()

    def _notify(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def update_config(self, cfg: VlmConfig) -> None:
        with self._cv:
            self._cfg = cfg
            self._next_dispatch_ts = 0.0
            self._cv.notify_all()

    @property
    def queue_depth(self) -> int:
        with self._cv:
            return len(self._queue) + len(self._inflight)

    def submit(self, candidate: Dict[str, Any], grid_bgr: np.ndarray, ring_frames: List[Tuple[float, bytes]]) -> None:
        item = (dict(candidate), grid_bgr.copy(), list(ring_frames))
        with self._cv:
            if time.monotonic() < self._next_dispatch_ts and self._queue:
                cam_id = item[0].get("camera_id")
                tid = item[0].get("track_id")
                for idx in range(len(self._queue) - 1, -1, -1):
                    q_cand = self._queue[idx][0]
                    if q_cand.get("camera_id") == cam_id and q_cand.get("track_id") == tid:
                        self._queue[idx] = item
                        self._cv.notify_all()
                        return
            self._queue.append(item)
            self._cv.notify_all()

    def stop(self, timeout_sec: float = 3.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
        self._executor.shutdown(wait=False)

    def _run(self) -> None:
        while True:
            completed: Optional[Tuple[Future, Dict[str, Any], np.ndarray, List[Tuple[float, bytes]], float]] = None
            with self._cv:
                while True:
                    # Dispatch up to max_concurrent requests while respecting pacing/cooldown
                    while len(self._inflight) < self._max_concurrent and self._queue and not self._stop:
                        cfg = self._cfg
                        min_interval = _effective_min_interval(cfg)
                        now_mono = time.monotonic()
                        if now_mono < self._next_dispatch_ts:
                            break
                        candidate, grid_bgr, ring_frames = self._queue.popleft()
                        if min_interval > 0.0:
                            self._next_dispatch_ts = now_mono + min_interval
                        prov_cur, base_cur, _ = resolve_vlm_endpoint(cfg)
                        seq_frames = candidate.get("sequence_frames")
                        will_use_native = (
                            prov_cur == "gemini"
                            and bool(getattr(cfg, "use_native_video", True))
                            and len(ring_frames) >= 2
                            and _resolve_gemini_native_url(base_cur, cfg.model_name) is not None
                        )
                        seq_cap = 6 if prov_cur == "gemini" else max(6, min(16, int(getattr(cfg, "max_sequence_frames", 14) or 14)))
                        candidate["_using_native_video"] = will_use_native
                        candidate["_vlm_media_mode"] = (
                            "native_video_mp4"
                            if will_use_native
                            else f"multi_frame_jpeg({min(len(seq_frames or []), seq_cap)})"
                        )
                        sys_prompt, user_prompt = build_vlm_prompts(candidate)
                        call_kwargs: Dict[str, Any] = {
                            "cfg": cfg,
                            "system_prompt": sys_prompt,
                            "user_prompt": user_prompt,
                            "grid_bgr": grid_bgr,
                            "sequence_frames": seq_frames if isinstance(seq_frames, list) else None,
                        }
                        if will_use_native:
                            call_kwargs["ring_frames"] = ring_frames
                        if candidate.get("has_reference_sheet"):
                            call_kwargs["has_reference_sheet"] = True
                        fut = self._executor.submit(call_cloud_vlm, **call_kwargs)
                        fut.add_done_callback(lambda _: self._notify())
                        self._inflight.append((fut, candidate, grid_bgr, ring_frames, time.monotonic()))

                    # Deliver completed verdicts per-track in FIFO order without cross-track head-of-line blocking
                    seen_tracks = set()
                    for idx, entry in enumerate(self._inflight):
                        fut_i, cand_i = entry[0], entry[1]
                        track_key = (cand_i.get("camera_id"), cand_i.get("track_id"))
                        if fut_i.done() and track_key not in seen_tracks:
                            completed = entry
                            del self._inflight[idx]
                            break
                        seen_tracks.add(track_key)
                    if completed is not None:
                        break

                    if self._stop and not self._inflight and not self._queue:
                        return
                    if self._stop and not self._inflight:
                        return

                    wait_timeout = 0.5
                    if self._queue and not self._stop:
                        rem = self._next_dispatch_ts - time.monotonic()
                        if rem > 0.0:
                            wait_timeout = min(0.5, max(0.01, rem))
                    self._cv.wait(timeout=wait_timeout)

            fut, candidate, grid_bgr, ring_frames, submitted_ts = completed
            candidate["_vlm_latency_sec"] = round(max(0.0, time.monotonic() - submitted_ts), 3)
            try:
                verdict = fut.result()
                self._on_verdict(candidate, verdict, grid_bgr, ring_frames)
            except VlmRateLimitError as rl:
                with self._cv:
                    cool = 60.0 if rl.is_daily_quota else max(15.0, min(60.0, float(rl.retry_after_sec or 15.0)))
                    self._next_dispatch_ts = max(self._next_dispatch_ts, time.monotonic() + cool)
                    self._cv.notify_all()
                try:
                    self._on_error(candidate, str(rl), grid_bgr)
                except Exception:
                    pass
            except Exception as exc:
                try:
                    self._on_error(candidate, str(exc), grid_bgr)
                except Exception:
                    pass
