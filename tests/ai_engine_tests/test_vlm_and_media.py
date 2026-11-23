import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.api.ws_events import EventBroadcaster
from ai_engine.config import AppConfig, EngineSettings, VlmConfig
from ai_engine.inference.gpu_scheduler import EngineOrchestrator
from ai_engine.media.clip_writer import RetentionManager, save_grid_jpeg, write_ring_clip_mp4
from ai_engine.vlm.vlm_worker import VlmWorker, call_cloud_vlm


def test_vlm_worker_dry_run_and_error_callback():
    verdicts = []
    errors = []
    done = threading.Event()

    def on_ok(cand, verdict, grid, ring):
        verdicts.append((cand["track_id"], verdict["confidence_score"]))
        if len(verdicts) >= 2:
            done.set()

    def on_err(cand, msg, grid):
        errors.append((cand["track_id"], msg))
        done.set()

    worker = VlmWorker(VlmConfig(provider="dry_run", dry_run_confidence=0.88), on_verdict=on_ok, on_error=on_err)
    grid = np.zeros((120, 180, 3), dtype=np.uint8)
    try:
        worker.submit({"camera_id": "cam_01", "track_id": 1}, grid, [])
        worker.submit({"camera_id": "cam_01", "track_id": 2}, grid, [])
        assert done.wait(timeout=3.0)
        assert verdicts == [(1, 0.88), (2, 0.88)]

        done.clear()
        worker.update_config(VlmConfig(provider="custom", base_url="", api_key=""))
        worker.submit({"camera_id": "cam_01", "track_id": 3}, grid, [])
        assert done.wait(timeout=3.0)
        assert len(errors) == 1 and "Empty base URL" in errors[0][1]
    finally:
        worker.stop()


def test_clip_writer_and_retention_pruning(tmp_path):
    ev_dir = tmp_path / "evidence"
    ev_dir.mkdir()

    frame = np.full((180, 320, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    ring = [(i * 0.125, buf.tobytes()) for i in range(16)]

    clip_path = write_ring_clip_mp4(ring, ev_dir / "test_clip.mp4", default_fps=8.0)
    assert clip_path is not None and Path(clip_path).is_file() and Path(clip_path).stat().st_size > 100

    grid_path = save_grid_jpeg(frame, ev_dir / "old_grid.jpg")
    old_mtime = time.time() - (45 * 86400)
    os.utime(grid_path, (old_mtime, old_mtime))

    mgr = RetentionManager(ev_dir, retention_days=30, max_evidence_gb=20.0)
    removed = mgr.prune_now()
    assert removed == 1
    assert not Path(grid_path).exists()
    assert Path(clip_path).exists()


def test_vlm_multiframe_json_mode_concurrency_and_four_fact_gate(tmp_path):
    # 1. Multi-frame chronological sequence + response_format with automatic 400 retry fallback
    posted_payloads = []
    resp_400 = MagicMock(status_code=400)
    resp_ok = MagicMock(status_code=200)
    resp_ok.raise_for_status.return_value = None
    resp_ok.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"person_detected":true,"item_removed_from_shelf":"yes","visible_destination":"personal_bag_or_clothing","likely_personal_object":false,"staff_or_byob_indicator":false,"behavior_category":"high_risk_concealment_attempt","confidence_score":0.92,"justification":"Placed item into backpack."}'
            }
        }]
    }

    def fake_post(url, headers=None, json=None, timeout=None):
        posted_payloads.append(dict(json))
        return resp_400 if len(posted_payloads) == 1 else resp_ok

    frame = np.full((64, 64, 3), 80, dtype=np.uint8)
    ok_j, jpg = cv2.imencode(".jpg", frame)
    assert ok_j
    seq = [jpg.tobytes(), jpg.tobytes(), jpg.tobytes()]

    with patch("requests.post", side_effect=fake_post):
        out = call_cloud_vlm(
            cfg=VlmConfig(provider="openai", api_key="sk-test", model_name="gpt-4o-mini"),
            system_prompt="sys",
            user_prompt="usr",
            grid_bgr=frame,
            sequence_frames=seq,
        )
    assert out["confidence_score"] == 0.92
    assert "response_format" in posted_payloads[0]
    assert "response_format" not in posted_payloads[1]
    img_parts = [p for p in posted_payloads[1]["messages"][-1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 3

    # 2. Concurrent VlmWorker runs jobs in parallel: cross-track delivers immediately, same-track stays FIFO
    delivered = []
    done = threading.Event()

    def on_ok(cand, verdict, grid, ring):
        delivered.append((cand["camera_id"], cand["track_id"], verdict.get("job_tag")))
        if len(delivered) == 4:
            done.set()

    def slow_call(*, cfg, system_prompt, user_prompt, grid_bgr, sequence_frames=None):
        # cam_01 Track 1's first job is slow (0.18s); fast cam_01 Track 2 & cam_02 Track 3 deliver ahead of it,
        # while cam_01 Track 1's second job waits for job_1a on that same track.
        if "Tag_1a" in user_prompt:
            time.sleep(0.18)
            tag = "job_1a"
        elif "Tag_1b" in user_prompt:
            time.sleep(0.02)
            tag = "job_1b"
        elif "Cam_2" in user_prompt:
            time.sleep(0.02)
            tag = "job_2"
        else:
            time.sleep(0.02)
            tag = "job_3"
        return {"behavior_category": "regular_browsing", "confidence_score": 0.1, "job_tag": tag}

    with patch("ai_engine.vlm.vlm_worker.call_cloud_vlm", side_effect=slow_call):
        worker = VlmWorker(VlmConfig(provider="dry_run"), on_verdict=on_ok, on_error=lambda *_: None, max_concurrent=4)
        try:
            worker.submit({"camera_id": "cam_01", "camera_name": "Tag_1a", "track_id": 1}, frame, [])
            worker.submit({"camera_id": "cam_01", "camera_name": "Cam_2", "track_id": 2}, frame, [])
            worker.submit({"camera_id": "cam_01", "camera_name": "Tag_1b", "track_id": 1}, frame, [])
            worker.submit({"camera_id": "cam_02", "camera_name": "Cam_3", "track_id": 3}, frame, [])
            assert done.wait(timeout=3.0)
            tags = [t for _, _, t in delivered]
            assert tags.index("job_2") < tags.index("job_1a")
            assert tags.index("job_3") < tags.index("job_1a")
            assert tags.index("job_1a") < tags.index("job_1b")
        finally:
            worker.stop()

    # 3. Four-fact consistency gate in EngineOrchestrator suppresses false-positive verdicts
    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            models_dir=str(ROOT / "models"),
            evidence_dir=str(tmp_path / "ev"),
            vlm=VlmConfig(provider="dry_run", alert_threshold_pct=65.0),
        ),
        cameras=[],
    )
    orch = EngineOrchestrator(cfg, ROOT, EventBroadcaster())
    try:
        emitted = []
        with patch.object(orch.broadcaster, "emit_sync", side_effect=lambda t, p: emitted.append((t, p))):
            cand = {"camera_id": "cam_01", "camera_name": "Aisle 1", "track_id": 5, "behaviors": ["reach_and_tuck"], "nearby_bags": []}
            # Suppressed: item went into store basket or was a personal object
            orch._handle_vlm_verdict(cand, {"behavior_category": "high_risk_concealment_attempt", "confidence_score": 0.91, "visible_destination": "store_basket_or_cart"}, frame, [])
            orch._handle_vlm_verdict(cand, {"behavior_category": "high_risk_concealment_attempt", "confidence_score": 0.91, "likely_personal_object": True}, frame, [])
            orch._handle_vlm_verdict(cand, {"behavior_category": "high_risk_concealment_attempt", "confidence_score": 0.91, "visible_destination": "unclear"}, frame, [])
            assert len(emitted) == 0

            # Confirmed theft -> ALERT emitted
            orch._handle_vlm_verdict(cand, {"behavior_category": "high_risk_concealment_attempt", "confidence_score": 0.91, "visible_destination": "personal_bag_or_clothing", "item_removed_from_shelf": "yes", "justification": "Concealed into backpack."}, frame, [])
            assert len(emitted) == 1 and emitted[0][0] == "ALERT"
    finally:
        orch.stop()


def test_orchestrator_cross_cam_reference_sheet_and_carried_openly_hook(tmp_path):
    from pathlib import Path
    import cv2
    import numpy as np
    from ai_engine.api.ws_events import EventBroadcaster
    from ai_engine.config import AppConfig, CameraConfig, EngineSettings, VlmConfig, ZoneConfig
    from ai_engine.inference.gpu_scheduler import EngineOrchestrator
    from ai_engine.vlm.prompt_builder import build_vlm_prompts

    real_models_dir = Path(__file__).resolve().parents[2] / "models"
    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            target_fps=8,
            models_dir=str(real_models_dir),
            evidence_dir=str(tmp_path / "evidence"),
            vlm=VlmConfig(provider="dry_run", dry_run_confidence=0.9),
        ),
        cameras=[
            CameraConfig(
                camera_id="cam1",
                name="Aisle 1",
                rtsp_url="",
                enabled=False,
                dwell_sec=0.4,
                release_wait_sec=0.4,
                cooldown_sec=12.0,
                zones=[ZoneConfig(name="shelf_snacks", role="shelf", rect_norm=[0.6, 0.2, 0.3, 0.6])],
            ),
            CameraConfig(
                camera_id="cam2",
                name="Corridor",
                rtsp_url="",
                enabled=False,
                dwell_sec=0.4,
                release_wait_sec=0.4,
                cooldown_sec=12.0,
                zones=[],
            ),
        ],
    )

    broadcaster = EventBroadcaster()
    emitted = []
    broadcaster.emit_sync = lambda et, payload: emitted.append((et, payload))
    orch = EngineOrchestrator(cfg, tmp_path, broadcaster)
    try:
        submitted = []
        orch.vlm_worker.submit = lambda cand, grid, frames: submitted.append((cand, grid, frames))

        def kps(cx, lw, rw):
            pts = [[0.0, 0.0, 0.0] for _ in range(17)]
            pts[0] = [cx, 100.0, 0.9]
            pts[5] = [cx - 25.0, 140.0, 0.9]
            pts[6] = [cx + 25.0, 140.0, 0.9]
            pts[7] = [cx - 30.0, 175.0, 0.9]
            pts[8] = [cx + 30.0, 175.0, 0.9]
            pts[9] = [float(lw[0]), float(lw[1]), 0.92]
            pts[10] = [float(rw[0]), float(rw[1]), 0.92]
            pts[11] = [cx - 18.0, 215.0, 0.9]
            pts[12] = [cx + 18.0, 215.0, 0.9]
            return pts

        from apex_vision.core.base_tool import ToolExecutionResult, ToolStatus

        state_pose = {"tracks": [], "bags": []}

        def _attach_flow_mocks(proc):
            def _pose_exec(ctx):
                dets = [
                    {"Box": t["bbox"], "Keypoints": t.get("keypoints", []), "Score": 0.95, "Class": "person", "Class_Id": 0}
                    for t in state_pose["tracks"]
                ]
                return ToolExecutionResult(
                    status=ToolStatus.OK,
                    outputs={
                        "Image": ctx.get("Image"),
                        "Detections": dets,
                        "Keypoints": [t.get("keypoints", []) for t in state_pose["tracks"]],
                        "Count": len(dets),
                        "Status": "OK",
                    },
                )

            def _bags_exec(ctx):
                bags = list(state_pose["bags"])
                return ToolExecutionResult(
                    status=ToolStatus.OK,
                    outputs={"Detections": bags, "Count": len(bags), "Cache_Hit": False, "Status": "OK"},
                )

            def _trk_exec(ctx):
                trks = list(state_pose["tracks"])
                return ToolExecutionResult(
                    status=ToolStatus.OK,
                    outputs={"Image": ctx.get("Image"), "Tracks": trks, "Tracked_Count": len(trks), "Status": "OK"},
                )

            proc.node("node_pose").execute = _pose_exec
            proc.node("node_bags").execute = _bags_exec
            proc.node("node_tracker").execute = _trk_exec

        _attach_flow_mocks(orch.solution["cam1"])
        _attach_flow_mocks(orch.solution["cam2"])

        def make_frame(shift):
            f = np.full((360, 640, 3), 75, dtype=np.uint8)
            cv2.rectangle(f, (260 + shift, 80), (350 + shift, 300), (20, 40, 210), -1)
            cv2.rectangle(f, (420, 90), (580, 260), (90, 130, 160), -1)
            return f

        # 1. Cam1 sees shopper tid=11 reach into shelf_snacks -> pickup pool populated
        state_pose["tracks"] = [{"track_id": 11, "bbox": [310, 80, 90, 220], "keypoints": kps(355, (300, 170), (430, 160))}]
        state_pose["bags"] = []
        for i in range(4):
            orch.process_single_frame("cam1", make_frame(i * 4), 10.0 + i * 0.25, ring_frames=[])

        active_pickups = orch.recent_pickups.get_active(11.0)
        assert len(active_pickups) == 1
        assert active_pickups[0].camera_id == "cam1"
        assert active_pickups[0].shelf_zone == "shelf_snacks"

        # 2. VLM reports carried_openly for cam1 tid=11 -> no ALERT emitted, pickup stays OPEN, cam1 Path A TTL extended
        orch._handle_vlm_verdict(
            {"camera_id": "cam1", "camera_name": "Aisle 1", "track_id": 11},
            {
                "behavior_category": "regular_browsing",
                "confidence_score": 0.25,
                "item_removed_from_shelf": "yes",
                "visible_destination": "carried_openly",
                "likely_personal_object": False,
                "staff_or_byob_indicator": False,
            },
            np.zeros((200, 200, 3), dtype=np.uint8),
            [],
        )
        assert [e for e in emitted if e[0] == "ALERT"] == []
        assert orch.recent_pickups.get_active(12.0)[0].resolution == "carried_openly"
        assert orch.solution["cam1"].node("node_behavior")._gate._tracks[11]["has_merchandise_until"] is not None

        # 3. Cam2 (no shelf ROI) sees shopper tid=51 show open hands then stuff into backpack -> Path B prepends ref sheet
        state_pose["bags"] = [{"Class": "backpack", "Box": [320, 160, 60, 85]}]
        for i in range(5):
            state_pose["tracks"] = [{"track_id": 51, "bbox": [260, 80, 90, 220], "keypoints": kps(305, (235, 170), (375, 170))}]
            orch.process_single_frame("cam2", make_frame(i * 3), 20.0 + i * 0.25, ring_frames=[])

        for i in range(5, 16):
            state_pose["tracks"] = [{"track_id": 51, "bbox": [260, 80, 90, 220], "keypoints": kps(305, (235, 170), (345, 195))}]
            orch.process_single_frame("cam2", make_frame(15 + i * 2), 20.0 + i * 0.25, ring_frames=[])
            if submitted:
                break

        assert len(submitted) == 1
        cand2, _grid2, _frames2 = submitted[0]
        assert cand2["trigger_mode"] == "shelf_free_tuck"
        assert cand2["has_reference_sheet"] is True
        assert len(cand2["sequence_frames"]) >= 2
        sheet_img = cv2.imdecode(np.frombuffer(cand2["sequence_frames"][0], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert sheet_img is not None and sheet_img.shape[0] >= 200

        sys_p, usr_p = build_vlm_prompts(cand2)
        assert "RECENT STORE PICKUPS" in sys_p
        assert "RECENT STORE PICKUPS reference sheet" in usr_p
        assert "shelf_free_tuck" in usr_p
    finally:
        orch.stop()


def test_gemini_thinking_budget_and_truncated_json_retry():
    from unittest.mock import patch
    import numpy as np
    from ai_engine.config import VlmConfig
    from ai_engine.vlm.vlm_worker import call_cloud_vlm

    assert VlmConfig().max_tokens == 2048

    posted = []

    class FakeResp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_post(url, headers=None, json=None, timeout=None):
        posted.append(dict(json or {}))
        if len(posted) == 1:
            return FakeResp({
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": '```json\n{\n  "person_detected": true,\n  "behavior_'},
                }]
            })
        return FakeResp({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": '{"person_detected": true, "item_removed_from_shelf": "no", "visible_destination": "returned_to_shelf", "likely_personal_object": false, "staff_or_byob_indicator": false, "behavior_category": "regular_browsing", "confidence_score": 0.12, "justification": "Returned item cleanly."}'
                },
            }]
        })

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with patch("requests.post", side_effect=fake_post):
        out = call_cloud_vlm(
            cfg=VlmConfig(provider="gemini", api_key="AIzaTest", model_name="gemini-2.5-flash", max_tokens=512),
            system_prompt="sys",
            user_prompt="usr",
            grid_bgr=frame,
            sequence_frames=[b"\xff\xd8\xff\xd9"],
        )

    assert len(posted) == 2
    assert posted[0]["max_tokens"] == 2048
    assert posted[0]["reasoning_effort"] == "low"
    assert posted[1]["max_tokens"] == 4096
    assert posted[1]["reasoning_effort"] == "none"
    assert out["behavior_category"] == "regular_browsing"
    assert out["confidence_score"] == 0.12


def test_vlm_429_retry_after_backoff_and_daily_quota_message():
    import threading
    import time
    from unittest.mock import patch
    import numpy as np
    import pytest
    from ai_engine.config import VlmConfig
    from ai_engine.vlm.vlm_worker import (
        VlmRateLimitError,
        VlmWorker,
        _effective_min_interval,
        call_cloud_vlm,
    )

    assert _effective_min_interval(VlmConfig(provider="gemini")) == 6.0
    assert _effective_min_interval(VlmConfig(provider="dry_run")) == 0.0
    assert _effective_min_interval(VlmConfig(provider="gemini", min_request_interval_sec=0.0)) == 0.0

    class FakeResp:
        def __init__(self, status_code, body, headers=None):
            self.status_code = status_code
            self._body = body
            self.headers = headers or {}
            self.text = str(body)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._body

    # 1. Transient RPM/TPM 429 with short Retry-After recovers automatically on retry
    calls = {"n": 0}

    def post_recover(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(
                429,
                {"error": {"message": "Quota exceeded for GenerateRequestsPerMinutePerProject, retry in 0.01s"}},
                headers={"Retry-After": "0.01"},
            )
        return FakeResp(
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": '{"person_detected": true, "behavior_category": "regular_browsing", "confidence_score": 0.1}'}}]},
        )

    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    with patch("requests.post", side_effect=post_recover):
        out = call_cloud_vlm(
            cfg=VlmConfig(provider="gemini", api_key="AIzaTest", model_name="gemini-2.5-flash"),
            system_prompt="sys",
            user_prompt="usr",
            grid_bgr=frame,
            sequence_frames=[b"\xff\xd8\xff\xd9"],
        )
    assert calls["n"] == 2
    assert out["behavior_category"] == "regular_browsing"

    # 2. Daily RPD exhaustion fails immediately with clear quota guidance & VlmWorker sets cooldown
    def post_daily_429(url, headers=None, json=None, timeout=None):
        return FakeResp(
            429,
            {"error": {"message": "Quota exceeded for quota metric Generate Content API requests per day (GenerateRequestsPerDayPerProjectPerModel-FreeTier), please retry in 5s"}},
        )

    with patch("requests.post", side_effect=post_daily_429):
        with pytest.raises(VlmRateLimitError) as exc_info:
            call_cloud_vlm(
                cfg=VlmConfig(provider="gemini", api_key="AIzaTest", model_name="gemini-2.5-flash"),
                system_prompt="sys",
                user_prompt="usr",
                grid_bgr=frame,
                sequence_frames=[b"\xff\xd8\xff\xd9"],
            )
    assert exc_info.value.is_daily_quota is True
    assert "daily free-tier RPD quota exhausted" in str(exc_info.value)

    errs = []
    done = threading.Event()

    def on_err(cand, msg, grid):
        errs.append(msg)
        done.set()

    with patch("ai_engine.vlm.vlm_worker.call_cloud_vlm", side_effect=VlmRateLimitError("HTTP 429 RPM", retry_after_sec=20.0, is_daily_quota=False)):
        w = VlmWorker(VlmConfig(provider="gemini", api_key="AIzaTest", min_request_interval_sec=0.0), on_verdict=lambda *_: None, on_error=on_err)
        try:
            t0 = time.monotonic()
            w.submit({"camera_id": "cam_01", "track_id": 1}, frame, [])
            assert done.wait(timeout=3.0)
            assert w._next_dispatch_ts >= t0 + 18.0
            assert "HTTP 429" in errs[0]
        finally:
            w.stop()


def test_deepseek_provider_400_fallback_and_vlm_audit_logger(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    import numpy as np
    from ai_engine.config import VlmConfig
    from ai_engine.vlm.audit_logger import VlmAuditLogger
    from ai_engine.vlm.vlm_worker import call_cloud_vlm, resolve_vlm_endpoint

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-env-secret")
    prov, url, key = resolve_vlm_endpoint(VlmConfig(provider="deepseek", api_key="", model_name="deepseek-flash"))
    assert prov == "deepseek"
    assert url == "https://api.deepseek.com"
    assert key == "ds-env-secret"

    posted = []

    class FakeResp:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body
            self.headers = {}
            self.text = str(body)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._body

    def fake_post(url, headers=None, json=None, timeout=None):
        posted.append(dict(json or {}))
        if len(posted) == 1:
            return FakeResp(400, {"error": {"message": "Unsupported parameter: max_tokens; use max_completion_tokens"}})
        return FakeResp(
            200,
            {"choices": [{"finish_reason": "stop", "message": {"content": '{"person_detected": true, "behavior_category": "high_risk_concealment_attempt", "visible_destination": "personal_bag_or_clothing", "confidence_score": 0.89, "justification": "Concealed item."}'}}]},
        )

    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    with patch("requests.post", side_effect=fake_post):
        out = call_cloud_vlm(
            cfg=VlmConfig(provider="deepseek", api_key="ds-key", model_name="deepseek-flash"),
            system_prompt="sys",
            user_prompt="usr",
            grid_bgr=frame,
            sequence_frames=[b"\xff\xd8\xff\xd9"],
        )
    assert len(posted) == 2
    assert "max_completion_tokens" in posted[1] and "max_tokens" not in posted[1]
    assert out["confidence_score"] == 0.89

    audit = VlmAuditLogger(tmp_path / "logs")
    audit.log_verdict(
        candidate={"camera_id": "cam_01", "camera_name": "Aisle 1", "track_id": 2, "trigger_mode": "shelf_reach_and_delta", "_vlm_latency_sec": 1.25},
        provider="deepseek",
        model_name="deepseek-flash",
        decision="ALERT",
        conf_pct=89.0,
        threshold_pct=65.0,
        verdict=out,
        event_id="evt_ds_1",
    )
    lines = audit.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["provider"] == "deepseek"
    assert parsed["decision"] == "ALERT"
    assert parsed["latency_sec"] == 1.25
    assert "ds-key" not in lines[0]


def test_gemini_native_video_mp4_upload_and_custom_base_url_fallback():
    from unittest.mock import patch
    import cv2
    import numpy as np
    from ai_engine.config import VlmConfig
    from ai_engine.media.clip_writer import encode_ring_clip_mp4_bytes
    from ai_engine.vlm.vlm_worker import call_cloud_vlm

    frame = np.full((180, 320, 3), 95, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    ring = [(i * 0.125, buf.tobytes()) for i in range(12)]
    mp4 = encode_ring_clip_mp4_bytes(ring, default_fps=8.0, max_dim=320)
    assert mp4 is not None and len(mp4) > 200 and b"ftyp" in mp4[:64]

    calls = []

    class FakeResp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body
            self.headers = {}
            self.text = str(body)

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, dict(headers or {}), dict(json or {})))
        if ":generateContent" in url:
            return FakeResp({
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"text": "thinking trace", "thought": True},
                            {"text": '{"person_detected":true,"item_removed_from_shelf":"yes","visible_destination":"personal_bag_or_clothing","likely_personal_object":false,"staff_or_byob_indicator":false,"behavior_category":"high_risk_concealment_attempt","confidence_score":0.94,"justification":"Observed theft in video clip."}'}
                        ]
                    },
                }]
            })
        return FakeResp({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"person_detected":true,"item_removed_from_shelf":"no","visible_destination":"unclear","likely_personal_object":false,"staff_or_byob_indicator":false,"behavior_category":"regular_browsing","confidence_score":0.15,"justification":"Proxy call."}'},
            }]
        })

    # 1. Default Google Gemini endpoint with ring_frames uses native generateContent + video/mp4 + x-goog-api-key header
    with patch("requests.post", side_effect=fake_post):
        out_native = call_cloud_vlm(
            cfg=VlmConfig(provider="gemini", api_key="AIzaNativeSecret", model_name="gemini-2.5-flash", use_native_video=True, video_fps=2.0),
            system_prompt="sys prompt",
            user_prompt="usr prompt",
            grid_bgr=frame,
            sequence_frames=[buf.tobytes()],
            ring_frames=ring,
        )
    assert len(calls) == 1
    u0, h0, p0 = calls[0]
    assert u0 == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert "AIzaNativeSecret" not in u0
    assert h0.get("x-goog-api-key") == "AIzaNativeSecret"
    video_part = p0["contents"][0]["parts"][0]
    assert video_part["inlineData"]["mimeType"] == "video/mp4"
    assert video_part["videoMetadata"]["fps"] == 2.0
    assert out_native["confidence_score"] == 0.94

    # 2. Custom non-Google proxy base_url stays on OpenAI-compatible /chat/completions
    with patch("requests.post", side_effect=fake_post):
        out_proxy = call_cloud_vlm(
            cfg=VlmConfig(provider="gemini", base_url="http://127.0.0.1:9999/v1", api_key="proxy-key", model_name="gemini-2.5-flash", use_native_video=True),
            system_prompt="sys prompt",
            user_prompt="usr prompt",
            grid_bgr=frame,
            sequence_frames=[buf.tobytes()],
            ring_frames=ring,
        )
    assert len(calls) == 2
    assert calls[1][0] == "http://127.0.0.1:9999/v1/chat/completions"
    assert out_proxy["confidence_score"] == 0.15


def test_prompt_builder_neutral_delta_and_chain_of_evidence():
    from ai_engine.vlm.prompt_builder import SYSTEM_PROMPT, build_vlm_prompts

    assert "object_seen_in_hand" in SYSTEM_PROMPT
    assert "action_near_bag_or_pocket" in SYSTEM_PROMPT
    assert "concealment probability" in SYSTEM_PROMPT.lower()
    assert "skeptical" not in SYSTEM_PROMPT.lower()

    _, usr_unverified = build_vlm_prompts({
        "camera_id": "cam_01",
        "trigger_mode": "shelf_reach_and_delta",
        "shelf_delta_status": "unchanged",
        "shelf_delta_score": 0.0,
        "sequence_frames": [b"a", b"b"],
    })
    assert "unchanged (score=0.00)" not in usr_unverified
    assert "unverified" in usr_unverified

    _, usr_confirmed = build_vlm_prompts({
        "camera_id": "cam_01",
        "trigger_mode": "shelf_reach_and_delta",
        "shelf_delta_status": "confirmed",
        "shelf_delta_score": 0.42,
        "sequence_frames": [b"a", b"b"],
    })
    assert "confirmed (score=0.42)" in usr_confirmed


def test_sequence_endpoint_preservation_on_pre_reach_and_ref_sheet(tmp_path):
    from ai_engine.api.ws_events import EventBroadcaster
    from ai_engine.config import AppConfig, CameraConfig, EngineSettings, VlmConfig, ZoneConfig
    from ai_engine.inference.gpu_scheduler import EngineOrchestrator
    from ai_engine.tracking.grid_builder import extract_sequence_frames

    def mk_jpg(val: int) -> bytes:
        img = np.full((64, 64, 3), int(val), dtype=np.uint8)
        ok_e, buf = cv2.imencode(".jpg", img)
        assert ok_e
        return buf.tobytes()

    pre_jpg = mk_jpg(15)
    rel_jpg = mk_jpg(245)
    samples = [(i * 0.3, mk_jpg(35 + i * 8), "TUCK" if 3 <= i <= 14 else "POST") for i in range(19)]
    samples.append((6.0, rel_jpg, "RELEASE"))
    seq = extract_sequence_frames(
        {"samples": samples, "pre_reach_jpg": pre_jpg, "reach_ts": 0.6, "tuck_ts": 1.5, "last_conceal_ts": 4.2},
        max_frames=14,
    )
    assert len(seq) == 14
    assert seq[0] == pre_jpg
    assert seq[-1] == rel_jpg

    real_models_dir = Path(__file__).resolve().parents[2] / "models"
    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            target_fps=8,
            models_dir=str(real_models_dir),
            evidence_dir=str(tmp_path / "evidence_end"),
            vlm=VlmConfig(provider="dry_run", dry_run_confidence=0.9, max_sequence_frames=6),
        ),
        cameras=[
            CameraConfig(
                camera_id="cam2",
                name="Corridor",
                rtsp_url="",
                enabled=False,
                dwell_sec=0.5,
                release_wait_sec=1.5,
                cooldown_sec=12.0,
                zones=[],
            ),
        ],
    )
    orch = EngineOrchestrator(cfg, tmp_path, EventBroadcaster())
    try:
        submitted = []
        orch.vlm_worker.submit = lambda cand, grid, frames: submitted.append((cand, grid, frames))

        def kps(cx, lw, rw):
            pts = [[0.0, 0.0, 0.0] for _ in range(17)]
            pts[0] = [cx, 100.0, 0.9]
            pts[5] = [cx - 25.0, 140.0, 0.9]
            pts[6] = [cx + 25.0, 140.0, 0.9]
            pts[7] = [cx - 30.0, 175.0, 0.9]
            pts[8] = [cx + 30.0, 175.0, 0.9]
            pts[9] = [float(lw[0]), float(lw[1]), 0.92]
            pts[10] = [float(rw[0]), float(rw[1]), 0.92]
            pts[11] = [cx - 18.0, 215.0, 0.9]
            pts[12] = [cx + 18.0, 215.0, 0.9]
            return pts

        from apex_vision.core.base_tool import ToolExecutionResult, ToolStatus

        state_pose = {"tracks": [], "bags": [{"Class": "backpack", "Box": [320, 160, 60, 85]}]}
        proc2 = orch.solution["cam2"]
        proc2.node("node_pose").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={
                "Image": ctx.get("Image"),
                "Detections": [{"Box": t["bbox"], "Keypoints": t.get("keypoints", []), "Score": 0.95} for t in state_pose["tracks"]],
                "Keypoints": [t.get("keypoints", []) for t in state_pose["tracks"]],
                "Count": len(state_pose["tracks"]),
                "Status": "OK",
            },
        )
        proc2.node("node_bags").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={"Detections": list(state_pose["bags"]), "Count": len(state_pose["bags"]), "Cache_Hit": False, "Status": "OK"},
        )
        proc2.node("node_tracker").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={"Image": ctx.get("Image"), "Tracks": list(state_pose["tracks"]), "Tracked_Count": len(state_pose["tracks"]), "Status": "OK"},
        )

        pickup_crop = np.full((120, 80, 3), 110, dtype=np.uint8)
        orch.recent_pickups.record_pickup(camera_id="cam1", track_id=9, ts=10.0, person_crop_bgr=pickup_crop, shelf_zone="Snacks")

        dark_frame = np.full((360, 640, 3), 45, dtype=np.uint8)
        release_frame = np.full((360, 640, 3), 240, dtype=np.uint8)

        # Open hands -> backpack stuff -> open hands settle (many >6 samples so ref sheet thinning must preserve [RELEASE])
        for i in range(6):
            state_pose["tracks"] = [{"track_id": 21, "bbox": [260, 80, 90, 220], "keypoints": kps(305, (235, 170), (375, 170))}]
            orch.process_single_frame("cam2", dark_frame, 12.0 + i * 0.30, ring_frames=[])
        for i in range(5):
            state_pose["tracks"] = [{"track_id": 21, "bbox": [260, 80, 90, 220], "keypoints": kps(305, (235, 170), (345, 195))}]
            orch.process_single_frame("cam2", dark_frame, 14.0 + i * 0.30, ring_frames=[])
        # Settled frames; dispatch fires when quiescence condition completes -> release_frame captured as RELEASE endpoint
        for i in range(6):
            state_pose["tracks"] = [{"track_id": 21, "bbox": [260, 80, 90, 220], "keypoints": kps(305, (235, 170), (375, 170))}]
            orch.process_single_frame("cam2", release_frame, 15.6 + i * 0.30, ring_frames=[])
            if submitted:
                break

        assert len(submitted) == 1
        cand_out = submitted[0][0]
        assert cand_out["has_reference_sheet"] is True
        seq_out = cand_out["sequence_frames"]
        assert len(seq_out) == 6
        last_img = cv2.imdecode(np.frombuffer(seq_out[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert last_img is not None and float(last_img.mean()) > 200.0
    finally:
        orch.stop()


def test_bag_infer_stride_caching_and_reset(tmp_path):
    from ai_engine.config import CameraConfig

    real_models_dir = Path(__file__).resolve().parents[2] / "models"
    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            target_fps=8,
            bag_infer_stride=3,
            models_dir=str(real_models_dir),
            evidence_dir=str(tmp_path / "ev_stride"),
            vlm=VlmConfig(provider="dry_run"),
        ),
        cameras=[
            CameraConfig(camera_id="cam_s", name="StrideCam", rtsp_url="", enabled=False, dwell_sec=3.0, release_wait_sec=3.5, cooldown_sec=12.0, zones=[]),
        ],
    )
    orch = EngineOrchestrator(cfg, tmp_path, EventBroadcaster())
    try:
        import sys
        from unittest.mock import patch
        from apex_vision.core.base_tool import ToolExecutionResult, ToolStatus

        bag_calls = 0
        state = {"poses": [{"bbox": [200, 80, 70, 220], "keypoints": []}]}

        class _FakeBagSess:
            def infer(self, img, **kw):
                nonlocal bag_calls
                bag_calls += 1
                return [{"Class": "backpack", "Box": [210, 150, 50, 70]}]

        proc = orch.solution["cam_s"]
        proc.node("node_motion").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={"Image": ctx.get("Image"), "Active": True, "Motion_ROI": [0, 0, 0, 0], "Inactive_Branches": [], "Status": "ACTIVE"},
        )
        proc.node("node_pose").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={
                "Image": ctx.get("Image"),
                "Detections": [{"Box": p["bbox"], "Keypoints": p.get("keypoints", []), "Score": 0.92} for p in state["poses"]],
                "Keypoints": [p.get("keypoints", []) for p in state["poses"]],
                "Count": len(state["poses"]),
                "Status": "OK",
            },
        )
        proc.node("node_tracker").execute = lambda ctx: ToolExecutionResult(
            status=ToolStatus.OK,
            outputs={
                "Image": ctx.get("Image"),
                "Tracks": [
                    {"track_id": idx + 1, "bbox": d["Box"], "keypoints": d.get("Keypoints", []), "confidence": 0.92}
                    for idx, d in enumerate(ctx.get("Detections") or ())
                ],
                "Tracked_Count": len(ctx.get("Detections") or ()),
                "Status": "OK",
            },
        )

        bag_mod = sys.modules["apex_plugin_theft_recognition_plugin.context_detector_tool"]
        frame = np.full((360, 640, 3), 70, dtype=np.uint8)

        with patch.object(bag_mod, "get_shared_onnx_bag_session", return_value=_FakeBagSess()):
            # Unarmed stride=3: run at 0, skip 1 & 2, run at 3
            orch.process_single_frame("cam_s", frame, 0.0, ring_frames=[])
            orch.process_single_frame("cam_s", frame, 0.125, ring_frames=[])
            orch.process_single_frame("cam_s", frame, 0.250, ring_frames=[])
            assert bag_calls == 1
            orch.process_single_frame("cam_s", frame, 0.375, ring_frames=[])
            assert bag_calls == 2

            # Pose count change refreshes immediately
            state["poses"] = [{"bbox": [200, 80, 70, 220], "keypoints": []}, {"bbox": [350, 80, 70, 220], "keypoints": []}]
            orch.process_single_frame("cam_s", frame, 0.500, ring_frames=[])
            assert bag_calls == 3

            # 0 poses clears cache; next 1-pose frame runs immediately
            state["poses"] = []
            orch.process_single_frame("cam_s", frame, 0.625, ring_frames=[])
            assert bag_calls == 3
            state["poses"] = [{"bbox": [200, 80, 70, 220], "keypoints": []}]
            orch.process_single_frame("cam_s", frame, 0.750, ring_frames=[])
            assert bag_calls == 4

            # Armed track halves effective stride to 2 (skip at 0.875, refresh at 1.000)
            proc.node("node_behavior")._gate._tracks[1]["armed_ts"] = 0.750
            with proc.manager._globals_lock if proc.manager else orch._lock:
                orch.solution.manager.global_variables["__armed_cam_s"] = True
            orch.process_single_frame("cam_s", frame, 0.875, ring_frames=[])
            assert bag_calls == 4
            orch.process_single_frame("cam_s", frame, 1.000, ring_frames=[])
            assert bag_calls == 5
    finally:
        orch.stop()
