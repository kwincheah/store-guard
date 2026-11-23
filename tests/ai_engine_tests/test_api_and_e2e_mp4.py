from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.config import AppConfig, CameraConfig, EngineSettings, VlmConfig, ZoneConfig, save_config_atomic
from ai_engine.main import create_app

TEST_MP4 = Path(r"C:\Users\winch\.gemini\antigravity\brain\7d3601f0-70b1-4ca6-8295-7780bcda97be\.user_uploaded\media_1795120909932_632393e2.mp4")


def test_fastapi_routes_hot_reload_and_real_mp4_e2e(tmp_path):
    cfg = AppConfig(
        engine=EngineSettings(
            device="auto",
            target_fps=24,
            models_dir=str(ROOT / "models"),
            evidence_dir=str(tmp_path / "evidence"),
            vlm=VlmConfig(provider="dry_run", dry_run_confidence=0.86, alert_threshold_pct=65.0),
        ),
        cameras=[
            CameraConfig(
                camera_id="cam_01",
                name="Aisle Test",
                rtsp_url=str(TEST_MP4),
                enabled=True,
                dwell_sec=3.0,
                release_wait_sec=3.5,
                zones=[],
            ),
            CameraConfig(
                camera_id="cam_02",
                name="Offline Cam",
                rtsp_url="",
                enabled=False,
            ),
        ],
    )
    cfg_path = tmp_path / "cameras.json"
    save_config_atomic(cfg_path, cfg)

    app = create_app(config_path=cfg_path, base_dir=ROOT, config_override=cfg)
    with TestClient(app) as client:
        # 1. Health & live CPU/GPU device switching
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        health = h_resp.json()
        assert health["status"] == "ready"
        assert health["active_streams"] == 2
        assert "cuda_available" in health
        assert health["runtime"] == "apex_vision"
        assert "cam_01" in health["procedures"]

        # Verify Hikrobot-style Apex SDK routes & zero secret serialization
        sol_resp = client.get("/sdk/solution")
        assert sol_resp.status_code == 200
        assert [p["name"] for p in sol_resp.json()["procedures"]] == ["cam_01", "cam_02"]

        procs_resp = client.get("/sdk/procedures")
        assert procs_resp.status_code == 200
        assert len(procs_resp.json()["procedures"]) == 2

        p_mut = client.put(
            "/sdk/procedures/cam_01/params/node_motion/diff_thresh",
            json={"value": 21},
        )
        assert p_mut.status_code == 200
        save_resp = client.post("/sdk/solution/save")
        assert save_resp.status_code == 200
        saved_xml = Path(save_resp.json()["solution_path"]).read_text(encoding="utf-8")
        assert "<ApexSolution" in saved_xml and "diff_thresh" in saved_xml
        assert "__vlm_api_key__" not in saved_xml

        d_cpu = client.put("/engine/device", json={"device": "cpu"})
        assert d_cpu.status_code == 200
        assert d_cpu.json()["requested_device"] == "cpu"
        assert d_cpu.json()["resolved_device"] == "cpu"

        d_gpu = client.put("/engine/device", json={"device": "cuda:0"})
        assert d_gpu.status_code == 200
        assert d_gpu.json()["requested_device"] == "cuda:0"
        assert client.get("/health").json()["requested_device"] == "cuda:0"

        d_auto = client.put("/engine/device", json={"device": "auto"})
        assert d_auto.status_code == 200
        assert d_auto.json()["requested_device"] == "auto"

        # 2. Hot-reload per-camera zones on cam_01 only without affecting cam_02
        new_zones = [
            ZoneConfig(name="shelf_1", role="shelf", rect_norm=[0.0, 0.1, 0.25, 0.8]).model_dump(),
            ZoneConfig(name="aisle_1", role="aisle", rect_norm=[0.18, 0.05, 0.65, 0.9]).model_dump(),
        ]
        z_resp = client.put("/cameras/cam_01/zones", json={"zones": new_zones})
        assert z_resp.status_code == 200
        assert z_resp.json()["zone_count"] == 2

        # Wait briefly for stream worker to decode first frame and grab snapshot
        snap_ok = False
        for _ in range(30):
            s_resp = client.get("/cameras/cam_01/snapshot?overlays=true")
            if s_resp.status_code == 200 and len(s_resp.content) > 500:
                snap_ok = True
                break
            time.sleep(0.1)
        assert snap_ok

        # 3. Reset zones to empty on cam_01, pause live reader so replay is deterministic, and verify real MP4 -> ALERT over WS
        client.put("/cameras/cam_01/zones", json={"zones": []})

        import cv2
        orch = app.state.storeguard["orch"]
        orch.pause_worker("cam_01")
        orch.reset_camera_state("cam_01")
        cap = cv2.VideoCapture(str(TEST_MP4))
        assert cap.isOpened()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        triggered_cands = []

        with client.websocket_connect("/ws/events") as ws:
            for idx in range(240):
                ok, frame = cap.read()
                if not ok:
                    break
                ok_j, buf = cv2.imencode(".jpg", frame)
                ring = [(i / fps, buf.tobytes()) for i in range(max(0, idx - 15), idx + 1)] if ok_j else []
                cand = orch.process_single_frame("cam_01", frame, idx / fps, ring_frames=ring)
                if cand is not None:
                    triggered_cands.append((idx, cand["track_id"], cand["behaviors"]))
            cap.release()

            assert len(triggered_cands) >= 1
            alert_msg = None
            for _ in range(20):
                msg = ws.receive_json()
                if msg.get("type") == "ALERT":
                    alert_msg = msg["payload"]
                    break
            assert alert_msg is not None
            assert alert_msg["camera_id"] == "cam_01"
            assert alert_msg["confidence_pct"] == 86.0
            assert Path(alert_msg["grid_path"]).is_file()
            assert Path(alert_msg["clip_path"]).is_file()


def test_live_snapshot_draws_person_bbox_and_id_tag():
    import numpy as np
    import cv2
    from ai_engine.rtsp.stream_worker import CameraStreamWorker

    worker = CameraStreamWorker(CameraConfig(camera_id="cam_t", name="Test Cam", rtsp_url="", enabled=False))
    blank = np.full((360, 640, 3), 30, dtype=np.uint8)
    with worker._lock:
        worker._latest_frame = blank
        worker._latest_ts = 1.0
        worker._seq = 1

    raw_jpg = worker.get_snapshot_jpeg(overlays=False)
    ov_jpg = worker.get_snapshot_jpeg(
        overlays=True,
        overlay_tracks=[
            {"track_id": 7, "bbox": [120, 80, 90, 180], "confidence": 0.92, "phase": "REACH", "armed": False},
            {"track_id": 9, "bbox": [340, 90, 85, 170], "confidence": 0.88, "phase": "TUCK", "armed": True},
        ],
    )
    assert raw_jpg is not None and ov_jpg is not None
    assert raw_jpg != ov_jpg

    raw_img = cv2.imdecode(np.frombuffer(raw_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    ov_img = cv2.imdecode(np.frombuffer(ov_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    diff = np.abs(ov_img.astype(np.int16) - raw_img.astype(np.int16)).sum(axis=2)
    # Check both bbox border region and top ID tag region were actually painted
    assert int(np.count_nonzero(diff[50:90, 115:235])) > 80
    assert int(np.count_nonzero(diff[55:95, 335:455])) > 80


def test_local_clip_detects_once_and_rerun_endpoint_resets_pass(tmp_path):
    import cv2
    import numpy as np
    from unittest.mock import patch

    clip_path = tmp_path / "short_clip.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 240))
    assert writer.isOpened()
    for i in range(6):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (40 + i * 20, 40), (110 + i * 20, 200), (200, 200, 200), -1)
        writer.write(img)
    writer.release()

    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            target_fps=12,
            models_dir=str(ROOT / "models"),
            evidence_dir=str(tmp_path / "evidence"),
            vlm=VlmConfig(provider="dry_run", dry_run_confidence=0.88, alert_threshold_pct=65.0),
        ),
        cameras=[
            CameraConfig(camera_id="cam_01", name="Local Clip", rtsp_url=str(clip_path), enabled=True, zones=[]),
        ],
    )
    cfg_path = tmp_path / "cameras.json"
    save_config_atomic(cfg_path, cfg)

    app = create_app(config_path=cfg_path, base_dir=ROOT, config_override=cfg)
    with TestClient(app) as client:
        orch = app.state.storeguard["orch"]
        worker = orch._workers["cam_01"]
        assert worker.is_local_file is True

        # Wait until natural loop #1 marks clip_detection_done=True
        done_seen = False
        for _ in range(80):
            h = client.get("/health").json()["cameras"][0]
            if h["is_local_file"] and h["clip_detection_done"]:
                done_seen = True
                break
            time.sleep(0.1)
        assert done_seen

        # In preview_only mode, process_single_frame must still populate track overlays without invoking BehaviorGate/VLM
        from apex_vision.core.base_tool import ToolExecutionResult, ToolStatus

        orch.pause_worker("cam_01")
        sample_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(sample_frame, (50, 40), (130, 200), (220, 220, 220), -1)
        proc = orch.solution["cam_01"]
        proc.node("node_motion").reset()
        gate = proc.node("node_behavior")._ensure_gate(None)
        fake_pose_res = ToolExecutionResult(
            status=ToolStatus.OK,
            message="mock pose",
            outputs={
                "Image": sample_frame,
                "Detections": [{"Box": [50, 40, 80, 160], "Score": 0.91, "Class_Id": 0, "Class": "person", "Keypoints": []}],
                "Keypoints": [[]],
                "Count": 1,
                "Status": "OK",
            },
        )
        with patch.object(proc.node("node_pose"), "execute", return_value=fake_pose_res), patch.object(gate, "step") as bg_step:
            cand1 = orch.process_single_frame("cam_01", sample_frame, 1.0)
            cand2 = orch.process_single_frame("cam_01", sample_frame, 1.2)
            assert cand1 is None and cand2 is None
            assert bg_step.call_count == 0
            assert len(orch._latest_overlays["cam_01"]["tracks"]) >= 1

        # Trigger manual /rerun -> resets clip_detection_done back to False for Pass 0
        resp = client.post("/cameras/cam_01/rerun")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "camera_id": "cam_01"}
        assert worker.clip_detection_done is False

        resp_missing = client.post("/cameras/unknown_cam/rerun")
        assert resp_missing.status_code == 404


def test_snapshot_sd_downscale_scales_overlays_and_uses_seq_cache(tmp_path):
    import cv2
    import numpy as np
    from ai_engine.rtsp.stream_worker import CameraStreamWorker

    # 1) Direct unit check on CameraStreamWorker: downscale + overlay coordinate scaling + cache hit/invalidation
    worker = CameraStreamWorker(CameraConfig(camera_id="cam_hd", name="HD Cam", rtsp_url="", enabled=False))
    hd_frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
    with worker._lock:
        worker._latest_frame = hd_frame
        worker._latest_ts = 1.0
        worker._seq = 1

    tracks = [{"track_id": 4, "bbox": [300, 180, 160, 320], "confidence": 0.91, "phase": "REACH", "armed": False}]
    sd_raw = worker.get_snapshot_jpeg(overlays=False, max_width=640, quality=72, cache_key=(0, (), False))
    sd_ov1 = worker.get_snapshot_jpeg(
        overlays=True,
        overlay_tracks=tracks,
        max_width=640,
        quality=72,
        cache_key=(1, (), False),
    )
    sd_ov2 = worker.get_snapshot_jpeg(
        overlays=True,
        overlay_tracks=tracks,
        max_width=640,
        quality=72,
        cache_key=(1, (), False),
    )
    assert sd_raw is not None and sd_ov1 is not None and sd_ov2 is not None
    assert sd_ov1 is sd_ov2  # exact seq + cache_key hit skips re-encode

    raw_img = cv2.imdecode(np.frombuffer(sd_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    ov_img = cv2.imdecode(np.frombuffer(sd_ov1, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert raw_img.shape == (360, 640, 3)
    assert ov_img.shape == (360, 640, 3)
    diff = np.abs(ov_img.astype(np.int16) - raw_img.astype(np.int16)).sum(axis=2)
    # Scaled bbox top-left is around (150, 90), width 80, height 160 in 640x360 coordinates
    assert int(np.count_nonzero(diff[60:115, 140:240])) > 60

    # Invalidating cache_key or incrementing _seq produces a fresh buffer
    sd_ov_new_key = worker.get_snapshot_jpeg(
        overlays=True,
        overlay_tracks=tracks,
        max_width=640,
        quality=72,
        cache_key=(2, (), False),
    )
    assert sd_ov_new_key is not None and sd_ov_new_key is not sd_ov1
    with worker._lock:
        worker._seq = 2
    sd_ov_new_seq = worker.get_snapshot_jpeg(
        overlays=True,
        overlay_tracks=tracks,
        max_width=640,
        quality=72,
        cache_key=(2, (), False),
    )
    assert sd_ov_new_seq is not None and sd_ov_new_seq is not sd_ov_new_key

    # 2) FastAPI /cameras/{id}/snapshot route check with max_width & Cache-Control header
    cfg = AppConfig(
        engine=EngineSettings(
            device="cpu",
            target_fps=8,
            models_dir=str(ROOT / "models"),
            evidence_dir=str(tmp_path / "evidence"),
            vlm=VlmConfig(provider="dry_run"),
        ),
        cameras=[CameraConfig(camera_id="cam_hd", name="HD Route Cam", rtsp_url="", enabled=False)],
    )
    cfg_path = tmp_path / "cameras.json"
    save_config_atomic(cfg_path, cfg)
    app = create_app(config_path=cfg_path, base_dir=ROOT, config_override=cfg)
    with TestClient(app) as client:
        orch = app.state.storeguard["orch"]
        w = orch._workers["cam_hd"]
        with w._lock:
            w._latest_frame = hd_frame
            w._latest_ts = 2.0
            w._seq = 5
        orch._cache_track_overlays("cam_hd", tracks, {4: {"phase": "REACH", "armed": False}}, preview_only=False)

        r_sd = client.get("/cameras/cam_hd/snapshot?overlays=true&max_width=640&quality=72")
        assert r_sd.status_code == 200
        assert r_sd.headers.get("cache-control") == "no-store"
        img_sd = cv2.imdecode(np.frombuffer(r_sd.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img_sd.shape == (360, 640, 3)

        r_hd = client.get("/cameras/cam_hd/snapshot?overlays=false&max_width=0")
        assert r_hd.status_code == 200
        img_hd = cv2.imdecode(np.frombuffer(r_hd.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img_hd.shape == (720, 1280, 3)

        # 3) MJPEG stream endpoint emits bounded multipart JPEG frames + /health reports scaling telemetry
        r_mjpeg = client.get("/cameras/cam_hd/mjpeg?overlays=true&max_width=480&quality=75&fps=15&max_frames=2")
        assert r_mjpeg.status_code == 200
        assert "multipart/x-mixed-replace" in (r_mjpeg.headers.get("content-type") or "")
        assert r_mjpeg.content.count(b"--frame\r\n") == 2
        assert b"\xff\xd8" in r_mjpeg.content

        health = client.get("/health").json()
        assert int(health["worker_threads"]) >= 2
        assert "tensorrt_available" in health
        assert "batch_metrics" in health
        assert "decoder_backend" in health["cameras"][0]
