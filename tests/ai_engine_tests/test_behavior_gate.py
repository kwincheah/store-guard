from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.config import ZoneConfig
from ai_engine.inference.gpu_scheduler import compute_aisle_crop
from ai_engine.tracking.behavior_gate import BehaviorGate
from ai_engine.tracking.shelf_delta import LocalizedShelfDeltaTracker


def _mk_track(tid: int, cx: float, *, reach_left: bool = False, tuck: bool = False):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[0] = [cx, 95.0, 0.9]
    kps[5] = [cx - 22.0, 130.0, 0.9]
    kps[6] = [cx + 22.0, 130.0, 0.9]
    kps[11] = [cx - 18.0, 210.0, 0.9]
    kps[12] = [cx + 18.0, 210.0, 0.9]
    if reach_left:
        kps[9] = [55.0, 140.0, 0.9]  # Inside left shelf ROI [0.0..0.2] on 640px wide frame (0..128)
        kps[10] = [cx + 22.0, 185.0, 0.9]
    elif tuck:
        kps[9] = [cx, 180.0, 0.9]
        kps[10] = [cx + 6.0, 185.0, 0.9]
    else:
        kps[9] = [cx - 25.0, 195.0, 0.9]
        kps[10] = [cx + 25.0, 195.0, 0.9]
    return {"track_id": tid, "bbox": [int(cx - 35), 80, 70, 220], "keypoints": kps}


def test_behavior_gate_multi_zone_loop_wrap_and_fifo_queue():
    zones = [
        ZoneConfig(name="shelf_left", role="shelf", rect_norm=[0.0, 0.1, 0.22, 0.8]),
        ZoneConfig(name="aisle_main", role="aisle", rect_norm=[0.18, 0.05, 0.72, 0.9]),
    ]
    gate = BehaviorGate("cam_01", "Cosmetics", dwell_sec=3.0, release_wait_sec=3.5, cooldown_sec=12.0, zones=zones)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    dispatched = []
    timeline = [
        (0.0, False, False), (1.0, False, False), (2.0, False, False), (3.2, False, False),
        (3.5, True, False), (4.5, False, True),
        (8.2, False, False),  # both tracks complete 3.5s release simultaneously
        (8.3, False, False),  # 2nd track dequeued on next step
        (0.0, False, False),  # video loop rewind should advance monotonically without error
        (0.1, False, False),
    ]
    for ts, reach, tuck in timeline:
        grid, cand, _ = gate.step(
            frame,
            ts,
            [_mk_track(1, 180.0, reach_left=reach, tuck=tuck), _mk_track(2, 420.0, reach_left=reach, tuck=tuck)],
            [],
        )
        if cand is not None:
            assert grid is not None and grid.shape == (768, 1152, 3)
            dispatched.append(cand["track_id"])

    assert dispatched == [1, 2]


def _make_textured_shelf_frame(*, include_a: bool = True, include_b: bool = True, shadow_center: bool = False) -> np.ndarray:
    img = np.full((360, 640, 3), 55, dtype=np.uint8)
    # Draw shelf backing grid texture in [100..520, 40..220]
    for y in range(40, 220, 12):
        img[y:y + 4, 100:520] = (85, 85, 90)
    for x in range(100, 520, 16):
        img[40:220, x:x + 4] = (75, 80, 85)
    if include_a:
        cv2.rectangle(img, (140, 80), (188, 136), (240, 220, 80), -1)
        cv2.rectangle(img, (146, 90), (182, 126), (40, 60, 210), -1)
    if include_b:
        cv2.rectangle(img, (416, 80), (464, 136), (70, 235, 190), -1)
        cv2.rectangle(img, (422, 92), (458, 124), (200, 40, 90), -1)
    if shadow_center:
        img[50:200, 245:365] = np.clip(img[50:200, 245:365].astype(np.float32) * 0.68 - 8.0, 0, 255).astype(np.uint8)
    return img


def test_localized_shelf_delta_multi_person_concurrent_occlusion_contention_shadow():
    tracker = LocalizedShelfDeltaTracker()
    shelf_rects = [(100, 40, 420, 180)]
    clean_frame = _make_textured_shelf_frame(include_a=True, include_b=True)

    # Warm up background ring (>= 0.7s)
    for ts in (0.0, 0.25, 0.5, 0.75):
        tracker.step(clean_frame, ts, shelf_rects, [])

    # 1. Track 1 reaches left spot (puts nothing in pocket), Track 2 concurrently takes Item B on right spot
    res_conc = {}
    for ts in (1.0, 1.2):
        res_conc = tracker.step(
            clean_frame,
            ts,
            shelf_rects,
            [
                {"tid": 1, "bbox": (130, 140, 70, 200), "wrists": [(164, 108), (160, 220)], "reach_active": True, "touched_shelf_idx": 0},
                {"tid": 2, "bbox": (406, 140, 70, 200), "wrists": [(440, 108), (438, 220)], "reach_active": True, "touched_shelf_idx": 0},
            ],
        )
    frame_b_gone = _make_textured_shelf_frame(include_a=True, include_b=False)
    for ts in (1.5, 1.7, 1.9):
        res_conc = tracker.step(
            frame_b_gone,
            ts,
            shelf_rects,
            [
                {"tid": 1, "bbox": (120, 245, 70, 110), "wrists": [(150, 260), (160, 260)], "reach_active": False, "touched_shelf_idx": None},
                {"tid": 2, "bbox": (410, 245, 70, 110), "wrists": [(440, 260), (450, 260)], "reach_active": False, "touched_shelf_idx": None},
            ],
        )
    assert res_conc[1]["status"] == "unchanged" and res_conc[1]["confirmed"] is False
    assert res_conc[2]["status"] == "confirmed" and res_conc[2]["confirmed"] is True

    # 2. Occlusion wait: Track 1 grabs Item A, then bystander Track 3 walks directly over Track 1's patch before clearing
    tracker.clear_episode(1)
    tracker.clear_episode(2)
    tracker.step(frame_b_gone, 2.4, shelf_rects, [])
    tracker.step(frame_b_gone, 2.8, shelf_rects, [])
    for ts in (3.1, 3.3):
        tracker.step(
            frame_b_gone,
            ts,
            shelf_rects,
            [{"tid": 1, "bbox": (130, 140, 70, 200), "wrists": [(164, 108), (160, 220)], "reach_active": True, "touched_shelf_idx": 0}],
        )
    frame_ab_gone = _make_textured_shelf_frame(include_a=False, include_b=False)
    for ts in (3.6, 3.8):
        r_occ = tracker.step(
            frame_ab_gone,
            ts,
            shelf_rects,
            [
                {"tid": 1, "bbox": (120, 245, 70, 110), "wrists": [(150, 260)], "reach_active": False, "touched_shelf_idx": None},
                {"tid": 3, "bbox": (125, 45, 85, 200), "wrists": [(155, 195), (185, 195)], "reach_active": False, "touched_shelf_idx": None},
            ],
        )
        assert r_occ[1]["status"] == "occluded_unverified" and r_occ[1]["confirmed"] is False
    for ts in (4.0, 4.2):
        r_clear = tracker.step(
            frame_ab_gone,
            ts,
            shelf_rects,
            [{"tid": 1, "bbox": (120, 245, 70, 110), "wrists": [(150, 260)], "reach_active": False, "touched_shelf_idx": None}],
        )
    assert r_clear[1]["status"] == "confirmed" and r_clear[1]["confirmed"] is True

    # 3. Same-spot multi-person contention (Tracks 4 & 5 touch same center area)
    for ts in (5.0, 5.2):
        r_cont = tracker.step(
            frame_ab_gone,
            ts,
            shelf_rects,
            [
                {"tid": 4, "bbox": (260, 140, 70, 200), "wrists": [(295, 110)], "reach_active": True, "touched_shelf_idx": 0},
                {"tid": 5, "bbox": (290, 140, 70, 200), "wrists": [(308, 112)], "reach_active": True, "touched_shelf_idx": 0},
            ],
        )
    assert r_cont[4]["status"] == "contested_multi_person" and r_cont[4]["confirmed"] is False

    # 4. Uniform shadow cast over clean textured center without removing an item -> rejected
    shadow_frame = _make_textured_shelf_frame(include_a=False, include_b=False, shadow_center=True)
    for ts in (6.5, 6.7):
        tracker.step(
            frame_ab_gone,
            ts,
            shelf_rects,
            [{"tid": 6, "bbox": (270, 140, 70, 200), "wrists": [(305, 115)], "reach_active": True, "touched_shelf_idx": 0}],
        )
    for ts in (7.0, 7.2, 7.4):
        r_shad = tracker.step(
            shadow_frame,
            ts,
            shelf_rects,
            [{"tid": 6, "bbox": (270, 250, 70, 100), "wrists": [(300, 265)], "reach_active": False, "touched_shelf_idx": None}],
        )
    assert r_shad[6]["status"] == "unchanged" and r_shad[6]["confirmed"] is False


def test_zone_exit_dispatch_phone_basket_suppression_and_union_crop():
    zones = [
        ZoneConfig(name="shelf_left", role="shelf", rect_norm=[0.02, 0.10, 0.20, 0.70]),
        ZoneConfig(name="aisle_main", role="aisle", rect_norm=[0.24, 0.15, 0.30, 0.75]),
    ]
    crop = compute_aisle_crop(zones, 1000, 500)
    # Union crop includes both shelf_left (x=20) AND aisle_main (x=540), never clipping shelf-reach arms
    assert crop is not None and crop[0] <= 20 and (crop[0] + crop[2]) >= 540

    # 1. Zone-exit immediately dispatches an armed track instead of dropping it
    gate = BehaviorGate("cam_z", "Cosmetics", dwell_sec=1.0, release_wait_sec=4.0, cooldown_sec=12.0, zones=zones)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for ts in (0.0, 0.6, 1.2):
        gate.step(frame, ts, [_mk_track(9, 250.0)], [])
    gate.step(frame, 1.5, [_mk_track(9, 250.0, reach_left=True)], [])
    gate.step(frame, 2.0, [_mk_track(9, 250.0, tuck=True)], [])
    # Track leaves aisle (cx=590 outside aisle_main [153..345]) -> >0.5s triggers zone_exit dispatch!
    gate.step(frame, 2.2, [_mk_track(9, 590.0)], [])
    grid, cand_exit, _ = gate.step(frame, 2.9, [_mk_track(9, 590.0)], [])
    assert cand_exit is not None
    assert cand_exit["dispatch_cause"] == "zone_exit"
    assert len(cand_exit.get("sequence_frames") or []) >= 2
    assert grid is not None and grid.shape == (768, 1152, 3)

    # 2. Benign shopping basket & smartphone suppress tuck_now (zero false alarm)
    gate_fp = BehaviorGate("cam_fp", "Cosmetics", dwell_sec=1.0, release_wait_sec=1.0, cooldown_sec=12.0, zones=zones)
    basket_det = [{"Class": "shopping basket", "Box": [210, 155, 65, 65]}]
    phone_det = [{"Class": "cell phone", "Box": [230, 165, 36, 40]}]
    for ts in (0.0, 0.6, 1.2):
        gate_fp.step(frame, ts, [_mk_track(10, 250.0), _mk_track(11, 250.0)], [])
    gate_fp.step(frame, 1.5, [_mk_track(10, 250.0, reach_left=True), _mk_track(11, 250.0, reach_left=True)], [])
    for ts in (2.0, 3.2, 4.0):
        _, c1, _ = gate_fp.step(frame, ts, [_mk_track(10, 250.0, tuck=True)], basket_det)
        assert c1 is None
    for ts in (4.5, 5.8, 6.5):
        _, c2, _ = gate_fp.step(frame, ts, [_mk_track(11, 250.0, tuck=True)], phone_det)
        assert c2 is None


def _kps_with_wrists(
    cx: float = 200.0,
    top: float = 80.0,
    w: float = 90.0,
    h: float = 220.0,
    lw=None,
    rw=None,
    sh_span: float = 52.0,
):
    pts = [[0.0, 0.0, 0.0] for _ in range(17)]
    sy = top + h * 0.25
    hy = top + h * 0.58
    pts[0] = [cx, top + h * 0.08, 0.9]
    pts[5] = [cx - sh_span * 0.5, sy, 0.9]
    pts[6] = [cx + sh_span * 0.5, sy, 0.9]
    pts[7] = [cx - sh_span * 0.65, sy + h * 0.14, 0.9]
    pts[8] = [cx + sh_span * 0.65, sy + h * 0.14, 0.9]
    pts[11] = [cx - 18.0, hy, 0.9]
    pts[12] = [cx + 18.0, hy, 0.9]
    if lw is not None:
        pts[9] = [float(lw[0]), float(lw[1]), 0.92]
    if rw is not None:
        pts[10] = [float(rw[0]), float(rw[1]), 0.92]
    return pts


def test_recent_pickups_pool_ttl_dedup_rush_hour_and_sheet():
    import cv2
    import numpy as np
    from ai_engine.tracking.recent_pickups import RecentPickupsPool, compute_upper_color_hist

    pool = RecentPickupsPool(ttl_sec=120.0, max_entries=16)

    def solid_person(bgr):
        img = np.zeros((160, 90, 3), dtype=np.uint8)
        img[16:120, 10:80] = bgr
        return img

    red_crop = solid_person((20, 20, 230))
    blue_crop = solid_person((230, 30, 20))
    green_crop = solid_person((30, 220, 30))

    # 1. Deduplication for same (camera_id, track_id) within 6s
    pool.record_pickup(camera_id="cam1", track_id=5, ts=10.0, person_crop_bgr=blue_crop, shelf_zone="Snacks")
    pool.record_pickup(camera_id="cam1", track_id=5, ts=12.0, person_crop_bgr=red_crop, shelf_zone="Drinks")
    assert len(pool.get_active(12.0)) == 1
    assert pool.get_active(12.0)[0].shelf_zone == "Drinks"

    # 2. Mark returned removes from active sheet; mark carried_openly retains with flag
    pool.mark_resolution("cam1", 5, "returned", now_ts=13.0)
    assert len(pool.get_active(13.0)) == 0

    pool.record_pickup(camera_id="cam1", track_id=9, ts=20.0, person_crop_bgr=red_crop, shelf_zone="Liquor")
    assert pool.mark_resolution("cam1", 9, "carried_openly", now_ts=25.0)
    sheet_bytes, meta = pool.build_reference_sheet(25.0, query_hist=compute_upper_color_hist(red_crop), max_tiles=6)
    assert sheet_bytes is not None
    assert len(meta) == 1 and meta[0]["resolution"] == "carried_openly"
    decoded = cv2.imdecode(np.frombuffer(sheet_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[0] >= 200 and decoded.shape[1] >= 400

    # 3. Rush-hour (> 6 active) keeps total <= 6 and color tie-breaker brings matching shirt into sheet
    for idx in range(7):
        t = 50.0 + idx
        c = green_crop if idx != 0 else red_crop
        pool.record_pickup(camera_id=f"cam_r{idx}", track_id=100 + idx, ts=t, person_crop_bgr=c, shelf_zone=f"Aisle{idx}")
    _, rush_meta = pool.build_reference_sheet(60.0, query_hist=compute_upper_color_hist(red_crop), max_tiles=6)
    assert len(rush_meta) == 6
    assert any(m["camera_id"] == "cam_r0" for m in rush_meta)

    # 4. 2-minute TTL expires entries older than 120s
    assert pool.get_active(200.0) == []


def test_dual_trigger_path_b_smart_tuck_and_resting_hands_suppression():
    import numpy as np
    from ai_engine.tracking.behavior_gate import BehaviorGate

    frame = np.full((360, 640, 3), 80, dtype=np.uint8)
    # Bright torso patch so upper_color_hist is populated
    frame[100:240, 165:235] = (30, 50, 210)
    p_box = [155, 80, 90, 220]

    # A. Hands-in-pocket from frame 0 (never open) must NOT arm Path B
    gate_rest = BehaviorGate("cam2", "Corridor", dwell_sec=0.6, release_wait_sec=0.5, cooldown_sec=4.0)
    rest_kps = _kps_with_wrists(cx=200, top=80, w=90, h=220, lw=(195, 195), rw=(205, 195))
    for i in range(12):
        g, c, _ = gate_rest.step(frame, 1.0 + i * 0.25, [{"track_id": 1, "bbox": p_box, "keypoints": rest_kps}], [])
        assert g is None and c is None

    # B. Open wrists -> sustained personal bag stuff (>= 0.50s) arms Path B
    gate_bag = BehaviorGate("cam2", "Corridor", dwell_sec=0.5, release_wait_sec=0.5, cooldown_sec=4.0)
    open_kps = _kps_with_wrists(cx=200, top=80, w=90, h=220, lw=(120, 175), rw=(275, 175))
    bag_det = [{"Class": "backpack", "Box": [210, 160, 65, 85]}]
    bag_kps = _kps_with_wrists(cx=200, top=80, w=90, h=220, lw=(125, 175), rw=(235, 195))
    dispatched = None
    for i in range(5):
        gate_bag.step(frame, i * 0.25, [{"track_id": 2, "bbox": p_box, "keypoints": open_kps}], bag_det)
    for i in range(5, 14):
        g, c, _ = gate_bag.step(frame, i * 0.25, [{"track_id": 2, "bbox": p_box, "keypoints": bag_kps}], bag_det)
        if c is not None:
            dispatched = c
            break
    assert dispatched is not None
    assert dispatched["trigger_mode"] == "shelf_free_tuck"
    assert "bag_stuff" in dispatched["behaviors"]

    # C. Open wrists -> slowed 2-handed garment_hide (>= 0.70s) arms Path B, while latched basket suppresses it
    gate_garm = BehaviorGate("cam2", "Corridor", dwell_sec=0.5, release_wait_sec=0.4, cooldown_sec=4.0)
    tuck_2h = _kps_with_wrists(cx=200, top=80, w=90, h=220, lw=(192, 188), rw=(210, 188))
    # Single-frame basket flicker must NOT latch has_store_vessel
    gate_garm.step(frame, 0.0, [{"track_id": 3, "bbox": p_box, "keypoints": open_kps}], [{"Class": "shopping basket", "Box": [170, 210, 60, 50]}])
    for i in range(1, 5):
        gate_garm.step(frame, i * 0.25, [{"track_id": 3, "bbox": p_box, "keypoints": open_kps}], [])
    cand_garm = None
    for i in range(5, 15):
        _, c, _ = gate_garm.step(frame, i * 0.25, [{"track_id": 3, "bbox": p_box, "keypoints": tuck_2h}], [])
        if c is not None:
            cand_garm = c
            break
    assert cand_garm is not None
    assert cand_garm["trigger_mode"] == "shelf_free_tuck"
    assert "garment_hide" in cand_garm["behaviors"]
    assert cand_garm["has_store_vessel"] is False

    # Now test 3-frame store basket latch suppresses torso garment_hide
    gate_basket = BehaviorGate("cam2", "Corridor", dwell_sec=0.5, release_wait_sec=0.4, cooldown_sec=4.0)
    b_det = [{"Class": "shopping basket", "Box": [160, 200, 70, 55]}]
    for i in range(4):
        gate_basket.step(frame, i * 0.25, [{"track_id": 4, "bbox": p_box, "keypoints": open_kps}], b_det)
    for i in range(4, 13):
        _, c, _ = gate_basket.step(frame, i * 0.25, [{"track_id": 4, "bbox": p_box, "keypoints": tuck_2h}], b_det)
        assert c is None

    # D. Path A carried_openly -> 60s carry_delayed triggers even after reach_ts (>7s) expired
    gate_delay = BehaviorGate("cam1", "Aisle1", dwell_sec=0.5, release_wait_sec=0.4, cooldown_sec=12.0)
    for i in range(4):
        gate_delay.step(frame, i * 0.25, [{"track_id": 7, "bbox": p_box, "keypoints": open_kps}], [])
    # Simulate open-carry confirmed at t=1.0 -> valid 60s until 61.0
    gate_delay.mark_carried_openly(7, ttl_sec=60.0)
    delayed_cand = None
    for i in range(10):
        t = 12.0 + i * 0.25  # well past 7s reach window, inside 60s open carry window
        _, c, _ = gate_delay.step(frame, t, [{"track_id": 7, "bbox": p_box, "keypoints": tuck_2h}], [])
        if c is not None:
            delayed_cand = c
            break
    assert delayed_cand is not None
    assert delayed_cand["trigger_mode"] == "carry_delayed"
    assert "carry_delayed" in delayed_cand["behaviors"]


def test_shelf_only_crop_preserves_head_and_sequential_flicker_not_contested():
    # 1. Shelf-only ROI expands generously above shelf so shopper torso/head are not clipped
    shelf_only = [ZoneConfig(name="shelf_1", role="shelf", rect_norm=[0.43, 0.49, 0.37, 0.45])]
    crop = compute_aisle_crop(shelf_only, 640, 480)
    assert crop is None or crop[1] <= 30

    # 2. Sequential BoT-SORT ID handoff (track 1 gone before track 2 appears) must NOT self-contest
    tracker = LocalizedShelfDeltaTracker()
    shelf_rects = [(100, 40, 420, 180)]
    clean_frame = _make_textured_shelf_frame(include_a=True, include_b=True)
    gone_frame = _make_textured_shelf_frame(include_a=False, include_b=True)
    for ts in (0.0, 0.25, 0.5, 0.75):
        tracker.step(clean_frame, ts, shelf_rects, [])

    # Track 1 touches spot A then vanishes (simulates brief ID drop within WRIST_HISTORY_SEC)
    tracker.step(
        clean_frame,
        1.0,
        shelf_rects,
        [{"tid": 1, "bbox": (120, 45, 85, 200), "wrists": [(164, 108)], "reach_active": True, "touched_shelf_idx": 0}],
    )
    res_new = {}
    for ts in (1.3, 1.6, 1.9, 2.2):
        unoccluded = ts > 1.35
        res_new = tracker.step(
            gone_frame,
            ts,
            shelf_rects,
            [{
                "tid": 2,
                "bbox": (122, 245, 85, 110) if unoccluded else (122, 45, 85, 200),
                "wrists": [(165, 260)] if unoccluded else [(165, 110)],
                "reach_active": not unoccluded,
                "touched_shelf_idx": None if unoccluded else 0,
            }],
        )
    assert res_new[2]["status"] == "confirmed" and res_new[2]["confirmed"] is True


def test_stage1_noise_reduction_motion_crop_and_ring_backfill():
    from ai_engine.rtsp.motion_gate import MotionGate
    from ai_engine.tracking.grid_builder import augment_sequence_with_ring_context

    # 1. Standalone blading (arms open, no bag, no shelf delta) must NOT arm Stage 1
    zones = [
        ZoneConfig(name="shelf_left", role="shelf", rect_norm=[0.0, 0.1, 0.22, 0.8]),
        ZoneConfig(name="aisle_main", role="aisle", rect_norm=[0.18, 0.05, 0.72, 0.9]),
    ]
    frame = np.full((360, 640, 3), 65, dtype=np.uint8)
    gate_blade = BehaviorGate("cam_b", "Aisle", dwell_sec=1.0, release_wait_sec=1.0, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.5, 1.1):
        gate_blade.step(frame, ts, [_mk_track(1, 250.0)], [])
    gate_blade.step(frame, 1.4, [_mk_track(1, 250.0, reach_left=True)], [])

    # Narrowed shoulders (blading) while keeping both wrists wide open outside torso & bag
    bladed_open = _kps_with_wrists(cx=250.0, top=80.0, w=70.0, h=220.0, lw=(195.0, 195.0), rw=(305.0, 195.0), sh_span=12.0)
    for ts in (1.8, 2.2, 2.8, 3.4, 4.0):
        _, c_blade, _ = gate_blade.step(frame, ts, [{"track_id": 1, "bbox": [215, 80, 70, 220], "keypoints": bladed_open}], [])
        assert c_blade is None

    # 2. Active browsing after arming (post_arm_reach_frames >= 2, no bag, no confirmed delta) disarms cleanly
    gate_browse = BehaviorGate("cam_br", "Aisle", dwell_sec=1.0, release_wait_sec=1.5, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.5, 1.1):
        gate_browse.step(frame, ts, [_mk_track(2, 250.0)], [])
    gate_browse.step(frame, 1.4, [_mk_track(2, 250.0, reach_left=True)], [])
    gate_browse.step(frame, 1.8, [_mk_track(2, 250.0, tuck=True)], [])
    for ts in (2.2, 2.7, 3.5, 4.2):
        _, c_br, _ = gate_browse.step(frame, ts, [_mk_track(2, 250.0, reach_left=True)], [])
        assert c_br is None

    # 3. MotionGate localized motion crop & reset
    mg = MotionGate()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    moved = blank.copy()
    cv2.rectangle(moved, (500, 300), (545, 390), (240, 240, 240), -1)
    assert mg.check(blank, 0.0) is True
    assert mg.check(moved, 0.5) is True
    mc = mg.get_motion_crop(640, 480, min_side=240)
    assert mc is not None and mc[0] <= 500 and mc[1] <= 300 and (mc[0] + mc[2]) >= 545 and (mc[1] + mc[3]) >= 390
    mg.reset()
    assert mg.get_motion_crop(640, 480) is None

    # 4. Ring-buffer pre-reach backfill prepends earlier frames chronologically
    f_early = np.full((240, 320, 3), 20, dtype=np.uint8)
    f_mid = np.full((240, 320, 3), 120, dtype=np.uint8)
    f_late = np.full((240, 320, 3), 220, dtype=np.uint8)
    _, j_early = cv2.imencode(".jpg", f_early)
    _, j_mid = cv2.imencode(".jpg", f_mid)
    _, j_late = cv2.imencode(".jpg", f_late)
    ring = [(0.5, j_early.tobytes()), (1.2, j_mid.tobytes()), (3.0, j_late.tobytes())]
    aug = augment_sequence_with_ring_context([j_late.tobytes()], ring, first_sample_ts=2.5, max_frames=6)
    assert len(aug) == 3
    dec0 = cv2.imdecode(np.frombuffer(aug[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    dec2 = cv2.imdecode(np.frombuffer(aug[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert float(dec0.mean()) < 60.0 and float(dec2.mean()) > 180.0

    # 5. Active-browse disarm clears unconfirmed has_merchandise_until so carry_delayed won't falsely re-arm
    gate_clear = BehaviorGate("cam_clr", "Aisle", dwell_sec=1.0, release_wait_sec=1.5, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.5, 1.1):
        gate_clear.step(frame, ts, [_mk_track(5, 250.0)], [])
    gate_clear.step(frame, 1.4, [_mk_track(5, 250.0, reach_left=True)], [])
    gate_clear.step(frame, 1.8, [_mk_track(5, 250.0, tuck=True)], [])
    for ts in (2.2, 2.7, 3.6):
        gate_clear.step(frame, ts, [_mk_track(5, 250.0, reach_left=True)], [])
    assert gate_clear._tracks[5]["armed_ts"] is None
    assert gate_clear._tracks[5]["has_merchandise_until"] is None
    # Later resting/tuck after active-browse disarm (no new reach) must NOT emit carry_delayed
    for ts in (4.5, 5.5, 6.5):
        _, c_none, _ = gate_clear.step(frame, ts, [_mk_track(5, 250.0, tuck=True)], [])
        assert c_none is None


def test_quiescence_dispatch_path_b_promotion_carry_escalation_and_bag_ttl():
    zones = [
        ZoneConfig(name="shelf_left", role="shelf", rect_norm=[0.0, 0.1, 0.22, 0.8]),
        ZoneConfig(name="aisle_main", role="aisle", rect_norm=[0.18, 0.05, 0.72, 0.9]),
    ]
    frame = np.full((360, 640, 3), 75, dtype=np.uint8)
    p_box = [215, 80, 70, 220]
    open_kps = _kps_with_wrists(cx=250.0, top=80.0, w=70.0, h=220.0, lw=(195.0, 170.0), rw=(305.0, 170.0))
    tuck_2h = _kps_with_wrists(cx=250.0, top=80.0, w=70.0, h=220.0, lw=(246.0, 180.0), rw=(256.0, 182.0))
    bag_det = [{"Class": "backpack", "Box": [230, 155, 60, 85]}]

    # 1a. Slow sustained concealment (~4.5s) holds off dispatch past 3.5s until post_quiet_sec after stopping
    gate_slow = BehaviorGate("cam_slow", "Aisle", dwell_sec=1.0, release_wait_sec=3.5, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.5, 1.1):
        gate_slow.step(frame, ts, [_mk_track(1, 250.0)], bag_det)
    gate_slow.step(frame, 1.4, [_mk_track(1, 250.0, reach_left=True)], bag_det)
    # Arm at t=1.8 and continue active bag concealment through t=6.2 (4.4s)
    for ts in (1.8, 2.5, 3.2, 4.0, 4.8, 5.5, 6.2):
        _, c_hold, _ = gate_slow.step(frame, ts, [_mk_track(1, 250.0, tuck=True)], bag_det)
        assert c_hold is None, f"Dispatched prematurely during active concealment at t={ts}"
    # Open arms -> stops concealing; should still wait < post_quiet_sec (1.75s) then dispatch cleanly (~t=8.0)
    _, c_early_q, _ = gate_slow.step(frame, 7.2, [{"track_id": 1, "bbox": p_box, "keypoints": open_kps}], [])
    assert c_early_q is None
    _, c_slow, _ = gate_slow.step(frame, 8.1, [{"track_id": 1, "bbox": p_box, "keypoints": open_kps}], [])
    assert c_slow is not None and c_slow["dispatch_cause"] == "natural_release"
    assert "backpack" in c_slow["nearby_bags"]

    # 1b. Fast tuck settles promptly after min_arm_sec (~2.275s) instead of lingering full 3.5s
    gate_fast = BehaviorGate("cam_fast", "Aisle", dwell_sec=1.0, release_wait_sec=3.5, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.5, 1.1):
        gate_fast.step(frame, ts, [_mk_track(2, 250.0)], bag_det)
    gate_fast.step(frame, 1.4, [_mk_track(2, 250.0, reach_left=True)], bag_det)
    gate_fast.step(frame, 1.8, [_mk_track(2, 250.0, tuck=True)], bag_det)
    _, c_too_soon, _ = gate_fast.step(frame, 3.6, [{"track_id": 2, "bbox": p_box, "keypoints": open_kps}], [])
    assert c_too_soon is None
    _, c_fast, _ = gate_fast.step(frame, 4.2, [{"track_id": 2, "bbox": p_box, "keypoints": open_kps}], [])
    assert c_fast is not None and c_fast["dispatch_cause"] == "natural_release"

    # 2a. Pre-reach Path B garment_hide promotes cleanly to Path A on fresh shelf reach without cooldown
    gate_prom = BehaviorGate("cam_prom", "Aisle", dwell_sec=0.5, release_wait_sec=2.0, cooldown_sec=12.0, zones=zones)
    for i in range(4):
        gate_prom.step(frame, i * 0.25, [{"track_id": 3, "bbox": p_box, "keypoints": open_kps}], [])
    # Sustained slowed 2-handed torso posture before reaching shelf arms Path B garment_hide (~t=1.75)
    for i in range(4, 9):
        gate_prom.step(frame, i * 0.25, [{"track_id": 3, "bbox": p_box, "keypoints": tuck_2h}], [])
    assert gate_prom._tracks[3]["armed_ts"] is not None
    assert gate_prom._tracks[3]["trigger_mode"] == "shelf_free_tuck"
    assert "garment_hide" in gate_prom._tracks[3]["reasons"]
    # Now reaches shelf -> aborts pre-reach Path B, anchors reach_ts, cooldown remains 0
    gate_prom.step(frame, 2.25, [_mk_track(3, 250.0, reach_left=True)], [])
    assert gate_prom._tracks[3]["armed_ts"] is None
    assert gate_prom._tracks[3]["reach_ts"] == 2.25
    assert gate_prom._tracks[3]["cooldown"] == 0.0
    # Immediately tucks -> arms Path A shelf_reach_and_delta and dispatches after quiescence
    gate_prom.step(frame, 2.75, [_mk_track(3, 250.0, tuck=True)], bag_det)
    cand_prom = None
    for ts in (3.2, 3.8, 4.4, 5.0):
        _, c, _ = gate_prom.step(frame, ts, [{"track_id": 3, "bbox": p_box, "keypoints": open_kps}], [])
        if c is not None:
            cand_prom = c
            break
    assert cand_prom is not None and cand_prom["trigger_mode"] == "shelf_reach_and_delta"

    # 2b. Prior bag_stuff Path B is NOT canceled by a subsequent shelf reach
    gate_keep = BehaviorGate("cam_keep", "Aisle", dwell_sec=0.5, release_wait_sec=2.0, cooldown_sec=12.0, zones=zones)
    for i in range(4):
        gate_keep.step(frame, i * 0.25, [{"track_id": 4, "bbox": p_box, "keypoints": open_kps}], bag_det)
    bag_in_kps = _kps_with_wrists(cx=250.0, top=80.0, w=70.0, h=220.0, lw=(195.0, 170.0), rw=(260.0, 195.0))
    for i in range(4, 8):
        gate_keep.step(frame, i * 0.25, [{"track_id": 4, "bbox": p_box, "keypoints": bag_in_kps}], bag_det)
    assert gate_keep._tracks[4]["armed_ts"] is not None
    assert "bag_stuff" in gate_keep._tracks[4]["reasons"]
    gate_keep.step(frame, 2.0, [_mk_track(4, 250.0, reach_left=True)], bag_det)
    assert gate_keep._tracks[4]["armed_ts"] is not None
    assert gate_keep._tracks[4]["trigger_mode"] == "shelf_free_tuck"

    # 3. mark_carried_openly unlocks exactly ONE strong-cue escalation inside 12s cooldown
    gate_esc = BehaviorGate("cam_esc", "Aisle", dwell_sec=0.5, release_wait_sec=0.8, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.3, 0.6):
        gate_esc.step(frame, ts, [_mk_track(6, 250.0)], [])
    gate_esc.step(frame, 0.9, [_mk_track(6, 250.0, reach_left=True)], [])
    gate_esc.step(frame, 1.2, [_mk_track(6, 250.0, tuck=True)], [])
    c_init = None
    for ts in (1.6, 2.0, 2.4):
        _, c, _ = gate_esc.step(frame, ts, [{"track_id": 6, "bbox": p_box, "keypoints": open_kps}], [])
        if c is not None:
            c_init = c
            break
    assert c_init is not None
    gate_esc.mark_carried_openly(6, ttl_sec=60.0)
    # Ordinary open-carry browsing during cooldown stays silent
    for ts in (3.0, 4.0, 5.0):
        _, c_open, _ = gate_esc.step(frame, ts, [{"track_id": 6, "bbox": p_box, "keypoints": open_kps}], [])
        assert c_open is None
    # Sustained in_personal_bag (>= 0.40s) inside cooldown triggers one escalation; then arms settle and release
    for ts in (5.5, 5.8, 6.1):
        _, c_arm, _ = gate_esc.step(frame, ts, [{"track_id": 6, "bbox": p_box, "keypoints": bag_in_kps}], bag_det)
        assert c_arm is None
    c_esc = None
    for ts in (6.6, 7.1, 7.6):
        _, c, _ = gate_esc.step(frame, ts, [{"track_id": 6, "bbox": p_box, "keypoints": open_kps}], [])
        if c is not None:
            c_esc = c
            break
    assert c_esc is not None and c_esc["trigger_mode"] == "carry_delayed"
    assert gate_esc._tracks[6]["carry_escalation_ready"] is False
    # Second attempt inside the new 12s cooldown does NOT bypass again
    for ts in (8.5, 9.5, 10.5, 11.5):
        _, c_blocked, _ = gate_esc.step(frame, ts, [{"track_id": 6, "bbox": p_box, "keypoints": bag_in_kps}], bag_det)
        assert c_blocked is None

    # 4. Bag proximity decays after BAG_RECENT_TTL_SEC (3.0s) so an old bystander bag doesn't block active-browse disarm
    gate_ttl = BehaviorGate("cam_ttl", "Aisle", dwell_sec=1.0, release_wait_sec=1.5, cooldown_sec=12.0, zones=zones)
    gate_ttl.step(frame, 0.0, [_mk_track(7, 250.0)], bag_det)
    assert "backpack" in gate_ttl._tracks[7]["bags"]
    for ts in (1.0, 2.0, 3.2):
        gate_ttl.step(frame, ts, [_mk_track(7, 250.0)], [])
    assert gate_ttl._tracks[7]["bags"] == set()
    gate_ttl.step(frame, 3.5, [_mk_track(7, 250.0, reach_left=True)], [])
    gate_ttl.step(frame, 3.9, [_mk_track(7, 250.0, tuck=True)], [])
    for ts in (4.3, 4.8, 5.6):
        _, c_ttl, _ = gate_ttl.step(frame, ts, [_mk_track(7, 250.0, reach_left=True)], [])
        assert c_ttl is None
    assert gate_ttl._tracks[7]["armed_ts"] is None

    # 5. High-FPS (20 fps) 8s active sequence retains earliest REACH sample without ring eviction
    gate_fps = BehaviorGate("cam_fps", "Aisle", dwell_sec=0.5, release_wait_sec=8.0, cooldown_sec=12.0, zones=zones)
    for i in range(160):
        t = i * 0.05
        if i < 12:
            tr = _mk_track(8, 250.0)
        elif i < 40:
            tr = _mk_track(8, 250.0, reach_left=True)
        else:
            tr = _mk_track(8, 250.0, tuck=True)
        gate_fps.step(frame, t, [tr], [])
    phases = [ph for _, _, ph in gate_fps._tracks[8]["samples"]]
    assert "REACH" in phases and len(phases) <= 80

    # 6a. Non-bag context objects ("bottle") must NEVER enter st["bags"] or trigger bag_stuff
    gate_bottle = BehaviorGate("cam_bot", "Aisle", dwell_sec=0.5, release_wait_sec=1.5, cooldown_sec=12.0, zones=zones)
    bottle_det = [{"Class": "bottle", "Box": [230, 155, 45, 75]}]
    for i in range(10):
        _, c_bot, _ = gate_bottle.step(frame, i * 0.25, [{"track_id": 9, "bbox": p_box, "keypoints": open_kps if i < 4 else bag_in_kps}], bottle_det)
        assert c_bot is None
    assert gate_bottle._tracks[9]["bags"] == set()
    assert "bottle" in gate_bottle._tracks[9]["vessels"]

    # 6b. Wearing a backpack without wrist-in-bag contact allows active-browse disarm; actual bag contact blocks disarm
    gate_wear = BehaviorGate("cam_wear", "Aisle", dwell_sec=1.0, release_wait_sec=1.5, cooldown_sec=12.0, zones=zones)
    wear_bag_det = [{"Class": "backpack", "Box": [175, 130, 35, 70]}]
    for ts in (0.0, 0.5, 1.1):
        gate_wear.step(frame, ts, [_mk_track(10, 250.0)], wear_bag_det)
    gate_wear.step(frame, 1.4, [_mk_track(10, 250.0, reach_left=True)], wear_bag_det)
    gate_wear.step(frame, 1.8, [_mk_track(10, 250.0, tuck=True)], wear_bag_det)
    assert gate_wear._tracks[10]["armed_bag_contact"] is False
    for ts in (2.2, 2.7, 3.6):
        _, c_disarm, _ = gate_wear.step(frame, ts, [_mk_track(10, 250.0, reach_left=True)], wear_bag_det)
        assert c_disarm is None
    assert gate_wear._tracks[10]["armed_ts"] is None

    # Now with real wrist-in-bag contact -> decoy reaches do NOT disarm it
    gate_touch = BehaviorGate("cam_touch", "Aisle", dwell_sec=0.5, release_wait_sec=1.2, cooldown_sec=12.0, zones=zones)
    for ts in (0.0, 0.3, 0.6):
        gate_touch.step(frame, ts, [{"track_id": 11, "bbox": p_box, "keypoints": open_kps}], bag_det)
    gate_touch.step(frame, 0.9, [_mk_track(11, 250.0, reach_left=True)], bag_det)
    gate_touch.step(frame, 1.2, [{"track_id": 11, "bbox": p_box, "keypoints": bag_in_kps}], bag_det)
    assert gate_touch._tracks[11]["armed_bag_contact"] is True
    dispatched_touch = None
    for ts in (1.5, 1.8, 2.2, 2.6):
        _, c_t, _ = gate_touch.step(frame, ts, [_mk_track(11, 250.0, reach_left=True)], bag_det)
        if c_t is not None:
            dispatched_touch = c_t
            break
    assert dispatched_touch is not None


