from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.config import ZoneConfig
from ai_engine.rtsp.motion_gate import MotionGate
from ai_engine.tracking.botsort_pool import BoTSORTPool


def test_motion_gate_aisle_mask_and_tail():
    mg = MotionGate(diff_thresh=20, ratio_thresh=0.01, tail_sec=3.0)
    aisle = [ZoneConfig(name="aisle_main", role="aisle", rect_norm=[0.5, 0.0, 0.5, 1.0])]

    base = np.zeros((180, 320, 3), dtype=np.uint8)
    assert mg.check(base, 0.0, aisle) is True  # initial warm-up tick

    # Change ONLY left half outside aisle ROI -> expires after tail
    moved_outside = base.copy()
    moved_outside[:, :140] = 255
    assert mg.check(moved_outside, 3.5, aisle) is False

    # Change right half INSIDE aisle ROI -> wakes motion gate with 3s tail
    moved_inside = moved_outside.copy()
    moved_inside[:, 170:] = 255
    assert mg.check(moved_inside, 4.0, aisle) is True
    assert mg.check(moved_inside, 6.5, aisle) is True   # still inside 3.0s tail
    assert mg.check(moved_inside, 7.2, aisle) is False  # tail expired cleanly


def test_botsort_pool_dark_frame_zero_norm_safe():
    pool = BoTSORTPool(fps=8, device="auto", enable_reid=True)
    dark = np.zeros((360, 640, 3), dtype=np.uint8)
    dets = [{"Box": [100, 80, 70, 200], "Score": 0.9, "Class": "person", "Class_Id": 0, "Keypoints": []}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        t1 = pool.update("cam_dark", dark, dets)
        t2 = pool.update("cam_dark", dark, dets)
    divide_warns = [w for w in caught if "invalid value encountered in divide" in str(w.message)]
    assert divide_warns == []
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0]["track_id"] == t2[0]["track_id"]
    assert pool._trackers["cam_dark"]._tracker.args.gmc_method is None
