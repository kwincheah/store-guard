from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.config import (
    AppConfig,
    CameraConfig,
    ZoneConfig,
    load_config,
    mask_rtsp_url,
    parse_apex_xml,
    save_config_atomic,
)
from ai_engine.inference.device_utils import resolve_onnx_providers, resolve_torch_device


def test_config_roundtrip_and_zone_clamping(tmp_path):
    cfg = AppConfig(
        cameras=[
            CameraConfig(
                camera_id="cam_01",
                name="Aisle 1",
                rtsp_url="rtsp://admin:secret@192.168.1.50:554/unicast/c1/s1/live",
                zones=[ZoneConfig(name="shelf_1", role="shelf", rect_norm=[-0.1, 0.2, 0.5, 1.5])],
            )
        ]
    )
    z = cfg.cameras[0].zones[0]
    assert z.rect_norm[0] == 0.0
    assert z.rect_norm[3] == 0.8

    out = tmp_path / "cameras.json"
    save_config_atomic(out, cfg)
    loaded = load_config(out)
    assert loaded.engine.retention_days == 30
    assert loaded.engine.max_evidence_gb == 20.0
    assert loaded.engine.vlm.alert_threshold_pct == 65.0
    assert loaded.cameras[0].camera_id == "cam_01"
    assert mask_rtsp_url(loaded.cameras[0].rtsp_url) == "rtsp://admin:***@192.168.1.50:554/unicast/c1/s1/live"


def test_parse_apex_xml_from_reference_solution(tmp_path):
    sample = tmp_path / "cam.apex"
    sample.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<ApexSolution name="Cam1Test">
  <Flows>
    <ApexFlow name="F">
      <Nodes>
        <Node tool_name="Image Source" name="01_RTSP_Camera">
          <Parameters>
            <Parameter name="stream_url" type="str">rtsp://user:pw@10.0.0.10:554/unicast/c2/s1/live</Parameter>
          </Parameters>
        </Node>
        <Node tool_name="ROI Zone Manager" name="01b_ROI_Zones">
          <Parameters>
            <Parameter name="zones" type="json">[{"name":"shelf_left","rect":[128,144,256,360]},{"name":"aisle_main","rect":[320,72,640,576]}]</Parameter>
          </Parameters>
        </Node>
      </Nodes>
    </ApexFlow>
  </Flows>
</ApexSolution>""",
        encoding="utf-8",
    )
    cam = parse_apex_xml(sample, camera_id="cam_02", ref_w=1280, ref_h=720)
    assert cam.rtsp_url == "rtsp://user:pw@10.0.0.10:554/unicast/c2/s1/live"
    assert len(cam.zones) == 2
    assert cam.zones[0].role == "shelf"
    assert cam.zones[0].rect_norm == [0.1, 0.2, 0.2, 0.5]
    assert cam.zones[1].role == "aisle"
    assert cam.zones[1].rect_norm == [0.25, 0.1, 0.5, 0.8]


def test_device_resolution_fallbacks():
    providers, _, label = resolve_onnx_providers("auto")
    assert "CPUExecutionProvider" in providers
    assert label in ("cpu", "cuda:0")
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("cuda:0") in ("cpu", "cuda:0")
