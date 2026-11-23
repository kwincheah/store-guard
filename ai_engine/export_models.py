"""Export YOLO11-Pose PyTorch weights to ONNX and verify ONNX Runtime tensor shapes."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


DEFAULT_PT = Path(__file__).resolve().parent.parent.parent / "ApexVision" / "yolo11n-pose.pt"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "yolo11n-pose.onnx"


def export_pose_onnx(pt_path: Path, out_path: Path, imgsz: int = 640) -> Path:
    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch pose weights not found: {pt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    from ultralytics import YOLO
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pt = Path(tmpdir) / pt_path.name
        shutil.copy2(pt_path, tmp_pt)
        model = YOLO(str(tmp_pt))
        exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
        if not exported.is_file():
            raise RuntimeError(f"Ultralytics export did not produce {exported}")
        shutil.copy2(exported, out_path)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inputs = [(i.name, list(i.shape), i.type) for i in sess.get_inputs()]
    outputs = [(o.name, list(o.shape), o.type) for o in sess.get_outputs()]
    print(f"Exported ONNX: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Inputs : {inputs}")
    print(f"Outputs: {outputs}")
    return out_path


DEFAULT_RETAIL_VOCAB = (
    "backpack",
    "handbag",
    "suitcase",
    "tote bag",
    "shopping basket",
    "shopping cart",
    "cell phone",
    "smartphone",
    "bottle",
    "product box",
)


def export_retail_vocab_onnx(
    out_path: Path,
    *,
    world_pt: str = "yolov8s-worldv2.pt",
    classes: tuple[str, ...] = DEFAULT_RETAIL_VOCAB,
    imgsz: int = 640,
) -> Path:
    """Bake zero-shot retail context vocabulary (baskets, carts, bags, phones) into ONNX."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    from ultralytics import YOLO
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd_old = os.getcwd()
        try:
            os.chdir(tmpdir)
            model = YOLO(world_pt)
            model.set_classes(list(classes))
            exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
            shutil.copy2(exported, out_path)
        finally:
            os.chdir(cwd_old)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print(f"Exported Retail Context ONNX: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Outputs: {[(o.name, list(o.shape)) for o in sess.get_outputs()]}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO11-Pose / Retail Context models to .onnx")
    parser.add_argument("--pt", type=Path, default=DEFAULT_PT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--export-retail-vocab", action="store_true", help="Also export zero-shot retail context YOLO-World ONNX")
    parser.add_argument("--retail-out", type=Path, default=DEFAULT_OUT.parent / "retail_context.onnx")
    args = parser.parse_args()
    export_pose_onnx(args.pt, args.out, args.imgsz)
    if args.export_retail_vocab:
        export_retail_vocab_onnx(args.retail_out, imgsz=args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
