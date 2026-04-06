from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run YOLOv13 on vehicle_night.mp4 and visualize the detection/tracking process."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(root / "vehicle_night.mp4"),
        help="Video path or camera index (default: vehicle_night.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(root / "yolov13s.pt"),
        help="Model weights path (default: yolov13s.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "track"],
        default="track",
        help="Use plain detection (predict) or tracking (track).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video to runs/ (default: off)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto | cpu | cuda | 0 | 0,1 ... (default: auto)",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    args = parse_args()

    from ultralytics import YOLO

    root = Path(__file__).resolve().parent
    source = args.source
    weights = args.weights

    if source.isdigit():
        source_val: str | int = int(source)
    else:
        source_val = str(Path(source))

    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = root / weights_path

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = resolve_device(args.device)

    model = YOLO(str(weights_path))

    if args.mode == "track":
        model.track(
            source=source_val,
            tracker="bytetrack.yaml",
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            show=True,
            save=args.save,
            stream=False,
        )
    else:
        model.predict(
            source=source_val,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            show=True,
            save=args.save,
            stream=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
