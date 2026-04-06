from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
  root = Path(__file__).resolve().parent

  parser = argparse.ArgumentParser(description="Train YOLOv13 on local datasets (vehicle_collision).")
  parser.add_argument("--data", type=str, default=str(root / "datasets" / "vehicle_night_rainy_foggy" / "data.yaml"), help="dataset yaml path")
  parser.add_argument(
    "--model",
    type=str,
    default=str(root / "runs" / "train" / "vehicle_night_rainy_foggy" / "weights" / "yolov13_original.pt"),
    help="initial weights (.pt) or model yaml",
  )
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--batch", type=int, default=16)
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--device", type=str, default="0", help='e.g. "0" or "cpu"')
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--name", type=str, default="vehicle_night_rainy_foggy_original")
  parser.add_argument("--project", type=str, default=str(root / "runs" / "train"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--test", action="store_true", help="also evaluate on test split after training")
  args = parser.parse_args()

  from ultralytics import YOLO

  model = YOLO(args.model)

  # Remove old cache files to avoid conflicts
  import glob
  for cache_file in glob.glob(str(root / "datasets" / "**" / ".cache"), recursive=True):
    try:
      Path(cache_file).unlink()
    except:
      pass
  for cache_file in glob.glob(str(root / "datasets" / "**" / "*.cache"), recursive=True):
    try:
      Path(cache_file).unlink()
    except:
      pass

  model.train(
    data=args.data,
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name=args.name,
    seed=args.seed,
    profile=False,
    verbose=True,
    amp=True,
    cache=False,
    save_period=10,
  )

  # Validate on val split
  model.val(data=args.data)

  # Optionally evaluate on test split (requires 	est: in data yaml)
  if args.test:
    try:
      model.val(data=args.data, split="test")
    except TypeError:
      # Older ultralytics versions may not support split=; in that case, skip.
      pass

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
