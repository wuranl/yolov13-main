from ultralytics import YOLO
from pathlib import Path
# Load the model
model = YOLO('yolov13n.pt')
print("Model loaded successfully!")

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / 'ultralytics' / 'assets' / 'bus.jpg'

# Perform inference on an image and save predictions to the runs folder (force CPU to avoid torchvision CUDA NMS errors)
try:
	results = model.predict(
		source=str(ASSET), save=True, device="cpu"
	)
	print("Prediction completed!")

	# The predictor saves outputs to its `save_dir` when `save=True`; print that directory
	try:
		save_dir = model.predictor.save_dir
	except Exception:
		save_dir = None

	if save_dir:
		print(f"Results saved to: {save_dir}")
	else:
		# Fallback: explicit per-result save (will write files like results_<name>.jpg to cwd)
		if isinstance(results, list):
			for r in results:
				r.save()
		else:
			results.save()
		print("Results saved (fallback) to current working directory")
except NotImplementedError as e:
	print("预测过程中发生 NotImplementedError：可能是 torchvision 与 CUDA 构建不兼容导致 NMS 在 CUDA 上不可用。")
	print("短期解决方案：将 `device='cpu'` 传入 `model.predict(...)` 或重新安装与 CUDA 匹配的 torchvision。")
	raise
