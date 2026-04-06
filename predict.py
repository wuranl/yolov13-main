from ultralytics import YOLO
from pathlib import Path

# Load the model
model = YOLO('yolov13n.pt')  # Replace with the desired model scale

# Perform inference on an image
ROOT = Path(__file__).resolve().parent
results = model.predict(source=str(ROOT / 'ultralytics' / 'assets' / 'bus.jpg'))

# Display results
for r in results:
	r.show()  # Displays the image with predictions
	r.save()  # Save predictions to 'runs/detect/exp'
