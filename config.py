# config.py
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "output")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# Scene detection config
DEFAULT_SCENE_THRESHOLD = 12.0

# Scroll detection config
MIN_SCROLL_SHIFT_PX = 15      # Minimum vertical translation to count as a scroll
MAX_SCROLL_OVERLAP = 0.90     # If overlap > 90%, it's practically static (or micro-scroll to ignore)
MIN_SCROLL_OVERLAP = 0.30     # If overlap < 30%, it's likely a complete slide change rather than a scroll

# AI Model config
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv3-large-100.onnx"
MODEL_PATH = os.path.join(MODELS_FOLDER, "mobilenetv3.onnx")