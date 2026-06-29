import cv2
import numpy
print(f"Numpy version: {numpy.__version__}")
try:
    import mediapipe.python.solutions.drawing_styles
    print("Mediapipe drawing styles imported successfully")
except ImportError as e:
    print(f"Failed to import mediapipe drawing styles: {e}")
    import traceback
    traceback.print_exc()
