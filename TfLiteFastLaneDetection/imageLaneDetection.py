import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

model_path = "models/model_float32.tflite"
model_type = ModelType.CULANE


image_path = "data/0.jpg"

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type)

# Read RGB images
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Detect the lanes
output_img = lane_detector.detect_lanes(img)

print(output_img.shape)

# Draw estimated depth
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
cv2.imshow("Detected lanes", output_img)
cv2.waitKey(0)

cv2.imwrite("output.jpg",output_img)
