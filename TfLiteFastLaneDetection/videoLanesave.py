import cv2
import numpy as np
import os
import time
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType


def get_files(path):
    list_files = []
    for root, subdirs, files in os.walk(path):
        print(os.walk(path))
        if len(files) > 0:
            for f in files:
                full_path = root + '/' + f
                list_files.append(full_path)
    return list_files


def img_to_video(path):
    image_files = get_files(path)

    img = cv2.imread(image_files[0])
    height, width, _ = img.shape
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f'./output.mp4', fourcc, fps, (width, height))

    for file in image_files:
        img = cv2.imread(file)
        writer.write(img)
    writer.release()


model_path = "models/model_float32.tflite"
model_type = ModelType.CULANE

# Initialize video
cap = cv2.VideoCapture("data/01.mp4")

# 비디오 파일 정보 설정
output_filename = 'output.avi'
frame_size = (1280, 720)
fps = 24

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type)

# cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)
video_list = []
count = 0

while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
    except:
        continue
    if ret:
        # Detect the lanes
        output_img = lane_detector.detect_lanes(frame)
        cv2.imwrite(f'./result/{count:04d}.jpg', output_img)
        count = count + 1

        #cv2.imshow("Detected lanes", output_img)
        video_list.append(output_img)
        # Write output frame to the video file

    else:
        break

    # # Press key q to stop
    # if cv2.waitKey(1) == ord('q'):
    #    break

img_to_video("./result")
cap.release()
cv2.destroyAllWindows()
