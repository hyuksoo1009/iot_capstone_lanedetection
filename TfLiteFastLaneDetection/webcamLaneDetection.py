import cv2
import os
import time
import argparse
import importlib.util

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color=(255, 0, 0)
    text_color_bg=(0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


model_path = "models/model_float32.tflite"
model_type = ModelType.CULANE

# #TPU 코드 복붙

# parser = argparse.ArgumentParser()
# parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
#                     required=True)
# parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
#                     default='detect.tflite')
# parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
#                     default='labelmap.txt')
# parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
#                     default=0.5)
# parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
#                     default='1280x720')
# parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
#                     action='store_true')

# args = parser.parse_args()

# MODEL_NAME = args.modeldir
# GRAPH_NAME = args.graph
# LABELMAP_NAME = args.labels
# # min_conf_threshold = float(args.threshold)
# # resW, resH = args.resolution.split('x')
# # imW, imH = int(resW), int(resH)
# use_TPU = args.edgetpu

# # Import TensorFlow libraries
# # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# # If using Coral Edge TPU, import the load_delegate library
# pkg = importlib.util.find_spec('tflite_runtime')
# if pkg:
#     from tflite_runtime.interpreter import Interpreter
#     if use_TPU:
#         from tflite_runtime.interpreter import load_delegate
#         print("tflite_runtime")
# else:
#     from tensorflow.lite.python.interpreter import Interpreter
#     if use_TPU:
#         from tensorflow.lite.python.interpreter import load_delegate
#         print("tensorflow")

# # If using Edge TPU, assign filename for Edge TPU model
# if use_TPU:
#     # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
#     #if (GRAPH_NAME == 'detect.tflite'):
#     GRAPH_NAME = 'model_float32_edgetpu.tflite'
#     print(GRAPH_NAME)

# # Get path to current working directory
# CWD_PATH = os.getcwd()

# # Path to .tflite file, which contains the model that is used for object detection
# PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# # Load the Tensorflow Lite model.
# # If using Edge TPU, use special load_delegate argument
# if use_TPU:
#     interpreter = Interpreter(model_path=PATH_TO_CKPT,
#                               experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
#     print(PATH_TO_CKPT)
# else:
#     interpreter = Interpreter(model_path=PATH_TO_CKPT)

# interpreter.allocate_tensors()


# ######여기까지

# Initialize lane detection model
print("1th confirm to PATH.TO.CKPT :webcam.py     ",model_path)
lane_detector = UltrafastLaneDetector(model_path, model_type)

# Initialize webcam
cap = cv2.VideoCapture(1)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

fps = cap.get(cv2.CAP_PROP_FPS)

time_per_frame_video = 1/fps
last_time = time.perf_counter()

while(True):
    ret, frame = cap.read()

    time_per_frame = time.perf_counter() - last_time
    time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
    time.sleep(time_sleep_frame)

    real_fps = 1 / (time.perf_counter() - last_time)
    last_time = time.perf_counter()

    text = '%.2f fps' % real_fps
    print(text)

    x = 30
    y = 50

    draw_text(frame, text, x, y)
    
   # cv2.imshow("Color", frame)

    # Detect the lanes
    output_img = lane_detector.detect_lanes(frame)

    
    cv2.imshow("Detected lanes", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break