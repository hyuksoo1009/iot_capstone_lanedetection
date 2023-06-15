import time
import cv2
import scipy.special
from enum import Enum
import numpy as np
import math
import importlib.util
pkg = importlib.util.find_spec('tflite_runtime')
try:
   from tflite_runtime.interpreter import Interpreter
   from tflite_runtime.interpreter import load_delegate
   print("tflite_runtime very good")
except:
   from tensorflow.lite.python.interpreter import Interpreter
   from tensorflow.lite.python.interpreter import load_delegate
   print("tensorflow , BAD, WRONG")


lane_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

class ModelType(Enum):
   TUSIMPLE = 0
   CULANE = 1


class ModelConfig():

   def __init__(self, model_type):

      if model_type == ModelType.TUSIMPLE:
         self.init_tusimple_config()
      else:
         self.init_culane_config()

   def init_tusimple_config(self):
      self.img_w = 1280
      self.img_h = 720
      self.row_anchor = tusimple_row_anchor
      self.griding_num = 100
      self.cls_num_per_lane = 56

   def init_culane_config(self):
      self.img_w = 1640
      self.img_h = 590
      self.row_anchor = culane_row_anchor
      self.griding_num = 200
      self.cls_num_per_lane = 18


class UltrafastLaneDetector():

   def __init__(self, model_path, model_type=ModelType.TUSIMPLE):

      self.fps = 10
      self.timeLastPrediction = time.time()
      self.frameCounter = 0

      # Load model configuration based on the model type
      self.cfg = ModelConfig(model_type)

      # Initialize model
      print("Model initialize 2econd check :ultra.py     ",model_path)
      self.model = self.initialize_model(model_path)

   def initialize_model(self, model_path):

      self.interpreter = Interpreter(model_path=model_path)
      # self.interpreter =  Interpreter(model_path=model_path,
      #                         experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
      self.interpreter.allocate_tensors()

      # Get model info
      self.getModel_input_details()
      self.getModel_output_details()

   def detect_lanes(self, image, draw_points=True, lane_points=None):

      input_tensor = self.prepare_input(image)

      # Perform inference on the image
      output = self.inference(input_tensor)

      # Process output data
      self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

      # # Draw depth image
      visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

      self.x_center = 1640 // 2
      self.lane_departure_count = 0  # 차선 이탈 횟수를 저장하는 변수

      self.is_previous_lane_departed = False  # 이전에 차선 이탈이 발생했는지 여부를 나타내는 변수

      # 1번차선 타입체크 후 postion 좌표랑, 밑에 써있는 좌표들끼리 거리계산
      if (self.lanes_points[1][0][0] is not None and self.lanes_points[1][0][1] is not None):
         if (self.lanes_points[2][0][0] is not None and self.lanes_points[2][0][1] is not None):
            x_left = self.lanes_points[1][0][0]
            x_right = self.lanes_points[2][0][0]

            left_dis = abs(self.x_center - x_left)
            right_dis = abs(x_right - self.x_center)

            lane_distance = x_right - x_left

            print("left_dis: ", left_dis)
            print("right_dis: ", right_dis)

         if (left_dis < lane_distance // 4) or (right_dis < lane_distance // 4):
            if not self.is_previous_lane_departed:
               self.lane_departure_count += 1
               self.is_previous_lane_departed = True
               print("Driving incorrectly, Lane Departure")
         else:
            self.is_previous_lane_departed = False
            print("Driving correctly")

      print("Lane Departure Count: ", self.lane_departure_count)
      return visualization_img

   def prepare_input(self, image):
      img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      self.img_height, self.img_width, self.img_channels = img.shape

      # Input values should be from -1 to 1 with a size of 288 x 800 pixels
      img_input = cv2.resize(img, (self.input_width, self.input_height)).astype(np.float32)

      # Scale input pixel values to -1 to 1
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]

      img_input = ((img_input / 255.0 - mean) / std).astype(np.float32)
      img_input = img_input[np.newaxis, :, :, :]

      #print('prepare   ',img_input.shape)
      return img_input

   def getModel_input_details(self):
      self.input_details = self.interpreter.get_input_details()
      input_shape = self.input_details[0]['shape']
      self.input_height = input_shape[1]
      self.input_width = input_shape[2]
      self.channels = input_shape[3]

   def getModel_output_details(self):
      self.output_details = self.interpreter.get_output_details()
      output_shape = self.output_details[0]['shape']
      self.num_anchors = output_shape[1]
      self.num_lanes = output_shape[2]
      self.num_points = output_shape[3]

   def inference(self, input_tensor):
      # Peform inference
      self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
      self.interpreter.invoke()
      output = self.interpreter.get_tensor(self.output_details[0]['index'])

      output = output.reshape(self.num_anchors, self.num_lanes, self.num_points)
      return output

   @staticmethod
   def process_output(output, cfg):

      # Parse the output of the model to get the lane information
      processed_output = output[:, ::-1, :]

      prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
      # idx = np.arange(cfg.griding_num) + 1
      idx = np.arange(200) + 1
      idx = idx.reshape(-1, 1, 1)
      loc = np.sum(prob * idx, axis=0)
      processed_output = np.argmax(processed_output, axis=0)
      loc[processed_output == cfg.griding_num] = 0
      processed_output = loc



      col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
      col_sample_w = col_sample[1] - col_sample[0]

      lane_points_mat = []
      lanes_detected = []

      max_lanes = processed_output.shape[1]

      for lane_num in range(max_lanes):
         lane_points = []
         # Check if there are any points detected in the lane
         if np.sum(processed_output[:, lane_num] != 0) > 2:

            lanes_detected.append(True)

            # Process each of the points for each lane
            for point_num in range(processed_output.shape[0]):
               if processed_output[point_num, lane_num] > 0:
                  lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1,
                                int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane - 1 - point_num] / 288)) - 1]
                  lane_points.append(lane_point)
         else:
            lanes_detected.append(False)

         # print("lanes_detected", lanes_detected)
         lane_points_mat.append(lane_points)

      threshold_distance = 580  # 기준 거리


      if (len(lane_points_mat[1]) > len(lane_points_mat[2])):

         if (len(lane_points_mat[1]) == 0 or len(lane_points_mat[2]) == 0):
            print("if 1-1") #에러 날때를 대비한 코드
            lane_points_mat[0] = [[0, 0] for _ in range(5)]
            lane_points_mat[1] = [[0, 0] for _ in range(5)]
            lane_points_mat[2] = [[0, 0] for _ in range(5)]
            lane_points_mat[3] = [[0, 0] for _ in range(5)]

         else:
            print("if 1-2", len(lane_points_mat[2]))
            number = len(lane_points_mat[2])
            lane_points_mat[1] = lane_points_mat[1][:number]
            lane_points_mat[0] = [[0, 0] for _ in range(number)]
            lane_points_mat[3] = [[0, 0] for _ in range(number)]

      elif (len(lane_points_mat[1]) < len(lane_points_mat[2])):
         if(len(lane_points_mat[1]) == 0 or len(lane_points_mat[2]) == 0):
            print("if 2-1")
            lane_points_mat[0] = [[0, 0] for _ in range(5)]
            lane_points_mat[1] = [[0, 0] for _ in range(5)]
            lane_points_mat[2] = [[0, 0] for _ in range(5)]
            lane_points_mat[3] = [[0, 0] for _ in range(5)]

         else:
            print("if 2-2")
            number = len(lane_points_mat[1])
            lane_points_mat[2] = lane_points_mat[2][:number]
            lane_points_mat[0] = [[0, 0] for _ in range(number)]
            lane_points_mat[3] = [[0, 0] for _ in range(number)]

      elif (len(lane_points_mat[1]) == len(lane_points_mat[2])):
         if(len(lane_points_mat[1])== 0 or len(lane_points_mat[2]) == 0):
            print("if 3-1")
            lane_points_mat[0] = [[0, 0] for _ in range(5)]
            lane_points_mat[1] = [[0, 0] for _ in range(5)]
            lane_points_mat[2] = [[0, 0] for _ in range(5)]
            lane_points_mat[3] = [[0, 0] for _ in range(5)]
         else:
            print("if 3-2")
            number = len(lane_points_mat[1])
            lane_points_mat[0] = [[0, 0] for _ in range(number)]
            lane_points_mat[3] = [[0, 0] for _ in range(number)]


      print("!!!!!!!!!!!!!!!!!!!!!")
      print("0 lane->", lanes_detected[0], "point", lane_points_mat[0], "len", len(lane_points_mat[0]))
      print("1 lane->", lanes_detected[1], "point", lane_points_mat[1], "len", len(lane_points_mat[1]))
      print("2 lane->", lanes_detected[2], "point", lane_points_mat[2], "len", len(lane_points_mat[2]))
      print("3 lane->", lanes_detected[3], "point", lane_points_mat[3], "len", len(lane_points_mat[3]))

      print("\n\n")

      # x_center = 1640//2
      # lane_departure_count = 0  # 차선 이탈 횟수를 저장하는 변수
      # is_previous_lane_departed = False  # 이전에 차선 이탈이 발생했는지 여부를 나타내는 변수
      #
      # # 1번차선 타입체크 후 postion 좌표랑, 밑에 써있는 좌표들끼리 거리계산
      # if (lane_points_mat[1][0][0] is not None and lane_points_mat[1][0][1] is not None):
      #    if (lane_points_mat[2][0][0] is not None and lane_points_mat[2][0][1] is not None):
      #       x_left = lane_points_mat[1][0][0]
      #       y_left = lane_points_mat[1][0][1]
      #
      #       x_right = lane_points_mat[2][0][0]
      #       y_right = lane_points_mat[2][0][1]
      #
      #       left_dis = abs(x_center - x_left)
      #       right_dis = abs(x_right - x_center)
      #
      #       lane_distance = x_right - x_left
      #
      #       print("left_dis: ", left_dis)
      #       print("right_dis: ", right_dis)
      #
      #    if (left_dis < lane_distance // 4) or (right_dis < lane_distance // 4):
      #       if not is_previous_lane_departed:
      #          lane_departure_count += 1
      #          is_previous_lane_departed = True
      #          print("차선 이탈함")
      #    else:
      #       is_previous_lane_departed = False
      #       print("차선 이탈하지 않음")

      return np.array(lane_points_mat), np.array(lanes_detected)

   @staticmethod
   def draw_lanes(input_img, lane_points_mat, lanes_detected, cfg, draw_points=True):
      # Write the detected line points in the image
      visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)

      # Draw a mask for the current lane
      if (lanes_detected[1] and lanes_detected[2]):
         lane_segment_img = visualization_img.copy()

         cv2.fillPoly(lane_segment_img, pts=[np.vstack((lane_points_mat[1], np.flipud(lane_points_mat[2])))],
                      color=(255, 191, 0))
         visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

      if (draw_points):
         for lane_num, lane_points in enumerate(lane_points_mat):
            for lane_point in lane_points:
               cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

      return visualization_img



