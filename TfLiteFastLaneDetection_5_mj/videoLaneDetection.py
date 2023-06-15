import cv2
import os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import time

def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 2
    text_color=(255, 0, 0)
    text_color_bg=(0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

def main():
    model_path = "models/model_float32.tflite"
    model_type = ModelType.CULANE # or ModelType.TUSIMPLE 

    # 사전 등록된 비디오 불러오기 (파라미터에 0을 넣어주면 컴퓨터에 연결된 기본 카메라로부터 비디오 가져올 수 있음)
    cap = cv2.VideoCapture("data/project.mp4")
    # _, img_list = video_to_img("D:/data/changed/03.mp4")
    #print("영상길이",len(img_list))

    # 모델 로드 및 초기화
    lane_detector = UltrafastLaneDetector(model_path, model_type)

    # ▽ 실시간으로 확인하고 싶을때 푸세요!! - 혁수 ▽
    cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
    
    video_list = []
    count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    time_per_frame_video = 1 / fps
    last_time = time.perf_counter()

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            time_per_frame = time.perf_counter() - last_time
            time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
            time.sleep(time_sleep_frame)

            real_fps = 1 / (time.perf_counter() - last_time)
            last_time = time.perf_counter()

            text = '%.2f fps' % real_fps

            x = 30
            y = 50

            # draw_text(frame, text, x, y)
        except:
            continue
        
        if ret:	
            # 차선 탐지 시작
            output_img = lane_detector.detect_lanes(frame)
            count = count + 1
            # ▽실시간으로 확인하고 싶을때 푸세요!! - 혁수 ▽
            cv2.imshow("Detected lanes", output_img)
            cv2.imwrite(f'./result/{count:04d}.jpg', output_img)
            video_list.append(output_img)
        else:
            break
        
        # Press key q to stop  # ▽실시간으로 확인하고 싶을때 푸세요!! - 혁수 ▽
        if cv2.waitKey(1) == ord('q'):
            break

    img_to_video("./result") 
    # 이미지를 비디오로 따로 저장할때 사용
    # imshow 쓰면 자체적으로 너무 느려서 영상파일 따로 저장하는 코드를 구현함
    cap.release()
    cv2.destroyAllWindows()
    
    
# -- 여기서부터 혁수 추가 --
def video_to_img(video_path): #비디오를 이미지로 분할하는 함수.
    video = cv2.VideoCapture(video_path)
    img_list = [] #영상에서 분리된 이미지 텐서값들 저장하는 리스트
    original = []
    if video.isOpened():
        while True:
            ret, img = video.read()
            if ret:
                original.append(img)
                img_list.append(img)
            else:
                break
    else:
        print("동영상 파일이 아니거나, 형식에서의 오류가 발생이 추정됩니다.")
    return original, img_list

 
def get_files(path):
    list_files = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            for f in files:
                full_path = root +'/' + f
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

if __name__ == "__main__": 
    main()
    