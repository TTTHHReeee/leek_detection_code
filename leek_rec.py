# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

# 导入 pyorbbecsdk 模块
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

# 强制优先从 examples 文件夹导入 utils，避免与系统中安装的 utils 包冲突
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples'))

from utils import frame_to_bgr_image

ESC_KEY = 27

model_path = "/home/jetson/pyorbbecsdk-main/examples/obb.engine"
MIN_length = 100  # pixels
def main():
    config = Config()
    pipeline = Pipeline()
    model = YOLO(model_path, task="obb")
    
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    
    
    # 设置窗口为全屏模式
    cv2.namedWindow("Color Viewer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Color Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    
    
    try:
        while True:
            # 这里使用降低等待时间，给 CPU 更多呼吸空间
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue

            # 使用降低图像分辨率来加快推理速度
            # 缩放图像为 320x240，如果你对小目标精确度要求极高，请慎用，或者缩放到模型的默认输入大小如 640x640
            resized_image = cv2.resize(color_image, (320, 240))
            
            # 使用半精度或缩放后的图像推理加快速度
            results = model(np.copy(resized_image), conf=0.45, iou=0.35, verbose=False, half=True)
            white_count = 0
            root_count = 0
            cut_ok = 0
            for result in results:
                if result.obb is None or len(result.obb) == 0:
                    continue

                for i, obb_points in enumerate(result.obb.xyxyxyxy):
                    # 注意：如果上面图像缩放了，这里的坐标需要按比例映射回原图
                    # 比如：x_rescaled = x * (640 / 320), y_rescaled = y * (480 / 240)
                    pts = obb_points.cpu().numpy().astype(np.float32)
                    pts[:, 0] *= (640.0 / 320.0)  # 还原 X 坐标
                    pts[:, 1] *= (480.0 / 240.0)  # 还原 Y 坐标
                    pts = pts.astype(np.int32)
                    
                    # 获取类别ID和置信度
                    cls_id = int(result.obb.cls[i].item())
                    conf = result.obb.conf[i].item()

                    # 按照你 obb.yaml 中真实的类别映射关系处理：0 -> leek_white, 1 -> leek_root
                    if cls_id == 0:
                        color = (0, 255, 0)  # 蓝色表示 leek_white
                        white_count += 1
                        display_name = "leek_white"
                    elif cls_id == 1:
                        color = (0, 0, 255)  # 绿色表示 leek_root
                        root_count += 1
                        display_name = "leek_root"
                    else:
                        color = (255, 255, 0) # 未知类别
                        display_name = f"unknown_{cls_id}"
                    
                    cut_ok = white_count - root_count
                    
                    # 绘制旋转多边形框
                    cv2.polylines(color_image, [pts], isClosed=True, color=color, thickness=2)
                    
                    # 在框上方添加文本标注 (分类名 + 置信度)
                    text_x = int(pts[0][0])
                    text_y = int(pts[0][1] - 10)
                    cv2.putText(
                        color_image,
                        f"{display_name} {conf:.2f}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # 在左上角显示当前画面中根部与葱白的瞬时计数
            cv2.putText(color_image, f"White: {white_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(color_image, f"Root: {root_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(color_image, f"cut_ok: {cut_ok}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            
            display_image = cv2.resize(color_image, (1920, 1080))
            cv2.imshow("Color Viewer", display_image)
            

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()