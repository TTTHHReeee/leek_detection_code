import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
try:
    from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline
    HAS_ORBBEC = True
except ImportError:
    print("Warning: pyorbbecsdk not found. Running in MOCK Mode (Simulated Camera).")
    HAS_ORBBEC = False

# ====== 消除 utils 依赖，直接嵌入核心图像转换函数 ======
def frame_to_bgr_image(frame):
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = None
    return image
# ===================================================

class CameraThread(QThread):
    # 定义信号：发送原图, 发送处理后的图
    update_frames_signal = pyqtSignal(np.ndarray, np.ndarray)
    # 定义信号：发送三个整数 (白葱数量, 葱根数量, 切除数量)
    update_data_signal = pyqtSignal(int, int, int)
    # 异常信号
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True 
        self.model_path = "./black_leek_obb_best.engine"

    def run(self):
        self._run_flag = True
        
        # simulated camera values for local testing
        if not HAS_ORBBEC:
            self._run_mock_camera()
            return
            
        config = Config()
        pipeline = Pipeline()
        
        try:
            model = YOLO(self.model_path, task="obb")
        except Exception as e:
            self.error_signal.emit(f"模型加载失败: {str(e)}")
            return

        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            except OBError as e:
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)

      
            
            pipeline.start(config)
        except Exception as e:
            self.error_signal.emit(f"相机启动失败: {str(e)}")
            return

        try:
            while self._run_flag:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                # 备份原图用于左侧显示
                raw_image = color_image.copy()
                # 结果图用于右侧画框显示
                result_image = color_image.copy()

                # 推理前缩放
                resized_image = cv2.resize(color_image, (320, 240))
                results = model(np.copy(resized_image), conf=0.45, iou=0.35, verbose=False, half=True)
                
                white_list = []
                root_list = []
                
                for result in results:
                    if result.obb is None or len(result.obb) == 0:
                        continue

                    for i, obb_points in enumerate(result.obb.xyxyxyxy):
                        pts = obb_points.cpu().numpy().astype(np.float32)
                        pts[:, 0] *= (640.0 / 320.0)
                        pts[:, 1] *= (480.0 / 240.0)
                        pts = pts.astype(np.int32)
                        
                        cls_id = int(result.obb.cls[i].item())
                        conf = result.obb.conf[i].item()

                        if cls_id == 0:
                            color = (0, 0, 255)
                            root_list.append((pts, conf))
                            display_name = "leek_root"
                        elif cls_id == 1:
                            color = (0, 255, 0)
                            white_list.append((pts, conf))
                            display_name = "leek_white"
                        else:
                            color = (255, 255, 0)
                            display_name = f"unknown_{cls_id}"
                        
                        # 画框在 result_image 上
                        cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)
                        text_x, text_y = int(pts[0][0]), int(pts[0][1] - 10)
                        cv2.putText(result_image, f"{display_name} {conf:.2f}", (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # === 全局贪心距离匹配算法 ===
                white_count = len(white_list)
                root_count = len(root_list)
                uncut_leeks = 0
                DISTANCE_THRESHOLD = 50.0  # 像素距离阈值

                if white_count > 0 and root_count > 0:
                    # 1. 提取葱白底部中心和葱须顶部中心，以及整体中心点
                    w_anchors = []
                    w_centers = []
                    for pts, _ in white_list:
                        w_bottom = pts[np.argsort(pts[:, 1])][2:] # 取y最大(最底)的两个点
                        w_anchors.append(w_bottom.mean(axis=0))
                        w_centers.append(pts.mean(axis=0))
                    
                    r_anchors = []
                    r_centers = []
                    for pts, _ in root_list:
                        r_top = pts[np.argsort(pts[:, 1])][:2]    # 取y最小(最顶)的两个点
                        r_anchors.append(r_top.mean(axis=0))
                        r_centers.append(pts.mean(axis=0))
                    
                    w_anchors = np.array(w_anchors)
                    r_anchors = np.array(r_anchors)
                    w_centers = np.array(w_centers)
                    r_centers = np.array(r_centers)

                    # 2. 计算距离矩阵 (W x R)
                    diff = w_anchors[:, np.newaxis, :] - r_anchors[np.newaxis, :, :]
                    dist_matrix = np.linalg.norm(diff, axis=2)

                    # 3. 全局贪心匹配
                    # 将距离矩阵展平，并获取升序索引
                    flat_indices = np.argsort(dist_matrix, axis=None)
                    
                    matched_w = set()
                    matched_r = set()

                    for idx in flat_indices:
                        w_idx, r_idx = np.unravel_index(idx, dist_matrix.shape)
                        
                        if w_idx in matched_w or r_idx in matched_r:
                            continue
                            
                        min_dist = dist_matrix[w_idx, r_idx]
                        if min_dist < DISTANCE_THRESHOLD:
                            # 匹配成功
                            matched_w.add(w_idx)
                            matched_r.add(r_idx)
                            uncut_leeks += 1
                            
                            # 在真实画面中画出匹配连线，连接两个框的中心点
                            ptw = tuple(w_centers[w_idx].astype(int))
                            ptr = tuple(r_centers[r_idx].astype(int))
                            cv2.line(result_image, ptw, ptr, (255, 0, 255), 3)
                            cv2.circle(result_image, ptw, 4, (0, 255, 0), -1)
                            cv2.circle(result_image, ptr, 4, (0, 0, 255), -1)
                        else:
                            # 由于距离已经排过序，当前最小距离已大于阈值，后续无需继续比较
                            break
                
                # 成功的 = 总发现的葱白体 - 未切除带根的葱白体
                cut_ok = max(0, white_count - uncut_leeks)
                
                # 发送信号更新 UI
                self.update_data_signal.emit(white_count, root_count, cut_ok)
                self.update_frames_signal.emit(raw_image, result_image)

        except Exception as e:
            self.error_signal.emit(f"推理运行错误: {str(e)}")
        finally:
            pipeline.stop()

    def _run_mock_camera(self):
        """本地测试用的模拟相机流（调用本地电脑摄像头）"""
        print("Starting local webcam stream...")
        import time
        
        # 打开默认摄像头 (通常是 0)
        cap = cv2.VideoCapture(0)
        
        # 你可以尝试加载一个本地模型（如果想在电脑上也运行 YOLO的话）
        # 这里为了演示，我们先尝试加载模型，如果有路径报错就跳过推理
        local_model = None
        try:
            # 你可以将 "obb.pt" 改成你电脑上的模型路径进行本地推理测试
            if os.path.exists("obb.pt"):
                local_model = YOLO("obb.pt", task="obb")
                print("成功加载本地模型 obb.pt")
        except Exception as e:
            print(f"本地模型加载失败或未找到: {e}")

        while self._run_flag:
            ret, color_image = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            raw_image = color_image.copy()
            result_image = color_image.copy()
            
            white_count = 0
            root_count = 0
            cut_ok = 0
            
            if local_model is not None:
                # ====== 本地真实的 YOLO 推理 ======
                resized_image = cv2.resize(color_image, (320, 240))
                results = local_model(np.copy(resized_image), conf=0.45, iou=0.35, verbose=False)
                
                white_list = []  # 存放 (中心点X, 中心点Y, 框坐标, conf)
                root_list = []   # 存放 (中心点X, 中心点Y, 框坐标, conf)
                
                # 第一步：分离识别结果，并计算每个框的中心点
                for result in results:
                    if result.obb is None or len(result.obb) == 0: continue
                    for i, obb_points in enumerate(result.obb.xyxyxyxy):
                        pts = obb_points.cpu().numpy().astype(np.float32)
                        pts[:, 0] *= (color_image.shape[1] / 320.0)
                        pts[:, 1] *= (color_image.shape[0] / 240.0)
                        pts = pts.astype(np.int32)
                        
                        cls_id = int(result.obb.cls[i].item())
                        conf = result.obb.conf[i].item()

                        if cls_id == 0:
                            white_list.append((pts, conf))
                            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
                            cv2.putText(result_image, f"white {conf:.2f}", (int(pts[0][0]), int(pts[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        elif cls_id == 1:
                            root_list.append((pts, conf))
                            cv2.polylines(result_image, [pts], True, (0, 0, 255), 2)
                            cv2.putText(result_image, f"root {conf:.2f}", (int(pts[0][0]), int(pts[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 第二步：基于中心点的全局贪心匹配 (快速且高效)
                white_count = len(white_list)
                root_count = len(root_list) # 画面里总的根数量（包含废料）
                uncut_leeks = 0  # 绑定成功的未切葱
                
                DISTANCE_THRESHOLD = 50.0  # 距离阈值
                
                if white_count > 0 and root_count > 0:
                    # 1. 提取葱白底端中心和葱须顶端中心点，以及整体中心点
                    w_anchors = []
                    w_centers = []
                    for pts, _ in white_list:
                        w_bottom = pts[np.argsort(pts[:, 1])][2:]
                        w_anchors.append(w_bottom.mean(axis=0))
                        w_centers.append(pts.mean(axis=0))
                        
                    r_anchors = []
                    r_centers = []
                    for pts, _ in root_list:
                        r_top = pts[np.argsort(pts[:, 1])][:2]
                        r_anchors.append(r_top.mean(axis=0))
                        r_centers.append(pts.mean(axis=0))
                        
                    w_anchors = np.array(w_anchors)
                    r_anchors = np.array(r_anchors)
                    w_centers = np.array(w_centers)
                    r_centers = np.array(r_centers)
                    
                    # 2. 计算距离矩阵 (W x R)
                    diff = w_anchors[:, np.newaxis, :] - r_anchors[np.newaxis, :, :]
                    dist_matrix = np.linalg.norm(diff, axis=2)
                    
                    # 3. 排序所有距离进行全局贪心匹配
                    flat_indices = np.argsort(dist_matrix, axis=None)
                    
                    matched_w = set()
                    matched_r = set()
                    
                    for idx in flat_indices:
                        w_idx, r_idx = np.unravel_index(idx, dist_matrix.shape)
                        
                        if w_idx in matched_w or r_idx in matched_r:
                            continue
                            
                        min_dist = dist_matrix[w_idx, r_idx]
                        if min_dist < DISTANCE_THRESHOLD:
                            matched_w.add(w_idx)
                            matched_r.add(r_idx)
                            uncut_leeks += 1
                            
                            # 画一条紫色的线连接中心点
                            pt1 = tuple(w_centers[w_idx].astype(int))
                            pt2 = tuple(r_centers[r_idx].astype(int))
                            cv2.line(result_image, pt1, pt2, (255, 0, 255), 3)
                            cv2.circle(result_image, pt1, 4, (0, 255, 0), -1)
                            cv2.circle(result_image, pt2, 4, (0, 0, 255), -1)
                        else:
                            break

                # 最终真实数据计算：
                # 成功的 = 总发现的葱白体 - 未切除带根的葱白体
                # 并且强制加上 max(0, x) 防止出现任何不合理的负数输出
                cut_ok = max(0, white_count - uncut_leeks)
            else:
                # ====== 没有模型时，仅仅显示画面，并在画面中间加个提示 ======
                cv2.putText(result_image, "Local Webcam (No Model Loaded)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            # 发送给界面
            self.update_data_signal.emit(white_count, root_count, cut_ok)
            self.update_frames_signal.emit(raw_image, result_image)
            
            time.sleep(0.01) # 限制渲染速度避免卡死界面
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
