import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QLineEdit, QPushButton)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent

# === 从其他文件模块导入各个类 ===
from camera_thread import CameraThread
from full_screen_viewer import FullScreenViewer

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("大葱切割智能识别分析系统")
        self.resize(1200, 800)
        self.setup_ui()
        
        self.cam_thread = None
        self.fullscreen_dialog = None
        self.current_fullscreen_target = None # 记录当前全屏的是 'raw' 还是 'result'

        # 为画面 Label 安装事件过滤器，捕获鼠标点击事件
        self.label_raw.installEventFilter(self)
        self.label_result.installEventFilter(self)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # ====== 上半部分：视频画面 ======
        video_layout = QHBoxLayout()
        
        # 原画面
        self.label_raw = QLabel("摄像头原画面 (点击全屏)")
        self.label_raw.setStyleSheet("background-color: #2b2b2b; color: white; border: 2px solid #555;")
        self.label_raw.setAlignment(Qt.AlignCenter)
        self.label_raw.setScaledContents(True)
        self.label_raw.setMinimumSize(480, 360)
        
        # 结果画面
        self.label_result = QLabel("AI识别画面 (点击全屏)")
        self.label_result.setStyleSheet("background-color: #2b2b2b; color: white; border: 2px solid #555;")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setScaledContents(True)
        self.label_result.setMinimumSize(480, 360)
        
        video_layout.addWidget(self.label_raw)
        video_layout.addWidget(self.label_result)
        
        # ====== 下半部分：数据与控制 ======
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(10, 20, 10, 10)
        
        # 数据区 (左下)
        data_layout = QVBoxLayout()
        self.edit_white = self.create_data_row(data_layout, "识别葱白数量:")
        self.edit_root = self.create_data_row(data_layout, "识别葱须数量:")
        self.edit_cut_ok = self.create_data_row(data_layout, "成功切除数量:")
        
        # 控制区 (右下)
        control_layout = QVBoxLayout()
        self.btn_open = QPushButton("▶ 打开摄像头")
        self.btn_open.setMinimumHeight(50)
        self.btn_open.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_open.clicked.connect(self.start_camera)
        
        self.btn_close = QPushButton("⏹ 关闭摄像头")
        self.btn_close.setMinimumHeight(50)
        self.btn_close.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #f44336; color: white;")
        self.btn_close.clicked.connect(self.stop_camera)
        self.btn_close.setEnabled(False) # 默认禁用，等打开相机后再启用
        
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_close)
        control_layout.addStretch()

        bottom_layout.addLayout(data_layout, stretch=2)
        bottom_layout.addLayout(control_layout, stretch=1)
        
        # 拼装
        main_layout.addLayout(video_layout, stretch=3)
        main_layout.addLayout(bottom_layout, stretch=1)

    def create_data_row(self, parent_layout, label_text):
        row_layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        edit = QLineEdit("0")
        edit.setReadOnly(True)  # 设置为只读，不允许用户修改
        edit.setAlignment(Qt.AlignCenter)
        edit.setStyleSheet("font-size: 16px; border: 1px solid #ccc; padding: 5px;")
        row_layout.addWidget(lbl)
        row_layout.addWidget(edit)
        parent_layout.addLayout(row_layout)
        return edit

    # ============= 核心控制逻辑 =============
    def start_camera(self):
        if self.cam_thread is None or not self.cam_thread.isRunning():
            self.cam_thread = CameraThread()
            # 连接核心信号槽
            self.cam_thread.update_frames_signal.connect(self.update_images)
            self.cam_thread.update_data_signal.connect(self.update_data)
            self.cam_thread.error_signal.connect(lambda e: print(f"[Error]: {e}"))
            
            self.cam_thread.start()
            
            self.btn_open.setEnabled(False)
            self.btn_close.setEnabled(True)

    def stop_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.btn_open.setEnabled(True)
            self.btn_close.setEnabled(False)
            self.label_raw.setText("已关闭")
            self.label_result.setText("已关闭")

    # ============= UI 数据更新与格式转换 =============
    def update_data(self, white, root, cut):
        self.edit_white.setText(str(white))
        self.edit_root.setText(str(root))
        self.edit_cut_ok.setText(str(cut))

    def update_images(self, raw_img, result_img):
        raw_pixmap = self.cv_to_qpixmap(raw_img)
        result_pixmap = self.cv_to_qpixmap(result_img)

        # 无论是否全屏，主界面的双画面都应该一直更新，这是主线程活着的最好证明！
        self.label_raw.setPixmap(raw_pixmap)
        self.label_result.setPixmap(result_pixmap)

        # 全屏模式检查
        if self.fullscreen_dialog is not None:
            # 如果进入了全屏，则额外向全屏视窗输送指定的图流
            if self.current_fullscreen_target == 'raw':
                self.fullscreen_dialog.label.setPixmap(raw_pixmap)
            else:
                self.fullscreen_dialog.label.setPixmap(result_pixmap)

    def cv_to_qpixmap(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    # ============= 鼠标全屏点击事件监控 =============
    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            # 只有当相机线程正在运作并且有画面时，才允许进入全屏
            if self.cam_thread and self.cam_thread.isRunning():
                if source is self.label_raw:
                    self.open_fullscreen('raw')
                    return True
                elif source is self.label_result:
                    self.open_fullscreen('result')
                    return True
        return super().eventFilter(source, event)

    def open_fullscreen(self, target):
        self.current_fullscreen_target = target
        self.fullscreen_dialog = FullScreenViewer()
        
        # 连接我们刚刚在子窗口里自定义的 closed_signal
        self.fullscreen_dialog.closed_signal.connect(self._on_fullscreen_closed)
        
        # Linux (Jetson Ubuntu) 非常需要先 show() 再进行窗口状态的改变
        self.fullscreen_dialog.show()
        # 强制变成全屏并升到最顶层
        self.fullscreen_dialog.showFullScreen()
        self.fullscreen_dialog.raise_()
        self.fullscreen_dialog.activateWindow()

    def _on_fullscreen_closed(self):
        # 恢复状态变量即可（窗口会在自身 close() 中释放）
        self.fullscreen_dialog = None
        self.current_fullscreen_target = None

    def closeEvent(self, event):
        # 如果点击了主窗口右上角的 X 关掉程序，务必在此安全杀掉相机内存！
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
