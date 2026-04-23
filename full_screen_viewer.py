from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal

class FullScreenViewer(QWidget):
    """全屏查看器子窗口"""
    # 自定义信号，用来通知主窗口全屏结束
    closed_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # 改用标准 QWidget，加入 WindowStaysOnTopHint 确保在最前面
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        # 允许拉伸
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.layout.addWidget(self.label)
        
    def mousePressEvent(self, event):
        # 鼠标点击时发送关闭信号
        self.closed_signal.emit()
        self.close()
