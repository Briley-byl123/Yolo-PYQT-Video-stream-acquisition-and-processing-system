#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口及主要界面交互逻辑。负责集成：
摄像头采集 (由 capture_thread.py 提供)
YOLO检测 (由 detection.py 提供)
视频回放 (由 video_player.py 提供)
辅助工具函数 (由 utils.py 提供)
"""

import sys
import time
from datetime import datetime
import os
from ultralytics import YOLO
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QSlider, QFileDialog,
    QMessageBox, QCheckBox, QLineEdit, QTextEdit, QSizePolicy, QFrame, QApplication
)
from PyQt5.QtCore import QFile, QTextStream
import numpy as np
from capture_thread import VideoCaptureThread
from video_player import VideoPlayer
from detection import YoloDetector
from utils import (
    SUPPORTED_RESOLUTIONS,
    SUPPORTED_FPS,
    detect_cameras,
    safe_release
)

# ------------------------- 默认配置 -------------------------
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_INTERVAL_MINUTES = 1  # 默认存储间隔(分钟)

class MainWindow(QMainWindow):
    """
    主界面窗口
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("多功能视频采集与播放 - 扩展版")
        self.setGeometry(50, 50, 1280, 720)

        # -------------------- 状态变量 --------------------
        self.isRecording = False
        self.timerInterval = DEFAULT_INTERVAL_MINUTES  # 存储间隔(分钟)
        self.lastRecordTime = time.time()
        self.saveFormat = "mp4"   # 默认存储格式
        self.recordOut = None
        self.recordFourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.currentWidth = DEFAULT_WIDTH
        self.currentHeight = DEFAULT_HEIGHT
        self.currentFps = DEFAULT_FPS

        self.baseFilePath = None  # 存储基本文件路径
        self.recordSegmentCounter = 0  # 分段计数器

        # YOLO检测相关
        self.detector = None
        self.useDetector = False  # 是否启用检测
        self.detectionClasses = ["person", "car"]  # 默认检测行人和车辆
        self.confThreshold = 0.3  # 默认置信度阈值

        # 捕获线程
        self.captureThread = None

        # 回放控制
        self.videoPlayer = None  # VideoPlayer实例
        self.isPlaying = False
        self.isPaused = False

        # 加载应用样式
        self.load_app_style()

        # -------------------- 初始化UI --------------------
        self._init_ui()

        # 加载检测器(若需要)
        try:
            self.detector = YoloDetector(model_path="./models/yolov5su.pt")
            self.logViewer.append("[INFO] YOLO模型加载成功。")
        except Exception as e:
            self.logViewer.append(f"[警告] 加载YOLO模型失败: {e}")
            self.detector = None

    def load_app_style(self):
        """加载应用样式表"""
        try:
            style_file = QFile("app_style.css")
            if style_file.open(QFile.ReadOnly | QFile.Text):
                stream = QTextStream(style_file)
                style_sheet = stream.readAll()
                self.setStyleSheet(style_sheet)
                style_file.close()
                print("样式文件加载成功")
            else:
                print(f"无法打开样式文件: {style_file.errorString()}")
        except Exception as e:
            print(f"加载样式文件失败: {e}")

    def _init_ui(self):
        """
        初始化界面：使用QTabWidget分成两部分
          1. 视频采集/显示/回放
          2. 设置/信息
        """
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        tabWidget = QTabWidget()

        # 1. 视频功能Tab
        videoTab = self._create_video_tab()
        tabWidget.addTab(videoTab, "视频")

        # 2. 设置Tab
        settingsTab = self._create_settings_tab()
        tabWidget.addTab(settingsTab, "设置")

        # 总布局
        layout = QVBoxLayout()
        layout.addWidget(tabWidget)
        centralWidget.setLayout(layout)

    def _create_video_tab(self):
        """
        创建视频采集/回放等主布局
        """
        tabWidget = QWidget()

        # 创建分组函数
        def create_group_frame(title):
            frame = QFrame()
            frame.setObjectName("groupFrame")
            frame.setFrameShape(QFrame.StyledPanel)
            
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            
            if title:
                label = QLabel(title)
                label.setProperty("class", "groupTitle")
                layout.addWidget(label)
            
            return frame, layout

        # 主布局 - 使用水平布局将控制区和视频区分开
        mainLayout = QHBoxLayout()
        
        # 左侧 - 视频显示区域 (70%的空间)
        videoContainer = QFrame()
        videoContainer.setObjectName("videoContainer")
        videoContainer.setFrameShape(QFrame.StyledPanel)
        videoContainer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        videoContainerLayout = QVBoxLayout(videoContainer)
        videoContainerLayout.setContentsMargins(0, 0, 0, 0)
        
        self.videoLabel = QLabel("视频显示区")
        self.videoLabel.setObjectName("videoLabel")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        videoContainerLayout.addWidget(self.videoLabel)
        
        # 右侧 - 控制区域 (30%的空间)
        controlPanel = QWidget()
        controlPanel.setObjectName("controlPanel")
        controlPanel.setMaximumWidth(350)  # 限制控制面板宽度
        controlPanelLayout = QVBoxLayout(controlPanel)

        # 摄像头选择下拉
        self.cameraComboBox = QComboBox()
        # 手机摄像头的 URL 列表
        mobile_camera_urls = [
            "http://192.168.1.100:8080/video",  # 替换为你的手机摄像头URL
        ]

        # 检测摄像头（包括本地和手机摄像头）
        camera_dict = detect_cameras(max_test=5, mobile_camera_urls=mobile_camera_urls)
        for idx, name in camera_dict.items():
            self.cameraComboBox.addItem(name, idx if isinstance(idx, int) else str(idx))
        self.cameraComboBox.setCurrentIndex(0)

        # 分辨率选择
        self.resolutionComboBox = QComboBox()
        for res_name in SUPPORTED_RESOLUTIONS.keys():
            self.resolutionComboBox.addItem(res_name)
        self.resolutionComboBox.setCurrentText("480P")

        # 帧率选择
        self.fpsComboBox = QComboBox()
        for fps_val in SUPPORTED_FPS:
            self.fpsComboBox.addItem(str(fps_val))
        self.fpsComboBox.setCurrentText(str(DEFAULT_FPS))

        # 定时存储
        self.intervalSpinBox = QSpinBox()
        self.intervalSpinBox.setRange(1, 60)
        self.intervalSpinBox.setValue(DEFAULT_INTERVAL_MINUTES)

        # 存储格式
        self.formatComboBox = QComboBox()
        self.formatComboBox.addItem("mp4")
        self.formatComboBox.addItem("avi")

        # 创建按钮并添加objectName
        self.btnStartCamera = QPushButton("启动摄像头")
        self.btnStartCamera.setObjectName("btnStartCamera")
        
        self.btnStopCamera = QPushButton("停止摄像头")
        self.btnStopCamera.setObjectName("btnStopCamera")
        
        self.btnStartRecord = QPushButton("开始存储")
        self.btnStartRecord.setObjectName("btnStartRecord")
        
        self.btnStopRecord = QPushButton("停止存储")
        self.btnStopRecord.setObjectName("btnStopRecord")
        
        self.btnLoadVideo = QPushButton("打开本地视频")
        self.btnLoadVideo.setObjectName("btnLoadVideo")
        
        self.btnToggleDetect = QPushButton("开启检测")
        self.btnToggleDetect.setObjectName("btnToggleDetect")
        
        self.btnConvertFormat = QPushButton("格式转换")
        self.btnConvertFormat.setObjectName("btnConvertFormat")

        # 回放控制
        self.playSlider = QSlider(Qt.Horizontal)
        self.playSlider.setMinimum(0)
        self.playSlider.setMaximum(100)
        self.playSlider.setValue(0)
        self.playSlider.setEnabled(False)

        self.btnPlay = QPushButton("播放")
        self.btnPlay.setObjectName("btnPlay")
        self.btnPlay.setEnabled(False)
        
        self.btnPause = QPushButton("暂停")
        self.btnPause.setObjectName("btnPause")
        self.btnPause.setEnabled(False)
        
        self.btnStop = QPushButton("停止")
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setEnabled(False)

        # 添加文件格式转换控件
        self.formatConvertComboBox = QComboBox()
        self.formatConvertComboBox.addItem("avi")
        self.formatConvertComboBox.addItem("mp4")
        self.formatConvertComboBox.addItem("mov")
        self.formatConvertComboBox.addItem("mkv")

        # 摄像头设置分组
        cameraGroupFrame, cameraGroupLayout = create_group_frame("摄像头设置")

        cameraLayout = QVBoxLayout()

        camSourceLayout = QHBoxLayout()
        camSourceLayout.addWidget(QLabel("摄像头:"))
        camSourceLayout.addWidget(self.cameraComboBox, 1)

        camResLayout = QHBoxLayout()
        camResLayout.addWidget(QLabel("分辨率:"))
        camResLayout.addWidget(self.resolutionComboBox, 1)
        camResLayout.addWidget(QLabel("FPS:"))
        camResLayout.addWidget(self.fpsComboBox)

        cameraLayout.addLayout(camSourceLayout)
        cameraLayout.addLayout(camResLayout)

        # 摄像头控制按钮放在一行
        cameraControlLayout = QHBoxLayout()
        cameraControlLayout.addWidget(self.btnStartCamera)
        cameraControlLayout.addWidget(self.btnStopCamera)
        # cameraControlLayout.addStretch()  # 添加弹性空间

        # 检测按钮单独放在一行
        detectControlLayout = QHBoxLayout()
        detectControlLayout.addWidget(self.btnToggleDetect)
        # detectControlLayout.addStretch()  # 添加弹性空间，让按钮靠左对齐

        # 将所有布局添加到摄像头分组
        cameraGroupLayout.addLayout(cameraLayout)
        cameraGroupLayout.addLayout(cameraControlLayout)
        cameraGroupLayout.addLayout(detectControlLayout)
        # 视频存储分组
        recordGroupFrame, recordGroupLayout = create_group_frame("视频存储")
        
        recordLayout = QVBoxLayout()
        
        recordSettingsLayout = QHBoxLayout()
        recordSettingsLayout.addWidget(QLabel("定时(分钟):"))
        recordSettingsLayout.addWidget(self.intervalSpinBox)
        recordSettingsLayout.addWidget(QLabel("格式:"))
        recordSettingsLayout.addWidget(self.formatComboBox)
        
        recordBtnLayout = QHBoxLayout()
        recordBtnLayout.addWidget(self.btnStartRecord)
        recordBtnLayout.addWidget(self.btnStopRecord)
        
        recordLayout.addLayout(recordSettingsLayout)
        recordLayout.addLayout(recordBtnLayout)
        
        recordGroupLayout.addLayout(recordLayout)

        # 视频回放分组
        playbackGroupFrame, playbackGroupLayout = create_group_frame("视频回放")
        
        #playbackControlLayout = QHBoxLayout()#水平布局
        playbackControlLayout = QVBoxLayout()#垂直布局
        playbackControlLayout.addWidget(self.btnLoadVideo)
        playbackControl2 = QHBoxLayout()  # 播放、暂停、停止按钮放在一行
        playbackControl2.addWidget(self.btnPlay)
        playbackControl2.addWidget(self.btnPause)
        playbackControl2.addWidget(self.btnStop)
        playbackControlLayout.addLayout(playbackControl2)
        
        playbackSliderLayout = QVBoxLayout()
        playbackSliderLayout.addWidget(self.playSlider)
        
        playbackGroupLayout.addLayout(playbackControlLayout)
        playbackGroupLayout.addLayout(playbackSliderLayout)

        # 格式转换分组
        convertGroupFrame, convertGroupLayout = create_group_frame("格式转换")
        
        convertLayout = QHBoxLayout()
        convertLayout.addWidget(QLabel("目标格式:"))
        convertLayout.addWidget(self.formatConvertComboBox)
        convertLayout.addWidget(self.btnConvertFormat)
        
        convertGroupLayout.addLayout(convertLayout)

        # 添加所有控制组件到控制面板
        controlPanelLayout.addWidget(cameraGroupFrame)
        controlPanelLayout.addWidget(recordGroupFrame)
        controlPanelLayout.addWidget(playbackGroupFrame)
        controlPanelLayout.addWidget(convertGroupFrame)
        controlPanelLayout.addStretch(1)  # 底部添加弹性空间

        # 将视频区和控制区添加到主布局
        mainLayout.addWidget(videoContainer, 7)  # 70% 的空间
        mainLayout.addWidget(controlPanel, 3)   # 30% 的空间

        tabWidget.setLayout(mainLayout)

        # 绑定信号
        self.btnStartCamera.clicked.connect(self.start_camera)
        self.btnStopCamera.clicked.connect(self.stop_camera)
        self.btnStartRecord.clicked.connect(self.start_recording)
        self.btnStopRecord.clicked.connect(self.stop_recording)
        self.btnLoadVideo.clicked.connect(self.load_video)
        self.btnPlay.clicked.connect(self.play_video)
        self.btnPause.clicked.connect(self.pause_video)
        self.btnStop.clicked.connect(self.stop_video)
        self.btnToggleDetect.clicked.connect(self.toggle_detection)

        self.resolutionComboBox.currentIndexChanged.connect(self.on_resolution_change)
        self.fpsComboBox.currentIndexChanged.connect(self.on_fps_change)
        self.intervalSpinBox.valueChanged.connect(self.on_interval_change)
        self.formatComboBox.currentIndexChanged.connect(self.on_format_change)
        self.btnConvertFormat.clicked.connect(self.convert_format)

        return tabWidget

    def _create_settings_tab(self):
        """
        创建一个Tab来展示更多的设置选项，如检测目标类别，置信度阈值，以及支持日志显示等
        """
        widget = QWidget()

        # 创建分组函数
        def create_group_frame(title):
            frame = QFrame()
            frame.setObjectName("groupFrame")
            frame.setFrameShape(QFrame.StyledPanel)
            
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            
            if title:
                label = QLabel(title)
                label.setProperty("class", "groupTitle")
                layout.addWidget(label)
            
            return frame, layout

        # 使用水平布局分割设置区和日志区
        mainLayout = QHBoxLayout()

        # 左侧 - 设置区域
        settingsPanel = QWidget()
        settingsPanelLayout = QVBoxLayout(settingsPanel)
        settingsPanelLayout.setContentsMargins(0, 0, 0, 0)

        # 检测设置分组
        detectionGroupFrame, detectionGroupLayout = create_group_frame("检测设置")
        
        # 示例：检测目标类别
        classesLabel = QLabel("检测目标类别(英文逗号分隔):")
        self.lineDetectClasses = QLineEdit()
        self.lineDetectClasses.setText("person,car")  # 默认检测行人和车辆
        
        # 置信度阈值
        confidenceLabel = QLabel("置信度阈值(%):")
        self.spinConfThreshold = QSpinBox()
        self.spinConfThreshold.setRange(1, 100)
        self.spinConfThreshold.setValue(50)
        
        # 添加保存设置按钮
        self.btnSaveSettings = QPushButton("保存检测设置")
        self.btnSaveSettings.setObjectName("btnSaveSettings")
        
        detectionGroupLayout.addWidget(classesLabel)
        detectionGroupLayout.addWidget(self.lineDetectClasses)
        detectionGroupLayout.addWidget(confidenceLabel)
        detectionGroupLayout.addWidget(self.spinConfThreshold)
        detectionGroupLayout.addWidget(self.btnSaveSettings)
        
        settingsPanelLayout.addWidget(detectionGroupFrame)
        settingsPanelLayout.addStretch(1)

        # 右侧 - 日志区域
        logGroupFrame, logGroupLayout = create_group_frame("系统日志")
        
        self.logViewer = QTextEdit()
        self.logViewer.setObjectName("logViewer")
        self.logViewer.setReadOnly(True)
        self.logViewer.setPlaceholderText("日志输出...")
        
        logGroupLayout.addWidget(self.logViewer)

        # 添加到主布局
        mainLayout.addWidget(settingsPanel, 3)  # 30% 的空间
        mainLayout.addWidget(logGroupFrame, 7)  # 70% 的空间

        # 连接信号
        self.btnSaveSettings.clicked.connect(self.save_detection_settings)
        self.spinConfThreshold.valueChanged.connect(self.on_conf_threshold_change)
        self.lineDetectClasses.textChanged.connect(self.on_detect_classes_change)

        widget.setLayout(mainLayout)
        return widget
    
    def save_detection_settings(self):
        """保存并应用检测设置"""
        self.confThreshold = self.spinConfThreshold.value() / 100.0
        classes_str = self.lineDetectClasses.text()
        # 确保类别正确格式化
        self.detectionClasses = [s.strip() for s in classes_str.split(",") if s.strip()] if classes_str else []
        
        # 输出调试信息查看类别格式
        self.logViewer.append(f"[DEBUG] 检测类别格式: {type(self.detectionClasses)}, 值: {self.detectionClasses}")       
        # 更新检测器设置
        if self.detector:
            # 假设YoloDetector类有这些属性，如果没有，需要修改检测器类
            if hasattr(self.detector, 'conf'):
                self.detector.conf = self.confThreshold
            
            self.logViewer.append(f"[INFO] 检测设置已更新: 置信度={self.confThreshold}, 类别={classes_str}")
            
            # 如果正在使用检测器，需要重启线程以应用新设置
            if self.useDetector and self.captureThread is not None:
                wasRecording = self.isRecording
                self.stop_camera()
                if wasRecording:
                    self.stop_recording()

                self.start_camera()
                if wasRecording:
                    self.start_recording()
        else:
            self.logViewer.append("[WARN] 检测器未加载，设置已保存但未应用")
            QMessageBox.warning(self, "警告", "YOLO模型未加载，设置已保存但未应用到检测器")
    
    def on_conf_threshold_change(self, value):
        """当置信度阈值改变时"""
        self.logViewer.append(f"[INFO] 置信度阈值已修改为: {value}%")
    
    def on_detect_classes_change(self, text):
        """当检测类别改变时"""
        self.logViewer.append(f"[INFO] 检测类别已修改为: {text}")

    # -------------------- 摄像头及录像逻辑 --------------------
    def start_camera(self):
        """ 打开摄像头，启动采集线程 """
        if self.captureThread is not None:
            QMessageBox.warning(self, "警告", "摄像头已在运行中!")
            return
        try:
            cameraIndex = self.cameraComboBox.currentData()
            # 检查是否为手机摄像头 URL
            if isinstance(cameraIndex, str) and cameraIndex.startswith("http"):
                self.captureThread = VideoCaptureThread(
                    cameraIndex=cameraIndex,  # 直接传入URL
                    width=self.currentWidth,
                    height=self.currentHeight,
                    fps=self.currentFps,
                    detector=self.detector if self.useDetector else None
                )
            else:
                self.captureThread = VideoCaptureThread(
                    cameraIndex=cameraIndex,
                    width=self.currentWidth,
                    height=self.currentHeight,
                    fps=self.currentFps,
                    detector=self.detector if self.useDetector else None
                )

            if isinstance(cameraIndex, str) and cameraIndex.startswith("http"):
                self.logViewer.append(f"[INFO] 正在连接手机摄像头: {cameraIndex}")
            else:
                self.logViewer.append(f"[INFO] 正在启动本地摄像头: {cameraIndex}")
            self.captureThread.frameCaptured.connect(self.update_frame)
            self.captureThread.cameraError.connect(self.on_camera_error)
            self.captureThread.start()
            self.logViewer.append("[INFO] 成功启动摄像头采集线程.")
        except Exception as e:
            self.logViewer.append(f"[ERROR] 无法启动摄像头采集：{e}")
            QMessageBox.critical(self, "错误", f"启动摄像头异常：{e}")

    def stop_camera(self):
        """ 停止摄像头 """
        if self.captureThread:
            self.captureThread.stop()
            self.captureThread = None
            self.videoLabel.clear()
            self.videoLabel.setText("视频显示区")
            self.logViewer.append("[INFO] 摄像头已停止.")

    def on_camera_error(self, errMsg):
        QMessageBox.critical(self, "摄像头错误", errMsg)
        self.logViewer.append(f"[ERROR] 摄像头错误：{errMsg}")
        self.stop_camera()

    def update_frame(self, frame):
        """
        从采集线程接收图像帧，用于显示 + 录像
        """
        # 定时存储判断
        if self.isRecording and frame is not None:
            now = time.time()
            if now - self.lastRecordTime >= (self.timerInterval * 60):
                # 到时间了，重新开始一个新的文件
                self.start_new_save_file()
                self.lastRecordTime = now

            if self.recordOut is not None:
                try:
                    self.recordOut.write(frame)
                except Exception as e:
                    self.logViewer.append(f"[ERROR] 保存视频帧异常: {e}")

        # 显示到GUI
        if frame is not None:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            
            # 使用KeepAspectRatio使视频保持比例但尽可能填满区域
            p = convertToQtFormat.scaled(
                self.videoLabel.width(),
                self.videoLabel.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.videoLabel.setPixmap(QPixmap.fromImage(p))

    def start_recording(self):
        """ 开始定时存储 """
        if self.captureThread is None:
            QMessageBox.warning(self, "警告", "请先启动摄像头！")
            return
        if not self.isRecording:
            self.isRecording = True
            
            # 重置分段计数器
            self.recordSegmentCounter = 0
            
            # 弹出对话框让用户选择第一个文件的保存位置
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = "./videos"
            
            # 如果目录不存在，创建它
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            # 获取基本文件路径
            self.baseFilePath, _ = QFileDialog.getSaveFileName(
                self, "保存视频", f"{directory}/record_{timestamp}.{self.saveFormat}", 
                f"视频文件 (*.{self.saveFormat})"
            )
            
            if not self.baseFilePath:
                self.isRecording = False
                self.logViewer.append("[INFO] 用户取消了保存操作。")
                return
                
            # 现在开始第一段的录制
            self.start_new_save_file()
            self.lastRecordTime = time.time()
            self.logViewer.append("[INFO] 开始存储视频。")

    def start_new_save_file(self):
        """ 打开一个新的输出文件，用于分段存储 """
        if self.baseFilePath is None:
            self.logViewer.append("[ERROR] 没有设置基本文件路径。")
            return
            
        # 从基本路径生成新的文件名（添加分段编号）
        base_name, ext = os.path.splitext(self.baseFilePath)
        new_file_path = f"{base_name}_part{self.recordSegmentCounter}{ext}"
        self.recordSegmentCounter += 1
        
        safe_release(self.recordOut)

        try:
            self.recordOut = cv2.VideoWriter(
                new_file_path,
                self.recordFourcc,
                float(self.currentFps),
                (self.currentWidth, self.currentHeight)
            )
            self.logViewer.append(f"[INFO] 新建视频存储文件: {new_file_path}")
        except Exception as e:
            self.logViewer.append(f"[ERROR] 创建VideoWriter失败: {e}")

    def stop_recording(self):
        """ 停止定时存储 """
        if self.isRecording:
            self.isRecording = False
            safe_release(self.recordOut)
            self.recordOut = None
            self.baseFilePath = None  # 重置基本文件路径
            self.logViewer.append("[INFO] 停止存储视频。")

    def convert_format(self):
        """ 文件格式转换 """
        filePath, _ = QFileDialog.getOpenFileName(
            self, "选择要转换的视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not filePath:
            return

        targetFormat = self.formatConvertComboBox.currentText()
        outputFilePath, _ = QFileDialog.getSaveFileName(
            self, "保存转换后的视频", f"converted_video.{targetFormat}", f"视频文件 (*.{targetFormat})"
        )
        if not outputFilePath:
            return

        try:
            # 使用 ffmpeg 或 OpenCV 转换视频格式
            cap = cv2.VideoCapture(filePath)
            fourcc = cv2.VideoWriter_fourcc(*'XVID') if targetFormat == "avi" else cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(outputFilePath, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()
            self.logViewer.append(f"[INFO] 文件转换成功: {outputFilePath}")
            QMessageBox.information(self, "成功", f"文件转换成功: {outputFilePath}")
        except Exception as e:
            self.logViewer.append(f"[ERROR] 文件转换失败: {e}")
            QMessageBox.critical(self, "错误", f"文件转换失败: {e}")


    # -------------------- 参数变更/格式处理 --------------------
    def on_resolution_change(self):
        res_name = self.resolutionComboBox.currentText()
        w, h = SUPPORTED_RESOLUTIONS.get(res_name, (640, 480))
        self.currentWidth, self.currentHeight = w, h
        self.logViewer.append(f"[INFO] 分辨率切换为: {w}x{h}")

    def on_fps_change(self):
        val = self.fpsComboBox.currentText()
        try:
            self.currentFps = int(val)
        except:
            self.currentFps = DEFAULT_FPS
        self.logViewer.append(f"[INFO] FPS切换为: {self.currentFps}")

    def on_interval_change(self):
        self.timerInterval = self.intervalSpinBox.value()
        self.logViewer.append(f"[INFO] 存储间隔切换为: {self.timerInterval}分钟")

    def on_format_change(self):
        fmt = self.formatComboBox.currentText()
        self.saveFormat = fmt
        if fmt == "avi":
            self.recordFourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            self.recordFourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.logViewer.append(f"[INFO] 视频格式切换为: {fmt}")

    # -------------------- 视频回放相关功能 --------------------
    def load_video(self):
        """ 选择本地视频进行回放 """
        filePath, _ = QFileDialog.getOpenFileName(
            self, "选择要播放的视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if filePath:
            self.stop_video()
            try:
                self.videoPlayer = VideoPlayer(
                    filePath=filePath,
                    mainWindow=self,  # 传给VideoPlayer，用于回调更新UI
                    fps=self.currentFps
                )
                if not self.videoPlayer.open():
                    QMessageBox.critical(self, "错误", "无法打开该视频文件")
                    self.videoPlayer = None
                    return
                # 设置进度条
                total_frames = self.videoPlayer.get_total_frames()
                self.playSlider.setEnabled(True)
                self.playSlider.setMinimum(0)
                self.playSlider.setMaximum(max(total_frames - 1, 0))
                self.playSlider.setValue(0)

                # 启用控制按钮
                self.btnPlay.setEnabled(True)
                self.btnPause.setEnabled(True)
                self.btnStop.setEnabled(True)

                self.logViewer.append(f"[INFO] 加载视频成功: {filePath}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载视频异常: {e}")
                self.logViewer.append(f"[ERROR] 加载视频异常: {e}")
                self.videoPlayer = None

    def play_video(self):
        """ 播放视频 """
        if self.videoPlayer is None:
            return
        self.isPlaying = True
        self.isPaused = False
        self.videoPlayer.start()

    def pause_video(self):
        """ 暂停视频 """
        self.isPaused = True
        if self.videoPlayer is not None:
            self.videoPlayer.pause()

    def stop_video(self):
        """ 停止视频 """
        self.isPlaying = False
        self.isPaused = False
        if self.videoPlayer:
            self.videoPlayer.stop()
            self.videoPlayer = None
        self.playSlider.setValue(0)
        self.playSlider.setEnabled(False)
        self.btnPlay.setEnabled(False)
        self.btnPause.setEnabled(False)
        self.btnStop.setEnabled(False)
        self.videoLabel.clear()
        self.videoLabel.setText("视频显示区")

    # 在回放时，更新进度条的位置
    def update_playback_position(self, pos):
        self.playSlider.setValue(pos)

    # 在回放时，接收图像并显示
    def update_playback_frame(self, frame):
        if self.useDetector and self.detector is not None:
            try:
                # 添加类型检查
                if isinstance(frame, str):
                    self.logViewer.append(f"[ERROR] 接收到字符串而不是图像帧")
                if frame is None or not isinstance(frame, np.ndarray):
                    self.logViewer.append(f"[ERROR] 无效的帧格式")
                    return
                    
                # 使用保存的检测设置
                frame = self.detector.detect_and_plot(
                    frame, 
                    conf_thres=self.confThreshold, 
                    classes=self.detectionClasses
                )
            except Exception as e:
                self.logViewer.append(f"[ERROR] 离线检测异常: {e}")
        
        # 添加此行将帧显示到界面上
        self.update_frame(frame)

    # -------------------- YOLO检测开关 --------------------
    def toggle_detection(self):
        if not self.detector:
            QMessageBox.warning(self, "警告", "YOLO模型不可用，无法启用检测！")
            return

        # 获取当前设置
        self.confThreshold = self.spinConfThreshold.value() / 100.0
        classes_str = self.lineDetectClasses.text()
        self.detectionClasses = [s.strip() for s in classes_str.split(",") if s.strip()] if classes_str else []
        
        # 切换检测状态
        if not self.useDetector:
            self.useDetector = True
            self.btnToggleDetect.setText("关闭检测")
            self.btnToggleDetect.setProperty("state", "active")
            self.logViewer.append(f"[INFO] 实时YOLO检测已开启。类别：{classes_str}, 置信度：{self.confThreshold}")
        else:
            self.useDetector = False
            self.btnToggleDetect.setText("开启检测")
            self.btnToggleDetect.setProperty("state", "normal")
            self.logViewer.append("[INFO] 实时YOLO检测已关闭.")
        
        # 强制刷新样式
        self.btnToggleDetect.style().unpolish(self.btnToggleDetect)
        self.btnToggleDetect.style().polish(self.btnToggleDetect)

        # 如果正在采集，需要重启线程
        if self.captureThread is not None:
            wasRecording = self.isRecording
            self.stop_camera()
            if wasRecording:
                self.stop_recording()

            self.start_camera()
            if wasRecording:
                self.start_recording()

    # -------------------- 窗口关闭处理 --------------------
    def closeEvent(self, event):
        if self.captureThread:
            self.captureThread.stop()
        safe_release(self.recordOut)
        if self.videoPlayer:
            self.videoPlayer.stop()
        event.accept()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())