#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 封装一个简单的视频回放控制类，使用 QTimer 来按帧读取并发给主窗口显示。
import cv2
from PyQt5.QtCore import QTimer
import numpy as np

class VideoPlayer:
    """
    用于本地视频回放的封装类
    """
    def __init__(self, filePath, mainWindow, fps=30):
        self.filePath = filePath
        self.mainWindow = mainWindow
        self.cap = None
        self.isPlaying = False
        self.isPaused = False
        self.fps = fps
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

    def open(self):
        self.cap = cv2.VideoCapture(self.filePath)
        if not self.cap.isOpened():
            return False
        return True

    def get_total_frames(self):
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def start(self):
        self.isPlaying = True
        self.isPaused = False
        self.timer.start(int(1000 / (self.fps + 1e-6)))

    def pause(self):
        self.isPaused = True
        self.isPlaying = False
        self.timer.stop()

    def stop(self):
        self.isPlaying = False
        self.isPaused = False
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _next_frame(self):
        """ 定时器调用此函数读取下一帧 """
        if not self.isPlaying or self.isPaused or (not self.cap):
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # 播放结束
            self.mainWindow.stop_video()
            return

        # 检查帧是否为有效的 numpy.ndarray
        if not isinstance(frame, np.ndarray):
            self.mainWindow.logViewer.append(f"[ERROR] 无效的帧格式: {type(frame)}")
            self.mainWindow.stop_video()
            return

        # 输出帧的形状，帮助调试
        self.mainWindow.logViewer.append(f"[DEBUG] 帧形状: {frame.shape}")

        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 更新主界面进度条显示
        self.mainWindow.update_playback_position(current_pos)

        # 通知主界面更新画面
        self.mainWindow.update_playback_frame(frame)