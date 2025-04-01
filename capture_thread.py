#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 实现摄像头采集线程，在单独的线程中循环读取摄像头帧，
# 并可选用YOLO检测后再发送给主界面进行显示与录像。

import time
import cv2
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class VideoCaptureThread(QThread):
    """
    在单独的线程中执行视频采集/检测，发出frameCaptured信号供主界面更新UI。
    """
    frameCaptured = pyqtSignal(object)   # 发送图像帧
    cameraError = pyqtSignal(str)        # 发送摄像头错误消息

    def __init__(self, cameraIndex=0, width=640, height=480, fps=30, detector=None):
        super().__init__()
        self.cameraIndex = cameraIndex
        self.width = width
        self.height = height
        self.fps = fps
        self.detector = detector
        self._running = True
        self.cap = None

    def run(self):
        """ 线程主体：打开摄像头，循环采集帧并检测 """
        try:
            # 有些平台需要CV_CAP_DSHOW等，做更多尝试
            self.cap = cv2.VideoCapture(self.cameraIndex, cv2.CAP_DSHOW)
        except:
            self.cap = cv2.VideoCapture(self.cameraIndex)

        # 设置分辨率 / 帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            self.cameraError.emit(f"无法打开摄像头(Index: {self.cameraIndex})")
            return

        while self._running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cameraError.emit("摄像头读取失败！")
                break

            # YOLO检测
            if self.detector:
                try:
                    frame = self.detector.detect_and_plot(frame)
                except Exception as e:
                    # 如果检测出错了，不中断摄像头读取
                    print(f"[ERROR] YOLO检测过程中出错: {e}")
                    traceback.print_exc()

            self.frameCaptured.emit(frame)

            # 控制帧率
            time.sleep(1 / (self.fps + 1e-6))

        if self.cap is not None:
            self.cap.release()

    def stop(self):
        """ 停止线程 """
        self._running = False
        self.quit()
        self.wait()