#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# YOLO模型的加载与推断逻辑。这里示例使用 ultralytics 包进行 YOLOv5(s) 预训练模型的载入；
# 如果使用 YOLOv8 或官方 yolov5 仓库，需要对应修改。

import cv2
import numpy as np
import time

# 如果安装了ultralytics，可直接使用 YOLO类
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[警告] 未安装ultralytics库，YOLO功能将无法使用！")

class YoloDetector:
    """
    封装用于加载YOLO模型并进行推断的类
    """
    def __init__(self, model_path="./models/yolov5s.pt", skip_frames=2):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics库不可用，无法创建YoloDetector.")

        self.model = YOLO(model_path)  # 加载预训练模型
        self.frame_count = 0
        self.skip_frames = skip_frames  # 跳帧数量，每处理1帧将跳过2帧
        self.last_result = None  # 存储上一次的推理结果

    def detect_and_plot(self, frame, conf_thres=0.25, classes=None):
        """
        使用YOLO模型对输入图像进行检测，并在画面上绘制检测框。
        实现跳帧处理，仅在特定帧上执行检测
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        
        # 计数器增加
        self.frame_count += 1
        
        # 如果不是需要处理的帧，返回上一次的结果(如果有的话)或原始帧
        if self.frame_count % (self.skip_frames + 1) != 1:
            if self.last_result is not None:
                # 如果有上一次的结果，可以选择将上次检测的边界框应用到当前帧
                return self.last_result
            return frame
        
        # 确保 classes 是正确的格式（None 或整数列表）
        if classes is not None:
            if isinstance(classes, str):
                classes = None
            elif not all(isinstance(item, int) for item in classes):
                classes = None
        
        # 调整分辨率以提高检测速度
        original_size = frame.shape[:2]
        resized_frame = cv2.resize(frame, (640, 480))  # 调整为较低分辨率
        
        try:
            # 注意这里修改了调用方式，确保 classes 参数格式正确
            start_time = time.time()
            results = self.model(resized_frame, conf=conf_thres, classes=classes, verbose=False)
            inference_time = time.time() - start_time
            # 可选：打印推理时间
            # print(f"推理时间: {inference_time:.4f}秒")
        except Exception as e:
            return frame

        # 取第一个结果
        if len(results) > 0:
            r = results[0]
            # 如果想自行处理 boxes，可以用 r.boxes.xyxy, r.boxes.conf 等
            annotated = r.plot()  # ultralytics自带方法，可绘制检测框
            self.last_result = annotated  # 保存当前结果
            return annotated
        else:
            self.last_result = frame
            return frame