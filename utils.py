#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#存放一些通用工具函数，如检测可用摄像头、释放资源、常量列表等。

import cv2
import requests

SUPPORTED_RESOLUTIONS = {
    "480P": (640, 480),
    "720P": (1280, 720),
    "1080P": (1920, 1080)
}

SUPPORTED_FPS = [15, 30, 60]

def detect_cameras(max_test=5, mobile_camera_urls=None):
    """
    检测可用的摄像头，返回 {index: name} 的dict
    Args:
        max_test: 本机摄像头最大检测数量。
        mobile_camera_urls: 可选，手机摄像头的URL列表（如通过IP Webcam提供的地址）。
    """
    camera_dict = {}

    # 检测本机摄像头
    for i in range(max_test):
        cap = None
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_dict[i] = f"Local Camera {i}"
        except:
            pass
        finally:
            if cap:
                cap.release()

    # 检测手机摄像头
    if mobile_camera_urls:
        for idx, url in enumerate(mobile_camera_urls, start=max_test):
            try:
                # 测试手机摄像头流是否可用
                response = requests.get(url, stream=True, timeout=2)
                if response.status_code == 200:
                    camera_dict[idx] = f"Mobile Camera {url}"
            except:
                pass

    return camera_dict

def safe_release(cap):
    """
    安全释放VideoCapture或VideoWriter
    """
    try:
        if cap is not None:
            cap.release()
    except:
        pass