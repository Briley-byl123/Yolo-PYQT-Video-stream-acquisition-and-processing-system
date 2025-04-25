![image](https://github.com/user-attachments/assets/7b3fff3a-2c6c-4c52-ae50-c768291fc604)


以下为项目的示例性源码说明和使用方法，项目文件结构如下：

```
PyQT-Yolo/
├── run.py                  # 程序入口
├── main_window.py          # 主窗口界面及相关逻辑
├── capture_thread.py       # 摄像头采集线程
├── video_player.py         # 本地视频回放类
├── detection.py            # YOLO检测封装
├── utils.py                # 工具函数
└── requirements.txt        # Python依赖包
└── models/                 #存放YOLO预训练模型
└── videos/                 #存放录制视频
```

#### 环境依赖

- Python 3.8+
- PyQt5 (图形界面)
- OpenCV (视频操作)
- ultralytics (YOLO 模型；或手动安装 yolov5, yolov8)
- FFmpeg (若需在系统命令行中进行视频格式转换)

可在命令行中执行以下命令安装主要依赖：

```bash
pip install pyqt5 opencv-python ultralytics
```

或使用本项目中的 `requirements.txt` 文件：

```
pip install -r requirements.txt
```

#### 运行方法

1. **下载或克隆本项目代码**，确保目录结构正确；

2. 在项目根目录（包含 run.py 的目录）打开命令行 / Terminal；

3. 执行：

   ```bash
   python run.py
   ```

4. 如果一切正常，PyQt5 窗口被启动，页面上展示摄像头选择、分辨率 / FPS、存储控制等功能。

#### 操作步骤

1. 启动摄像头
   - 在主界面“视频”选项卡，选择所需的摄像头（若有多个），然后点击“启动摄像头”；
   - 若成功，会在界面中央出现实时预览画面。
2. 开启YOLO检测（可选）
   - 在界面点击“开启检测”，如果模型加载成功，会对每帧进行检测并在画面上绘制检测框。
3. 开始录制 / 定时存储
   - 设置分辨率、帧率、定时存储间隔，点击“开始存储”，系统会默认输出 .mp4 (或 .avi) 文件。
   - 当已录制到达设定的时间间隔，会自动切换到新的文件。
4. 停止录制 / 摄像头
   - 可随时点击“停止存储”或“停止摄像头”。
5. 视频回放
   - 可点击“打开本地视频”按钮，在文件对话框中选择任意的视频文件（.mp4/.avi等）进行加载；
   - 之后点击“播放”启动回放，可使用“暂停”/“停止”以及进度条随意跳转。
   - 可以在回放时开启检测，对离线视频进行标注。
6. 设置
   - 在“设置”选项卡里可填写目标类别、置信度阈值，并查看日志信息（如错误提醒、状态提示等）。
