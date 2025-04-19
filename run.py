#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#运行入口脚本，主要完成应用程序启动逻辑。
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()