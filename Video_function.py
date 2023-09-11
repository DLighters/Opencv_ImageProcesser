# OpenCVPyqt10.py
# Demo05 of GUI by PyQt5
# Crated：2023-02-20

import sys
import cv2 as cv
import imageio.v2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import Make
import TextCaptureClass
import textCapture
from uiDemo10 import Ui_MainWindow  # 导入 uiDemo10.py 中的 Ui_MainWindow 界面类


class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow 类和 Ui_MainWindow 界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.videoPath = None
        self.frame = None
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.timerCam = QtCore.QTimer()  # 定时器，毫秒
        self.cap = None  #
        self.frameNum = 1  # 视频帧数初值

        self.textCapture = TextCaptureClass.Dialog()

        # 菜单栏
        self.actionOpen.triggered.connect(self.openVideo)  # 连接并执行 openSlot 子程序
        # self.actionHelp.triggered.connect(self.trigger_actHelp)  # 连接并执行 trigger_actHelp 子程序
        self.actionQuit.triggered.connect(self.close)  # 连接并执行 trigger_actHelp 子程序
        self.actionSave.triggered.connect(self.saveVideo)

        # 通过 connect 建立信号/槽连接，点击按钮事件发射 triggered 信号，执行相应的子程序 click_pushButton
        # self.pushButton_1.clicked.connect(self.openVideo)  # 打开视频文件
        self.pushButtonPlay.clicked.connect(self.playVideo)  # 播放视频文件
        self.pushButtonPause.clicked.connect(self.pauseVideo)  # 停止视频播放
        # self.pushButton_4.clicked.connect(self.trigger_actHelp)  # 按钮触发
        # self.pushButton_5.clicked.connect(self.close)  # 点击 # 按钮触发：关闭
        self.timerCam.timeout.connect(self.refreshFrame)  # 计时器结束时调用槽函数刷新当前帧

        # 初始化
        return

    def openVideo(self):  # 读取视频文件，点击 pushButton_1 触发
        self.videoPath, _ = QFileDialog.getOpenFileName(self, "Open Video", "../images/", "*.gif *.mp4 *.avi *.flv")
        print("Open Video: ", self.videoPath)
        if self.videoPath == "":
            return
        if not self.timerCam.isActive():
            self.cap = cv.VideoCapture(self.videoPath)
            self.refreshFrame()
        # self.refreshFrame()
        return

    def openVideo_path(self, path):  # 读取视频文件，点击 pushButton_1 触发
        if path == "":
            return
        self.cap = cv.VideoCapture(path)
        self.refreshFrame()
        return

    def saveVideo(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        curPath = QDir.currentPath()
        title = "保存图片"
        filt = "所有文件(*.*);;图片文件(*.gif *.mp4 *.avi *.flv)"
        savename, filtUsed = QFileDialog.getSaveFileName(self, title, curPath, filt)

        frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # 将BGR图像转换为RGB图像
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        imageio.v2.mimsave(savename, frames, format='GIF', duration=100)

    def playVideo(self):  # 播放视频文件，点击 pushButtonPlay 触发
        # 实例化 VideoCapture 类
        if self.cap.isOpened():  # 检查视频捕获是否成功
            self.timerCam.start(20)  # 设置计时间隔并启动，定时结束将触发刷新当前帧
        else:  #
            self.timerCam.stop()  # 停止定时器
            self.cap.release()  # 关闭读取视频文件
            self.label.clear()  # 清除显示内容

    def pauseVideo(self):
        self.timerCam.blockSignals(False)  # 取消信号阻塞，恢复定时器
        if self.timerCam.isActive() and self.frameNum % 2 == 1:
            self.timerCam.blockSignals(True)  # 信号阻塞，暂停定时器
            self.pushButtonPause.setText("继续")  # 点击"继续"，恢复播放
            print("信号阻塞，暂停播放。", self.frameNum)
        else:
            self.pushButtonPause.setText("暂停")  # 点击"暂停"，暂停播放
            print("取消阻塞，恢复播放。", self.frameNum)
        self.frameNum = self.frameNum + 1

    def closeEvent(self, event):
        if self.timerCam.isActive():
            self.timerCam.stop()

    def refreshFrame(self):  # 刷新视频图像
        ret, self.frame = self.cap.read()  # 读取下一帧视频图像
        if not ret:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, self.frame = self.cap.read()

        qImg = self.cvToQImage(self.frame)  # OpenCV 转为 PyQt 图像格式
        self.label.setPixmap((QPixmap.fromImage(qImg)))  # 加载 PyQt 图像
        return

    def cvToQImage(self, image):
        # 8-bits unsigned, NO. OF CHANNELS=1
        if image is None:
            return
        if image.dtype == np.uint8:
            channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 3:  # CV_8UC3
            # Create QImage with same dimensions as input Mat
            qImg = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            return qImg.rgbSwapped()
        elif channels == 1:
            # Create QImage with same dimensions as input Mat
            qImg = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Indexed8)
            return qImg
        else:
            QtCore.qDebug("ERROR: numpy.ndarray could not be converted to QImage. Channels = %d" % image.shape[2])
            return QImage()

    def on_actionTrans_triggered(self):
        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        img1Name, filtUsed1 = QFileDialog.getOpenFileName(self, title, curPath, filt)

        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        img2Name, filtUsed2 = QFileDialog.getOpenFileName(self, title, curPath, filt)

        if img1Name == '' or img2Name == '':
            return
        img_1 = cv.imdecode(np.fromfile(img1Name, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
        img_2 = cv.imdecode(np.fromfile(img2Name, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)

        path = Make.ImageTrans(img_1, img_2)
        self.openVideo_path(path)

    def on_actionRowCaption_triggered(self):
        self.textCapture.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
