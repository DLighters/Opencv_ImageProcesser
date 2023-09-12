import time

import cv2
import imageio
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from textCapture import Ui_Dialog

class Dialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # 连接槽函数
        self.ui.pushButtonContinue.clicked.connect(self.on_pushButtonContinue_clicked)

    def on_pushButtonContinue_clicked(self):
        # 在这里编写按钮点击后的逻辑
        self.close()
        subtitle = self.ui.textEdit.toPlainText()
        font_size = 1.0
        scroll_speed = 30  # 滚动速度，单位为毫秒
        scroll_direction = "left"  # 滚动方向，"up"表示从下往上滚动，"down"表示从上往下滚动，"left"表示从右往左滚动，"right"表示从左往右滚动

        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        imgName, filtUsed = QFileDialog.getOpenFileName(self, title, curPath, filt)

        if imgName == '':
            return
        img = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)

        scrollText(img, subtitle)


def scroll_subtitle(image, subtitle, font_size, scroll_speed, scroll_direction):
    # 设置字体、字号、字体颜色等参数
    outputPath = 'scrolling_subtitle.gif'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size
    font_color = (255, 255, 255)  # 白色
    thickness = 2

    # 获取文字的宽度和高度
    (text_width, text_height), _ = cv2.getTextSize(subtitle, font, font_scale, thickness)

    # 计算滚动文字的起始位置
    if scroll_direction == "up":
        x = (image.shape[1] - text_width) // 2
        y = image.shape[0] + text_height
        y_end = -text_height
    elif scroll_direction == "down":
        x = (image.shape[1] - text_width) // 2
        y = -text_height
        y_end = image.shape[0] + text_height
    elif scroll_direction == "left":
        x = image.shape[1]
        y = (image.shape[0] + text_height) // 2
        x_end = -text_width
    elif scroll_direction == "right":
        x = -text_width
        y = (image.shape[0] + text_height) // 2
        x_end = image.shape[1]

    # 创建一个空白图像作为背景
    background = image.copy()
    frames = []
    while True:
        # 在背景图像上绘制滚动文字
        cv2.putText(background, subtitle, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 显示图像
        cv2.imshow('Scrolling Subtitle', background)
        frame = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        cv2.waitKey(scroll_speed)

        # 移动文字的位置
        if scroll_direction == "up":
            y -= 5
            if y <= y_end:
                break
        elif scroll_direction == "down":
            y += 5
            if y >= y_end:
                break
        elif scroll_direction == "left":
            x -= 5
            if x <= x_end:
                break
        elif scroll_direction == "right":
            x += 5
            if x >= x_end:
                break

        # 清空背景图像
        background = image.copy()

    imageio.mimsave(outputPath, frames, 'GIF', duration=0.02)
    cv2.destroyAllWindows()
    return outputPath


def scrollText(img, Text):
    Font = None
    Font_scale = None
    Font_color = None

    def ChangeFont(value):
        nonlocal Font
        if value == 1:
            Font = cv2.FONT_HERSHEY_SIMPLEX
        elif value == 2:
            Font = cv2.FONT_HERSHEY_PLAIN
        elif value == 3:
            Font = cv2.FONT_HERSHEY_DUPLEX
        elif value == 4:
            Font = cv2.FONT_HERSHEY_COMPLEX
        elif value == 5:
            Font = cv2.FONT_HERSHEY_TRIPLEX
        elif value == 6:
            Font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        elif value == 7:
            Font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        elif value == 8:
            Font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    def ChangeFontColor(value):
        nonlocal Font_color
        if value == 1:
            Font_color = (255, 255, 255)
        elif value == 2:
            Font_color = (255, 0, 0)
        elif value == 3:
            Font_color = (0, 255, 0)
        elif value == 4:
            Font_color = (0, 0, 255)

    def ChangeFontScale(value):
        nonlocal Font_scale
        Font_scale = value

    def scroll_subtitle(image, subtitle, scroll_speed, scroll_direction):
        # 设置字体、字号、字体颜色等参数
        nonlocal Font
        nonlocal Font_scale
        nonlocal Font_color
        outputPath = 'scrolling_subtitle.gif'
        cv2.namedWindow("Scrolling Subtitle")
        cv2.createTrackbar("Font", "Scrolling Subtitle", 1, 8, ChangeFont)
        cv2.createTrackbar("Font_scale", "Scrolling Subtitle", 1, 5, ChangeFontScale)
        cv2.createTrackbar("Font_color", "Scrolling Subtitle", 1, 4, ChangeFontColor)
        # font = Font # 字体
        # font_scale = Font_scale # 字体大小
        # font_color = Font_color  # 白色
        thickness = 2

        # 获取文字的宽度和高度
        (text_width, text_height), _ = cv2.getTextSize(subtitle, Font, Font_scale, thickness)

        # # 计算滚动文字的起始位置
        # if scroll_direction == "up":
        #     x = (image.shape[1] - text_width) // 2
        #     y = image.shape[0] + text_height
        #     y_end = -text_height
        # elif scroll_direction == "down":
        #     x = (image.shape[1] - text_width) // 2
        #     y = -text_height
        #     y_end = image.shape[0] + text_height
        # elif scroll_direction == "left":
        #     x = image.shape[1]
        #     y = (image.shape[0] + text_height) // 2
        #     x_end = -text_width
        # elif scroll_direction == "right":
        #     x = -text_width
        #     y = (image.shape[0] + text_height) // 2
        #     x_end = image.shape[1]
        #
        # # 创建一个空白图像作为背景
        # background = image.copy()
        # frames = []

        # time.sleep(10)
        while True:
            # 计算滚动文字的起始位置
            if scroll_direction == "up":
                x = (image.shape[1] - text_width) // 2
                y = image.shape[0] + text_height
                y_end = -text_height
            elif scroll_direction == "down":
                x = (image.shape[1] - text_width) // 2
                y = -text_height
                y_end = image.shape[0] + text_height
            elif scroll_direction == "left":
                x = image.shape[1]
                y = (image.shape[0] + text_height) // 2
                x_end = -text_width
            elif scroll_direction == "right":
                x = -text_width
                y = (image.shape[0] + text_height) // 2
                x_end = image.shape[1]

            # 创建一个空白图像作为背景
            background = image.copy()
            frames = []

            while True:
                # 在背景图像上绘制滚动文字
                cv2.putText(background, subtitle, (x, y), Font, Font_scale, Font_color, thickness, cv2.LINE_AA)
                # 显示图像
                cv2.imshow('Scrolling Subtitle', background)
                frame = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                cv2.waitKey(scroll_speed)
                # 移动文字的位置
                if scroll_direction == "up":
                    y -= 3
                    if y <= y_end:
                        break
                elif scroll_direction == "down":
                    y += 3
                    if y >= y_end:
                        break
                elif scroll_direction == "left":
                    x -= 3
                    if x <= x_end:
                        break
                elif scroll_direction == "right":
                    x += 3
                    if x >= x_end:
                        break
                # 清空背景图像
                background = image.copy()

            if cv2.waitKey(1000) == 13:
                break

        imageio.mimsave(outputPath, frames, 'GIF', duration=0.02)
        cv2.destroyAllWindows()
        return outputPath

    scroll_subtitle(img, Text, 1, "left")