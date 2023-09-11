import numpy as np
import cv2
import imageio

def scroll_subtitle(image, subtitle, font_size, scroll_speed, scroll_direction):
    # 设置字体、字号、字体颜色等参数
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
    frame = []
    while True:
        # 在背景图像上绘制滚动文字
        cv2.putText(background, subtitle, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 显示图像
        cv2.imshow('Scrolling Subtitle', background)
        frame.append(background.copy())
        cv2.waitKey(scroll_speed)

        # 移动文字的位置
        if scroll_direction == "up":
            y -= 1
            if y <= y_end:
                break
        elif scroll_direction == "down":
            y += 1
            if y >= y_end:
                break
        elif scroll_direction == "left":
            x -= 1
            if x <= x_end:
                break
        elif scroll_direction == "right":
            x += 1
            if x >= x_end:
                break

        # 清空背景图像
        background = image.copy()
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()