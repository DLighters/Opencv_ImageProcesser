import cv2
import numpy as np
import math

def mergeImg(img, img_2):
    window_name = 'Object Insertion'
    scale_min = 0.1
    scale_max = 2.0

    object_alpha = img_2.astype(float) / 255.0

    # 初始化物体的位置和缩放
    object_pos = (0, 0)
    object_scale = 1.0

    # 图片是否显示并跟随鼠标
    flag = False

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal object_pos, flag

        if event == cv2.EVENT_LBUTTONDOWN:
            flag = not flag
            object_pos = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            object_pos = (x, y)

    # 滚动条回调函数
    def scale_callback(value):
        nonlocal object_scale

        object_scale = scale_min + (value / 100) * (scale_max - scale_min)

    # 创建窗口并注册回调函数
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 创建滚动条
    cv2.createTrackbar('Scale', window_name, 0, 100, scale_callback)

    while True:
        # 复制背景图像
        img_copy = img.copy()

        # 计算物体的缩放尺寸
        scaled_width = int(img_2.shape[1] * object_scale)
        scaled_height = int(img_2.shape[0] * object_scale)
        scaled_object_img = cv2.resize(img_2, (scaled_width, scaled_height))

        if flag:
            # 调整大小后的图片长宽数值
            object_height, object_width, _ = scaled_object_img.shape
            # 当图片运动到边缘时超出边界部分应消失，而不能出现报错，因此叠加范围需限制在背景图片内
            y_min = min(object_pos[1] + object_height, img.shape[0])
            x_min = min(object_pos[0] + object_width, img.shape[1])
            img_copy[object_pos[1]:y_min, object_pos[0]:x_min] = scaled_object_img[0:y_min - object_pos[1],
                                                            0:x_min - object_pos[0]]

        # 显示图像
        cv2.imshow(window_name, img_copy)

        # 监听键盘事件
        key = cv2.waitKey(1) & 0xFF

        # 按下 'Enter' 键确认并保存图像
        if key == 13:
            # cv2.imwrite('composite_image.jpg', img_copy)
            return img_copy
            break

        # 按下 'Esc' 键退出
        if key == 27:
            return img
            break

    cv2.destroyAllWindows()


img_path = "input_image.jpg"
img_test = cv2.imread(img_path)
img_bg = cv2.imread("background_image.jpg")
# mergeImg(img_bg, img_test)