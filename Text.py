import cv2
import numpy as np

def insertText(img, textline):
    # 定义全局变量
    window_name = 'Insert Text'
    color_rect_pos = (10, 10, 200, 50)  # 颜色矩形位置和大小
    text_color = (0, 0, 255)  # 文字颜色，默认红色
    text_size = 1  # 文字大小，默认为1
    text_thickness = 1  # 文字粗细，默认为1

    # 鼠标回调函数
    def insert_text(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(img, textline, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness,
                        cv2.LINE_AA)

    # 创建窗口
    cv2.namedWindow(window_name)

    # 创建滑动条
    cv2.createTrackbar('R', window_name, text_color[0], 255, lambda x: None)
    cv2.createTrackbar('G', window_name, text_color[1], 255, lambda x: None)
    cv2.createTrackbar('B', window_name, text_color[2], 255, lambda x: None)
    cv2.createTrackbar('Size', window_name, int(text_size * 10), 50, lambda x: None)
    cv2.createTrackbar('Thickness', window_name, text_thickness, 5, lambda x: None)

    # 注册鼠标回调函数
    cv2.setMouseCallback(window_name, insert_text)

    while True:
        canvas = np.zeros_like(img)

        # 在画布上绘制颜色矩形
        cv2.rectangle(canvas, color_rect_pos[:2],
                      (color_rect_pos[0] + color_rect_pos[2], color_rect_pos[1] + color_rect_pos[3]), text_color, -1)

        # 将图像和画布水平拼接
        display_img = np.hstack((img, canvas))

        cv2.imshow(window_name, display_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # 按下 'Esc' 键退出
            break

        # 更新文字颜色
        text_color = (cv2.getTrackbarPos('R', window_name),
                      cv2.getTrackbarPos('G', window_name),
                      cv2.getTrackbarPos('B', window_name))

        # 更新文字大小
        text_size = cv2.getTrackbarPos('Size', window_name) / 10.0

        # 更新文字粗细
        text_thickness = cv2.getTrackbarPos('Thickness', window_name)

    cv2.destroyAllWindows()
    return img


def deleteText(img):
    # 创建掩膜图像，将文字区域标记为白色
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.namedWindow('Original')
    roi = cv2.selectROI('Original', img, False, False)
    x, y, w, h = roi
    if w == 0 and h == 0:
        return img
    mask[y:y + h, x:x + w] = 255

    # 使用inpaint函数进行图像修复
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # 显示原始图像和修复后的图像
    cv2.imshow('Original', img)
    cv2.imshow('Inpainted', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


image_path = "input_image_text.jpg"
img = cv2.imread(image_path)
# deleteText(img)