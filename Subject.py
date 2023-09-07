import cv2
import numpy as np


def selectForeground(img):
    # 定义全局变量
    window_name = 'Select Foreground'
    foreground_color = (0, 255, 0)  # 前景目标颜色 (B, G, R)
    transparent_color = (0, 0, 0, 0)  # 透明颜色 (B, G, R, A)
    drawing = False  # 是否正在绘制

    # 读取输入图像
    img_copy = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 前景目标的遮罩图像

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, mask, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), 10, (255), -1)
                cv2.circle(img_copy, (x, y), 10, foreground_color, -1)

    # 创建窗口并注册鼠标回调函数
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # transparent_img = img
        cv2.imshow(window_name, img_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 按下 'Enter' 键保存透明图像并退出
            alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha
            cv2.imwrite('OutputImg/transparent_image.png', img)

            img[np.where(img[:, :, 3] == 0)] = [0, 0, 0, 0]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite('ToolImg/transparent_image.jpg', img)

            cv2.destroyAllWindows()
            return True
            break

        elif key == 27:  # 按下 'Esc' 键退出
            break

    # 销毁窗口
    cv2.destroyAllWindows()
    return False


def rectSelect(img):
    return False


img_path = "input_image.jpg"
img_test = cv2.imread(img_path)
# selectForeground(img_test)