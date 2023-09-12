import cv2
import numpy as np
import tkinter as tk


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


# 在图像中隐藏信息
def hideImg(img1, img2):
    img = cv2.resize(img2, (224, 224))

    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    print(binary.shape)
    print(ret)
    print(binary)
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)

    def insert_pic(event, x, y, flags, param):
        nonlocal ix, iy, drawing
        # 当按下左键是返回起始位置坐标

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            lena_copy = img1.copy()
            lena_copy[y - 112:y + 112, x - 112:x + 112] = binary
            cv2.imshow('image', lena_copy)
            cv2.waitKey(1)

        # 当鼠标左键按下并移动是绘制图形。 event 可以查看移动， flag 查看是否按下
        if event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # 绘制圆圈，小圆点连在一起就成了线， size 代表了笔画的粗细
                ix, iy = x, y
                lena_copy = img1.copy()
                lena_copy[y - 112:y + 112, x - 112:x + 112] = binary
                cv2.imshow('image', lena_copy)

            # 当鼠标松开停止绘画。

    drawing = False
    ix, iy = -1, -1
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("image")
    cv2.setMouseCallback('image', insert_pic)
    cv2.imshow('image', img1)

    if cv2.waitKey(0) & 0xFF == 13:
        drawing = False
        # 读取原始载体图像的 shape 值
        r, c = img1.shape
        print(r, c)
        watermark = np.zeros((r, c), dtype=np.uint8)
        watermark[iy - 112:iy + 112, ix - 112:ix + 112] = binary
        w = watermark[:, :] > 0
        watermark[w] = 1

        # =========嵌入过程========
        # 生成元素值都是 254 的数组
        t254 = np.ones((r, c), dtype=np.uint8) * 254
        # 获取 img1 图像的高七位
        lenaH7 = cv2.bitwise_and(img1, t254)
        # 将 watermark 嵌入 lenaH7 内
        e = cv2.bitwise_or(lenaH7, watermark)
        cv2.imshow("image", e)
        e = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
        return e
        if cv2.waitKey(0) & 0xFF == 13:
            # ======提取过程=========
            # 生成元素值都是 1 的数组
            t1 = np.ones((r, c), dtype=np.uint8)
            # 从载体图像内提取水印图像
            wm = cv2.bitwise_and(e, t1)
            # 将水印图像内的值 1 处理为 255， 以方便显示
            w = wm[:, :] > 0
            wm[w] = 255

            cv2.imshow("extract", wm)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def colorDivide(img):
    # 创建一个窗口，放置6个滑动条
    cv2.namedWindow("ColorDivision")
    # cv2.resizeWindow("ColorDivision", 640, 240)
    cv2.createTrackbar("Hue Min", "ColorDivision", 0, 179, lambda x: x)
    cv2.createTrackbar("Hue Max", "ColorDivision", 19, 179, lambda x: x)
    cv2.createTrackbar("Sat Min", "ColorDivision", 110, 255, lambda x: x)
    cv2.createTrackbar("Sat Max", "ColorDivision", 240, 255, lambda x: x)
    cv2.createTrackbar("Val Min", "ColorDivision", 153, 255, lambda x: x)
    cv2.createTrackbar("Val Max", "ColorDivision", 255, 255, lambda x: x)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    while True:
        # 调用回调函数，获取滑动条的值
        h_min = cv2.getTrackbarPos("Hue Min", "ColorDivision")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorDivision")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorDivision")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorDivision")
        v_min = cv2.getTrackbarPos("Val Min", "ColorDivision")
        v_max = cv2.getTrackbarPos("Val Max", "ColorDivision")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        # 获得指定颜色范围内的掩码
        mask = cv2.inRange(imgHSV, lower, upper)
        # 对原图图像进行按位与的操作，掩码区域保留
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("ColorDivision", imgResult)

        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.destroyAllWindows()
    return imgResult


# 图像阈值化
def threshold(img):
    mode = None
    thresh = None
    type = 0

    def changeMode(value):
        nonlocal mode
        nonlocal type
        type = value
        if type == 1:
            mode = cv2.THRESH_BINARY
        elif type == 2:
            mode = cv2.THRESH_BINARY_INV
        elif type == 3:
            mode = cv2.THRESH_TRUNC
        elif type == 4:
            mode = cv2.THRESH_TOZERO_INV
        elif type == 5:
            mode = cv2.THRESH_TOZERO
        elif type == 6:
            mode = cv2.THRESH_MASK

    def changeThresh(value):
        nonlocal thresh
        thresh = value

    cv2.namedWindow("Threshold")
    cv2.createTrackbar("Mode", "Threshold", 1, 6, changeMode)
    cv2.createTrackbar("Thresh", "Threshold", 100, 300, changeThresh)

    while True:
        retval, dst = cv2.threshold(img, thresh, 255, mode)
        cv2.imshow("Threshold", dst)
        if cv2.waitKey(1) == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst


def changeBackground(background_image, original_image):
    # 调整新背景图像的大小为原图像的大小
    background_image_resized = cv2.resize(background_image, (original_image.shape[1], original_image.shape[0]))

    # 使用颜色分离方法选择背景区域
    lower_color = np.array([100, 0, 0], dtype=np.uint8)
    upper_color = np.array([250, 80, 80], dtype=np.uint8)
    mask = cv2.inRange(original_image, lower_color, upper_color)

    # 将原图中的背景像素值设为0
    original_image_no_background = original_image.copy()
    original_image_no_background[np.where(mask == 255)] = 0

    # 对mask取反
    mask_inverse = cv2.bitwise_not(mask)

    # 将背景图像中对mask取反后对应的像素值设为0
    background_image_no_mask = background_image_resized.copy()
    background_image_no_mask[np.where(mask_inverse == 255)] = 0

    # 处理后的前景和背景图像相加，得到换背景后的图像
    result_image = cv2.add(original_image_no_background, background_image_no_mask)

    # 显示结果图像
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image


def makeIDcard(image):
    def update_background_color(*args):
        nonlocal background_color, result

        # 获取滚动条的值
        red = red_scale.get()
        green = green_scale.get()
        blue = blue_scale.get()

        # 更新背景颜色
        background_color = (blue, green, red)

        # 创建新的背景图像
        background_image = np.full_like(image, background_color)
        result = np.where(mask2[:, :, np.newaxis] == 1, image, background_image)
        cv2.imshow("Segmented Image", result)

    # 加载图像
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = cv2.selectROI(image, showCrosshair=True, fromCenter=False)
    x, y, w, h = rect
    if w == 0 and h == 0:
        return
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 创建背景图像
    background_color = (0, 255, 0)  # 初始背景颜色为绿色
    background_image = np.full_like(image, background_color, dtype=np.uint8)
    result = np.where(mask2[:, :, np.newaxis] == 1, image, background_image)

    # 创建滚动条控制界面
    window = tk.Tk()
    window.title("Background Color Control")

    # 创建红色滚动条
    red_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Red", command=update_background_color)
    red_scale.set(background_color[2])
    red_scale.pack()

    # 创建绿色滚动条
    green_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Green",
                           command=update_background_color)
    green_scale.set(background_color[1])
    green_scale.pack()

    # 创建蓝色滚动条
    blue_scale = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Blue", command=update_background_color)
    blue_scale.set(background_color[0])
    blue_scale.pack()

    # 显示结果
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 启动滚动条控制界面的事件循环
    window.mainloop()
    return result


img_path = "input_image.jpg"
img_test = cv2.imread(img_path)
img_bg = cv2.imread("background_image.jpg")
# mergeImg(img_bg, img_test)
