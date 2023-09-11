import cv2
import numpy as np
import math


def affineTrans(img):
    window_name = 'Polygon Blurring'
    point_color = (0, 0, 255)  # 多边形线条颜色 (B, G, R)
    polygon_points = []  # 四边形顶点
    flag = 0

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal polygon_points, flag

        if event == cv2.EVENT_LBUTTONDOWN and flag < 3:
            polygon_points.append((x, y))
            # cv2.circle(img_copy, (x, y), 3, point_color, -1)
            flag += 1

    # 读取输入图像
    img_copy = img.copy()

    # 创建窗口并注册鼠标回调函数
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, img)

        height, width = img.shape[:2]
        src = np.float32([[0, 0], [0, width], [height, 0], ])
        dst = np.float32(polygon_points)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 按下 'Enter' 键确认多边形顶点
            if len(polygon_points) == 3:
                flag = False
                retval = cv2.getAffineTransform(src, dst)
                img = cv2.warpAffine(img, retval, (height, width))
                # Point 1: 生成与白色部分对应的mask图像
                mask = np.all(img[:, :, :] == [0, 0, 0], axis=-1)

                # Point 2: 将图片从三通道转为四通道
                trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                # Point3:  以mask图像为基础，使白色部分透明化
                trans_img[mask, 3] = 0
                cv2.imwrite('OutputImg/affine_trans.png', trans_img)
        elif key == 27:  # 按下 'Esc' 键退出
            return img
            break

    # 销毁窗口
    cv2.destroyAllWindows()
    return img


def perspectiveTrans(img):
    r, c = img.shape[:2]
    point = []
    num = 0

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        nonlocal num, point
        if event == cv2.EVENT_LBUTTONDOWN and num < 4:
            num += 1
            xy = "%d,%d" % (x, y)
            point.append([x, y])
            # cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
            print('pointers are:')
            print(x, y)
        else:
            pass

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 13:
        p1 = np.float32([[0, 0], [c, 0], [c, r], [0, r]])
        p2 = np.float32(point)
        M = cv2.getPerspectiveTransform(p1, p2)
        dst = cv2.warpPerspective(img, M, (c, r))

        b = np.array(point)
        print(b)
        mask = np.zeros((r, c), dtype=np.uint8)
        cv2.polylines(mask, [b], 1, 255)  # 描绘边缘
        cv2.fillPoly(mask, [b], 255)  # 填充

        bgra = cv2.cvtColor(dst, cv2.COLOR_BGR2BGRA)
        b, g, r, a = cv2.split(bgra)
        a[:, :] = np.where(mask > 0, 255, 0).astype(np.uint8)
        bgra = cv2.merge([b, g, r, a])

        cv2.imshow("perspective transform", bgra)
        cv2.imwrite("OutputImg/perspective_trans.png", bgra)
        cv2.waitKey(0)
        return dst

    cv2.destroyAllWindows()
    return img


def convexLens(img):
    def convex_lens_filter(img, center_x, center_y, scale):
        h, w, c = img.shape
        radius = min(center_x, center_y)

        # 生成行和列索引数组
        i, j = np.meshgrid(np.arange(h), np.arange(w))

        # 计算各点到中心的距离
        dx = i - center_x
        dy = j - center_y
        distance = dx ** 2 + dy ** 2

        # 计算映射后的坐标
        dist = np.sqrt(distance)
        new_x = np.floor(center_x + (dx * (dist <= radius ** 2)) * (dist / (radius * scale))).astype(int)
        new_y = np.floor(center_y + (dy * (dist <= radius ** 2)) * (dist / (radius * scale))).astype(int)
        new_x = np.clip(new_x, 0, h - 1)
        new_y = np.clip(new_y, 0, w - 1)

        # 生成结果图像
        out = np.zeros_like(img)
        row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w))
        out[row_idx, col_idx] = img[new_x, new_y]

        return out

    # 读取图片
    img = cv2.imread('input_image.jpg')

    # 设置滑块变量
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    scale = 10

    # 创建窗口和滑块
    cv2.namedWindow('image')
    cv2.createTrackbar('center_x', 'image', center_x, img.shape[1], lambda x: x)
    cv2.createTrackbar('center_y', 'image', center_y, img.shape[0], lambda x: x)
    cv2.createTrackbar('scale', 'image', scale, 50, lambda x: x)

    while True:

        # 获取滑块位置
        center_x = cv2.getTrackbarPos('center_x', 'image')
        center_y = cv2.getTrackbarPos('center_y', 'image')
        scale = round(cv2.getTrackbarPos('scale', 'image') / 10, 1)

        # 实现滤镜效果
        result = convex_lens_filter(img, center_x, center_y, scale)

        # 对比显示
        cv2.imshow('image', np.hstack([img, result]))
        if cv2.waitKey(1) == 13:
            cv2.destroyAllWindows()
            return result
            break

    cv2.destroyAllWindows()
    return img

# 在图像中隐藏信息
def hide_binary_image(img1, img2):
    # 读入RGB图像图像
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
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
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
    return e

def color_divide(img):
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
    return img


#图像阈值化
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


def outline(img):

    # 描绘轮廓
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    img2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) + 255
    img2 = cv2.drawContours(img2, contours, -1, (0, 0, 0), 2)
    cv2.imshow('Contours', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img2

def concaveLens(img):
    result = np.zeros_like(img)
    def on_trackbar_change(val):
        nonlocal result
        # 获取滚动条当前值


    # 挤压滤镜函数
    def squeeze_filter(image, center_x, center_y, squeeze_factor):
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # 计算相对中心坐标
        x_rel = x - center_x
        y_rel = y - center_y

        # 计算极坐标
        theta = np.arctan2(y_rel, x_rel)
        radius = np.sqrt(x_rel ** 2 + y_rel ** 2)

        # 计算挤压程度
        squeeze_radius = radius / (1 + squeeze_factor * radius)

        # 转换回直角坐标
        result_x = center_x + squeeze_radius * np.cos(theta)
        result_y = center_y + squeeze_radius * np.sin(theta)

        # 进行向量运算
        result = cv2.remap(image, result_x, result_y, cv2.INTER_LINEAR)

        return result

    # 创建窗口
    cv2.namedWindow("Squeeze Filter")

    # 创建滚动条
    cv2.createTrackbar("Center X", "Squeeze Filter", img.shape[1] // 2, img.shape[1], lambda x: x)
    cv2.createTrackbar("Center Y", "Squeeze Filter", img.shape[0] // 2, img.shape[0], lambda x: x)
    cv2.createTrackbar("Squeeze Factor", "Squeeze Filter", 20, 100, lambda x: x)

    # 初始化滚动条回调函数
    while True:
        center_x = cv2.getTrackbarPos("Center X", "Squeeze Filter")
        center_y = cv2.getTrackbarPos("Center Y", "Squeeze Filter")
        squeeze_factor = cv2.getTrackbarPos("Squeeze Factor", "Squeeze Filter") / 10000

        # 应用挤压滤镜效果
        result = squeeze_filter(img, center_x, center_y, squeeze_factor)

        # 显示原图和处理后的图像
        cv2.imshow('Squeeze Filter', np.hstack([img, result]))
        if cv2.waitKey(1) == 27:
            return result
            break

    # 清理资源
    cv2.destroyAllWindows()
    return result


def sinTrans(img):
    window_name = 'sin'

    alpha = 50
    beta = 50
    degree_x = 20
    degree_y = 20

    def degree_x_callback(value):
        nonlocal degree_x
        degree_x = value

    def degree_y_callback(value):
        nonlocal degree_y
        degree_y = value

    def alpha_callback(value):
        nonlocal alpha
        alpha = value

    def beta_callback(value):
        nonlocal beta
        beta = value

    cv2.namedWindow(window_name)
    cv2.createTrackbar('degree_x', window_name, 0, 100, degree_x_callback)
    cv2.createTrackbar('degree_y', window_name, 0, 100, degree_y_callback)
    cv2.createTrackbar('alpha', window_name, 1, 200, alpha_callback)
    cv2.createTrackbar('beta', window_name, 1, 200, beta_callback)

    def sin_trans(img, alpha, beta, degree_x, degree_y):
        row, col, channel = img.shape
        center_x = (col - 1) / 2.0
        center_y = (row - 1) / 2.0
        y_mask, x_mask = np.indices((row, col))
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        x = degree_x * np.sin(2 * math.pi * yy_dif / alpha) + xx_dif
        y = degree_y * np.cos(2 * math.pi * xx_dif / beta) + yy_dif
        x_new = x + center_x
        y_new = center_y - y
        x_new = x_new.astype(np.float32)
        y_new = y_new.astype(np.float32)
        dst = cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR)
        return dst

    while True:
        result = sin_trans(img, alpha, beta, degree_x, degree_y)
        cv2.imshow(window_name, np.hstack([result, img]))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return result
            break

    cv2.destroyAllWindows()
    return result


def vortexFilter(img):
    window_name = 'sin'

    row, col, channel = img.shape
    center_x = (col - 1) / 2.0
    center_y = (row - 1) / 2.0
    degree = 70

    def center_x_callback(value):
        nonlocal center_x
        value = (-50 + value) / 100
        center_x = (1 + value) * (col - 1) / 2.0

    def center_y_callback(value):
        nonlocal center_y
        value = (-50 + value) / 100
        center_y = (1 + value) * (col - 1) / 2.0

    def degree_callback(value):
        nonlocal degree
        degree = value

    cv2.namedWindow(window_name)
    cv2.createTrackbar('center_x', window_name, 50, 100, center_x_callback)
    cv2.createTrackbar('center_y', window_name, 50, 100, center_y_callback)
    cv2.createTrackbar('degree', window_name, 90, 180, degree_callback)

    def vortex(img):
        nonlocal center_y, center_x, degree
        img_out = img * 1.0
        y_mask, x_mask = np.indices((row, col))
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        theta = np.arctan(yy_dif / xx_dif)
        mask_1 = xx_dif < 0
        theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
        theta = theta + r / degree
        x_new = r * np.cos(theta) + center_x
        y_new = center_y - r * np.sin(theta)
        x_new = x_new.astype(np.float32)
        y_new = y_new.astype(np.float32)
        dst = cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR)
        return dst

    while True:
        result = vortex(img)
        cv2.imshow(window_name, np.hstack([result,img]))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return result
            break

    cv2.destroyAllWindows()
    return result


def styleConversion(image):
    type = None
    style = None

    def ChangeStyle(value):
        nonlocal type
        nonlocal style
        type = value
        if type == 1:
            style = cv2.COLORMAP_AUTUMN
        elif type == 2:
            style = cv2.COLORMAP_BONE
        elif type == 3:
            style = cv2.COLORMAP_CIVIDIS
        elif type == 4:
            style = cv2.COLORMAP_COOL
        elif type == 5:
            style = cv2.COLORMAP_DEEPGREEN
        elif type == 6:
            style = cv2.COLORMAP_HOT
        elif type == 7:
            style = cv2.COLORMAP_HSV
        elif type == 8:
            style = cv2.COLORMAP_INFERNO
        elif type == 9:
            style = cv2.COLORMAP_JET
        elif type == 10:
            style = cv2.COLORMAP_MAGMA
        elif type == 11:
            style = cv2.COLORMAP_OCEAN
        elif type == 12:
            style = cv2.COLORMAP_PARULA
        elif type == 13:
            style = cv2.COLORMAP_PINK
        elif type == 14:
            style = cv2.COLORMAP_PLASMA
        elif type == 15:
            style = cv2.COLORMAP_RAINBOW
        elif type == 16:
            style = cv2.COLORMAP_SPRING
        elif type == 17:
            style = cv2.COLORMAP_SUMMER
        elif type == 18:
            style = cv2.COLORMAP_TURBO
        elif type == 19:
            style = cv2.COLORMAP_TWILIGHT
        elif type == 20:
            style = cv2.COLORMAP_TWILIGHT_SHIFTED
        elif type == 21:
            style = cv2.COLORMAP_VIRIDIS
        elif type == 22:
            style = cv2.COLORMAP_WINTER

    def LUTColorStyleChange(image):
        cv2_luts = [lut for lut in dir(cv2) if lut.startswith("COLORMAP_")]
        all_lut_imgs = [(lut, cv2.applyColorMap(image, eval("cv2." + lut))) for lut in cv2_luts]
        add_text_imgs = [cv2.putText(lut_img[1], lut_img[0], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                         for
                         lut_img in all_lut_imgs]
        col1 = np.vstack(tuple(add_text_imgs[0:11]))
        col2 = np.vstack(tuple(add_text_imgs[11:22]))
        result = np.hstack((col1, col2))

        res = cv2.applyColorMap(image, style)
        return res

    cv2.namedWindow("ColorStyleChange")
    cv2.createTrackbar("Style", "ColorStyleChange", 1, 22, ChangeStyle)
    while True:
        res = LUTColorStyleChange(image)
        cv2.imshow("ColorStyleChange", res)
        if cv2.waitKey(1) == 13:
            return res
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


img_path = "input_image.jpg"
img_test = cv2.imread(img_path)
threshold(img_test)

