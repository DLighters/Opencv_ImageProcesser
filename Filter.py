import cv2
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk

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



def regionBlur(img):
    window_name = 'Polygon Blurring'
    polygon_color = (0, 0, 255)  # 多边形线条颜色 (B, G, R)
    blur_kernel_size = (25, 25)  # 模糊核大小

    polygon_points = []  # 多边形顶点

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal polygon_points

        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            cv2.circle(img, (x, y), 3, polygon_color, -1)

    # 创建窗口并注册鼠标回调函数
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # 按下 'Enter' 键确认多边形顶点
            if len(polygon_points) >= 3:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                # 多边形顶点数组
                points = np.array(polygon_points, np.int32)
                # 填充多边形
                cv2.fillPoly(mask, [points], 255)

                # 对多边形区域进行模糊处理
                blurred_img = cv2.blur(img, blur_kernel_size)

                # 将多边形区域应用到原始图像上
                img = np.where(mask[:, :, np.newaxis] == 255, blurred_img, img)

            polygon_points = []

        elif key == 27:  # 按下 'Esc' 键退出
            break

    # 销毁窗口
    cv2.destroyAllWindows()
    return img


def sketchFilter(img):
    def generate_sketch(x):
        # 获取滚动条的当前值
        sketch_type = cv2.getTrackbarPos('Sketch Type', 'Sketch Generation')

        # 根据滚动条的值生成对应的草图
        sketch = generate_sketch_image(sketch_type)

        # 将原图和草图在水平方向上连接起来
        combined_img = np.hstack((img, sketch))

        # 显示合并图像
        cv2.imshow('Sketch Generation', combined_img)
        return sketch

    # 定义生成草图的函数
    def generate_sketch_image(sketch_type):
        if sketch_type == 0:
            # Canny边缘检测
            edges = cv2.Canny(img, 100, 200)
            sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif sketch_type == 1:
            # Laplacian边缘检测
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U)
            sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            # 其他草图生成方法...
            sketch = np.zeros_like(img)

        return sketch

    # 创建窗口并显示原图像
    cv2.namedWindow('Sketch Generation')

    # 创建滚动条以选择草图生成方法
    cv2.createTrackbar('Sketch Type', 'Sketch Generation', 0, 2, generate_sketch)

    # 初始化草图生成
    sketchImg = generate_sketch(0)

    # 等待按键操作
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sketchImg


def mosaic(img):
    # 选择矩形区域
    roi = cv2.selectROI('Select Region', img)

    # 获取选定区域的坐标和尺寸
    x, y, w, h = roi

    if w == 0 or h == 0:
        return img
    # 提取选定区域图像
    roi_image = img[y:y + h, x:x + w]

    # 缩小选定区域图像
    small_roi = cv2.resize(roi_image, (10, 10))

    # 放大缩小后的图像
    mosaic_roi = cv2.resize(small_roi, roi_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # 将马赛克区域放回原图像中
    img[y:y + h, x:x + w] = mosaic_roi

    # 显示结果图像
    cv2.imshow('Mosaic Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img



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

def vignetting(img):
    # 输入图像
    image_copy = img.copy()

    # 创建窗口
    cv2.namedWindow("image")

    # Global variables to store the coordinates of rectangle
    roi_pts = []
    drawing = False

    def draw_roi(event, x, y, flags, param):
        nonlocal roi_pts, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_pts = [(x,y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_pts.append((x, y))
            cv2.rectangle(image_copy, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    cv2.setMouseCallback("image", draw_roi)

    while True:
        cv2.imshow("image", image_copy)
        key = cv2.waitKey(1) & 0xFF

        # Press "r" to reset the ROI
        if key == ord("r"):
            image_copy = cv2.imread(img)
            roi_pts = []

        # Press "enter" to continue with the selected ROI
        elif key == 13:
            break
    print(roi_pts)
    # Get image dimensions
    height, width = img.shape[:2]
    print(height, width)
    # 求ROI中心点坐标
    # 横坐标为（左边横坐标+右边横坐标）/2
    if len(roi_pts) == 0:
        return img
    center_y = (roi_pts[0][0] + roi_pts[1][0]) // 2
    # 纵坐标为（上边纵坐标+下边纵坐标）/2
    center_x = (roi_pts[0][1] + roi_pts[1][1]) // 2
    print(center_x, center_y)
    x = center_x
    y = center_y
    # Define vignette radius

    # Close all windows
    cv2.destroyAllWindows()


    # 设置滚动条回调函数为无操作
    def nothing(x):
        return x

    cv2.namedWindow("image")
    # 创建滚动条
    cv2.createTrackbar("threshold", "image", 200, 400, lambda x: x)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        vignette_level = cv2.getTrackbarPos("threshold", "image")
        # vignette_level = 200
        # # 防止初始时滚动条0位置使结果清零
        if vignette_level == 0:
           vignette_level = 1
        # # Create empty mask
        # mask = np.zeros((height, width), dtype=np.uint8)
        # # Create circular mask
        # cv2.circle(mask, (center_x, center_y), radius, (255), -1)
        #
        # # Apply mask to the image
        # result = cv2.bitwise_and(img, img, mask=mask)
        # 用高斯核生成渐晕掩码
        rows, cols = img.shape[:2]
        if(x<=rows/2 and y<=cols/2):
            kernel_x = cv2.getGaussianKernel(int(2*(rows-x)), vignette_level)
            kernel_y = cv2.getGaussianKernel(int(2*(cols-y)), vignette_level)
            kernel = kernel_x * kernel_y.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = mask[int(2*(rows-x))-rows:int(2*(rows-x)), int(2*(cols-y))-cols:int(2*(cols-y))]
        elif(x>=rows/2 and y<=cols/2):
            kernel_x = cv2.getGaussianKernel(int(2*(x)), vignette_level)
            kernel_y = cv2.getGaussianKernel(int(2*(cols-y)), vignette_level)
            kernel = kernel_x * kernel_y.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = mask[0:rows, int(2*(cols-y))-cols:int(2*(cols-y))]
        elif (x <= rows / 2 and y >= cols / 2):
            kernel_x = cv2.getGaussianKernel(int(2 * (rows-x)), vignette_level)
            kernel_y = cv2.getGaussianKernel(int(2 * (y)), vignette_level)
            kernel = kernel_x * kernel_y.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = mask[int(2 * (rows-x)) - rows:int(2 * (rows-x)), 0:cols]
        else:
            kernel_x = cv2.getGaussianKernel(int(2 * (x)), vignette_level)
            kernel_y = cv2.getGaussianKernel(int(2 * (y)), vignette_level)
            kernel = kernel_x * kernel_y.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = mask[0:rows, 0:cols]


        result = np.copy(img)

        # Adjust vignette effect intensity
        # intensity_factor = 0.7
        # result = result.astype(float)
        # result *= intensity_factor
        # result = np.clip(result, 0, 255).astype(np.uint8)


        # 设置大小为滚动条的位置
        for i in range(3):
            # result[:, :, i] = result[:, :, i] * (vignette_level / 255)
            result[:, :, i] = result[:, :, i] * mask
        concat = np.concatenate((img, result), axis=1)
        # 在和滚动条同一个窗口显示图像，以显示在滚动条下方
        cv2.imshow('image', concat)
        # 按q结束循环
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # 关闭所有窗口
        return result


def relief(img):
    def update_emboss(*args):
        # 获取滚动条的值
        filter_size = filter_size_var.get()
        direction = direction_var.get()

        # 应用浮雕效果
        embossed_image = apply_emboss(img, filter_size, direction)

        # 显示处理后的图像
        embossed_pil = Image.fromarray(embossed_image)
        embossed_tk = ImageTk.PhotoImage(embossed_pil)
        embossed_label.configure(image=embossed_tk)
        embossed_label.image = embossed_tk

    def apply_emboss(image, filter_size, direction):
        # 创建一个浮雕滤波器
        kernel = np.zeros((filter_size, filter_size), np.float32)
        kernel[0, 0] = -1
        kernel[filter_size - 1, filter_size - 1] = 1

        # 根据方向旋转滤波器
        if direction == 0:  # 垂直方向
            kernel = np.rot90(kernel)
        # elif direction == 1:  # 水平方向
        #     kernel = np.flipud(kernel)

        # 应用滤波器
        embossed = cv2.filter2D(image, -1, kernel)

        return embossed

    # 创建Tkinter窗口
    root = tk.Tk()

    # 创建滚动条控制变量
    filter_size_var = tk.IntVar()
    direction_var = tk.IntVar()

    # 创建滚动条和标签
    filter_size_label = tk.Label(root, text="滤波器大小")
    filter_size_label.pack()
    filter_size_scale = tk.Scale(root, from_=3, to=15, orient=tk.HORIZONTAL, variable=filter_size_var,
                                 command=update_emboss)
    filter_size_scale.pack()

    direction_label = tk.Label(root, text="浮雕方向\n(0: 垂直, 1: 水平)")
    direction_label.pack()
    direction_scale = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, variable=direction_var, command=update_emboss)
    direction_scale.pack()

    # 显示原图
    original_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    original_tk = ImageTk.PhotoImage(original_pil)
    original_label = tk.Label(root, image=original_tk)
    original_label.pack()

    # 显示处理后的图像
    embossed_image = apply_emboss(img, filter_size_var.get(), direction_var.get())
    embossed_pil = Image.fromarray(embossed_image)
    embossed_tk = ImageTk.PhotoImage(embossed_pil)
    embossed_label = tk.Label(root, image=embossed_tk)
    embossed_label.pack()

    # 运行Tkinter主循环
    root.mainloop()

    return embossed_image

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
# threshold(img_test)

