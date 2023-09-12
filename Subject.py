import cv2
import numpy as np
import sys


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
            return img

        elif key == 27:  # 按下 'Esc' 键退出
            break

    # 销毁窗口
    cv2.destroyAllWindows()
    return img


def rectSelect(img):
    window_name = 'cut_img'

    while True:
        # cv2.imshow(window_name, img)
        roi = cv2.selectROI(img, showCrosshair=True, fromCenter=False)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, roi, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        mask = np.all(img[:, :, :] == [0, 0, 0], axis=-1)
        trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        trans_img[mask, 3] = 0
        cv2.imwrite('OutputImg/grabcut_image.png', trans_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            # return img
            break

    cv2.destroyAllWindows()
    return img


def object_removal(img_input):
    x_init, y_init = 0, 0

    top_left_pt, bottom_right_pt = (0, 0), (0, 0)
    img_orig = np.copy(img_input)
    img_output = np.copy(img_input)

    def draw_rectangle(event, x, y, flags, params):
        nonlocal x_init, y_init, drawing, top_left_pt, bottom_right_pt, img_orig, img_output

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x_init, y_init = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
                img[y_init:y, x_init:x] = 255 - img_orig[y_init:y, x_init:x]
                cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
            img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
            cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            rect_final = (x_init, y_init, x - x_init, y - y_init)
            remove_object(img_orig, rect_final)

    def compute_energy_matrix_modified(img, rect_roi):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        energy_matrix = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        x, y, w, h = rect_roi
        energy_matrix[y:y + h, x:x + w] = 0

        return energy_matrix

    def compute_energy_matrix(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    def find_vertical_seam(img, energy):
        rows, cols = img.shape[:2]

        seam = np.zeros(img.shape[0])
        dist_to = np.zeros(img.shape[:2]) + sys.maxsize
        dist_to[0, :] = np.zeros(img.shape[1])
        edge_to = np.zeros(img.shape[:2])

        # reduce 1
        for row in range(1, rows):
            # dist_left = np.copy(dist_to)
            # energy_left = np.copy(energy)
            #
            # dist_left[row - 1, 1:cols] = dist_left[row-1, :cols-1]
            # energy_left[row - 1, 1:cols] = energy_left[row-1, :cols-1]

            dist_from_left = np.copy(dist_to)
            dist_from_up = np.copy(dist_to)
            dist_from_right = np.copy(dist_to)
            # dist_from_left[row, :] = dist_left[row - 1, :] + energy[row - 1, :]

            dist_from_left[row, 1:] = dist_to[row - 1, :cols - 1] + energy[row, 1:]
            dist_from_up[row, :] = dist_to[row - 1, :] + energy[row, :]
            dist_from_right[row, :cols - 1] = dist_to[row - 1, 1:] + energy[row, :cols - 1]

            dist_to[row, :] = np.minimum(dist_from_left[row, :], dist_from_up[row, :])
            dist_to[row, :] = np.minimum(dist_to[row, :], dist_from_right[row, :])

            edge_to[row, :] = np.where(dist_to[row, :] == dist_from_left[row, :], -1, 1)
            edge_to[row, :] = np.where(dist_to[row, :] == dist_from_up[row, :], 0, edge_to[row, :])

        # Retrace the min-path and update the 'seam' vector
        seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
        for i in (x for x in reversed(range(rows)) if x > 0):
            seam[i - 1] = seam[i] + edge_to[i, int(seam[i])]

        return seam

    def add_vertical_seam(img, seam, num_iter):
        seam = seam + num_iter
        rows, cols = img.shape[:2]
        zero_col_mat = np.zeros((rows, 1, 3), dtype=np.uint8)
        img_extended = np.hstack((img, zero_col_mat))

        # reduce 1
        for row in range(rows):
            img_extended[row, int(seam[row]) + 1:cols + 1] = img[row, int(seam[row]):cols]

            for i in range(3):
                v1 = img_extended[row, int(seam[row]) - 1, i]
                v2 = img_extended[row, int(seam[row]) + 1, i]
                img_extended[row, int(seam[row]), i] = (int(v1) + int(v2)) / 2
            #
            # # reduce 1
            # img_extended[row, int(seam[row]), :] = \
            #     ((img_extended[row, int(seam[row])-1, :]+img_extended[row, int(seam[row])+1, :]).astype(int))/2

        return img_extended

    def remove_vertical_seam(img, seam):
        rows, cols = img.shape[:2]
        # reduce 1
        for row in range(rows):
            img[row, int(seam[row]):cols - 1] = img[row, int(seam[row]) + 1:]

        img = img[:, 0:cols - 1]
        return img

    def remove_object(img, rect_roi):
        nonlocal img_output
        num_seams = rect_roi[2] + 10
        energy = compute_energy_matrix_modified(img, rect_roi)

        for i in range(num_seams):
            seam = find_vertical_seam(img, energy)
            img = remove_vertical_seam(img, seam)
            x, y, w, h = rect_roi
            energy = compute_energy_matrix_modified(img, (x, y, w - i, h))
            print('Number of seams removed =', i + 1)

        img_output = np.copy(img)
        img_carved_backup = np.copy(img)

        for i in range(num_seams):
            seam = find_vertical_seam(img, energy)
            img = remove_vertical_seam(img, seam)
            img_output = add_vertical_seam(img_output, seam, i)
            energy = compute_energy_matrix(img)
            print('Number of seams added =', i + 1)

        cv2.imshow('Output', img_output)
        cv2.imwrite("img_output.jpg", img_output)
        cv2.waitKey()

    drawing = False
    img = np.copy(img_input)
    img_orig = np.copy(img_input)

    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', draw_rectangle)

    while True:
        cv2.imshow('Input', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cv2.destroyAllWindows()
    return img_output


def object_resize(img_input):
    x_init, y_init = 0, 0

    top_left_pt, bottom_right_pt = (0, 0), (0, 0)
    img_orig = np.copy(img_input)

    def draw_rectangle(event, x, y, flags, params):
        nonlocal x_init, y_init, drawing, top_left_pt, bottom_right_pt, img_orig

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x_init, y_init = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
                img[y_init:y, x_init:x] = 255 - img_orig[y_init:y, x_init:x]
                cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
            img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
            cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
            rect_final = (x_init, y_init, x - x_init, y - y_init)

            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_orig, mask, rect_final, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img2 = img_orig * mask2[:, :, np.newaxis]
            while True:
                cv2.imshow('Output', img2)
                c = cv2.waitKey(1)
                if c == 27:
                    break
            cv2.destroyAllWindows()
            cv2.imwrite("frontal.jpg", img2)
            remove_object(img_orig, rect_final)

    def compute_energy_matrix_modified(img, rect_roi):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        energy_matrix = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        x, y, w, h = rect_roi
        energy_matrix[y:y + h, x:x + w] = 0

        return energy_matrix

    def compute_energy_matrix(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    def find_vertical_seam(img, energy):
        rows, cols = img.shape[:2]

        seam = np.zeros(img.shape[0])
        dist_to = np.zeros(img.shape[:2]) + sys.maxsize
        dist_to[0, :] = np.zeros(img.shape[1])
        edge_to = np.zeros(img.shape[:2])

        # reduce 1
        for row in range(1, rows):
            # dist_left = np.copy(dist_to)
            # energy_left = np.copy(energy)
            #
            # dist_left[row - 1, 1:cols] = dist_left[row-1, :cols-1]
            # energy_left[row - 1, 1:cols] = energy_left[row-1, :cols-1]

            dist_from_left = np.copy(dist_to)
            dist_from_up = np.copy(dist_to)
            dist_from_right = np.copy(dist_to)
            # dist_from_left[row, :] = dist_left[row - 1, :] + energy[row - 1, :]

            dist_from_left[row, 1:] = dist_to[row - 1, :cols - 1] + energy[row, 1:]
            dist_from_up[row, :] = dist_to[row - 1, :] + energy[row, :]
            dist_from_right[row, :cols - 1] = dist_to[row - 1, 1:] + energy[row, :cols - 1]

            dist_to[row, :] = np.minimum(dist_from_left[row, :], dist_from_up[row, :])
            dist_to[row, :] = np.minimum(dist_to[row, :], dist_from_right[row, :])

            edge_to[row, :] = np.where(dist_to[row, :] == dist_from_left[row, :], -1, 1)
            edge_to[row, :] = np.where(dist_to[row, :] == dist_from_up[row, :], 0, edge_to[row, :])

        # Retrace the min-path and update the 'seam' vector
        seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
        for i in (x for x in reversed(range(rows)) if x > 0):
            seam[i - 1] = seam[i] + edge_to[i, int(seam[i])]

        return seam

    def add_vertical_seam(img, seam, num_iter):
        seam = seam + num_iter
        rows, cols = img.shape[:2]
        zero_col_mat = np.zeros((rows, 1, 3), dtype=np.uint8)
        img_extended = np.hstack((img, zero_col_mat))

        # reduce 1
        for row in range(rows):
            img_extended[row, int(seam[row]) + 1:cols + 1] = img[row, int(seam[row]):cols]

            for i in range(3):
                v1 = img_extended[row, int(seam[row]) - 1, i]
                v2 = img_extended[row, int(seam[row]) + 1, i]
                img_extended[row, int(seam[row]), i] = (int(v1) + int(v2)) / 2
            #
            # # reduce 1
            # img_extended[row, int(seam[row]), :] = \
            #     ((img_extended[row, int(seam[row])-1, :]+img_extended[row, int(seam[row])+1, :]).astype(int))/2

        return img_extended

    def remove_vertical_seam(img, seam):
        rows, cols = img.shape[:2]
        # reduce 1
        for row in range(rows):
            img[row, int(seam[row]):cols - 1] = img[row, int(seam[row]) + 1:]

        img = img[:, 0:cols - 1]
        return img

    def remove_object(img, rect_roi):
        num_seams = rect_roi[2] + 10
        energy = compute_energy_matrix_modified(img, rect_roi)

        for i in range(num_seams):
            seam = find_vertical_seam(img, energy)
            img = remove_vertical_seam(img, seam)
            x, y, w, h = rect_roi
            energy = compute_energy_matrix_modified(img, (x, y, w - i, h))
            print('Number of seams removed =', i + 1)

        img_output = np.copy(img)
        img_carved_backup = np.copy(img)

        for i in range(num_seams):
            seam = find_vertical_seam(img, energy)
            img = remove_vertical_seam(img, seam)
            img_output = add_vertical_seam(img_output, seam, i)
            energy = compute_energy_matrix(img)
            print('Number of seams added =', i + 1)

        cv2.imwrite('background.jpg', img_output)
        cv2.waitKey()

    drawing = False
    img = np.copy(img_input)
    img_orig = np.copy(img_input)

    # load background image
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', draw_rectangle)
    while True:
        cv2.imshow('Input', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Load the background and moving images
    background_image = cv2.imread("background.jpg")
    moving_image = cv2.imread("frontal.jpg")

    height, width = moving_image.shape[:2]

    def nothing(x):
        pass

    cv2.namedWindow("size")
    # 创建滚动条
    cv2.createTrackbar("threshold", "size", 0, 255, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        size = cv2.getTrackbarPos("threshold", "size")
        if size == 0:
            size = 100
        print(size)
        cv2.imshow('size', img)
        if cv2.waitKey(1) == 27:
            break

    size1 = (int(width * size / 100), int(height * size / 100))
    print(size1)
    print(moving_image.shape)
    moe = cv2.resize(moving_image, size1)

    if size > 100:
        moving_image[:, :] = [0, 0, 0]
        init_x = (size1[0] - width) // 2
        init_y = (size1[1] - height) // 2

        canvas = np.copy(background_image)

        moving_image[:, :] = moe[init_y:height + init_y, init_x:width + init_x]
        canvas[:, :] = np.where(moving_image[:, :] == [0, 0, 0], canvas[:, :], moving_image[:, :])
    else:

        moving_image[:, :] = [0, 0, 0]
        init_x = (width - size1[0]) // 2
        init_y = (height - size1[1]) // 2

        canvas = np.copy(background_image)

        moving_image[init_y:size1[1] + init_y, init_x:size1[0] + init_x] = moe[:, :]

        while True:
            cv2.imshow('size', moving_image)
            if cv2.waitKey(1) == 27:
                break

        canvas[:, :] = np.where(moving_image[:, :] == [0, 0, 0], canvas[:, :], moving_image[:, :])

    # 关闭所有窗口
    cv2.destroyAllWindows()
    print(size)

    while True:
        cv2.imshow('Output', canvas)
        if cv2.waitKey(1) == 27:
            break
    cv2.imwrite('resize.jpg', canvas)
    cv2.destroyAllWindows()

    return canvas


img_path = "saved.jpg"
img_test = cv2.imread(img_path)
# img = object_resize(img_test)
