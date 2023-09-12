import numpy as np
import cv2
import imageio
import sys

def ImageTrans(img1,img2):
    mode = None
    bool1 = False
    bool2 = False
    bool3 = False
    bool4 = False
    bool5 = False
    duration = 1
    path = ""

    def ChangeMode(value):
        nonlocal bool1
        nonlocal bool2
        nonlocal bool3
        nonlocal bool4
        nonlocal bool5
        nonlocal mode
        mode = value
        if mode == 1:
            bool1 = True
        elif mode == 2:
            bool2 = True
        elif mode == 3:
            bool3 = True
        elif mode == 4:
            bool4 = True
        else:
            bool5 = True

    def ChangeDuration(value):
        nonlocal duration
        duration = value

    def percent_func_gen(a, b, time, n, mode):
        """
        高次多项式计算函数生成器
        :param a: 起始百分比（如：0.25）
        :param b: 结束百分比
        :param time: 动画持续时间
        :param n: 多项式次数
        :param mode: faster（越来越快）、slower（越来越慢）
        :return: 每个时刻到达百分比的计算函数
        """
        if mode == "slower":
            a, b = b, a
        delta = abs(a - b)
        sgn = 1 if b - a > 0 else (-1 if b - a < 0 else 0)

        def percent_calc(ti):
            if mode == "slower":
                ti = time - ti
            return sgn * delta / (time ** n) * (ti ** n) + a

        return percent_calc

    def MoveLeft(img1, img2):
        nonlocal duration, path
        rows, cols = img1.shape[:2]
        img = np.hstack([img1, img2])
        images = []
        load_f = 20
        tim = 0.3
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=2, mode="faster")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            x = int(percent * cols)
            M = np.float32([[1, 0, -x], [0, 1, 0]])
            res = cv2.warpAffine(img, M, (rows, cols))
            b, g, r = cv2.split(res)
            images.append(cv2.merge((r, g, b)))
        res = imageio.mimsave("MoveLeft.gif", images, duration=duration, loop=0)
        # '''关闭窗口'''
        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        path = "MoveLeft.gif"
        return res

    def FlashBlack(img1, img2):
        nonlocal duration, path
        rows, cols = img1.shape[:2]
        img_shows = []
        load_f = 20
        tim = 1
        percent_func1 = percent_func_gen(a=1, b=0, time=tim, n=1, mode="null")
        percent_func2 = percent_func_gen(a=0, b=1, time=tim, n=1, mode="null")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func1(t * load_f / 1000)
            img_show = cv2.multiply(img1, (1, 1, 1, 1), scale=percent)
            b, g, r = cv2.split(img_show)
            img_shows.append(cv2.merge((r, g, b)))
            cv2.imshow("show", img_show)
            cv2.waitKey(load_f)
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func2(t * load_f / 1000)
            img_show = cv2.multiply(img2, (1, 1, 1, 1), scale=percent)
            b, g, r = cv2.split(img_show)
            img_shows.append(cv2.merge((r, g, b)))
            cv2.imshow("show", img_show)
            cv2.waitKey(load_f)
        res = imageio.mimsave("FlashBlack.gif", img_shows, duration=duration, loop=0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        path = "FlashBlack.gif"
        return res

    def EraseDown(img1, img2):
        nonlocal duration, path
        rows, cols = img1.shape[:2]
        img_shows = []
        load_f = 20
        tim = 0.3
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=1, mode="null")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            height = int(percent * rows)
            img1[:height, :] = img2[:height, :]
            cv2.imshow("show", img1)
            cv2.waitKey(load_f)
            b, g, r = cv2.split(img1)
            img_shows.append(cv2.merge((r, g, b)))

        res = imageio.mimsave('EraseDown.gif', img_shows, duration=duration, loop=0)
        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        path = 'EraseDown.gif'
        return res

    def HorizontalCurtain(img1, img2):
        nonlocal duration, path
        rows, cols = img1.shape[:2]
        img_shows = []
        load_f = 20
        tim = 0.3
        half = int(rows / 2)
        percent_func = percent_func_gen(a=0, b=0.5, time=tim, n=1, mode="null")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            width = int(percent * rows)
            ys, ye = half - width, half + width
            img1[:, ys:ye] = img2[:, ys:ye]
            cv2.imshow("show", img1)
            cv2.waitKey(load_f)
            b, g, r = cv2.split(img1)
            img_shows.append(cv2.merge((r, g, b)))

        res = imageio.mimsave("HorizontalCurtain.gif", img_shows, duration=duration, loop=0)

        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        path = "HorizontalCurtain.gif"
        return res

    def Spin(img1, img2):
        nonlocal duration, path
        rows, cols = img1.shape[:2]
        img_shows = []
        img1_ru = cv2.flip(img1, 1)
        img1_ld = cv2.flip(img1, 0)
        img1_rd = cv2.flip(img1, -1)
        img1_u = np.hstack([img1, img1_ru])
        img1_d = np.hstack([img1_ld, img1_rd])
        img1_res_tmp = np.vstack([img1_u, img1_d])
        img1_res_tmp = np.hstack([img1_res_tmp] * 3)
        img1_res = np.vstack([img1_res_tmp] * 3)

        img2_lu = cv2.flip(img2, 1)
        img2_rd = cv2.flip(img2, 0)
        img2_ld = cv2.flip(img2, -1)
        img2_u = np.hstack([img2_lu, img2])
        img2_d = np.hstack([img2_ld, img2_rd])
        img2_res_tmp = np.vstack([img2_u, img2_d])
        img2_res_tmp = np.hstack([img2_res_tmp] * 3)
        img2_res = np.vstack([img2_res_tmp] * 3)

        res_rows, res_cols = img1_res.shape[:2]

        load_f = 20
        tim = 0.2
        angle_all = 150
        point1 = (rows * 3, cols * 3)
        point2 = (rows * 3, cols * 4)
        percent_func1 = percent_func_gen(a=0, b=1, time=tim, n=4, mode="faster")
        percent_func2 = percent_func_gen(a=1, b=0, time=tim, n=4, mode="slower")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func1(t * load_f / 1000)
            angle = percent * angle_all
            M1 = cv2.getRotationMatrix2D(point1, angle, 1)
            res = cv2.warpAffine(img1_res, M1, (res_rows, res_cols))
            M2 = np.float32([[1, 0, -cols * 2], [0, 1, -rows * 2]])
            res = cv2.warpAffine(res, M2, (rows, cols))
            cv2.imshow("show", res)
            cv2.waitKey(load_f)
            b, g, r = cv2.split(res)
            img_shows.append(cv2.merge((r, g, b)))

        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func2(t * load_f / 1000)
            angle = -percent * angle_all
            M2 = cv2.getRotationMatrix2D(point2, angle, 1)
            res = cv2.warpAffine(img2_res, M2, (res_rows, res_cols))
            M2 = np.float32([[1, 0, -cols * 3], [0, 1, -rows * 2]])
            res = cv2.warpAffine(res, M2, (rows, cols))
            cv2.imshow("show", res)
            cv2.waitKey(load_f)
            b, g, r = cv2.split(res)
            img_shows.append(cv2.merge((r, g, b)))

        dst = imageio.mimsave('Spin.gif', img_shows, duration=duration, loop=0)

        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        path = "Spin.gif"
        return dst

    cv2.namedWindow("PictureTransition")
    cv2.createTrackbar("Mode", "PictureTransition", 1, 5, ChangeMode)
    cv2.createTrackbar("Duration", "PictureTransition", 1, 4, ChangeDuration)

    while True:
        if bool1 == True:
            res = MoveLeft(img1, img2)
            # cv2.imshow("picture",res)
            bool1 = False
        elif bool2 == True:
            res = FlashBlack(img1, img2)
            # cv2.imshow("picture",res)
            bool2 = False
        elif bool3 == True:
            res = EraseDown(img1, img2)
            # cv2.imshow("picture",res)
            bool3 = False
        elif bool4 == True:
            res = HorizontalCurtain(img1, img2)
            # cv2.imshow("picture",res)
            bool4 = False
        else:
            res = Spin(img1, img2)
            # cv2.imshow("picture",res)
            bool5 = False
        if cv2.waitKey(0) == 13:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return path

def move(img_input):
    x_init, y_init = 0, 0
    pos_x, pos_y =0, 0
    top_left_pt, bottom_right_pt = (0,0),(0,0)
    img2=np.copy(img_input)
    def draw_rectangle(event, x, y, flags, params):
        nonlocal x_init, y_init, drawing, top_left_pt, bottom_right_pt, img_orig, img2

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


    def draw_path(event, x, y, flags, params):
        nonlocal drawing, pos_x, pos_y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            pos_x, pos_y = x, y
            cv2.circle(img, (pos_x, pos_y), 20, (0, 255, 0), -1)
            while True:
                cv2.imshow('Output', img)
                c = cv2.waitKey(1)
                if c == 27:
                    break


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
    cv2.destroyAllWindows()

    # set destination
    cv2.namedWindow('Path')
    cv2.setMouseCallback('Path', draw_path)
    while True:
        cv2.imshow('Path', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Load the background and moving images
    background_image = cv2.imread("background.jpg")
    moving_image = cv2.imread("frontal.jpg")

    # print(moving_image.shape)
    # Determine the size and position for the moving image
    # moving_height, moving_width = moving_image.shape[:2]

    moving_height = pos_y - y_init
    moving_width = pos_x - x_init

    start_x = 0  # Starting X position of the moving image
    start_y = 0  # Starting Y position of the moving image


    fps = 30
    duration = 5
    def nothing():
        pass

    cv2.namedWindow("fps")
    # 创建滚动条
    cv2.createTrackbar("threshold", "fps", 0, 60, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        angle = cv2.getTrackbarPos("threshold", "fps")

        if cv2.waitKey(1) == 27:
            break
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('fps=', fps)

    cv2.namedWindow("duration")
    # 创建滚动条
    cv2.createTrackbar("threshold", "duration", 0, 10, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        duration = cv2.getTrackbarPos("threshold", "duration")

        if cv2.waitKey(1) == 27:
            break
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('duration=', duration)


    # Set up the parameters for the GIF
    # fps = 30  # Frames per second
    # duration = 5  # Duration of the GIF in seconds
    num_frames = fps * duration

    # Create a list to store the frames
    frames = []

    # Create the animation frames
    for i in range(num_frames):
        # Create a blank canvas
        canvas = np.copy(background_image)

        # Calculate the position of the moving image for the current frame
        offset_x = int((i / num_frames) * (moving_width - start_x))
        offset_y = int((i / num_frames) * (moving_height - start_y))
        pos_x = start_x + offset_x
        pos_y = start_y + offset_y

        if pos_x < 0:
            init_x = - pos_x
            final_x = canvas.shape[1]
            init_x2 = 0
            final_x2 = canvas.shape[1] + pos_x
        else:
            init_x = 0
            final_x = canvas.shape[1] - pos_x
            init_x2 = pos_x
            final_x2 = canvas.shape[1]
        if pos_y < 0:
            init_y = - pos_y
            final_y = canvas.shape[0]
            init_y2 = 0
            final_y2 = canvas.shape[0] + pos_y
        else:
            init_y = 0
            final_y = canvas.shape[0] - pos_y
            init_y2 = pos_y
            final_y2 = canvas.shape[0]

        moe = moving_image[init_y:final_y, init_x:final_x]
        print(init_y,final_y, init_x,final_x)
        # roi = canvas[pos_y:lim_y, pos_x:lim_x]
        # cv2.addWeighted(moe, 1, roi, 0, 0, roi)
        # print(canvas.shape, moe.shape)
        print(init_y2, final_y2, init_x2, final_x2)
        roi = canvas[init_y2:final_y2, init_x2:final_x2]

        roi[:, :] = np.where(moe[:, :] == [0, 0, 0], roi[:, :], moe[:, :])
        # Overlay the moving image onto the canvas
        # alpha = 0.5  # Opacity of the moving image

        b, g, r = cv2.split(canvas)

        # Append the frame to the list of frames
        frames.append(cv2.merge((r, g, b)))
        # frames.append(canvas)
    # Save the frames as a GIF
    imageio.mimsave("result.gif", frames, duration=5000 / fps, loop=1)

    cv2.destroyAllWindows()
    return "result.gif"

def spin(img_input):
    x_init, y_init = 0, 0
    pos_x, pos_y = 0, 0
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


    def draw_center(event, x, y, flags, params):
        nonlocal drawing, pos_x, pos_y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            pos_x, pos_y = x, y
            cv2.circle(img, (pos_x, pos_y), 20, (0, 255, 0), 2)
            while True:
                cv2.imshow('Output', img)
                c = cv2.waitKey(1)
                if c == 27:
                    break


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

    # set destination
    cv2.namedWindow('Center')
    cv2.setMouseCallback('Center', draw_center)
    while True:
        cv2.imshow('Center', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Load the background and moving images
    background_image = cv2.imread("background.jpg")
    moving_image = cv2.imread("frontal.jpg")

    print(moving_image.shape)
    # Determine the size and position for the moving image
    moving_height, moving_width = moving_image.shape[:2]
    start_x = 0  # Starting X position of the moving image
    start_y = 0  # Starting Y position of the moving image

    fps = 30
    duration = 5

    def nothing():
        pass

    cv2.namedWindow("fps")
    # 创建滚动条
    cv2.createTrackbar("threshold", "fps", 0, 60, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        angle = cv2.getTrackbarPos("threshold", "fps")

        cv2.imshow('fps', img)
        if cv2.waitKey(1) == 27:
            break
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('fps=', fps)

    cv2.namedWindow("duration")
    # 创建滚动条
    cv2.createTrackbar("threshold", "duration", 0, 10, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        duration = cv2.getTrackbarPos("threshold", "duration")

        cv2.imshow('duration', img)
        if cv2.waitKey(1) == 27:
            break
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('duration=', duration)

    # Set up the parameters for the GIF
    # fps = 30  # Frames per second
    # duration = 1  # Duration of the GIF in seconds
    num_frames = fps * duration

    # Create a list to store the frames
    frames = []

    angle = 0

    cv2.namedWindow("angle")
    # 创建滚动条
    cv2.createTrackbar("threshold", "angle", 0, 360, nothing)

    # 循环实现用户对滚动条的持续操作
    while True:
        # 获取滚动条的值
        angle = cv2.getTrackbarPos("threshold", "angle")

        cv2.imshow('angle', img)
        if cv2.waitKey(1) == 27:
            break
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('angle=',angle)
    # angle = 20 # 旋转角度

    center = (pos_x, pos_y)
    # center = (top_left_pt[0], bottom_right_pt[1])  # 旋转中心

    # Create the animation frames
    for i in range(num_frames):
        # Create a blank canvas
        canvas = np.copy(background_image)

        # Calculate the position of the moving image for the current frame
        offset_x = 0
        offset_y = 0
        pos_x = start_x + offset_x
        pos_y = start_y + offset_y

        lim_x = moving_width + pos_x
        lim_y = moving_height + pos_y
        if lim_x > canvas.shape[1]:
            lim_x = canvas.shape[1]
        if lim_y > canvas.shape[0]:
            lim_y = canvas.shape[0]

        if i < duration*fps//2:
            b = angle * i / (duration * fps // 2 - 1)
        else:
            b = angle * (duration*fps - i) / (duration * fps // 2 - 1)
        print(b)
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, b, 1.0)

        # 进行图像旋转
        rotated_image = cv2.warpAffine(moving_image, rotation_matrix, (moving_width, moving_height))
        moe = rotated_image[0:lim_y - pos_y, 0:lim_x - pos_x]

        # roi = canvas[pos_y:lim_y, pos_x:lim_x]
        # cv2.addWeighted(moe, 1, roi, 0, 0, roi)
        print(canvas.shape, moe.shape)
        roi = canvas[pos_y:lim_y, pos_x:lim_x, :]
        roi[:, :] = np.where(moe[:, :] == [0, 0, 0], roi[:, :], moe[:, :])
        # Overlay the moving image onto the canvas
        # alpha = 0.5  # Opacity of the moving image

        b, g, r = cv2.split(canvas)

        # Append the frame to the list of frames
        frames.append(cv2.merge((r, g, b)))
        # frames.append(canvas)
    # Save the frames as a GIF
    imageio.mimsave("result.gif", frames, duration=1000 / fps, loop=0)

    cv2.destroyAllWindows()

    return "result.gif"
