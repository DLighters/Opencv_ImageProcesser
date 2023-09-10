import numpy as np
import cv2
import imageio

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
