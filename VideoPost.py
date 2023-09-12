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

def video2Image(vedio):
    fps = 20

    def read_video(video_path):
        nonlocal fps
        video_cap = cv2.VideoCapture(video_path)
        # fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        all_frames = []
        while True:
            for k in range(fps):
                ret, frame = video_cap.read()
                if ret is False:
                    break
            if ret is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break
            frame_count += 1
            print(frame_count)
        video_cap.release()
        cv2.destroyAllWindows()
        print('===>', len(all_frames))

        return all_frames

    def frame_to_gif(frame_list):
        nonlocal fps
        gif = imageio.mimsave('output.gif', frame_list, 'GIF', duration=1000 * 1 / fps)

    # duration 表示图片间隔

    # fps = int(input('请输入采样频率'))
    frame_list = read_video('test.gif')
    i = 0
    for img in frame_list:
        cv2.imwrite('ImageSequence/img' + str(i) + '.jpg', img)
        i += 1

    frame_to_gif(frame_list)
    output_path = 'output.gif'
    return output_path

def cutVideo(vedio):
    return