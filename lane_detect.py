import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

delta_k = 0.17  # 判断直线平行
delta_line = 5  # 车道线宽度范围
w_line = 5  # 车道线宽度
w_lines = 120  # 两车道线距离
delta_lines = 35  # 两车道线距离允许误差
point1 = np.array([[250, 1000], [1250, 1000], [670, 680], [760, 680]], dtype="float32")
# point1 = np.array([[160,250],[450,250],[241,150],[324,150]],dtype="float32")
cnt = 1  # 用于记录是否kalman检测
# cnt2 = 0 #用于判断是否存在行道线
# kalman滤波
kalman = [cv2.KalmanFilter(4, 2), cv2.KalmanFilter(4, 2)]

pre = [[0, 0], [0, 0]]  # k,x


# 判断是否平行
def K(i, j):
    if abs(i[-3] - j[-3]) <= delta_k:
        return 1
    else:
        return 0


# 判断是否同一车道线
def D1(i, j):
    if abs(abs(i[-2] - j[-2]) - w_line) <= delta_line:
        return 2
    else:
        return 0


# 判断是否为两条车道线
def D2(i, j):
    if abs(abs(i[-2] - j[-2]) - w_lines) < delta_lines:
        return 5
    else:
        return 0


# 判断是否在中间
def L(i, j, w):
    if abs(i[-2] - w // 2) <= w_lines // 2:
        return 5
    else:
        return 0


# 逆透视变换
def IPM(img):
    w, h = img.shape
    point2 = np.array([[h // 8 * 3, w], [h - h // 8 * 3, w], [h // 8 * 3, 0], [h - h // 8 * 3, 0]], dtype='float32')
    M = cv2.getPerspectiveTransform(point1, point2)
    return cv2.warpPerspective(img, M, (h, w))


def IPM2(img):
    w, h = img.shape
    point2 = np.array([[h // 8 * 3, w], [h - h // 8 * 3, w], [h // 8 * 3, 0], [h - h // 8 * 3, 0]], dtype='float32')
    M = cv2.getPerspectiveTransform(point2, point1)
    return cv2.warpPerspective(img, M, (h, w))


# 求上面的点
def uppoint(x1, y1, k, up):
    y2 = int(x1 - (y1 - up) * k)
    return (y2, up)


# 生成掩膜
def gene_mask(img, area):
    mask = np.zeros_like(img)
    vertices = np.array(area, dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    img = cv2.bitwise_and(img, mask)
    return img


# 霍夫直线检测
def get_lines(img):
    max_y = img.shape[0]
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, np.array([]), 20, 5)
    if lines is None:
        return []
    # 霍夫变换
    newlines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (x2 - x1) == 0:
                slope = 1000
            else:
                slope = (y2 - y1) / (x2 - x1)
            if abs(slope) >= 2:
                newlines.append([x1, y1, x2, y2, (x2 - x1) / (y2 - y1), int(x2 + (max_y - y2) / slope), 0])
    for i in range(len(newlines)):
        for j in range(i + 1, len(newlines)):
            a = newlines[i]
            b = newlines[j]
            score = L(a, b, img.shape[1]) + K(a, b) * (D1(a, b) + D2(a, b))
            newlines[i][-1] += score
            newlines[j][-1] += score
    # 判断那些是线
    Newlines = []
    while (len(newlines)):
        newlines.sort(key=lambda x: x[-1], reverse=True)
        a = newlines[0].copy()
        sm = a[-2]
        m = 1
        if a[-1] == 0: break
        newlines[0][-1] = 0
        for i in range(len(newlines)):
            b = newlines[i]
            if b[-1] == 0: continue
            if K(a, b) * D1(a, b) == 2:
                newlines[i][-1] = 0
                sm += b[-2]
                m += 1
        a[-2] = sm / m
        Newlines.append(a)
    return Newlines


def annotate_image_array(img):
    global cnt, pre
    img3 = img
    n = max(1, int(img.shape[1] / 500 + 0.5))  # 图片缩放比例
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # plt.hist(img.ravel(),256)
    img = IPM(img)
    img = cv2.resize(img, [img.shape[1] // n, img.shape[0] // n])
    img2 = np.zeros_like(img)
    img = cv2.blur(img, (3, 3))
    img = cv2.blur(img, (3, 3))
    img = cv2.Canny(img, 90, 160)
    area = []
    if cnt >= 20:
        area = []
        for i in range(2):
            xl = int(pre[i][1] - kalman[i].errorCovPost[1][1])
            xr = int(pre[i][1] + kalman[i].errorCovPost[1][1])
            area.append([(xl, img.shape[0]), uppoint(xl, img.shape[0], pre[i][0], img.shape[0] // 2),
                         uppoint(xr, img.shape[0], pre[i][0], img.shape[0] // 2), (xr, img.shape[0])])
        img = gene_mask(img, area)
    else:
        img = gene_mask(img, [[(img.shape[1] // 3, img.shape[0]), (img.shape[1] // 3, img.shape[0] // 2),
                               (img.shape[1] // 3 * 2, img.shape[0] // 2), (img.shape[1] // 3 * 2, img.shape[0])]])
    max_y = img.shape[0]
    middle_x = img.shape[1] / 2
    Newlines = get_lines(img)
    final_line = []
    flag = True
    if not Newlines:
        # cnt2 -= 1
        flag = False
        # if cnt < 20:
        #     return img3
        # else:
        final_line = [(pre[0][1], max_y, 0, 0, pre[0][0], pre[0][1], 0),
                      (pre[1][1], max_y, 0, 0, pre[1][0], pre[1][1], 0)]
    for a in Newlines:
        for b in Newlines:
            if K(a, b) * D2(a, b) == 5:
                final_line = [a, b]
                break
        if final_line:
            break
    if not final_line:
        final_line = [Newlines[0]]
        # cnt2 -= 1
    # else: cnt2 = 0
    # if cnt2 < -20 : return img3
    real_lines = []
    for X in final_line:
        x1, y1, x2, y2, a, b, c = X
        x1, y1 = img.shape[0], int(b)
        y2, x2 = uppoint(y1, x1, a, img.shape[0] // 3)
        cv2.line(img2, (y1, x1), (y2, x2), [255, 0, 0], 3)
        real_lines.append([y1, (y1 - y2) / (x1 - x2)])
        if len(final_line) == 1:
            # flag = False
            if y1 < img.shape[1] // 2 - 30:
                cv2.line(img2, (y1 + w_lines, x1), (y2 + w_lines, x2), [255, 0, 0], 3)
                real_lines.append([y1 + w_lines, (y1 - y2) / (x1 - x2)])
            elif y1 > img.shape[1] // 2 + 30:
                cv2.line(img2, (y1 - w_lines, x1), (y2 - w_lines, x2), [255, 0, 0], 3)
                real_lines.append([y1 - w_lines, (y1 - y2) / (x1 - x2)])
            else:
                if abs(pre[0][1] - y1) < abs(pre[1][1] - y1):
                    cv2.line(img2, (y1 + w_lines, x1), (y2 + w_lines, x2), [255, 0, 0], 3)
                    real_lines.append([y1 + w_lines, (y1 - y2) / (x1 - x2)])
                else:
                    cv2.line(img2, (y1 - w_lines, x1), (y2 - w_lines, x2), [255, 0, 0], 3)
                    real_lines.append([y1 - w_lines, (y1 - y2) / (x1 - x2)])
    # plt.imshow(img,'gray')
    # plt.show()
    real_lines.sort(key=lambda x: x[0])
    if flag and abs(pre[0][0] - real_lines[0][1]) < kalman[0].errorCovPost[0][0] and abs(pre[0][1] - real_lines[0][0]) < \
            kalman[0].errorCovPost[1][1] and abs(pre[1][0] - real_lines[1][1]) < kalman[1].errorCovPost[0][0] and abs(
        pre[1][1] - real_lines[1][0]) < kalman[1].errorCovPost[1][1]:
        if cnt < 20:
            cnt += 1
        else:
            cnt = 20
    else:
        if cnt < 20:
            cnt = 0
        else:
            cnt = (cnt + 1) % 40
    for i in range(2):
        k = real_lines[i][1]
        x = real_lines[i][0]
        if pre[i] == [0, 0]:
            kalman[i].processNoiseCov = np.array([[k, 0, 0, 0], [0, x, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                 np.float32) * 0.04
        current_measurement = np.array([[np.float32(k)], [np.float32(x)]])
        kalman[i].correct(current_measurement)
        pre[i] = kalman[i].predict()
        pre[i] = [pre[i][0][0], pre[i][1][0]]
    img2 = cv2.resize(img2, [img3.shape[1], img3.shape[0]])
    img2 = IPM2(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # img2[:,:,1] *= 0
    img2[:, :, 2] *= 0
    h = img2.shape[0] // 5 * 4
    pl = []
    pr = []
    for i in range(img2.shape[1]):
        if img2[h][i][0] > 0:
            pl = [i, h]
            break
    for i in range(img2.shape[1] - 1, -1, -1):
        if img2[h][i][0] > 0:
            pr = [i, h]
            break
    if not pl:
        return img3
    cv2.line(img2, pl, pr, [0, 255, 255], 20)
    cv2.line(img2, pl, [pl[0], pl[1] - 10], [0, 255, 255], 30)
    cv2.line(img2, pr, [pr[0], pr[1] - 10], [0, 255, 255], 30)
    cv2.line(img2, [(pl[0] + pr[0]) // 2, pr[1]], [(pl[0] + pr[0]) // 2, pr[1] - 15], [0, 255, 0], 25)
    lenth = (pr[0] - pl[0]) * ((img2.shape[1] // 2 - (pl[0] + pr[0]) // 2) / img2.shape[1])
    lenth = int(lenth) - 60
    cv2.line(img2, [(pl[0] + pr[0]) // 2 + lenth, pr[1] + 15], [(pl[0] + pr[0]) // 2 + lenth, pr[1] - 15], [255, 0, 0],
             25)
    img = cv2.addWeighted(img3, 1, img2, 0.5, 0)
    k = img.shape[1] // 40
    if (pr[0] - pl[0]) / (abs(lenth) + 1) < 6:
        if lenth < 0:
            x = pl[0]
            y = pl[1]
            cv2.line(img, (x, y + img.shape[0] // 15), (x + img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0],
                     img.shape[0] // 40)
            cv2.line(img, (x + img.shape[1] // 10 - k, y + img.shape[0] // 15 + k),
                     (x + img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0], img.shape[0] // 50)
            cv2.line(img, (x + img.shape[1] // 10 - k, y + img.shape[0] // 15 - k),
                     (x + img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0], img.shape[0] // 50)
        else:
            x = pr[0]
            y = pr[1]
            cv2.line(img, (x, y + img.shape[0] // 15), (x - img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0],
                     img.shape[0] // 40)
            cv2.line(img, (x - img.shape[1] // 10 + k, y + img.shape[0] // 15 + k),
                     (x - img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0], img.shape[0] // 50)
            cv2.line(img, (x - img.shape[1] // 10 + k, y + img.shape[0] // 15 - k),
                     (x - img.shape[1] // 10, y + img.shape[0] // 15), [0, 255, 0], img.shape[0] // 50)
    # plt.imshow(img)
    # plt.show()
    # cv2.KalmanFilter
    return img


def annotate_video(input_file, output_file):
    for i in range(2):
        # 设置测量矩阵
        kalman[i].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 设置转移矩阵
        kalman[i].transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 设置过程噪声协方差矩阵
        kalman[i].measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32)
        kalman[i].errorCovPost = np.array([[9, 0, 0, 0], [0, 16, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]], np.float32)

    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image_array)
    annotated_video.write_videofile(output_file, audio=False)

# annotate_video("data_Trim.mp4","out.mp4")
# img = cv2.imread("7.jpg")
# cv2.line(img,(img.shape[1],img.shape[0]//2),(0,img.shape[0]//2,),[255,0,0],1)
# img = annotate_image_array(img)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = IPM(img)
# plt.imshow(img)
# plt.show()
