import cv2
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab

def test():
    print("hello world")

def capture_image():
    # 捕获摄像头图像
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ret, frame = cap.read()
    cap.release()
    return frame

def screen_shoot():
    # 如果不知道要截取屏幕的区域可以使用mouseinfo这个工具获取屏幕的像素坐标点，然后再放入bbox中
    # import mouseinfo
    # mouseinfo.mouseInfo()
    # 使用Pillow进行截图 这个区域要计算
    bbox = (2560, 0, 4000, 2560)
    pil_image = ImageGrab.grab(bbox)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image

def check_camera_orientation(frame, pattern_size=(7, 6)):
    # 检测棋盘格角点
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # 计算棋盘格的中心
        center = np.mean(corners, axis=0)
        img_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        dx = center[0][0] - img_center[0]
        dy = center[0][1] - img_center[1]

        if abs(dx) < 10 and abs(dy) < 10:
            return "Camera is correctly oriented."
        else:
            if dx > 0:
                horizontal_feedback = "Move camera to the right."
            else:
                horizontal_feedback = "Move camera to the left."

            if dy > 0:
                vertical_feedback = "Move camera upwards."
            else:
                vertical_feedback = "Move camera downwards."

            return horizontal_feedback + ' ' + vertical_feedback
    else:
        return "Cannot detect chessboard. Make sure it is visible."


def slope(image_path=None, img_frame=None):
    # 检查输入的是图片路径还是帧
    if img_frame == None and image_path != None:
        img = cv2.imread(image_path)
    elif img_frame != None and image_path == None:
        img = img_frame
    else:
        print("输入有误，检查输入")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将边界包围起来了
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = float('inf')  # vertical line
        else:
            slope = (y2 - y1) / (x2 - x1)

        slopes.append(slope)

    print(slopes)


"""
几个问题
1、像素的大小转换为实际的距离的大小 向左向右移动
2、像素的大小转换为实际的角度大小，旋转

思考： 用调整的方式来使得值在一个目标的值附近移动，直到趋于理想
3、由于旋转的装置的圆心与摄像头的光心不重合 导致在旋转的时候其他方向的也会发生变

"""


def hough(image_path=None, img_frame=np.array([])):
    # 检查输入的是图片路径还是帧
    if img_frame.size == 0 and image_path != None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img_frame.size != 0 and image_path == None:
        img = img_frame
    else:
        print("输入有误，检查输入")
        return

    # 求图片的中心位置
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2

    img_res = img[center_y - 100:center_y + 100,center_x - 100:center_x + 100]  # 感兴趣区域先是纵向，然后是横向
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("灰度图片",gray)
    # cv2.waitKey(0)
    # 将灰度图转为二值图
    thresh = 127
    _, binary_image = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # 边缘检测算法
    edges = cv2.Canny(gray, 50, 150)
    # 2.霍夫直线变换
    lines_ = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
    lines_select=lines_[:,0,1]
    count_first = lines_.shape[0]
    # 创建KMeans模型，设置聚类数为2，因为已知是两条故聚类设置为2
    kmeans = KMeans(n_clusters=2, n_init=10)
    lines = lines_select.reshape(count_first, 1)
    lines_42 = lines_.reshape(count_first, 2)

    # 拟合模型
    kmeans.fit(lines)
    # 获取聚类标签
    labels = kmeans.labels_
    theta_sum_1 = 0
    roh_sum_1 = 0
    count_1 = 0
    theta_sum_0 = 0
    roh_sum_0 = 0
    count_0 = 0
    for i, label in enumerate(labels):
        # if label not in clusters:
        #     clusters[label] = []
        # clusters[label].append(lines[i])
        if label == 1:
            theta_sum_1 += lines_42[i][1]
            roh_sum_1 += lines_42[i][0]
            count_1 += 1
        else:
            theta_sum_0 += lines_42[i][1]
            roh_sum_0 += lines_42[i][0]
            count_0 += 1
    # 获得旋转平均值
    theta_mean_1 = theta_sum_1 / count_1
    theta_mean_0 = theta_sum_0 / count_0
    # 获取roh的平均值
    roh_mean_0 = roh_sum_0 / count_0
    roh_mean_1 = roh_sum_1 / count_1
    # 旋转调整
    print("顺时针旋转" + str(min(theta_mean_0, theta_mean_1) * 180 / np.pi) + "度")

    # 求两条直线的交点的坐标
    A = np.array([[np.cos(theta_mean_1), np.sin(theta_mean_1)], [np.cos(theta_mean_0), np.sin(theta_mean_0)]])
    b = np.array([roh_mean_1, roh_mean_0])
    res_equation = np.linalg.solve(A, b)
    intersection_x = res_equation[0]
    intersection_y = res_equation[1]

    height_res, width_res, _ = img_res.shape
    center_res_x = width_res // 2
    center_res_y = height_res // 2

    # 如果要将摄像头的中心位置移动到交点
    x_distance = intersection_x - center_res_x  # 向左移动像素点
    y_distance = intersection_y - center_res_y  # 向下移动像素点
    print("向左移动" + str(x_distance) + "个像素点" + '\n' + "向下移动" + str(y_distance) + "个像素点")

    # 调整相机的pitch角和yaw角，根据图片的横纵中轴线位置上下的像素点的个数来推断 用百分比的差
    up_row = binary_image[:center_y]
    down_row = binary_image[center_y:]
    left_column = binary_image[:, :center_x]
    right_column = binary_image[:, center_x:]

    up_count = np.sum(up_row > 100)
    down_count = np.sum(down_row > 100)
    left_count = np.sum(left_column > 100)
    right_count = np.sum(right_column > 100)

    delta_up_down_count = up_count - down_count
    delta_left_right_count = left_count - right_count
    sum_255 = up_count + down_count

    print("pitch角需要向下转" + str(delta_up_down_count / sum_255 * 100) + '%' + '\n' + "yaw角需要向右转" + str(
        delta_left_right_count / sum_255 * 100) + '%' + '\n')

    # 最后调整相机的远近 这个要看需求让相机的fov覆盖屏幕多少范围


if __name__ == "__main__":
    while True:
        #从本地读取图片
        calibration_image_path = r"calibration_picture.jpg"
        hough(image_path=calibration_image_path)

        # 从摄像头读取图片
        # camera_frame = capture_image()
        # cv2.imshow("camera", camera_frame)
        # hough(img_frame=camera_frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        time.sleep(0.2)
