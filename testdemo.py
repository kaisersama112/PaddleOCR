import cv2
import numpy as np


def extract_cell_coordinates(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 将图像转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)  # 膨胀
    # 使用阈值处理将图像转换为二值图像
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓及其层次结构
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells_coordinates = []

    # 遍历轮廓
    for i, contour in enumerate(contours):
        # 获取轮廓的层次结构信息
        hierarchy_info = hierarchy[0][i]

        # 如果轮廓没有子轮廓，说明是最底层的单元格
        if hierarchy_info[2] == -1:
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤掉小面积的轮廓，以排除噪音
            area = cv2.contourArea(contour)
            print(area)
            if area > 500:  # 调整阈值以适应实际情况
                # 存储每个单元格的坐标
                cells_coordinates.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
                color = tuple(np.random.randint(0, 255, 3).tolist())
                # 在原始图像上绘制矩形
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    # 保存带有矩形标记的图像
    cv2.imwrite('../marked_image.png', img)

    return cells_coordinates


# 替换为你的图像路径
image_path = '../merge.png'

# 获取所有单元格的坐标
cells_coordinates = extract_cell_coordinates(image_path)
print(cells_coordinates)