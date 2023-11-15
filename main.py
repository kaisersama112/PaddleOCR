import os

import cv2
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch", rec_model_dir="./inference/ch_PP-OCRv4_rec_infer")


class TableOCR(object):
    def __init__(self, result_file):
        self.img_file = ''
        if not result_file:
            self.result_file = 'to_excel/result.xlsx'
        else:
            self.result_file = result_file

    def get_sorted_rect(self, rect):
        '''
        获取排序的四个坐标
        @param rect:
        @return:按照左上 右上 右下 左下排列返回
        '''
        mid_x = (max([x[1] for x in rect]) - min([x[1] for x in rect])) * 0.5 + min([x[1] for x in rect])  # 中间点坐标
        left_rect = [x for x in rect if x[1] < mid_x]
        left_rect.sort(key=lambda x: (x[0], x[1]))
        right_rect = [x for x in rect if x[1] > mid_x]
        right_rect.sort(key=lambda x: (x[0], x[1]))
        sorted_rect = left_rect[0], left_rect[1], right_rect[1], right_rect[0]

        return sorted_rect

    def get_table(self, gray, min_table_area=0):
        '''
        从灰度图获取表格坐标，[[右下→左下→左上→右上],..]
        边缘检测+膨胀---》找最外围轮廓点，根据面积筛选---》根据纵坐标排序---》计算轮廓的四个点，再次筛选
        @param gray:灰度图 如果是二值图会报错
        @return:image
        '''
        canny = cv2.Canny(gray, 200, 255)  # 第一个阈值和第二个阈值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        canny = cv2.dilate(canny, kernel)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not min_table_area:
            min_table_area = gray.shape[0] * gray.shape[1] * 0.01  # 50000  # 最小的矩形面积阈值
        candidate_table = [cnt for cnt in contours if cv2.contourArea(cnt) > min_table_area]  # 计算该轮廓的面积
        candidate_table = sorted(candidate_table, key=cv2.contourArea, reverse=True)
        area_list = [cv2.contourArea(cnt) for cnt in candidate_table]
        table = []
        for i in range(len(candidate_table)):
            # 遍历所有轮廓
            # cnt是一个点集
            cnt = candidate_table[i]
            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)
            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            sorted_box = self.get_sorted_rect(box)
            result = [sorted_box[2], sorted_box[3], sorted_box[0], sorted_box[1]]  # 右下 左下 左上 右上
            result = [x.tolist() for x in result]
            table.append(result)

        return table

    def perTran(self, image, rect):
        '''
        做透视变换
        @params:image 图像
                rect  四个顶点位置:左上 右上 右下 左下
        @return:image
        '''
        tl, tr, br, bl = rect
        # 计算宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # 计算高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # 定义变换后新图像的尺寸
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype='float32')
        # 变换矩阵
        rect = np.array(rect, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        # 透视变换
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def fastNlMeansDenoisingColored(self, merge, nums=30):
        converted_img2 = cv2.cvtColor(merge, cv2.COLOR_GRAY2BGR)
        merge = cv2.fastNlMeansDenoisingColored(converted_img2, None, nums, nums, 7, 21)
        merge = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)
        return merge

    def sort_cells(self, cells_coordinates, vertical_error=7):
        sorted_cells = sorted(cells_coordinates, key=lambda x: x['y'])
        rows = []
        current_row = [sorted_cells[0]]
        for cell in sorted_cells[1:]:
            # 允许一定的垂直误差
            if abs(cell['y'] - current_row[-1]['y']) <= vertical_error:
                current_row.append(cell)
            else:
                rows.append(current_row)
                current_row = [cell]
        rows.append(current_row)
        sorted_rows = [sorted(row, key=lambda x: x['x']) for row in rows]
        return sorted_rows

    def recognize_bgkx(self, binary):
        rows, cols = binary.shape
        scale = 40
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated_col = cv2.dilate(eroded, kernel, iterations=1)
        contours_col, _ = cv2.findContours(dilated_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours_col = [cnt for cnt in contours_col if cv2.arcLength(cnt, True) > 100]
        scale = 30
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated_row = cv2.dilate(eroded, kernel, iterations=1)
        cv2.imshow("dilated_row", dilated_row)
        cv2.waitKey()
        contours_row, _ = cv2.findContours(dilated_row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours_row = [cnt for cnt in contours_row if cv2.arcLength(cnt, True) > 100]

        dilated_col_filtered = np.zeros_like(dilated_col)
        cv2.drawContours(dilated_col_filtered, filtered_contours_col, -1, (255), thickness=1)
        dilated_row_filtered = np.zeros_like(dilated_row)
        cv2.drawContours(dilated_row_filtered, filtered_contours_row, -1, (255), thickness=1)
        print(dilated_col_filtered)

        # bitwise_and = cv2.bitwise_and(dilated_col_filtered, dilated_row_filtered)
        bitwise_and = cv2.bitwise_and(dilated_col_filtered, dilated_row_filtered)
        kernel = np.ones((3, 3), np.uint8)
        bitwise_and = cv2.dilate(bitwise_and, kernel, iterations=1)
        bitwise_and = cv2.dilate(bitwise_and, kernel, iterations=2)  # 膨胀

        # 标识表格轮廓
        merge = cv2.add(dilated_col_filtered, dilated_row_filtered)
        eroded = cv2.dilate(merge, kernel, iterations=1)
        merge = cv2.erode(eroded, kernel, iterations=1)
        merge = cv2.dilate(merge, kernel, iterations=2)
        cv2.imshow("merge",merge)
        cv2.waitKey()
        contours, hierarchy = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cells_coordinates = []
        img = cv2.imread("../gray_z.png")
        for i, contour in enumerate(contours):
            hierarchy_info = hierarchy[0][i]
            if hierarchy_info[2] == -1:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                print(area)
                if area > 150:
                    cells_coordinates.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'original_position': (x, y, x + w, y + h)
                    })
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.imwrite('../marked_image.png', img)
        sorted_cells = self.sort_cells(cells_coordinates)
        return sorted_cells

    def recognize_text_by_loop(self, gray, cells_coordinates):
        print(cells_coordinates)
        data = [[] for i in range(len(cells_coordinates))]
        for i in range(len(cells_coordinates)):
            for j in range(len(cells_coordinates[i])):
                item = cells_coordinates[i][j]
                cell_region = gray[item['y']:item['y'] + item['height'],
                              item['x']:item['x'] + item['width']]
                # cv2.imshow('cell_region', cell_region)
                # cv2.waitKey()
                cv2.imwrite("../test_image1/{}.png".format("image" + str(i) + str(j)), cell_region)
                result = ocr.ocr(cell_region)
                text = ""
                if result[0]:
                    if len(result) > 1:
                        for resulti in result:
                            text += resulti[-1][0]
                    else:
                        for resulti in result:
                            for resultj in resulti:
                                text += resultj[-1][0]
                    data[i].append(text)
                else:
                    data[i].append("")
        print(data)
        df = pd.DataFrame(data)
        print(df)
        return df

    def ocr(self, img_file):
        self.img_file = img_file
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = self.get_table(gray)
        if not rect:  # 检查rect是否为空
            print("No table found in the image.")
            return
        print(rect)
        sorted_rect = self.get_sorted_rect(rect[0])
        gray_z = self.perTran(gray, sorted_rect)
        cv2.imwrite('../gray_z.png', gray_z)
        binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, -5)
        cells_coordinates = self.recognize_bgkx(binary_z)
        df = self.recognize_text_by_loop(gray_z, cells_coordinates)
        # print(self.result_file)
        df.to_excel(self.result_file, index=False, header=False)


tableOCR = TableOCR(result_file='to_excel/132.xlsx')
tableOCR.ocr('./test_image/132.jpg')
