# -*- coding=GBK -*-
import os
import subprocess

import cv2 as cv


def transform(x):
    a, b = 150, 240  # ����������ֵ���м䲿�ֱ任Ϊ255
    dst = x.copy()
    dst[(x[:, :] >= a) & (x[:, :] <= b)] = 255  # �м�任255������Ĳ���
    return dst
def threshold_image(image):
    # dst = transform(image)
    # binary_path= "test_image/test2.jpg"
    # cv.imwrite(binary_path,dst)
    cmd='python3 tools/infer/predict_system.py --image_dir={image_dir} --det_model_dir="./inference/ch_PP-OCRv4_det_infer/"  --rec_model_dir="./inference/rec_ppocrv3_kaiser2/Student/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True --use_gpu=False'.format(image_dir=image)
    os.system(cmd)


if __name__ == '__main__':
    src = cv.imread("./test_image/4.jpg",0)
    threshold_image("./test_image/4.jpg")

