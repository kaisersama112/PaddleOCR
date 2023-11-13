"""
import cv2
import numpy as np


def transform1(x):
    a, b = 130, 240  # 定义两个阈值，中间部分变换为255
    dst = x.copy()
    dst[(x[:, :] >= a) & (x[:, :] <= b)] = 0  # 中间变换为255
    dst[(x[:, :] < a) | (x[:, :] > b)] = 255  # 其余的变换为0
    return dst


def transform2(x):
    a, b = 150, 240  # 定义两个阈值，中间部分变换为255
    dst = x.copy()
    dst[(x[:, :] >= a) & (x[:, :] <= b)] = 255  # 中间变换255，其余的不变
    return dst


gray = cv2.imread('./doc/imgs/test1.jpg', 0)
dst1 = transform1(gray)
dst2 = transform2(gray)
hh = np.hstack((gray, dst1, dst2))
cv2.imshow('img', hh)
cv2.imwrite('./doc/imgs/imagetest1.png', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
"""
import subprocess




result = subprocess.run([
    'python', './ppstructure/table/predict_table.py',
    '--det_model_dir','./inference/ch_PP-OCRv4_det_server_infer',
    '--rec_model_dir','./inference/rec_ppocrv3_kaiser2/Student',
    '--table_model_dir','./inference/ch_ppstructure_mobile_v2.0_SLANet_infer',
    '--rec_char_dict_path','./ppocr/utils/ppocr_keys_v1.txt',
    '--table_char_dict_path','./ppocr/utils/dict/table_structure_dict_ch.txt',
    '--vis_font_path','../doc/fonts/chinese_cht.ttf',
    '--image_dir','./test_image/4.jpg',
    '--output','./output'
])
print(result)

"""
表单识别

--det_model_dir=./inference/ch_PP-OCRv4_det_infer
--rec_model_dir=./inference/ch_PP-OCRv4_rec_infer
--table_model_dir=./inference/ch_ppstructure_mobile_v2.0_SLANet_infer
--rec_char_dict_path=./ppocr/utils/ppocr_keys_v1.txt
--table_char_dict_path=./ppocr/utils/dict/table_structure_dict_ch.txt
--image_dir=./doc/imgs/testdemo1.jpg
--output=./output

"""
"""
TableRec-RARE模型
--det_model_dir=./inference/ch_PP-OCRv4_det_server_infer
--rec_model_dir=./inference/ch_PP-OCRv4_rec_server_infer
--table_model_dir=./inference/ch_ppstructure_mobile_v2.0_SLANet_infer
--rec_char_dict_path=./ppocr/utils/ppocr_keys_v1.txt
--table_char_dict_path=./ppocr/utils/dict/table_structure_dict.txt,
--image_dir=./doc/imgs/test1.jpg
--output=./output
--merge_no_span_structure=False


RE
python3 kie/predict_kie_token_ser_re.py \
  --kie_algorithm=LayoutXLM \
  --re_model_dir=../inference/re_vi_layoutxlm_xfund_infer \
  --ser_model_dir=../inference/ser_vi_layoutxlm_xfund_infer \
  --use_visual_backbone=False \
  --image_dir=../doc/imgs/1024.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
"""
