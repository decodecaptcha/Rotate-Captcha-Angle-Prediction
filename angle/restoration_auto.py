# -*- coding: utf-8 -*-
# 文件: restoration_auto.py
# 自动预测角度并恢复图片

import os
import time
import uuid
from restoration import Restoration


if __name__ == '__main__':
    st = time.time()

    hashlib_path = 'pickles/hashlib.pickle'
    file_dir = 'captcha/test_captcha'

    imager = Restoration(hashlib_path)
    image_paths = imager.get_images_all(file_dir)
    print(image_paths)

    for image_path in image_paths:
        filedir, name = os.path.split(image_path)
        output_dir = os.path.join(filedir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        uuid_name = str(uuid.uuid1())
        output_img = os.path.join(output_dir, uuid_name + '.png')
        angle = imager.get_best_angle(image_path, rotate_step=2)
        imager.restoration(angle, image_path, output_img=None, show_img=True)

    print("总耗时: ", time.time() - st, "秒")
