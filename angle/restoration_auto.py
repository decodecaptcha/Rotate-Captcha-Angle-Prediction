# -*- coding: utf-8 -*-
# 文件: restoration_auto.py
# 4. 自动预测角度并恢复图片

import os
import time
import uuid
# from restoration import Restoration
# -*- coding: utf-8 -*-
# 文件: restoration.py
# 3. 预测角度并恢复图片

import os
import time
import uuid
import cv2
from predict_angle import PredictAngle
from image_hash import ImageHash


class Restoration(PredictAngle, ImageHash):

    """
    旋转图像恢复
    """

    def __init__(self, path=None):
        self.hashlib = []
        if path is not None:
            self.hashlib = self.load_hashlib(path)

    def restoration(self, angle, input_img, output_img=None, show_img: bool = False):
        """图像恢复

        :param angle: 预测角度
        :param input_img: _description_
        :param output_img: _description_, defaults to None
        :param show_img: 是否显示图像
        """

        filedir, name = os.path.split(input_img)
        print("图像名: ", name)
        print("预测角度: ", angle)
        print("图像保存路径: ", output_img)

        angle = int(angle)
        content = self.read_content(input_img)
        image_obj = self.content2image(content)
        cropped_image_obj = self.rotate_image(angle, image_obj)

        if output_img is not None:
            # 保存image
            cv2.imwrite(output_img, cropped_image_obj)

        if show_img:
            # 显示image
            cv2.imshow('img', cropped_image_obj)
            cv2.waitKey()


if __name__ == '__main__':
    st = time.time()

    hashlib_path = os.path.join(os.path.dirname(__name__), 'pickles', 'hashlib.pickle')
    print(hashlib_path)
    file_dir = os.path.join(os.path.dirname(__name__), 'captcha', 'test_captcha')
    print(file_dir)

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
