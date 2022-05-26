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
    path: 特征库路径, 可选
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
    image_path = 'captcha/test_captcha/1-127.png'
    angle = 127

    # image_path = 'captcha/test_captcha/2-323.png'
    # angle = 323

    imager = Restoration()
    imager.restoration(angle, image_path, output_img=None, show_img=True)
