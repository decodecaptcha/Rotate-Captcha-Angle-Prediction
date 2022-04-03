# -*- coding: utf-8 -*-
# @Author : 艾登Aiden
# @Email : aidenlen@163.com
# @Date : 2022-01-24

import cv2
import random
import numpy as np
from math import *
import os
import uuid


# 读取当前目录下的所有文件
def file_name(file_dir):
    files = []
    for filename in os.listdir(file_dir):
        files.append(filename)
        # print(filename)
    return files


def rotate_bound(angle, input_img_path, output_img_path=''):
    """
    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # input_img_path = 'C:/Users/admin/Desktop/RotateCaptchaBreak/data/captcha3/3-127.png'
    # output_img_path = 'C:/Users/admin/Desktop/RotateCaptchaBreak/data/captcha3/5-127.png'
    if output_img_path == '':
        output_img_path = input_img_path
    image = cv2.imread(input_img_path)
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    #计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # print('new W: ', nW)
    # print('new H: ', nH)

    #调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    img = cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))
    # perform the actual rotation and return the image
    # output_img_path = 'C:/Users/admin/Desktop/RotateCaptchaBreak/data/captcha3/2-250-.png'
    cv2.imwrite(output_img_path, img)

    # # 裁剪坐标为[y0:y1, x0:x1]
    start_W = int((nW / 2) - 75)
    start_H = int((nH / 2) - 75)
    
    end_W = int((nH / 2) + 75)
    end_H = int((nH / 2) + 75)
    
    print(start_W, start_H, end_W, end_H)

    cropped = img[start_W:end_W, start_H:end_H]
    cv2.imwrite(output_img_path, cropped)


if __name__ == '__main__':

    # 导入图片目录
    file_dir = 'C:/Users/admin/Desktop/RotateCaptchaBreak/data/captcha'

    # 根据导入目录生成导出目录
    output_dir = os.path.join(file_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extensions = ['.jpg', '.jpeg', '.bmp', '.png']
    image_paths = []
    angles = []

    for filename in file_name(file_dir):
        file_path = os.path.join(file_dir, filename)
        if os.path.splitext(file_path)[1].lower() in extensions:
            # print(file_path)
            image_paths.append(file_path)
            angle = filename.split('-')[1].split('.')[0]
            angles.append(angle)

    for input_path, angle in zip(image_paths, angles):
        # print(input_path)
        # print(angle)
        angle = int(angle)
        # angle = 127
        # uuid_name = str(uuid.uuid1())
        uuid_name = str(uuid.uuid4())
        output_img_path = os.path.join(output_dir, uuid_name + '.png')
        # print(output_img_path)
        rotate_bound(angle, input_path, output_img_path)