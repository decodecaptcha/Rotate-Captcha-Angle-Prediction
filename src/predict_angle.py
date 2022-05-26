# -*- coding: utf-8 -*-
# @Author : 艾登Aiden
# @Email : aidenlen@163.com
# @Date : 2022-02-09

# 文件: predict_angle.py
# 2.预测角度实现过程:
# 2.1 获取一张倾斜的验证码
# 2.2 把这张倾斜的验证码 旋转360度, 每x度生成一张图片, 当x=2时, 一共生成180张图片
# 2.3 计算这180张图片的hash值 与 特征库 中的 hash 存为 列表[(0.96xx, 角度), (0.95xx, 角度), xxx ]  (长度180*20=3600),
#     最后按相似度排序 [(0.96xx, 角度), (0.95xx, 角度)]  , 取 top 1 (0.96xx, 角度)
# 2.6 得到最大相似度的角度 (0.96xx, 角度)


import os
import time
import cv2
import numpy as np
from image_hash import ImageHash

class PredictAngle(ImageHash):

    """
    预测旋转角度
    path: 特征库路径, 可选
    """

    def __init__(self, path=None):
        self.hashlib = []
        if path is not None:
            self.hashlib = self.load_hashlib(path)

    def content2image(self, content):
        """转 image 对象"""
        image = np.asarray(bytearray(content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def rotate_image(self, angle, image):
        """
        :param image: 原图像image
        :param angle: 旋转角度
        :return: 旋转后的图像image 对象
        """
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵以考虑平移
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # borderValue 缺失背景填充色彩，此处为白色，可自定义
        img = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

        # # 裁剪坐标为[y0:y1, x0:x1]
        start_W = int((nW / 2) - 75)
        start_H = int((nH / 2) - 75)

        end_W = int((nH / 2) + 75)
        end_H = int((nH / 2) + 75)

        cropped_image = img[start_W:end_W, start_H:end_H]
        # cv2.imwrite(output_img_path, cropped)
        return cropped_image

    def pHash(self, image):
        # 感知哈希算法
        # 缩放32*32
        image = cv2.resize(image, (32, 32))

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 将灰度图转为浮点型，再进行dct变换
        dct = cv2.dct(np.float32(gray))
        # opencv实现的掩码操作
        dct_roi = dct[0:8, 0:8]

        hash = []
        avreage = np.mean(dct_roi)
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    def cmpHash(self, hash1, hash2):
        # Hash值对比
        # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
        # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
        # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            print("hash长度不同, 传参出错, -1")
            return -1
        # 遍历判断
        # print(len(hash1))
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        return n

    def get_same(self, hash1, hash2):
        """计算相似度"""
        # print(hash1, hash2)
        n = self.cmpHash(hash1, hash2)
        # print('感知哈希算法相似度 pHash：', n)
        similarity = 1 - (n / 64)
        return similarity

    def create_hashlib(self, file_dir, save_path):
        """生成特征库"""
        image_paths = self.get_images_all(file_dir)
        self.save_hashlib(image_paths, save_path)

    def get_best_angle(self, image_filename, rotate_step: int):
        """
        获取最佳角度
        :image_filename: 图片路径
        :rotate_step: 图片旋转360度, 每 rotate_step 度生成一张图片, 
                        rotate_step 越小越精确, 最小为整数1
        """
        same_and_angles = []
        content = self.read_content(image_filename)
        image = self.content2image(content)
        for angle in range(0, 360, rotate_step):
            sames = []
            new_image = self.rotate_image(angle, image)
            image_hash = self.pHash(new_image)
            for hash in self.hashlib:
                same = self.get_same(image_hash, hash)
                sames.append(same)
            sames.sort()
            same_and_angles.append((sames[-1], angle))

        same_and_angles.sort()
        best_same, best_angle = same_and_angles[-1]
        return best_angle


if __name__ == "__main__":
    st = time.time()

    hashlib_path = 'pickles/hashlib.pickle'
    image_filename = 'captcha/test_captcha/1-127.png'
    # image_filename = 'captcha/test_captcha/2-323.png'

    ang = PredictAngle(hashlib_path)
    best_angle = ang.get_best_angle(image_filename, rotate_step=2)
    # print(best_angle)
    dir, filename = os.path.split(image_filename)
    print("图片名: ", filename)
    print("预测角度: ", best_angle)

    print("总耗时: ", time.time() - st, "秒")