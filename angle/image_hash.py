# -*- coding: utf-8 -*-
# 文件: image_hash.py
# 1. 特征库生成
# 1.1 准备矫正后的旋转验证码图片 (100%矫正最好)
# 1.2 使用 image_hash.py 生成特征库
# 1.3 从特征库 .pickle 文件读取全部 hash


import os
import cv2
import numpy as np
import pickle


class ImageHash(object):

    """
    预测旋转角度
    path: 特征库路径, 可选
    """

    def __init__(self, path=None):
        self.hashlib = []
        if path is not None:
            self.hashlib = self.load_hashlib(path)

    def read_all_filename(self, file_dir):
        """读取当前目录下的所有文件"""
        files = []
        for filename in os.listdir(file_dir):
            files.append(filename)
        return files

    def read_content(self, name):
        """读取文件流"""
        try:
            with open(name, 'rb') as f:
                content = f.read()
            return content
        except Exception as e:
            print(e)

    def get_images_all(self, file_dir, extensions=['.jpg', '.jpeg', '.bmp', '.png']):
        """只获取指定后缀的图片"""
        image_paths = []
        for filename in self.read_all_filename(file_dir):
            file_path = os.path.join(file_dir, filename)
            if os.path.splitext(file_path)[1].lower() in extensions:
                image_paths.append(file_path)
        return image_paths

    def write_pickle(self, texts: list, path):
        """Python对象序列化 为 文件对象"""
        with open(path, 'wb') as f:
            pickle.dump(texts, f)
        print(f"Success, write {path}")

    def read_pickle(self, path):
        """文件对象反序列化 为 Python对象"""
        with open(path, 'rb')as f:
            hash_list = pickle.load(f)
        print(f"Success, read {path}")
        return hash_list

    def delete_duplicate(self, data: list):
        return list(set(data))

    def pHash(self, image):
        """感知哈希算法"""
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

    def generate_hash(self, image_name: str):
        """生成特征字符串"""
        image = cv2.imread(image_name)
        hash = self.pHash(image)
        hash_str = ''
        for i in hash:
            hash_str += str(i)
        return hash_str

    def write_hashlib(self, file_dir, save_path):
        """批量生成特征库"""
        image_paths = self.get_images_all(file_dir)
        hashs = []
        for image_path in image_paths:
            hash_str = self.generate_hash(image_path)
            hashs.append(hash_str)
        # print("去重前, hash_list: ", len(hashs))
        dup_hashs = self.delete_duplicate(hashs)
        # print("去重后, hash_list: ", len(dup_hashs))
        self.write_pickle(dup_hashs, save_path)

    def load_hashlib(self, path):
        """加载特征库"""
        read_list = self.read_pickle(path)

        hash_list = []
        for hash in read_list:
            i_list = []
            for i in hash:
                i_list.append(int(i))
            hash_list.append(i_list)
        return hash_list


if __name__ == "__main__":
    # 这是手动矫正后的验证码图片 (100%矫正最好)的存放目录 , 用于生成特征库
    captcha_dir = 'captcha/train_captcha'

    hashlib_path = 'pickles/hashlib.pickle'
    ihash = ImageHash()
    # 生成特征库
    ihash.write_hashlib(captcha_dir, hashlib_path)
    print("保存路径: ", hashlib_path)

    # 读取特征库
    hash_list = ihash.read_pickle(hashlib_path)
    print(hash_list)
    print(len(hash_list))
