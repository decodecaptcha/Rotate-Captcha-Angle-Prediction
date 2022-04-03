# -*- coding: utf-8 -*-
# @Author : 艾登Aiden
# @Email : aidenlen@163.com
# @Date : 2022-01-26
import json
import re
import cv2
import numpy as np
import pickle


def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    # print(len(hash1))
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def get_similarity(hash1, hash2):
    print(hash1, hash2)
    n = cmpHash(hash1, hash2)
    # print('感知哈希算法相似度 pHash：', n)
    return 1 - (n / 64)


# def write_file(texts:list, save_file_path):
#     """对象序列化到文件对象, 保存文件"""
#     with open(save_file_path, 'wb') as f:
#         pickle.dump(texts, f)
#     print("write_file Success")


# def read_file(save_path):
#     """对象反序列化, 从文件读取数据"""
#     with open(save_path, 'rb')as f:
#         hash_list = pickle.load(f)
#     print("read_file Success")
#     return hash_list
    

# def save_hash_file(img_paths:list, save_path):
#     """保存 CaptchaSet 文件"""
#     hash_list = []
#     for para in img_paths:
#         img1 = cv2.imread(para)
#         hash1 = pHash(img1)
#         hash_str = ''
#         for i in hash1:
#             hash_str += str(i)
#         print(hash_str)
#         hash_list.append(hash_str)
#     write_file(hash_list, save_path)


# def read_hash_file(path):
#     """保存 CaptchaSet 文件"""
#     read_list = read_file(path)
#     # print(read_list)
#     hash_list = []
#     for hash in read_list:
#         i_list = []
#         for i in hash:
#             i_list.append(int(i))
#         hash_list.append(i_list)
#     print(hash_list)
#     print(type(hash_list))
#     return hash_list

if __name__ == "__main__":
    pass
    # para1 = r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\61.png"
    # para2= r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\60.png"
    # img1 = cv2.imread(para1)
    # img2 = cv2.imread(para2)
    # hash1 = pHash(img1)
    # hash2 = pHash(img2)
    # n3 = similarity(hash1, hash2)
    # print("两张图片相似度为:%s" % n3)

    # save_path = r'C:\Users\admin\Desktop\RotateCaptchaBreak\data\CaptchaSet.txt'
    # image_paths = [
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\1.png", 
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\2.png",
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\3.png"
    # ]
    # save_hash_file(image_paths, save_path)

    # hash_list = read_hash_file(save_path)
    # n3 = get_similarity(hash_list[0], hash_list[1])
    # print("两张图片相似度为:%s" % n3)