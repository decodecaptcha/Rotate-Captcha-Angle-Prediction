# -*- coding: utf-8 -*-
# @Author : 艾登Aiden
# @Email : aidenlen@163.com
# @Date : 2022-02-09

# 1.从特征库里获取全部图片的 img_hash_all
# 2.获取一张新图片
# 3.旋转360度, 每2度生成一张图片, 共生成180张图片
# 4.计算这180张图片的hash值, 存为列表 rotate_hash_and_angle(长度180) = []
# 5.rotate_hash_and_angle 与 img_hash_all 对比
# 6.第一次对比, 从 rotate_hash_and_angle 中取第一个hash, 与 img_hash_all 对比, 按相似度排序, 取 top 20 存为 one_similarity_top = [hash1, hash2...] (长度20)
# 7.第二次对比, 遍历 rotate_hash_and_angle, 与 similarity_top20 依次对比, 存为 two_similarity_top (长度180*20=3600), 
#   按相似度排序 [(0.96xx, hash, 角度), (0.95xx, hash, 角度)]  , 取 top 1 (0.96xx, hash, 角度)
# 8.得到最大相似度的角度 angle


import json
import os
import re
import time
import cv2
import numpy as np
import pickle



def file_name(file_dir):
    """读取当前目录下的所有文件"""
    files = []
    for filename in os.listdir(file_dir):
        files.append(filename)
        # print(filename)
    return files

# 过滤文件目录和其他文件
def get_images_all(file_dir):
    """获取所有图片的路径"""
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']
    image_paths = []
    for filename in file_name(file_dir):
        file_path = os.path.join(file_dir, filename)
        if os.path.splitext(file_path)[1].lower() in extensions:
            # print(file_path)
            image_paths.append(file_path)
    return image_paths


def write_file(texts:list, save_file_path):
    """对象序列化到文件对象, 保存文件"""
    with open(save_file_path, 'wb') as f:
        pickle.dump(texts, f)
    print("write_file Success")

def read_file(save_path):
    """对象反序列化, 从文件读取数据"""
    with open(save_path, 'rb')as f:
        hash_list = pickle.load(f)
    print("read_file Success")
    return hash_list

def save_hash_file(img_paths:list, save_path):
    """保存 CaptchaSet 文件"""
    hash_list = []
    for para in img_paths:
        img1 = cv2.imread(para)
        hash1 = pHash(img1)
        hash_str = ''
        for i in hash1:
            hash_str += str(i)
        # print(hash_str)
        hash_list.append(hash_str)
    print("去重前, hash_list: ", len(hash_list))
    hash_list = list(set(hash_list))
    print("去重后, hash_list: ", len(hash_list))
    write_file(hash_list, save_path)


# 1.从特征库里获取全部图片的 img_hash_all
def read_hash_file(path):
    """保存 CaptchaSet 文件"""
    read_list = read_file(path)
    # print(read_list)
    hash_list = []
    for hash in read_list:
        i_list = []
        for i in hash:
            i_list.append(int(i))
        hash_list.append(i_list)
    # print(hash_list)
    # print(type(hash_list))
    return hash_list


# 2.获取一张新图片
def get_new_image(content):
    """new img content to image"""
    return content2image(content)


def content2image(content):
    """content to image"""
    image = np.asarray(bytearray(content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# 3.旋转360度, 每2度生成一张图片, 共生成180张图片
def rotate_bound(angle, image):
    """
    :param image: 原图像image
    :param angle: 旋转角度
    :return: 旋转后的图像image 对象
    """
    # if output_img_path == '':
    #     output_img_path = input_img_path
    # grab the dimensions of the image and then determine the center
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

    # # 裁剪坐标为[y0:y1, x0:x1]
    start_W = int((nW / 2) - 75)
    start_H = int((nH / 2) - 75)
    
    end_W = int((nH / 2) + 75)
    end_H = int((nH / 2) + 75)
    
    # print(start_W, start_H, end_W, end_H)

    cropped = img[start_W:end_W, start_H:end_H]
    # cv2.imwrite(output_img_path, cropped)
    return cropped


# 4.计算这180张图片的hash值, 存为列表 img_hash_180
def pHash(image):
    # 感知哈希算法
    # 缩放32*32
    image = cv2.resize(image, (32, 32))  # , interpolation=cv2.INTER_CUBIC

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


def cmpHash(hash1, hash2):
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


# 5.img_hash_180 与 img_hash_all 对比
def get_similarity(hash1, hash2):
    # print(hash1, hash2)
    n = cmpHash(hash1, hash2)
    # print('感知哈希算法相似度 pHash：', n)
    return 1 - (n / 64)


def get_angle(content):
    # 1.从特征库里获取全部图片的 img_hash_all
    img_hash_all = read_hash_file(CAPTCHA_SET_FILE)
    # print(img_hash_all)
    # print(len(img_hash_all))

    # 2.获取一张新图片
    # 3.旋转360度, 每2度生成一张图片, 共生成180张图片
    # 4.计算这180张图片的 hash 值, 存为列表 rotate_img_hash = [(hash, 角度)]
    rotate_hash_and_angle = []
    image = content2image(content)
    for i in range(0, 360, 1):
        rotate_image = rotate_bound(i, image)
        hash = pHash(rotate_image)
        rotate_hash_and_angle.append((hash, i))
    # print(rotate_hash_and_angle)
    # print(len(rotate_hash_and_angle))

    # rotate_hash_and_angle = []
    # img_hash_all = []
    # 5.rotate_hash_and_angle 与 img_hash_all 对比
    # 6.第一次对比, 从 rotate_hash_and_angle 中取第一个hash, 与 img_hash_all 对比, 
    #   按相似度排序, 取 top 20 存为 one_similarity_top = [hash1, hash2...] (长度20)
    
    hash_and_angle0 = rotate_hash_and_angle[0]
    # print(hash_and_angle0)
    hash0 = hash_and_angle0[0]
    # print(hash0)
    # print(len(hash0))

    one_similarity_top = []
    for img_hash in img_hash_all:
        one_simi = get_similarity(hash0, img_hash)
        one_similarity_top.append(img_hash)
    # print(one_similarity_top)

    # 升序排序
    one_similarity_top.sort()
    one_similarity_top = one_similarity_top[-1000:]
    # print(one_similarity_top)
    # print(len(one_similarity_top))


    # 7.第二次对比, 遍历rotate_img_hash, 与 similarity_top20 依次对比, 存为 two_similarity_top (长度180*20=3600),
    #   按相似度排序 [(0.96xx, hash, 角度), (0.95xx, hash, 角度)]  , 取 top 1 (0.96xx, hash, 角度), 即最大相似度的角度 angle
    two_similarity_top = []
    for hash1 in one_similarity_top:
        for hash2, angle in rotate_hash_and_angle:
            two_simi = get_similarity(hash1, hash2)
            two_similarity_top.append((two_simi, hash, angle))
    # print(two_similarity_top)
    # 相似度对比次数
    print("相似度计算次数: ", len(two_similarity_top))
    two_similarity_top.sort()
    angles = two_similarity_top[-1]
    # print(angles)
    print("相似度计算次数: ", angles[0])
    print("hash值: ", angles[1])
    angle = angles[-1]
    return angle


def create_CaptchaSet(file_dir, save_path):
    """生成特征库"""
    image_paths = get_images_all(file_dir)
    # image_paths = [
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\1.png", 
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\2.png",
    #     r"C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-similarity\3.png"
    # ]
    # print("image_paths", len(image_paths))
    save_hash_file(image_paths, save_path)
    


if __name__ == "__main__":
    st = time.time()
    
    # FILE_DIR = r'C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha2000'
    # 存储特征库
    # create_CaptchaSet(FILE_DIR, CAPTCHA_SET_FILE)
    # 读取特征库
    # hash_list = read_hash_file(CAPTCHA_SET_FILE)
    # print(hash_list)
    # print(len(hash_list))

    CAPTCHA_SET_FILE = r'C:\Users\admin\Desktop\RotateCaptchaBreak\RotateCaptchaSet.pickle'
    input_img_path = r'C:\Users\admin\Desktop\RotateCaptchaBreak\data\captcha-test50\490-259.png'
    with open(input_img_path, 'rb') as f:
        input_img_content = f.read()
    angle = get_angle(input_img_content)

    dir, filename = os.path.split(input_img_path)
    print("图片名: ", filename)
    print("预测角度: ", angle)
    print("总耗时: ", time.time() - st, "秒")