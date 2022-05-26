旋转验证码角度预测
=======
### 简介

1. 本项目可对旋转验证码角度预测, 可绕过常见的旋转类型验证码, 如百度旋转验证码

2. 项目基于 图像特征库, 是绕过旋转验证码的一种方案

3. 图像特征库 .pickle文件 采用 图像感知哈希算法 构建

### 依赖

    安装依赖: 
```bash
pip install -r requirements.txt
```

* 支持版本: ![](https://img.shields.io/badge/Python-3.6+-blue.svg)

### 原理

    文件: image_hash.py

1. 特征库生成
1.1 准备矫正后的旋转验证码图片 (100%矫正最好)
1.2 使用 image_hash.py 生成特征库
1.3 从特征库 .pickle 文件读取全部 hash


    文件: predict_angle.py

2. 预测角度实现过程:
2.1 获取一张倾斜的验证码
2.2 把这张倾斜的验证码 旋转360度, 每x度生成一张图片, 当x=2时, 一共生成180张图片
2.3 计算这180张图片的hash值 与 特征库 中的 hash 存为 列表[(0.96xx, 角度), (0.95xx, 角度), xxx ]  (长度180*20=3600),
    最后按相似度排序 [(0.96xx, 角度), (0.95xx, 角度)]  , 取 top 1 (0.96xx, 角度)
2.6 得到最大相似度的角度 (0.96xx, 角度)

    文件: restoration.py

3. 预测角度并恢复图片

    文件: restoration_auto.py

4. 自动预测角度并恢复图片

    生产环境实测 预测角度准确率在 70-90%


### 如何使用
```bash
git clone git@github.com:aiden2048/Rotate-Captcha-Angle-Prediction.git
```

```bash
cd Rotate-Captcha-Angle-Prediction
```


```bash
pip install -r requirements.txt
```

运行 angle/restoration.py


### 报错和解决

    欢迎在 Issues 中提交 bug (或新功能)描述，我会尽力改进
