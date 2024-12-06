# 等比缩放图片
import os

from PIL import Image
import shutil
import numpy as np

import yaml
from easydict import EasyDict as ed
from email.mime.text import MIMEText

def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)

    # print(img.size)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    # print("mask:", mask.size)
    # print("img:", img.size)
    mask = mask.resize(size)
    return mask


# 移动文件
def move_file(src, dst):
    # shutil.move(src, dst)
    # print("1 :",os.listdir(src))
    for file in os.listdir(src):
        path = os.path.join(src, file)  # 文件夹路径
        print("file: ", file)
        for file_1 in os.listdir(path):
            path_1 = os.path.join(path, file_1)  # 图片路径
            dst_images = os.path.join(dst, "images")
            dst_labels = os.path.join(dst, "labels")
            if "mask" in file_1:
                shutil.move(path_1, dst_labels)
            else:
                shutil.move(path_1, dst_images)

            # print(file_1)


def insert_str(src, str):  # str: 要插入的值
    list_i = list(src)  # str -> list
    list_i.insert(-4, str)
    str_i = ''.join(list_i)  # list -> str

    return str_i


if __name__ == '__main__':
    keep_image_size_open("./data/2.jpg")
