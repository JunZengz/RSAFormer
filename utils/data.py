from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
import cv2
import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


class MyDataset(Dataset):
    def __init__(self, path, transforms, augmentation=None):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'masks'))
        self.transforms = transforms
        self.class_rgb_values = [[255, 255, 255], [0, 0, 0]]
        self.class_name = ['polyp', 'background']
        self.augmentation = augmentation
        self.randomflip = RandomFlip()
        self.randomrotate = RandomRotate()


        # print("path && name")
        # print(self.path)
        # print(self.name)

    def __len__(self):
        return  len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]     # xx.png
        mask_path = os.path.join(self.path, 'masks', segment_name)
        image_path = os.path.join(self.path, 'images', segment_name)
        mask = Image.open(mask_path).convert('L')
        image = Image.open(image_path).convert('RGB')
        shape = mask.size

    # Data Augmentation
        if self.augmentation is not None:
            mask = np.asarray(mask)
            image = np.asarray(image)
            pair = self.augmentation(image=image, mask=mask)
            image, mask = pair['image'], pair['mask']

        return self.transforms['image'](image), self.transforms['mask'](mask), shape

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:, ::-1], mask[:, ::-1]
        else:
            return image, mask

class RandomRotate(object):
    def __call__(self, image, mask):
        degree = 10
        rows, cols, channels = image.shape
        random_rotate = random.random() * 2 * degree - degree

        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), random_rotate, 1)
        '''
        第一个参数：旋转中心点
        第二个参数：旋转角度
        第三个参数：缩放比例
        '''
        image = cv2.warpAffine(image, rotate, (cols, rows))
        mask = cv2.warpAffine(mask, rotate, (cols, rows))
        # contour = cv2.warpAffine(contour, rotate, (cols, rows))

        return image, mask



