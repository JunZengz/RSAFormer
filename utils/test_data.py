import os
from PIL import Image
import torchvision.transforms as transforms

class test_dataset:
    def __init__(self, image_root, gt_root, mode=None):
        self.mode = mode
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((352, 352)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((352, 352))])
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.png'))
        if self.mode is None:
            image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index] + '.png'))
        elif self.mode == 'evaluate_train':
            image = self.rgb_loader(os.path.join(self.image_root, self.img_list[self.index] + '.png'))
            image = self.transform(image).unsqueeze(0)
            trans_gt = self.gt_transform(gt).unsqueeze(0)
            self.index += 1
            return image, gt, trans_gt

        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

