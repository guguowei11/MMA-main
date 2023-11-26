import glob, os, torch
import random
from PIL import Image
from utils import *
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F 
import matplotlib.pyplot as plt


def imagestransforms(img, label):
    img = transforms.Resize(224, interpolation=Image.BILINEAR)(img)
    label = transforms.Resize(224, interpolation=Image.NEAREST)(label)

    # 以0.2的概率进行,随机旋转，水平翻转，垂直翻转操作
    if random.random() < 0.2:
        img = F.hflip(img)
        label = F.hflip(label)

    if random.random() < 0.2:
        img = F.vflip(img)
        label = F.vflip(label)

    if random.random() < 0.2:
        angle = transforms.RandomRotation.get_params([-180, 180])
        img = img.rotate(angle, resample=Image.BILINEAR)
        label = label.rotate(angle, resample=Image.NEAREST)

    img = F.to_tensor(img).float()
    return img, label


class Cloud_Data(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, csv_path, mode='train'):
        '''
        继承Dataset类,加载数据
        :param img_path: 图片路径
        :param label_path: 标签路径
        :param csv_path: colormap(每一个类的RGB值)
        :param mode:
        :param transform: 数据增强
        '''
        super().__init__()
        self.mode = mode
        self.img_list = glob.glob(os.path.join(img_path,  '*.png'))  # 读取所有png文件
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))

        self.label_colormap = get_label_colormap(csv_path)


    def __getitem__(self, index):
        '''
        把rgb图转化为标签图
        :param index:
        :return: torch类型的img和label
        '''
        img = Image.open(self.img_list[index]).convert('RGB')  # 这里.convert('RGB')可加可不加 把图片转化成RGB模式
        label = Image.open(self.label_list[index]).convert('RGB')

        if self.mode == 'train':
            img , label = imagestransforms(img, label)
        else:
            img = transforms.Resize(224, interpolation=Image.BILINEAR)(img)
            img = transforms.ToTensor()(img).float()
            label = transforms.Resize(224, interpolation=Image.NEAREST)(label)

        label = image2label(label, self.label_colormap)
        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.img_list)






