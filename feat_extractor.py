# -*- coding: utf-8 -*-
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil

data_dir = './train'
features_dir = './Resnet_features'
# shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))

import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()
    np.savetxt(saved_path, y, delimiter=',')


if __name__ == '__main__':

    training_data = CustomImageDataset(
        root="./data/nerf_synthetic/",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    files_list = []
    x = os.walk(data_dir)
    for path,d,filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))

    print(files_list)
    resnet50_feature_extractor = models.resnet50(pretrained=True)
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()

    for x_path in files_list:
        print("x_path" + x_path)
        file_name = x_path.split('/')[-1]
        fx_path = os.path.join(features_dir, file_name + '.txt')
        print(fx_path)
        extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)