# -*- coding: utf-8 -*-
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import shutil
import itertools
# shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))

import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from models import MHE

from tqdm import tqdm

import json

import imageio


class CustomImageDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.transform = transform

        with open(
                os.path.join(root, "transforms_{}.json".format(split)), "r"
        ) as fp:
            meta = json.load(fp)

        self.frames = []

        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = os.path.join(root, frame["file_path"] + ".png")

            self.frames.append(fname)

        self.transform = transforms.Compose([
            # transforms.Scale(256),
            transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.frames)

    @torch.no_grad()
    def __getitem__(self, index):
        fname = self.frames[index]
        img = Image.open(fname)
        img = self.transform(img)

        color_bkgd = torch.ones(3, 1, 1)
        pixels, alpha = torch.split(img, [3, 1], dim=0)
        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)

        return pixels


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

    n_views = 3

    train_dataset = CustomImageDataset(root="./data/nerf_synthetic/lego/",
                                       split="train",
                                       transform=None)

    train_loader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)

    resnet50_feature_extractor = models.resnet50(pretrained=True)
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    N = len(train_dataset)
    features = torch.zeros((N, 2048))

    resnet50_feature_extractor = resnet50_feature_extractor.cuda()

    for idx, data in enumerate(train_loader):
        data = data.cuda()
        feat = resnet50_feature_extractor(data)
        features[idx, :] = feat

    indices = list(range(0, N))
    com_indices = list(itertools.combinations(indices, n_views))

    losses = []

    mhe_loss = MHE(n_views)

    for i in tqdm(com_indices):
        selected_features = features[i, :]
        loss = mhe_loss(selected_features)

        losses.append(loss.item())

    from operator import itemgetter
    max_idx, max_loss = max(enumerate(losses), key=itemgetter(1))
    min_idx, min_loss = min(enumerate(losses), key=itemgetter(1))
    max_item = com_indices[max_idx]
    min_item = com_indices[min_idx]

    print(max_loss, max_item, min_loss, min_item)
