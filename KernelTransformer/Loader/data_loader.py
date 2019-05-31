#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import h5py
import numpy
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ..cfg import Config
from ..cfg import LAYERS
from ..cfg import DATA_DIR


class FeatureMapLoader(Dataset):

    def __init__(self, src_dir, dst_dir, ids):
        super(FeatureMapLoader, self).__init__()

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        feature_id = self.ids[idx]
        src = self.load_src(feature_id)
        dst = self.load_dst(feature_id)
        return src, dst

    def load_src(self, feature_id):
        src_path = os.path.join(self.src_dir, "{}.h5".format(feature_id))
        src = load_featuremap(src_path)
        src = torch.from_numpy(src)
        with torch.no_grad():
            src = F.relu(src)
        return src

    def load_dst(self, feature_id):
        dst_path = os.path.join(self.dst_dir, "{}.h5".format(feature_id))
        dst = load_featuremap(dst_path)
        dst = torch.from_numpy(dst)
        return dst


class ImageLoader(FeatureMapLoader):

    def load_src(self, feature_id):
        src_path = os.path.join(self.src_dir, "{}.jpg".format(feature_id))
        src = cv2.imread(src_path)
        # hand coded image size here
        src = cv2.resize(src, (640, 320))
        src = src - numpy.array([103.939, 116.779, 123.68])
        src = numpy.transpose(src, (2,0,1))
        dtype = Config["FloatType"]
        src = torch.from_numpy(src).type(dtype)
        return src


def load_ids(split):
    ids_path = os.path.join(DATA_DIR, "{}.txt".format(split))
    if not os.path.isfile(ids_path):
        raise IOError("{} does not exist".format(ids_path))
    ids = set()
    with open(ids_path, 'r') as fin:
        for line in fin:
            ids.add(line.rstrip())
    return ids

def load_featuremap(path):
    feature_id = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, 'r') as hf:
        feature = hf[feature_id][:]
    feature = feature.transpose([2, 0, 1])
    return feature

def merge_dataset(samples):
    srcs = []
    dsts = []
    for src, dst in samples:
        srcs.append(src)
        dsts.append(dst)
    N = len(samples)
    dtype = Config["FloatType"]

    size = list(src.size())
    size.insert(0, N)
    src_tensor = torch.FloatTensor(*size)
    srcs = torch.stack(srcs, out=src_tensor).type(dtype)

    size = list(dst.size())
    size.insert(0, N)
    dst_tensor = torch.FloatTensor(*size)
    dsts = torch.stack(dsts, out=dst_tensor).type(dtype)
    return srcs, dsts

def prepare_dataset(dst, src=None, src_cnn="pascal", batch_size=4):
    if src is None:
        i = LAYERS.index(dst)
        if i == 0:
            src = "pixel"
        else:
            src = LAYERS[i-1]
    if src == "pixel":
        CLS_LOADER = ImageLoader
    else:
        CLS_LOADER = FeatureMapLoader

    def build_directory(layer):
        if layer == "pixel":
            directory = os.path.join(DATA_DIR, layer)
        else:
            directory = os.path.join(DATA_DIR, "{0}{1}".format(src_cnn, layer))
        return directory
    src_dir = build_directory(src)
    dst_dir = build_directory(dst)
    sys.stderr.write("Read source from {}\n".format(src_dir))
    sys.stderr.write("Read target from {}\n".format(dst_dir))

    train_ids = load_ids(split="train")
    train_dataset = CLS_LOADER(src_dir, dst_dir, train_ids)
    valid_ids = load_ids(split="valid")
    valid_dataset = CLS_LOADER(src_dir, dst_dir, valid_ids)

    NUM_WORKERS = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, collate_fn=merge_dataset, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, collate_fn=merge_dataset, pin_memory=True)
    return train_loader, valid_loader

