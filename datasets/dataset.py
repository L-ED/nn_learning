import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as t

from PIL import Image

import os
import cv2
import csv
import time
import numpy as np


class GenericDataset(Dataset):
    def __init__(self, filefolder, transform):

        assert filefolder is not None

        self.filefolder = filefolder
        self.transform = transform

    def __getitem__(self, index):

        filepath, label = self.filefolder[index]

        image = np.array(Image.open(filepath))
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filefolder)


default_val_transform = t.Compose([
    t.ToTensor(), t.Resize(256), t.CenterCrop(224),
    t.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))])

default_train_transform = t.Compose([
    t.ToTensor(), t.RandomRotation((-5, 5)), t.RandomResizedCrop(224, (0.80, 1.0)),
    t.RandomHorizontalFlip(), t.RandomVerticalFlip(),
    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def generic_set(annotation, transformation=None, split=None):

    assert annotation is not None

    if transformation is None:
        if split.lower() == "train":
            transformation = default_train_transform
        elif "test" in split.lower() or "val" in split.lower():
            transformation = default_val_transform
        else:
            raise ValueError(
                "When transformation is None, split should be 'train','test' or 'val'")

    filefolder = create_folder(
        annotation
    )

    return GenericDataset(
        filefolder=filefolder,
        transform=transformation
    )


def generic_set_one_annotation(annotation_path, transformations_dict=None):

    assert annotation_path is not None

    if transformations_dict is None:
        train_transform = default_train_transform
        val_transform = default_val_transform
    else:
        train_transform = transformations_dict["train"]
        val_transform = transformations_dict["val"]

    trainfolder, valfolder = create_folders(
        annotation_path
    )

    trainset = GenericDataset(
        filefolder=trainfolder,
        transform=train_transform
    )

    valset = GenericDataset(
        filefolder=valfolder,
        transform=val_transform
    )

    return trainset, valset


def create_folder(annotation_path):
    filefolder = []
    with open(annotation_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        source_path = os.path.dirname(annotation_path)

        filefolder = list(map(
            lambda line: (
                os.path.join(source_path, line[0]),  # relativepath
                int(line[1])  # label
            ), csvreader
        ))

    return filefolder


def create_folders(annotation_path):
    trainfolder = []
    valfolder = []

    with open(annotation_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader)
        source_path = os.path.dirname(annotation_path)

        for line in csvreader:

            imgpath = os.path.join(source_path, line[0])
            class_id = int(line[1])

            if line[2] == "True":
                valfolder.append(
                    [imgpath,
                     class_id]
                )
            else:
                trainfolder.append(
                    [imgpath,
                     class_id]
                )

    return trainfolder, valfolder
