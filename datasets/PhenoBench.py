"""
Dataset parser
"""

import glob
import os
import random

import cv2
import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils.transforms import Transforms


class PhenoBench(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg["data"]["root_dir"]
        self.setup()

    def setup(self, stage=None):
        self.data_train = Data(self.root_dir, "train")
        self.data_val = Data(self.root_dir, "val")
        self.data_test = Data(self.root_dir, "test")

    def train_dataloader(self):
        loader = DataLoader(
            self.data_train,
            batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"]["n_gpus"],
            num_workers=self.cfg["train"]["workers"],
            pin_memory=True,
            shuffle=True,
        )
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.data_val,
            batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"]["n_gpus"],
            num_workers=self.cfg["train"]["workers"],
            pin_memory=True,
            shuffle=False,
        )
        self.len = self.data_val.__len__()
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.data_test,
            batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"]["n_gpus"],
            num_workers=self.cfg["train"]["workers"],
            pin_memory=True,
            shuffle=True,
        )
        self.len = self.data_test.__len__()
        return loader


class Data(Dataset):
    def __init__(self, root_dir, type_):
        super().__init__()
        self.root_dir = root_dir
        self.type = type_

        self.datapath = os.path.join(self.root_dir, "{}/images".format(self.type))

        image_list = glob.glob(os.path.join(self.root_dir, "{}/images".format(self.type), "*.png"))
        image_list.sort()
        self.image_list = image_list

        sem_annotations_path = self.datapath.replace("images", "semantics")
        sem_list = [os.path.join(sem_annotations_path, x) for x in os.listdir(sem_annotations_path)]
        sem_list.sort()
        self.sem_list = sem_list

        self.size = None
        self.real_size = len(self.image_list)
        self.transform = Transforms()

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        image = Image.open(self.image_list[index])
        width, height = image.size

        sample["name"] = self.image_list[index].split("/")[-1]

        sample["image"] = np.array(image)

        sample["semantics"] = np.array(Image.open(self.sem_list[index]))
        sample["semantics"][sample["semantics"] == 3] = 1
        sample["semantics"][sample["semantics"] == 4] = 2

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def save_img(self, img, name):
        id_to_color = {
            0: [0, 0, 128],  # navy
            1: [0, 128, 0],  # green
            2: [200, 0, 0],  # red
            3: [192, 192, 192],
        }  # gray

        toplot = np.zeros((img.shape[0], img.shape[1], 3))
        toplot[img == 0] = id_to_color[0]
        toplot[img == 1] = id_to_color[1]
        toplot[img == 2] = id_to_color[2]
        toplot[img == 3] = id_to_color[3]
        toplot = toplot.astype(np.uint8)
        cv2.imwrite(name, cv2.cvtColor(toplot, cv2.COLOR_RGB2BGR))
