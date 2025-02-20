"""
Dataset parser
"""

import glob
import os
import random

import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils.transforms import Transforms


class MapData(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg["data"]["root_dir"]
        self.setup()

    def setup(self, stage=None):
        self.data_train = Data(self.root_dir)
        # one can either use the same data as validation
        # or create a new Dataset class to load manually annotated data as val set
        self.data_val = Data(self.root_dir)

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
        pass  # no test should be done on generated labels


class Data(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.datapath = os.path.join(self.root_dir, "images")
        self.image_list = glob.glob(os.path.join(self.datapath, "*.png"))
        sem_annotations_path = self.datapath.replace("images", "semantics")
        self.image_list.sort()
        self.sem_list = [os.path.join(sem_annotations_path, x) for x in os.listdir(sem_annotations_path)]
        self.sem_list.sort()
        self.size = None
        self.real_size = len(self.image_list)
        self.transform = Transforms()

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # image resize and normalization
        image = np.array(Image.open(self.image_list[index]).convert("RGB").resize((1024, 1024)))
        width = image.shape[0]
        height = image.shape[1]
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        r = ((r - r.mean()) / (r.std() + 1e-17)).astype(np.float32)
        g = ((g - g.mean()) / (g.std() + 1e-17)).astype(np.float32)
        b = ((b - b.mean()) / (b.std() + 1e-17)).astype(np.float32)

        sample["image"] = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis]), -1)

        semantics = np.array(Image.open(self.sem_list[index]).resize((1024, 1024)))
        # we save labels as blue = soil (class = 0), green = crops (class = 1), red = weeds (class = 2), everything else is unknown (class = 3)
        sample["semantics"] = np.ones((width, height)) * 3
        sample["semantics"][semantics[:, :, -1] != 0] = 0
        sample["semantics"][semantics[:, :, 0] != 0] = 2
        sample["semantics"][semantics[:, :, 1] != 0] = 1

        sample["name"] = self.image_list[index].split("/")[-1]
        sample["semantics"] = sample["semantics"].astype(np.uint8)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
