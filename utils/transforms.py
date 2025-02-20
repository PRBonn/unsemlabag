import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F


class Transforms:
    def __init__(self):
        self.rotation = RandomRotation(p=0.75)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, data):
        data = self.rotation(data)
        data["semantics"] = np.array(data["semantics"]).astype(np.int16)
        for k, v in data.items():
            if k == "name":
                continue
            data[k] = self.to_tensor(v)

        data["semantics"] = (
            torch.nn.functional.one_hot((data["semantics"]).type(torch.long), num_classes=4).movedim(-1, 1).float()
        )
        return data


class RandomRotation(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            rotation_angle = random.choice([0.0, 90.0, 180.0, 270.0])
            data["image"] = F.rotate(
                Image.fromarray(data["image"].astype(np.uint8)),
                angle=rotation_angle,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
            data["semantics"] = F.rotate(
                Image.fromarray(data["semantics"].astype(np.uint8)),
                angle=rotation_angle,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
        return data
