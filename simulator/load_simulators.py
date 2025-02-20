import os
from typing import Dict

import cv2

from simulator import Simulator
from simulator.ortho_simulator import OrthoSimulator


def get_simulator(cfg: Dict) -> Simulator:
    path_to_orthomosaic = cfg["path_to_orthomosaic"]
    resize_flag = cfg["resize_flag"]
    resize_factor = cfg["resize_factor"]

    if not os.path.exists(path_to_orthomosaic):
        raise FileNotFoundError(f"RGB ortho file '{path_to_orthomosaic}' not found")

    orthomosaic = cv2.imread(path_to_orthomosaic)
    orthomosaic = cv2.cvtColor(orthomosaic, cv2.COLOR_BGR2RGB)

    if resize_flag:
        orig_height, orig_width, _ = orthomosaic.shape
        resized_height, resized_width = int(orig_height / resize_factor), int(orig_width / resize_factor)
        orthomosaic = cv2.resize(orthomosaic, (resized_height, resized_width))

    return OrthoSimulator(orthomosaic, cfg["world_range"], cfg["gsd"], cfg["sensor"])
