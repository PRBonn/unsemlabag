from typing import Dict

import cv2
import numpy as np

from simulator import Sensor, Simulator
from utils import utils


class RGBSensor(Sensor):
    def __init__(self, angle: np.array, resolution: np.array):
        super(RGBSensor, self).__init__(angle, resolution)


class OrthoSimulator(Simulator):
    def __init__(self, world: np.array, world_range: np.array, gsd: float, sensor_cfg: Dict):
        super(OrthoSimulator, self).__init__(world_range, gsd, sensor_cfg)

        self.simulator_name = "ortho-simulator"
        self.world = world

    def setup_sensor(self, sensor_cfg: Dict) -> RGBSensor:
        return RGBSensor(sensor_cfg["angle"], sensor_cfg["resolution"])

    def get_measurement(self, pose: np.array) -> Dict:
        fov_info = utils.get_fov(pose, self.sensor.angle, self.gsd, self.world_range)

        fov_corner, range_list = fov_info
        gsd = [
            (np.linalg.norm(fov_corner[1] - fov_corner[0])) / self.sensor.resolution[0],
            (np.linalg.norm(fov_corner[3] - fov_corner[0])) / self.sensor.resolution[1],
        ]
        rgb_image_raw = self.world[range_list[2] : range_list[3], range_list[0] : range_list[1], :]
        rgb_image = cv2.resize(rgb_image_raw, tuple(self.sensor.resolution))

        return {
            "image": rgb_image,
            "fov": fov_corner,
            "range": range_list,
            "gsd": gsd,
            "pose": pose,
        }
