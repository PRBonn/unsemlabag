from typing import Dict

import numpy as np


class Sensor:
    def __init__(self, angle: np.array, resolution: np.array):
        self.angle = np.array(angle)
        self.resolution = resolution


class Simulator:
    def __init__(self, world_range: np.array, gsd: float, sensor_cfg: Dict):
        self.simulator_name = "base-simulator"

        self.gsd = gsd  # m/pixel
        self.world_range = world_range  # pixel
        self.sensor = self.setup_sensor(sensor_cfg)

    def setup_sensor(self, sensor_cfg: Dict) -> Sensor:
        raise NotImplementedError(f"Simulator '{self.simulator_name}' does not implement 'setup_sensor()' function!")

    def get_measurement(self, pose: np.array) -> Dict:
        raise NotImplementedError(f"Simulator '{self.simulator_name}' does not implement 'get_measurement()' function!")
