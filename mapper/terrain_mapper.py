from typing import Dict, List, Tuple

import cv2
import numpy as np

from mapper import CountMap, DiscreteVariableMap, Mapper
from simulator import Sensor
from utils import utils


class TerrainMapper(Mapper):
    def __init__(self, sensor: Sensor, map_boundary: List, ground_resolution: List, class_num: int):
        super(TerrainMapper, self).__init__(sensor, map_boundary, ground_resolution, class_num)

    def init_map(self) -> Tuple[DiscreteVariableMap, CountMap]:
        hit_map = CountMap(self.map_boundary)
        semantic_map = DiscreteVariableMap(self.map_boundary, self.class_num)
        return semantic_map, hit_map

    def find_map_index(self, data_point) -> Tuple[float, float]:
        x_index = np.floor(data_point[0] / self.ground_resolution[0]).astype(int)
        y_index = np.floor(data_point[1] / self.ground_resolution[1]).astype(int)
        return x_index, y_index

    def update_map(self, data_source: Dict):
        semantics = data_source["semantics"]
        fov = data_source["fov"]
        gsd = data_source["gsd"]
        _, m_y_dim, m_x_dim = semantics.shape

        measurement_indices = np.array(np.meshgrid(np.arange(m_y_dim), np.arange(m_x_dim))).T.reshape(-1, 2).astype(int)
        x_ground = fov[0][0] + (0.5 + np.arange(m_x_dim)) * gsd[0]
        y_ground = fov[0][1] + (0.5 + np.arange(m_y_dim)) * gsd[1]
        ground_coords = np.array(np.meshgrid(y_ground, x_ground)).T.reshape(-1, 2)
        map_indices = np.floor(ground_coords / np.array(self.ground_resolution)).astype(int)

        self.hit_map_semantic.update(map_indices)

        semantics_proj = semantics[:, measurement_indices[:, 0], measurement_indices[:, 1]]
        self.semantic_map.update(map_indices, semantics_proj)

    def get_map_state(self, pose: np.array, depth: np.array = None, sensor: Sensor = None) -> Tuple[np.array, np.array]:
        if sensor is None:
            sensor = self.sensor

        fov_corners, _ = utils.get_fov(pose[:3], sensor.angle, self.ground_resolution[0], self.map_boundary)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.find_map_index(lu)
        rd_x, rd_y = self.find_map_index(rd)

        hit_count_map = self.hit_map_semantic.count_map[lu_y:rd_y, lu_x:rd_x]

        semantic_map_probs = (
            self.semantic_map.get_prob_map()[:, lu_y:rd_y, lu_x:rd_x].transpose(1, 2, 0).astype(np.float32)
        )
        semantic_map_probs = cv2.resize(semantic_map_probs, sensor.resolution).transpose(2, 0, 1)

        return semantic_map_probs, hit_count_map
