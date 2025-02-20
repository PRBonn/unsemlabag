from typing import Dict, List, Tuple

import numpy as np

from simulator import Sensor


def get_map_index_tuple(map_indices: np.array) -> Tuple:
    if map_indices.shape[1] == 3:
        return map_indices[:, 0], map_indices[:, 1], map_indices[:, 2]

    return map_indices[:, 0], map_indices[:, 1]


def get_map_boundary_tuple(map_boundary: np.array) -> Tuple:
    if len(map_boundary) == 3:
        return map_boundary[0], map_boundary[1], map_boundary[2]

    return map_boundary[1], map_boundary[0]


class CountMap:
    def __init__(self, map_boundary: np.array):
        self.map_boundary = map_boundary
        self.count_map = self.init_map()

    def init_map(self) -> np.array:
        return np.zeros(get_map_boundary_tuple(self.map_boundary), dtype=np.int16)

    def update(self, map_indices: np.array):
        self.count_map[get_map_index_tuple(map_indices)] += 1


class DiscreteVariableMap:
    def __init__(self, map_boundary: np.array, num_classes: int):
        self.num_classes = num_classes
        self.map_boundary = map_boundary
        self.log_odds_map, self.prob_map, self.mle_map = self.init_map()
        self.prob_map_outdated, self.mle_map_outdated = False, False

    def init_map(self) -> Tuple[np.array, np.array, np.array]:
        log_odds_prior_map = self.log_odds_prior_const * np.ones(
            (self.num_classes, *get_map_boundary_tuple(self.map_boundary)),
            dtype=np.float16,
        )
        prob_prior_map = (1 / self.num_classes) * np.ones(
            (self.num_classes, *get_map_boundary_tuple(self.map_boundary)),
            dtype=np.float16,
        )
        mle_prior_map = np.zeros(get_map_boundary_tuple(self.map_boundary), dtype=np.float16)
        return log_odds_prior_map, prob_prior_map, mle_prior_map

    @property
    def log_odds_prior_const(self) -> float:
        prob_prior = 1 / self.num_classes
        return np.log(prob_prior / (1 - prob_prior))

    def get_prob_map(self) -> np.array:
        if self.prob_map_outdated:
            self.set_prob_map()
            self.prob_map_outdated = False

        return self.prob_map

    def set_mle_map(self):
        self.mle_map_outdated = False
        self.mle_map = np.argmax(self.log_odds_map, axis=0)

    def set_prob_map(self):
        self.prob_map = 1 - (1 / (1 + np.exp(self.log_odds_map)))

    def update(self, map_indices: np.array, probs_measured: np.array, **kwargs):
        self.mle_map_outdated = True
        self.prob_map_outdated = True

        map_index_tuple_sliced = (slice(None), *get_map_index_tuple(map_indices))
        probs_measured = np.clip(probs_measured, a_min=10 ** (-2), a_max=1 - 10 ** (-2))
        probs_measured /= np.sum(probs_measured, axis=0)
        log_odds_measured = np.log(probs_measured / (1 - probs_measured))
        self.log_odds_map[map_index_tuple_sliced] += log_odds_measured - self.log_odds_prior_const


class Mapper:
    def __init__(self, sensor: Sensor, map_boundary: List, ground_resolution: List, class_num: int):
        self.map_name = "base-mapper"
        self.sensor = sensor
        self.class_num = class_num
        self.map_boundary = map_boundary
        self.ground_resolution = np.array(ground_resolution)
        self.semantic_map, self.hit_map_semantic = self.init_map()

    def init_map(self) -> Tuple[DiscreteVariableMap, CountMap]:
        raise NotImplementedError("'init_map' function not implemented!")

    def update_map(self, data_source: Dict):
        raise NotImplementedError("'update_map' function not implemented!")

    def get_map_state(self, pose: np.array, depth: np.array = None, sensor: Sensor = None) -> Tuple[np.array, np.array]:
        raise NotImplementedError("'get_map_state' function not implemented!")
