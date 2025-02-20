import os
from typing import Dict

import click
import numpy as np
import yaml
from PIL import Image

from mapper.terrain_mapper import TerrainMapper
from simulator.load_simulators import get_simulator
from utils.label_generation import generate_poses, generate_single_pose_label
from utils.utils import imap2rgb, save_images


def read_config_files(config_file_path: str) -> Dict:
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Cannot find config file '{config_file_path}'!")

    if not config_file_path.endswith((".yaml", ".yml")):
        raise ValueError(f"Config file is not a yaml-file! Only '.yaml' or '.yaml' file endings allowed!")

    with open(config_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    return cfg


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
)
def main(config: str):
    cfg = read_config_files(config)
    current_map = cfg["data"]["maps"]
    cfg["simulator"]["path_to_orthomosaic"] = current_map
    simulator = get_simulator(cfg["simulator"])
    mapper = TerrainMapper(
        simulator.sensor,
        cfg["mapper"]["map_boundary"],
        cfg["mapper"]["ground_resolution"],
        cfg["mapper"]["class_number"],
    )

    poses = generate_poses(cfg["mapper"]["poses"][0], cfg["mapper"]["poses"][1])

    rho_old = []
    theta_old = -1
    x_old = []
    prev_col_exceeding_lines = np.zeros(cfg["mapper"]["poses"][1] + 1, dtype=dict)

    print("Starting to move over the map...\n")
    for it, pose in enumerate(poses):
        rho_old, theta_old, x_old, prev_col_exceeding_lines = generate_single_pose_label(
            simulator, mapper, pose, rho_old, theta_old, x_old, prev_col_exceeding_lines, cfg["mapper"]["poses"][1]
        )

    map_rgb = imap2rgb(np.argmax(mapper.semantic_map.get_prob_map(), axis=0), "chw").transpose((1, 2, 0)).astype(int)
    print("Ready to save the labels...\n")

    save_images(cfg["data"]["map_out_name"], map_rgb, np.array(Image.open(cfg["simulator"]["path_to_orthomosaic"])))


if __name__ == "__main__":
    main()
