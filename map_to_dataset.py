import os

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


@click.command()
@click.option("--export", "-e", type=str, help="path to export folder to save images", default="./results/generated")
@click.option("--config", "-c", type=str, help="path to config file with map info", default="./config/config.yaml")
@click.option("--dpi", "-d", type=int, help="dpi for saved images", default=300)
def main(export, config, dpi):
    cfg = yaml.safe_load(open(config))
    # get paths to map and labels
    path_to_map = cfg["data"]["maps"]
    path_to_annos = cfg["data"]["map_out_name"]
    if not os.path.exists(path_to_annos):
        raise ValueError("There is no annotated map, we cannot produce the dataset.")
    # if export does not exists, we create it
    if not os.path.isdir(export):
        os.mkdir(export)
    # we then access or create the folders for images and semantics
    export_img = os.path.join(export, "images")
    export_sem = os.path.join(export, "semantics")
    # if they do not exist, we create them
    if not os.path.isdir(export_img):
        os.mkdir(export_img)
    if not os.path.isdir(export_sem):
        os.mkdir(export_sem)
    # get size of maps, poses, size for saving the image given dpi
    map_boundary = cfg["mapper"]["map_boundary"]
    poses = cfg["mapper"]["poses"]
    size = [cfg["simulator"]["sensor"]["resolution"][0], cfg["simulator"]["sensor"]["resolution"][1]]
    size_inches = [cfg["simulator"]["sensor"]["resolution"][0] / dpi, cfg["simulator"]["sensor"]["resolution"][1] / dpi]
    # counter of images for name
    counter = 0
    # list of maps
    current_map = np.array(Image.open(path_to_map))
    current_annos = np.array(Image.open(path_to_annos))

    for row in range(poses[1]):
        idx = row * size[1]
        if idx + size[1] > map_boundary[1]:
            break
        for col in range(poses[0]):
            idy = col * size[0]
            if idy + size[0] > map_boundary[0]:
                break
            current_img = current_map[idx : idx + size[1], idy : idy + size[0]]
            current_anno_img = current_annos[idx : idx + size[1], idy : idy + size[0]]

            fig = plt.figure(frameon=False)
            fig.set_size_inches(size_inches[1], size_inches[0])
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(current_img, aspect="auto")
            plt.savefig(os.path.join(export_img, str(counter) + ".png"), dpi=dpi)
            plt.close()

            fig = plt.figure(frameon=False)
            fig.set_size_inches(size_inches[1], size_inches[0])
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(current_anno_img, aspect="auto")
            plt.savefig(os.path.join(export_sem, str(counter) + ".png"), dpi=dpi)
            plt.close()
            counter += 1


if __name__ == "__main__":
    main()
