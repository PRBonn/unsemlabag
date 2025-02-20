import cv2
import numpy as np

from utils.utils import get_hough_labels, imap2rgb


def generate_single_pose_label(
    simulator, mapper, pose, rho_old, theta_old, x_old, prev_col_exceeding_lines, turning_point
):
    measurement = simulator.get_measurement(pose)
    if prev_col_exceeding_lines[len(x_old)] != 0:
        prediction, rho, theta, x, exceed_horiz, lines = get_hough_labels(
            measurement["image"], rho_old, theta_old, x_old, prev_col_exceeding_lines[len(x_old)], one_hot_encoded=True
        )
    else:
        prediction, rho, theta, x, exceed_horiz, lines = get_hough_labels(
            measurement["image"], rho_old, theta_old, x_old, {"start": 0, "end": 0, "size": 0}, one_hot_encoded=True
        )
    theta_old, x_old, rho_old, prev_col_exceeding_lines = update(
        theta_old, theta, x_old, x, rho_old, rho, prev_col_exceeding_lines, exceed_horiz
    )

    prediction = label_weeds(prediction, lines)
    # enable for experiments with no "unknown" class
    # prediction[2] += prediction[3]
    mapper.update_map({"semantics": prediction, "fov": measurement["fov"], "gsd": measurement["gsd"]})

    prediction_rgb = imap2rgb(np.argmax(prediction, axis=0), "chw").transpose((1, 2, 0)).astype(int)
    if len(x_old) == turning_point:
        x_old, rho_old = reset()

    return rho_old, theta_old, x_old, prev_col_exceeding_lines


def generate_poses(rows, cols):
    zero_step = 0.512
    add_step = 1.536 - 0.512

    poses = []
    for i in range(0, rows):
        for j in range(0, cols):
            poses.append(np.array([0.1 + zero_step + (add_step / 2) * i, 0.1 + zero_step + add_step * j, zero_step]))
    return poses


def reset():
    return [], []


def update(theta_old, theta, x_old, x, rho_old, rho, prev_col_exceeding_lines, exceed_horiz):

    if theta_old == -1:
        theta_old = theta
    else:
        theta_old = (0.5 * theta_old + 1.5 * theta) / 2

    rho_old.append(rho)
    x_old.append(x)

    if exceed_horiz == 1:
        prev_col_exceeding_lines[len(x_old) - 1] = exceed_horiz
    elif prev_col_exceeding_lines[len(x_old) - 1] != 0:  # if we are not substituting stuff, we override the prev column
        prev_col_exceeding_lines[len(x_old) - 1] = 0

    return theta_old, x_old, rho_old, prev_col_exceeding_lines


def label_weeds(prediction, line_mask):
    unknown = prediction[3]
    unknown = cv2.connectedComponentsWithStats(unknown.astype(np.uint8))

    crops = cv2.connectedComponentsWithStats(prediction[1].astype(np.uint8))
    crops_std = crops[2][1:, 3].std()

    for k in range(1, unknown[0] - 1):  # skip bg
        dists = np.abs(unknown[3][k].reshape(-1, 1) - np.where(line_mask.T != 0))
        dists = np.sqrt((dists**2).sum(0))
        closer_line = np.min(dists)

        if closer_line > 3 * crops_std:
            prediction[3][unknown[1] == k] = 0
            prediction[2][unknown[1] == k] = 1

    return prediction
