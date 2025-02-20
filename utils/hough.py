import cv2
import numpy as np
from PIL import Image


class Hough:
    def __init__(self, pixel_res=1, angle_res=np.pi / 180, min_theta=-0.02, max_theta=0.02, min_voting=800):
        self.pixel_res = pixel_res
        self.angle_res = angle_res
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_voting = min_voting

    def forward(self, image, rho_old=[], theta_old=-1, x_old=[], horizontal_exceed={}):
        self.extract_vegetation_mask(image)
        if self.binary_mask.sum() == 0:
            return self.binary_mask, horizontal_exceed, -1, -1, -1, -1
        try:
            rho, theta = cv2.HoughLines(
                self.binary_mask,
                self.pixel_res,
                self.angle_res,
                self.min_voting,
                min_theta=self.min_theta,
                max_theta=self.max_theta,
            )[0][0]
        except:
            try:
                rho, theta = cv2.HoughLines(
                    self.binary_mask,
                    self.pixel_res,
                    self.angle_res,
                    int(self.min_voting / 10),
                    min_theta=self.min_theta,
                    max_theta=self.max_theta,
                )[0][0]
            except:
                # no line has been found
                return np.zeros_like(image, dtype=np.uint8)[:, :, 0], horizontal_exceed, -1, -1, -1, -1

        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)
        y1 = int(0)
        x1 = int(rho / np.cos(theta))
        y2 = int(image.shape[1])
        x2 = int((rho - y2 * np.sin(theta)) / np.cos(theta))

        line_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 15)

        # include lines propagated from the y axis
        line_mask = self.propagate_vertical_rows(line_mask, x1, x_old, rho_old, theta_old)

        # include lines propagated from the x axis
        line_mask = cv2.line(
            line_mask,
            (5, horizontal_exceed["start"]),
            (5, image.shape[1] - horizontal_exceed["end"]),
            (255, 255, 255),
            15,
        )
        line_mask = line_mask[:, :, 0].astype(bool)
        final_mask, horizontal_dict = self.generate_hough_line_label(line_mask)
        return final_mask, horizontal_dict, rho, theta, x2, line_mask

    def propagate_vertical_rows(self, line_mask, x1, x_old, rho_old, theta_old):
        if x_old != []:
            for it in range(len(x_old)):
                if x_old[it] == -1 or abs(x_old[it] - x1) < 200:
                    continue
                x_new = int(
                    (rho_old[it] - line_mask.shape[0] * (it + 2) * np.sin(theta_old)) / np.cos(theta_old)
                    + (150 + (len(x_old) - it) * 25)
                )
                line_mask = cv2.line(
                    line_mask,
                    (x_old[it] + (150 + (len(x_old) - it) * 25), 0),
                    (x_new, line_mask.shape[1]),
                    (255, 255, 255),
                    15,
                )
        return line_mask

    def generate_hough_line_label(self, mask):  # , real_mask):
        # mask has one line that is my "row crop line"
        # we need to define which pixels in this line are vegetation using self.binary_mask
        line_crop = self.binary_mask * mask
        line_soil = ~self.binary_mask
        label_mask = np.ones_like(mask) * (3)  # 3 is the ignore index
        veg_components = cv2.connectedComponentsWithStats(self.binary_mask)

        # identify crops as veg components intersecting the line
        for _id in range(1, veg_components[0]):
            if (mask * (veg_components[1] == _id)).sum() != 0:
                label_mask[(veg_components[1] == _id)] = 1

        label_mask[(line_soil == 255)] = 0  # soil
        label_mask, horizontal_dict = self.check_horizontal(label_mask)
        return label_mask, horizontal_dict

    def check_horizontal(self, mask):
        return mask, {
            "start": mask[-int(mask.shape[0] / 2), :].argmax(),
            "end": np.flip(mask[-int(mask.shape[0] / 2), :]).argmax(),
            "size": mask[-int(mask.shape[0] / 2), :].sum(),
        }

    def weakly_supervised_mask(self, name):
        mask = np.array(Image.open(name))
        self.binary_mask = (mask != 0).astype(np.uint8)
        return mask

    def extract_vegetation_mask(self, image):
        # paper method
        # cv2.imwrite('geometric_segmentation/fig.pnm', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # os.system("cd geometric_segmentation; ./segment 1 500 50 fig.pnm fig.pnm" )
        # self.binary_mask = np.array(Image.open("geometric_segmentation/fig.pnm")).sum(-1)
        # colors = np.unique(self.binary_mask)
        # for color in colors:
        #    current = image[ self.binary_mask == color ].sum(0)/(self.binary_mask == color).sum()
        #    if (2*current[1] - current[0] - current[2] < 100) and (2*current[1] - current[0] - current[2] > 35):
        #        self.binary_mask[ self.binary_mask == color ] = 0

        # faster but requires a lot of manual finetuning
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        r = (r - r.mean()) / (r.std() + 1e-15)
        g = (g - g.mean()) / (g.std() + 1e-15)
        b = (b - b.mean()) / (b.std() + 1e-15)
        exg = 2 * g - r - b
        self.binary_mask = (exg > 0.3) * (r < g) * (b < g) * (r * 0.5 > b)
        comp = cv2.connectedComponentsWithStats(self.binary_mask.astype(np.uint8))
        for num in range(comp[0]):
            if comp[2][num][-1] < 100:  # fitler our very small components, usually this veg mask has noise
                self.binary_mask[comp[1] == num] = 0

        self.binary_mask[self.binary_mask != 0] = 1
        self.binary_mask = self.binary_mask.astype(np.uint8)
