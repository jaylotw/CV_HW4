import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image

import time
from tqdm import tqdm

PROGRESS_BAR = True
DEBUG_SAVE_IMG = False


class BinaryCost:
    def __init__(self, img1, img2, pair, half_wd=13, num_pair=4096):
        self.img1 = img1
        self.img2 = img2
        self.num_pair = num_pair

        self.first_y = pair[0]
        self.first_x = pair[1]
        self.second_y = pair[2]
        self.second_x = pair[3]
        self.half_wd = half_wd

    def set_bits(self):
        h, w = self.img1.shape
        self.left_bits = np.zeros((h, w, self.num_pair), dtype=bool)
        self.right_bits = np.zeros((h, w, self.num_pair), dtype=bool)

        if PROGRESS_BAR:
            pbar = tqdm(desc="set bits", total=h - 2 * self.half_wd, position=0)

        for y in range(self.half_wd, h - self.half_wd):
            for x in range(self.half_wd, w - self.half_wd):
                left_bits, right_bits = self.get_bits(y, x, 0)
                self.left_bits[y - self.half_wd, x - self.half_wd] = left_bits
                self.right_bits[y - self.half_wd, x - self.half_wd] = right_bits
                del left_bits, right_bits
            if PROGRESS_BAR:
                pbar.update()
        if PROGRESS_BAR:
            pbar.close()

    def get_bits(self, y_pos, x_pos, sft):
        self.select_patch_11 = self.img1[y_pos + self.first_y, x_pos + self.first_x]
        self.select_patch_12 = self.img1[y_pos + self.second_y, x_pos + self.second_x]

        self.select_patch_21 = self.img2[
            y_pos + self.first_y, x_pos - sft + self.first_x
        ]
        self.select_patch_22 = self.img2[
            y_pos + self.second_y, x_pos - sft + self.second_x
        ]

        result_patch_1 = self.select_patch_11 > self.select_patch_12
        result_patch_2 = self.select_patch_21 > self.select_patch_22
        del (
            self.select_patch_11,
            self.select_patch_12,
            self.select_patch_21,
            self.select_patch_22,
        )

        return result_patch_1, result_patch_2


def get_random_pair(num_pair=4096):

    pair_seq = np.random.normal(0.0, 4.0, num_pair * 4)
    pair_seq = np.clip(pair_seq, -13, 13).astype(np.int_)

    first_y = pair_seq[:num_pair]
    first_x = pair_seq[num_pair : num_pair * 2]
    second_y = pair_seq[num_pair * 2 : num_pair * 3]
    second_x = pair_seq[num_pair * 3 :]

    return (first_y, first_x, second_y, second_x)


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def normalize(patch):
    nor_patch = (patch - np.mean(patch)) / np.std(patch) + 10e-8
    return nor_patch


def evaluate(input_path, gt_path, scale_factor, threshold=1.0):
    disp_gt = cv2.imread(gt_path, -1)
    disp_gt = np.int32(disp_gt / scale_factor)
    disp_input = cv2.imread(input_path, -1)
    disp_input = np.int32(disp_input / scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1

    return float(nr_error) / nr_pixel


def computeDisp(Il, Ir, max_disp):

    h, w, ch = Il.shape
    kernel_size = 26
    half_kernel_width = int(kernel_size / 2)

    # bilateral filter to smooth the image for a better result
    Il_bilateral = cv2.bilateralFilter(src=Il, d=5, sigmaColor=21, sigmaSpace=21)
    Ir_bilateral = cv2.bilateralFilter(src=Ir, d=5, sigmaColor=21, sigmaSpace=21)

    # convert images to greyscale
    Il_grey = cv2.cvtColor(Il_bilateral, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ir_grey = cv2.cvtColor(Ir_bilateral, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # border padding
    Il_pad = cv2.copyMakeBorder(
        src=Il_grey,
        top=half_kernel_width,
        bottom=half_kernel_width,
        left=half_kernel_width,
        right=half_kernel_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # black
    )
    Ir_pad = cv2.copyMakeBorder(
        src=Ir_grey,
        top=half_kernel_width,
        bottom=half_kernel_width,
        left=half_kernel_width,
        right=half_kernel_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # black
    )

    pairs = get_random_pair()
    bincost = BinaryCost(
        img1=Il_pad, img2=Ir_pad, pair=pairs, half_wd=half_kernel_width
    )
    bincost.set_bits()

    # cost
    cost_map = np.full((h, w, max_disp + 1), fill_value=np.inf, dtype=np.float32)

    if PROGRESS_BAR:
        pbar = tqdm(desc="cost", total=h, position=0)

    for y in range(half_kernel_width, half_kernel_width + h):
        for x in range(half_kernel_width, half_kernel_width + w):
            top = y - half_kernel_width
            left = x - half_kernel_width
            bot = y + half_kernel_width + 1
            right = x + half_kernel_width + 1
            left_bits = bincost.left_bits[y - half_kernel_width, x - half_kernel_width]

            for shift in range(max_disp):
                if left - shift >= 0:
                    right_bits = bincost.right_bits[
                        y - half_kernel_width, x - half_kernel_width - shift
                    ]
                    match_cost = np.sum(np.logical_xor(left_bits, right_bits))
                    cost_map[
                        y - half_kernel_width, x - half_kernel_width, shift
                    ] = match_cost
                    del right_bits, match_cost

        if PROGRESS_BAR:
            pbar.update()
    if PROGRESS_BAR:
        pbar.close()

    smooth_cost_map = np.empty_like(cost_map)
    for offset in range(max_disp + 1):
        # smooth the cost_map with median filter
        smooth_cost_map[:, :, offset] = cv2.medianBlur(cost_map[:, :, offset], ksize=3)

    min_cost_map = np.argmin(smooth_cost_map, axis=2)

    # smooth the disparity with median filter
    disp_map = cv2.medianBlur(min_cost_map.astype(np.uint8), ksize=3)
    # get rid of the leftmost black pixel
    for offset in range(max_disp):
        disp_map[:, offset] = disp_map[:, max_disp + 5]

    if DEBUG_SAVE_IMG:
        cv2.imwrite("./debug/tsukuba_min_cost.png", np.uint8(min_cost_map * 16))
        cv2.imwrite("./debug/tsukuba_disp.png", np.uint8(disp_map * 16))

    return disp_map.astype(np.uint8)


def main():
    img_left = cv2.imread("./testdata/tsukuba/im3.png")
    img_right = cv2.imread("./testdata/tsukuba/im4.png")
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite("./debug/tsukuba.png", np.uint8(labels * scale_factor))

    print("[Bad Pixel Ratio]")
    res = evaluate(
        "./debug/tsukuba.png", "./testdata/tsukuba/disp3.pgm", scale_factor=16
    )
    print("Tsukuba: %.2f%%" % (res * 100))


if __name__ == "__main__":
    PROGRESS_BAR = True
    DEBUG_SAVE_IMG = True
    main()
