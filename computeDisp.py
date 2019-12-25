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
    kernel_size = 15  # should be odd
    half_kernel_width = int(kernel_size / 2)

    # bilateral filter to smooth the image for a better result
    Il_bilateral = cv2.bilateralFilter(src=Il, d=3, sigmaColor=21, sigmaSpace=21)
    Ir_bilateral = cv2.bilateralFilter(src=Ir, d=3, sigmaColor=21, sigmaSpace=21)

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
        left=half_kernel_width + max_disp,
        right=half_kernel_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],  # black
    )

    # cost
    sd_cost = np.zeros((h, w, max_disp + 1), dtype=np.float32)
    windows_cost = np.zeros((h, w, max_disp + 1), dtype=np.float32)

    if PROGRESS_BAR:
        pbar = tqdm(desc="SSD", total=max_disp + 1, position=0)

    # ----- cost computation -----
    for offset in range(max_disp + 1):
        # differences
        img_diff = (
            Il_pad
            - Ir_pad[
                :, max_disp - offset : max_disp - offset + w + 2 * half_kernel_width
            ]
        )
        # squared differences
        img_sd = img_diff ** 2
        # img_sd = np.abs(img_diff)

        sd_cost[:, :, offset] = img_sd[
            half_kernel_width:-half_kernel_width, half_kernel_width:-half_kernel_width
        ]

        census_x = half_kernel_width
        census_y = half_kernel_width

        for y in range(half_kernel_width, half_kernel_width + h):
            for x in range(half_kernel_width, half_kernel_width + w):
                # kernel window of both images
                Il_window = Il_pad[
                    y - half_kernel_width : y + half_kernel_width,
                    x - half_kernel_width : x + half_kernel_width,
                ]
                Ir_window = Ir_pad[
                    y - half_kernel_width : y + half_kernel_width,
                    x
                    + max_disp
                    - half_kernel_width
                    - offset : x
                    + max_disp
                    + half_kernel_width
                    - offset,
                ]

                window_cost_temp = (Il_window != Ir_window).sum()
                windows_cost[
                    y - half_kernel_width, x - half_kernel_width, offset
                ] = window_cost_temp

        cost_map = 2 - np.exp(-sd_cost / 10) - np.exp(-windows_cost / 10)

        if PROGRESS_BAR:
            pbar.update()
    if PROGRESS_BAR:
        pbar.close()

    # ----- cost aggregation -----
    smooth_cost_map = np.empty_like(cost_map)
    for offset in range(max_disp + 1):
        # smooth the cost_map with guided filter and median filter
        smooth_cost_map[:, :, offset] = cv2.ximgproc.guidedFilter(
            guide=Il_grey, src=cost_map[:, :, offset], radius=8, eps=100, dDepth=-1
        )
        smooth_cost_map[:, :, offset] = cv2.medianBlur(
            smooth_cost_map[:, :, offset], ksize=3
        )

    # ----- disparity optimization -----
    min_cost_map = np.argmin(smooth_cost_map, axis=2)

    # ----- disparity refinement -----
    # smooth the disparity with median filter
    disp_map = cv2.medianBlur(min_cost_map.astype(np.uint8), ksize=3)
    # get rid of the leftmost black pixel
    for offset in range(max_disp):
        disp_map[:, offset] = disp_map[:, max_disp + 8]

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
