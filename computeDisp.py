import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image

import time
from tqdm import tqdm
import math

DEBUG = True


def SSD(Il, Ir, kernel_size, max_offset):
    """
    cost computation: sum of squared differences(SSD) method
    """

    h, w, ch = Il.shape
    half_kernel_width = int(kernel_size / 2)

    Il_grey = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ir_grey = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Il_grey = cv2.bilateralFilter(src=Il_grey, d=5, sigmaColor=21, sigmaSpace=21)
    # Ir_grey = cv2.bilateralFilter(src=Ir_grey, d=5, sigmaColor=21, sigmaSpace=21)

    # Il_grey = cv2.copyMakeBorder(
    #     src=Il_grey,
    #     top=half_kernel_width,
    #     bottom=half_kernel_width,
    #     left=half_kernel_width,
    #     right=half_kernel_width,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=BLACK,
    # )
    # Ir_grey = cv2.copyMakeBorder(
    #     src=Ir_grey,
    #     top=half_kernel_width,
    #     bottom=half_kernel_width,
    #     left=half_kernel_width,
    #     right=half_kernel_width,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=BLACK,
    # )

    # depth map
    depth_map = np.zeros_like(Il_grey)

    if DEBUG:
        bar_y = tqdm(desc="SSD", total=h, position=0)

    for y in range(half_kernel_width, h - half_kernel_width):
        for x in range(half_kernel_width, w - half_kernel_width):
            best_offset = 0
            prev_ssd = 65534  # 256*256

            for offset in range(max_offset):
                # initialization
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-half_kernel_width, half_kernel_width):
                    for u in range(-half_kernel_width, half_kernel_width):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(Il_grey[y + v, x + u]) - int(
                            Ir_grey[y + v, (x + u) - offset]
                        )
                        ssd += ssd_temp * ssd_temp

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth_map output for the (x, y) pixel to the best match
            depth_map[y, x] = best_offset

        if DEBUG:
            bar_y.update()
    if DEBUG:
        bar_y.close()

    # map depth map output to 0-255 range
    depth_map *= 255 / max_offset

    return depth_map


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    # Il = Il.astype(np.float32)
    # Ir = Ir.astype(np.float32)

    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    if DEBUG:
        print(" ----- Cost Computation ----- ")
        t_start = time.time()

    labels = SSD(Il, Ir, kernel_size=6, max_offset=max_disp)

    if DEBUG:
        t_end = time.time()
        print("Time: {:.2f} sec.".format(t_end - t_start))

    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    if DEBUG:
        print(" ----- Cost Aggregation ----- ")
        t_start = time.time()

    if DEBUG:
        t_end = time.time()
        print("Time: {:.2f} sec.".format(t_end - t_start))

    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    if DEBUG:
        print(" ----- Disparity Optimization ----- ")
        t_start = time.time()

    if DEBUG:
        t_end = time.time()
        print("Time: {:.2f} sec.".format(t_end - t_start))

    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    if DEBUG:
        print(" ----- Disparity Refinement ----- ")
        t_start = time.time()

    if DEBUG:
        t_end = time.time()
        print("Time: {:.2f} sec.".format(t_end - t_start))

    return labels.astype(np.uint8)


def main():
    img_left = cv2.imread("./testdata/tsukuba/im3.png")
    img_right = cv2.imread("./testdata/tsukuba/im4.png")
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite("./debug/tsukuba.png", np.uint8(labels * scale_factor))


if __name__ == "__main__":
    main()
