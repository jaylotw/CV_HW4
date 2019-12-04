import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    

  
    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
   

    
    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.


    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering


    return labels.astype(np.uint8)