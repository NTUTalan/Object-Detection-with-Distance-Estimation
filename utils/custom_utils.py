# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only

object_real_height = {
    'Car': 1.676, 
    'Van': 1.676, 
    'Truck': 4, 
    'Pedestrian': 1.671
}

def PlotBoxWithDistance(x, img, color=None, label=None, line_thickness=3):
    object_real_height = {'Car': 1.676, 'Van': 1.676, 'Truck': 4, 'Pedestrian': 1.671}
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    w = int(x[2]) - int(x[0])
    h = int(x[3]) - int(x[1])
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    ### Caculate Distance and plot
    if label:
        tf = max(tl - 1, 1)  # font thickness
        # distance = str("\"{:.2f} Inches\"".format((2 * 3.14 * 180) / ( w + h * 360) * 1000 + 3)) ### Distance measuring in Inch 
        try:
            distance = 512 * object_real_height[label] / h
            distance_str = str("{:.2f} Meters".format(distance)) ### 目前使用手機計算
        except:
            distance =  ''
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        d_size = cv2.getTextSize(distance_str, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + d_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, distance_str, (c1[0] + t_size[0] + 1, c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return ("Side Test", distance)