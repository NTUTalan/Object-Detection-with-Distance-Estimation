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

# (影像, [(畫面位置Ex: “left”, 距離), (), (), () … ])
def CustomPlotBox(x: list, img, label: str=None, box_color: list=None, line_thickness: int=3) -> list[tuple]:
    '''
    從image的shape來判斷在左還是在右
    center_x = (shape[0] - 1 ) / 2
    '''
    img_width = img.shape[1]
    img_height = img.shape[0]
    tl = line_thickness or round(0.002 * (img_width + img_height) / 2) + 1  # line/font thickness
    box_color = box_color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) #左上、右下
    width = int(x[2]) - int(x[0])
    height = int(x[3]) - int(x[1])
    cv2.rectangle(img, c1, c2, box_color, thickness=tl, lineType=cv2.LINE_AA)

    # test area-------------------------------------------------------------------------------------------------------------------------------
    # print(img.shape)
    lb_start = [img_width // 3, 0]
    lb_end = [img_width // 3, img_height]
    rb_start = [img_width * 2 // 3, 0]
    rb_end = [img_width * 2 // 3, img_height]
    cv2.line(img, lb_start, lb_end, [153, 255, 255], 2)
    cv2.line(img, rb_start, rb_end, [153, 255, 255], 2)
    # cv2.line(img, [img_width, 0], [img_width, img_height], [255, 255, 153], 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------

    ### Caculate Distance and plot
    if label:
        tf = max(tl - 1, 1)  # font thickness
        distance = 512 * object_real_height[label] / height ### Distance measuring in Inch 
        try:
            str_distance = str("{:.2f} Meters".format(distance)) ### 目前使用手機計算
        except:
            distance =  ''
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        d_size = cv2.getTextSize(str_distance, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + d_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, str_distance, (c1[0] + t_size[0] + 1, c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return (GetPosition(x, img_width), distance)

def GetPosition(x: list, img_width: int):
    box_center_x = (int(x[2]) + int(x[0])) / 2
    left_bound = img_width / 3
    right_bound = img_width * 2 / 3
    print(box_center_x > left_bound)
    if(box_center_x < left_bound):
        return "left"
    elif(box_center_x > right_bound):
        return "right"
    else:
        return "center"