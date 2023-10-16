import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

class RoadLaneDetect():
    def __init__(self):
        self.lowcanny = 120
        self.highcanny = 150
        self.hough_threshold = 180

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def merge_lines(self, lines, max_dist=50):
        merged_lines = []
        if lines is not None:
            if len(lines) > 0:
                prev_x1, prev_y1, prev_x2, prev_y2 = lines[0][0]
                for line in lines[1:]:
                    x1, y1, x2, y2 = line[0]
                    dist = np.sqrt((x1 - prev_x2) ** 2 + (y1 - prev_y2) ** 2)
                    if dist < max_dist:
                        prev_x2, prev_y2 = x2, y2
                    else:
                        merged_lines.append([(prev_x1, prev_y1, prev_x2, prev_y2)])
                        prev_x1, prev_y1, prev_x2, prev_y2 = x1, y1, x2, y2
                merged_lines.append([(prev_x1, prev_y1, prev_x2, prev_y2)])

        return merged_lines

    def filter_lines(self, lines, slope_threshold):
        filtered_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    slope = np.inf 
                
                if abs(slope) > slope_threshold:
                    filtered_lines.append(line)
            
            return filtered_lines
        
        
    def detect(self, frame):
        # 預處理步驟，灰度化、高斯模糊、Canny邊緣檢測等
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 調整對比度和亮度
        alpha = 1.5  # 控制對比度
        beta = 30    # 控制亮度
        adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

        edges = cv2.Canny(adjusted, 120, 150)
        #cv2.imshow('edge',edges)

        # 選擇感興趣的區域
        height, width = frame.shape[:2]
        roi_vertices = [(0, height), (width, height), (width, height // 2), (0, height // 2)]
        roi_edges = self.region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 新增中下方矩形 ROI 區域
        bottom_roi = np.array([[(width//4, height-50),(width//2, height *3//4), ((width*3)//4, height-50)]], np.int32)
        cv2.polylines(frame, [bottom_roi], isClosed=True, color=(0, 0, 255), thickness=6)

        # 將 ROI 定義為多邊形
        roi_polygon = Polygon(bottom_roi[0])

        # 霍夫變換來偵測直線
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=180, minLineLength=100, maxLineGap=50)

        # 使用斜率篩選
        filtered_lines = self.filter_lines(lines, slope_threshold=0.5)

        # 合併虛線
        merged_lines = self.merge_lines(filtered_lines)
        return roi_polygon, merged_lines
