from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer 
from opencv_engine import opencv_engine
from detect import Detector
import numpy as np
import cv2
from shapely.geometry import LineString, Polygon
from road_lane_dec import RoadLaneDetect
# videoplayer_state_dict = {
#  "stop":0,   
#  "play":1,
#  "pause":2     
# }

class video_controller(object):
    def __init__(self, video_path, ui):
        self.video_path = video_path
        self.ui = ui
        self.qpixmap_fix_width = 800 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 450
        self.current_frame_no = 0
        self.videoplayer_state = "stop"
        self.init_video_info()
        self.set_video_player()
        self.detector = Detector(source=self.video_path)
        self.roadLaneDetector = RoadLaneDetect()
        # 防止模型來不及回傳圖片結果
        self.last_frame = None

    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"] 
        self.video_fps = videoinfo["fps"] 
        self.video_total_frame_count = videoinfo["frame_count"] 
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"] 

        self.ui.slider_videoframe.setRange(0, self.video_total_frame_count-1)
        self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)

        # self.ui.slider_videoframe

    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        self.timer.start(int(1000 / self.video_fps)) # start Timer, here we set '1000ms//Nfps' while timeout one time
        # self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)

    def __get_frame_from_frame_no(self, frame_no):
        webcam = False
        if webcam:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            polygon, lines = self.roadLaneDetector.detect(frame)
            img, infos = self.detector.predict(frame, self.ui.nightMode)
            if self.ui.rdMode:
                img = self.PostRoadLane(img, lines, polygon)
            self.warningUser(infos)
            return img
        else:
            self.setslidervalue(frame_no)
            self.vc.set(1, frame_no)
            ret, frame = self.vc.read()
            self.ui.label_framecnt.setText(f"frame number: {frame_no}/{self.video_total_frame_count}")
            polygon, lines = self.roadLaneDetector.detect(frame)
            img, infos = self.detector.predict(frame, self.ui.nightMode)
            if self.ui.rdMode:
                img = self.PostRoadLane(img, lines, polygon)
            self.warningUser(infos)
            return img
    
    def __get_next_frame(self):
        ret, frame = self.vc.read();
        # img = self.detector.predict(frame)
        return frame
    
    def set_current_frame_no(self, frame_no):
        self.vc.set(1, frame_no) # bottleneck
    
    def __update_label_frame(self, frame):     
        if frame is None:
            frame = self.last_frame
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.last_frame = frame
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.label_videoframe.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center


    def play(self):
        self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"

    def timer_timeout_job(self):
        if(self.current_frame_no != self.video_total_frame_count):
            frame = self.__get_frame_from_frame_no(self.current_frame_no)
            self.__update_label_frame(frame)
            if (self.videoplayer_state == "play"):
                # self.__update_label_frame(self.__get_next_frame())
                self.current_frame_no += 1

            if (self.videoplayer_state == "stop"):
                self.current_frame_no = 0

            if (self.videoplayer_state == "pause"):
                self.current_frame_no = self.current_frame_no

    def PostRoadLane(self, img, merged_lines, bottom_roi):

        height, width = img.shape[:2]

        # 新增中下方矩形 ROI 區域
        cv2.polylines(img, [bottom_roi], isClosed=True, color=(0, 0, 255), thickness=6)
        # 將 ROI 定義為多邊形
        roi_polygon = Polygon(bottom_roi[0])
        
        # 檢查車道線是否與 ROI 區域相交，如果是，則顯示警告文字
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            line_coords = LineString([(x1, y1), (x2, y2)])
            if roi_polygonhttps://youtu.be/r4WOus-2Nh0?si=AhWB0POic4htOQvM.intersects(line_coords):
                cv2.putText(img, "Warning", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)

        # 繪製車道線
        line_img = np.zeros_like(img)
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # 合併偵測結果和原始影片
        return cv2.addWeighted(img, 1, line_img, 1, 0)

    def warningUser(self, position):
        self.ui.warning_left.setHidden(position[0])
        self.ui.warning_middle.setHidden(position[1])
        self.ui.warning_right.setHidden(position[2])

    def getslidervalue(self):
        self.current_frame_no = self.ui.slider_videoframe.value()
        self.set_current_frame_no(self.current_frame_no)

    def setslidervalue(self, value):
        self.ui.slider_videoframe.setValue(self.current_frame_no)