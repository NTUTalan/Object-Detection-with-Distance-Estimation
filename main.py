from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5 import QtWidgets
# from PyQt5.QtCore import QThread, pyqtSignal

import time
import os

initial_style = """QPushButton
{
   color:white;
	background-color:black;
	border:2px solid gray;
}

QPushButton:hover {
background-color: rgba(165, 205, 255,70%);
border:2px outset rgba(36, 36, 36,0);
} 
QPushButton:pressed {
	background-color: rgba(220, 225, 255,90%);
	border:4px outset rgba(36, 36, 36,0);
}
"""
yellow_style = "background-color: yellow;"


from UI import Ui_MainWindow
from video_controller import video_controller

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.openrd = False
        self.opennight = False

    def setup_control(self):
        self.ui.button_openfile.clicked.connect(self.open_file)    
        self.ui.button_rd.clicked.connect(self.openroadlanedetect) 
        self.ui.button_warning.clicked.connect(self.opennightdetect) 
        
    #左、中、右警示圖示隱藏
    def disable_left_warning(self):
        self.warning_left.setHidden(True)
    def disable_middle_warning(self):
        self.warning_middle.setHidden(True)
    def disable_right_warning(self):
        self.warning_right.setHidden(True)
    #左、中、右警示圖出現
    def enable_left_warning(self):
        self.warning_left.setHidden(False)
    def enable_middle_warning(self):
        self.warning_middle.setHidden(False)
    def enable_right_warning(self):
        self.warning_right.setHidden(False)     

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi *.mkv)") # start path        
        if filename:
            self.video_path = filename
            self.video_controller = video_controller(video_path=self.video_path,
                                                     ui=self.ui)
            self.ui.label_filepath.setText(f"video path: {self.video_path}")
            self.ui.button_play.clicked.connect(self.video_controller.play) # connect to function()
            self.ui.button_stop.clicked.connect(self.video_controller.stop)
            self.ui.button_pause.clicked.connect(self.video_controller.pause)
        else:
            print("no file selected")
    def init_video_info(self):
        self.ui.slider_videoframe.setRange(0, self.video_total_frame_count-1)
        self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)

    def __get_frame_from_frame_no(self, frame_no):
        self.setslidervalue(frame_no)

    def getslidervalue(self):
        self.current_frame_no = self.ui.slider_videoframe.value()

    def setslidervalue(self, value):
        self.ui.slider_videoframe.setValue(self.current_frame_no)
    
    def openroadlanedetect(self):
        if self.openrd == False:
            self.ui.button_rd.setStyleSheet(yellow_style)
        else:
            self.ui.button_rd.setStyleSheet(initial_style)
        self.openrd = not self.openrd
        self.ui.rdMode = self.openrd
    
    def opennightdetect(self):
        if self.opennight == False:
            self.ui.button_warning.setStyleSheet(yellow_style)
        else:
            self.ui.button_warning.setStyleSheet(initial_style)
        self.opennight = not self.opennight
        self.ui.nightMode = self.opennight

# start.py
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())