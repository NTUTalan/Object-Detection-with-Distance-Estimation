from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import*

import os
import datetime

import cv2
import sys

from UI import Ui_MainWindow
#x=0

class myMainWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        Ui_MainWindow.__init__(self)
        #Ui_MainWindow.__init__(self)
        #uic.loadUi("untitled.ui",self)
        self.count=1
        self.setup()
        self.makeConnections()
        self.setWindowTitle("Video Player")

        # 獲取當前時間
        current_time = datetime.datetime.now()
        self.formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        
        
    def setup(self):
        self.videoOutput = self.makeVideoWidget()
        self.mediaPlayer = self.makeMediaPlayer()

    def makeMediaPlayer(self):
        mediaPlayer = QMediaPlayer(self)
        mediaPlayer.setVideoOutput(self.videoOutput)
        return mediaPlayer
    
    def makeVideoWidget(self):
        videoOutput = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(videoOutput)
        self.video_widget.setLayout(vbox)
        return videoOutput
    
    def makeConnections(self):
        self.button_openfile.clicked.connect(self.onActionAbrirTriggered)
        self.button_play.clicked.connect(self.mediaPlayer.play)
        #self.button_play.clicked.connect(self.enable_left_warning)
        self.button_pause.clicked.connect(self.mediaPlayer.pause)
        #self.button_pause.clicked.connect(self.disable_left_warning)
        self.button_stop.clicked.connect(self.mediaPlayer.stop)

        self.button_shot.clicked.connect(self.capture_picture)
        #影片進度條更新
        self.mediaPlayer.mediaStatusChanged.connect(self.onMediaStatusChanged)

    #左、中、右警示圖亮
    def enable_left_warning(self):
        self.warning_left.setEnabled(True)
    def enable_middle_warning(self):
        self.warning_middle.setEnabled(True)
    def enable_right_warning(self):
        self.warning_right.setEnabled(True)
    #左、中、右警示圖暗
    def disable_left_warning(self):
        self.warning_left.setEnabled(False)
    def disable_middle_warning(self):
        self.warning_middle.setEnabled(False)
    def disable_right_warning(self):
        self.warning_right.setEnabled(False)
        

    def onActionAbrirTriggered(self):
        
        path = QFileDialog.getOpenFileName(self)
        filepath =path[0]
        if filepath == "":
            return
        self.mediaPlayer.setMedia(QMediaContent(QUrl(filepath)))
        
        self.mediaPlayer.play()


    def updateSlider(self,position):
        self.slider_videoframe.setValue(position)
    

    def setPosition(self,position):
        self.mediaPlayer.setPosition(position)
    
    def onMediaStatusChanged(self):
        #print(QMediaPlayer.MediaStatus.LoadedMedia,"yo")
        if QMediaPlayer.MediaStatus.LoadedMedia == 3 :
            self.slider_videoframe.setRange(0, self.mediaPlayer.duration())
            self.mediaPlayer.positionChanged.connect(self.updateSlider)
            self.slider_videoframe.sliderMoved.connect(self.setPosition)
            #print("影片總長度：", duration)
    def capture_picture(self):
        screen = QScreen.grabWindow(QApplication.primaryScreen(), self.winId())
        image = QPixmap(screen)

        new_filename = str(self.count)+".png"
        self.count+=1

        output_folder = self.formatted_time
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        new_filename = os.path.join(output_folder, new_filename)

        # 存圖
        image.save(new_filename)

        print(f"圖片已保存為: {new_filename}")

        
        


# start.py
if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())