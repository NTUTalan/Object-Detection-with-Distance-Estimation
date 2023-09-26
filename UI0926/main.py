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

        # 获取当前时间
        current_time = datetime.datetime.now()
        # 格式化时间为字符串，例如："2023-09-26_14-30-15"
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
        self.button_pause.clicked.connect(self.mediaPlayer.pause)
        self.button_stop.clicked.connect(self.mediaPlayer.stop)

        self.button_shot.clicked.connect(self.capture_picture)
        #影片進度條更新
        self.mediaPlayer.mediaStatusChanged.connect(self.onMediaStatusChanged)
        

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

        # 生成新的文件名，例如："2023-09-26_14-30-15.jpg"
        new_filename = str(self.count)+".png"
        self.count+=1

        output_folder = self.formatted_time
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 生成新的文件名，例如："2023-09-26_14-30-15.jpg"，保存在"image"文件夹中
        new_filename = os.path.join(output_folder, new_filename)

        # 保存图像
        image.save(new_filename)

        print(f"图像已保存为: {new_filename}")

        
        


# start.py
if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())