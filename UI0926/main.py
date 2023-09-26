from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import*

import cv2
import sys

from UI import Ui_MainWindow


class myMainWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        Ui_MainWindow.__init__(self)
        #Ui_MainWindow.__init__(self)
        #uic.loadUi("untitled.ui",self)
        self.setup()
        self.makeConnections()
        self.setWindowTitle("Video Player")
        
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
        video_frame = self.mediaPlayer.currentMedia()
        print(video_frame)
        #image = self.video_frame_to_image(video_frame)
        #cv2.imwrite('captured_frame.png', image)
        #print('Frame captured and saved as captured_frame.png')
    # def video_frame_to_image(self, frame):
    #     image = QImage(
    #         frame.bits(),
    #         frame.width(),
    #         frame.height(),
    #         frame.bytesPerLine(),
    #         QImage.Format_RGB888
    #     )

        # Convert QImage to OpenCV format
        # cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # return cv_image


# start.py
if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())