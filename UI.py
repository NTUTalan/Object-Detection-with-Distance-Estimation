# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI3.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_stop = QtWidgets.QPushButton(self.centralwidget)
        self.button_stop.setGeometry(QtCore.QRect(650, 480, 111, 41))
        self.button_stop.setObjectName("button_stop")
        self.button_pause = QtWidgets.QPushButton(self.centralwidget)
        self.button_pause.setGeometry(QtCore.QRect(520, 480, 111, 41))
        self.button_pause.setObjectName("button_pause")
        self.button_play = QtWidgets.QPushButton(self.centralwidget)
        self.button_play.setGeometry(QtCore.QRect(390, 480, 111, 41))
        self.button_play.setObjectName("button_play")
        self.button_openfile = QtWidgets.QPushButton(self.centralwidget)
        self.button_openfile.setGeometry(QtCore.QRect(70, 370, 111, 41))
        self.button_openfile.setObjectName("button_openfile")
        self.label_videoframe = QtWidgets.QLabel(self.centralwidget)
        self.label_videoframe.setGeometry(QtCore.QRect(80, 50, 641, 271))
        self.label_videoframe.setStyleSheet("")
        self.label_videoframe.setObjectName("label_videoframe")
        self.label_framecnt = QtWidgets.QLabel(self.centralwidget)
        self.label_framecnt.setGeometry(QtCore.QRect(520, 380, 221, 21))
        self.label_framecnt.setStyleSheet("")
        self.label_framecnt.setObjectName("label_framecnt")
        self.label_filepath = QtWidgets.QLabel(self.centralwidget)
        self.label_filepath.setGeometry(QtCore.QRect(80, 480, 221, 21))
        self.label_filepath.setStyleSheet("")
        self.label_filepath.setObjectName("label_filepath")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_stop.setText(_translate("MainWindow", "Stop"))
        self.button_pause.setText(_translate("MainWindow", "Pause"))
        self.button_play.setText(_translate("MainWindow", "Play"))
        self.button_openfile.setText(_translate("MainWindow", "OpenFile"))
        self.label_videoframe.setText(_translate("MainWindow", "video_player"))
        self.label_framecnt.setText(_translate("MainWindow", "current_frame/total_frame"))
        self.label_filepath.setText(_translate("MainWindow", "file path:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())