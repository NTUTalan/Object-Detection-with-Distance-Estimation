# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1163, 822)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.slider_videoframe = QtWidgets.QSlider(self.centralwidget)
        self.slider_videoframe.setGeometry(QtCore.QRect(370, 660, 691, 51))
        self.slider_videoframe.setOrientation(QtCore.Qt.Horizontal)
        self.slider_videoframe.setObjectName("slider_videoframe")
        self.button_stop = QtWidgets.QPushButton(self.centralwidget)
        self.button_stop.setGeometry(QtCore.QRect(230, 590, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(24)
        self.button_stop.setFont(font)
        self.button_stop.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"border:2px solid gray;")
        self.button_stop.setObjectName("button_stop")
        self.button_shot = QtWidgets.QPushButton(self.centralwidget)
        self.button_shot.setGeometry(QtCore.QRect(1010, 590, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(24)
        self.button_shot.setFont(font)
        self.button_shot.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"\n"
"color:white;\n"
"border:2px solid gray;\n"
"\n"
"")
        self.button_shot.setObjectName("button_shot")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 1151, 821))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel{\n"
"    background-color:grey;\n"
"\n"
"}")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_framecnt = QtWidgets.QLabel(self.centralwidget)
        self.label_framecnt.setGeometry(QtCore.QRect(520, 720, 251, 31))
        self.label_framecnt.setStyleSheet("QLabel{\n"
"    border-radius: 5px;\n"
"    background-color :gainsboro\n"
"\n"
"}")
        self.label_framecnt.setObjectName("label_framecnt")
        self.label_filepath = QtWidgets.QLabel(self.centralwidget)
        self.label_filepath.setGeometry(QtCore.QRect(680, 580, 281, 31))
        self.label_filepath.setStyleSheet("QLabel{\n"
"    border-radius: 5px;\n"
"    background-color :gainsboro\n"
"\n"
"}")
        self.label_filepath.setObjectName("label_filepath")
        self.button_warning = QtWidgets.QPushButton(self.centralwidget)
        self.button_warning.setGeometry(QtCore.QRect(790, 590, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.button_warning.setFont(font)
        self.button_warning.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"border:2px solid gray;")
        self.button_warning.setObjectName("button_warning")
        self.button_rd = QtWidgets.QPushButton(self.centralwidget)
        self.button_rd.setGeometry(QtCore.QRect(604, 590, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.button_rd.setFont(font)
        self.button_rd.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"border:2px solid gray;")
        self.button_rd.setObjectName("button_rd")
        self.button_pause = QtWidgets.QPushButton(self.centralwidget)
        self.button_pause.setGeometry(QtCore.QRect(414, 590, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.button_pause.setFont(font)
        self.button_pause.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"border:2px solid gray;")
        self.button_pause.setObjectName("button_pause")
        self.button_play = QtWidgets.QPushButton(self.centralwidget)
        self.button_play.setGeometry(QtCore.QRect(44, 590, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(24)
        self.button_play.setFont(font)
        self.button_play.setStyleSheet("\n"
"color:white;\n"
"background-color:black;\n"
"\n"
"color:white;\n"
"border:2px solid gray;\n"
"\n"
"")
        self.button_play.setObjectName("button_play")
        self.button_openfile = QtWidgets.QPushButton(self.centralwidget)
        self.button_openfile.setGeometry(QtCore.QRect(170, 660, 111, 41))
        self.button_openfile.setObjectName("button_openfile")
        self.warning_left = QtWidgets.QLabel(self.centralwidget)
        self.warning_left.setHidden(True)
        self.warning_left.setGeometry(QtCore.QRect(150, 70, 181, 151))
        self.warning_left.setStyleSheet("")
        self.warning_left.setText("")
        self.warning_left.setPixmap(QtGui.QPixmap("resource/img/warning.png"))
        self.warning_left.setObjectName("warning_left")
        self.warning_middle = QtWidgets.QLabel(self.centralwidget)
        self.warning_middle.setHidden(True)
        self.warning_middle.setGeometry(QtCore.QRect(510, 70, 171, 151))
        self.warning_middle.setText("")
        self.warning_middle.setPixmap(QtGui.QPixmap("resource/img/warning.png"))
        self.warning_middle.setObjectName("warning_middle")
        self.warning_right = QtWidgets.QLabel(self.centralwidget)
        self.warning_right.setHidden(True)
        self.warning_right.setGeometry(QtCore.QRect(880, 70, 181, 151))
        self.warning_right.setText("")
        self.warning_right.setPixmap(QtGui.QPixmap("resource/img/warning.png"))
        self.warning_right.setObjectName("warning_right")
        self.label_videoframe = QtWidgets.QLabel(self.centralwidget)
        self.label_videoframe.setGeometry(QtCore.QRect(50, 70, 1091, 511))
        self.label_videoframe.setStyleSheet("QLabel{\n"
"    border-radius: 2px;\n"
"    background-color :gainsboro\n"
"\n"
"\n"
"}")
        self.label_videoframe.setText("")
        self.label_videoframe.setObjectName("label_videoframe")
        self.label_filepath.raise_()
        self.label.raise_()
        self.button_warning.raise_()
        self.slider_videoframe.raise_()
        self.label_framecnt.raise_()
        self.button_shot.raise_()
        self.button_rd.raise_()
        self.button_pause.raise_()
        self.button_play.raise_()
        self.button_openfile.raise_()
        self.button_stop.raise_()
        self.label_videoframe.raise_()
        self.warning_middle.raise_()
        self.warning_right.raise_()
        self.warning_left.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1163, 25))
        self.menubar.setObjectName("menubar")
        self.menuArchive = QtWidgets.QMenu(self.menubar)
        self.menuArchive.setObjectName("menuArchive")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbrir = QtWidgets.QAction(MainWindow)
        self.actionAbrir.setObjectName("actionAbrir")
        self.menuArchive.addAction(self.actionAbrir)
        self.menubar.addAction(self.menuArchive.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_stop.setText(_translate("MainWindow", "⟳"))
        self.button_shot.setText(_translate("MainWindow", "📷"))
        self.label_framecnt.setText(_translate("MainWindow", "  current_frame/total_frame"))
        self.label_filepath.setText(_translate("MainWindow", "file path:"))
        self.button_warning.setText(_translate("MainWindow", "⚠️"))
        self.button_rd.setText(_translate("MainWindow", "RD"))
        self.button_pause.setText(_translate("MainWindow", "||"))
        self.button_play.setText(_translate("MainWindow", "▶️"))
        self.button_openfile.setText(_translate("MainWindow", "OpenFile"))
        self.menuArchive.setTitle(_translate("MainWindow", "Archive"))
        self.actionAbrir.setText(_translate("MainWindow", "Abrir"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
