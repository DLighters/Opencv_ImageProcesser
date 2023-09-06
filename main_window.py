# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1098, 672)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.labelImg = QtWidgets.QLabel(self.centralwidget)
        self.labelImg.setGeometry(QtCore.QRect(40, 50, 701, 451))
        self.labelImg.setFrameShape(QtWidgets.QFrame.Box)
        self.labelImg.setFrameShadow(QtWidgets.QFrame.Plain)
        self.labelImg.setText("")
        self.labelImg.setObjectName("labelImg")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(790, 520, 271, 30))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonBox = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButtonBox.setObjectName("pushButtonBox")
        self.horizontalLayout_2.addWidget(self.pushButtonBox)
        self.horizontalSliderBox = QtWidgets.QSlider(self.layoutWidget)
        self.horizontalSliderBox.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderBox.setObjectName("horizontalSliderBox")
        self.horizontalLayout_2.addWidget(self.horizontalSliderBox)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(790, 480, 271, 30))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonBlur = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButtonBlur.setObjectName("pushButtonBlur")
        self.horizontalLayout.addWidget(self.pushButtonBlur)
        self.horizontalSliderAverage = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSliderAverage.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderAverage.setObjectName("horizontalSliderAverage")
        self.horizontalLayout.addWidget(self.horizontalSliderAverage)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(40, 520, 700, 30))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(398, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.pushButtonShowRaw = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButtonShowRaw.setObjectName("pushButtonShowRaw")
        self.horizontalLayout_3.addWidget(self.pushButtonShowRaw)
        self.pushButtonReturnRaw = QtWidgets.QPushButton(self.layoutWidget2)
        self.pushButtonReturnRaw.setObjectName("pushButtonReturnRaw")
        self.horizontalLayout_3.addWidget(self.pushButtonReturnRaw)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(790, 50, 261, 88))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.horizontalSliderHue = QtWidgets.QSlider(self.widget)
        self.horizontalSliderHue.setMinimum(1)
        self.horizontalSliderHue.setMaximum(100)
        self.horizontalSliderHue.setProperty("value", 50)
        self.horizontalSliderHue.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderHue.setObjectName("horizontalSliderHue")
        self.horizontalLayout_4.addWidget(self.horizontalSliderHue)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.horizontalSliderSaturation = QtWidgets.QSlider(self.widget)
        self.horizontalSliderSaturation.setMinimum(1)
        self.horizontalSliderSaturation.setMaximum(100)
        self.horizontalSliderSaturation.setProperty("value", 50)
        self.horizontalSliderSaturation.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderSaturation.setObjectName("horizontalSliderSaturation")
        self.horizontalLayout_5.addWidget(self.horizontalSliderSaturation)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.horizontalSliderValue = QtWidgets.QSlider(self.widget)
        self.horizontalSliderValue.setMinimum(1)
        self.horizontalSliderValue.setMaximum(100)
        self.horizontalSliderValue.setProperty("value", 50)
        self.horizontalSliderValue.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderValue.setObjectName("horizontalSliderValue")
        self.horizontalLayout_6.addWidget(self.horizontalSliderValue)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1098, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionRevoke = QtWidgets.QAction(MainWindow)
        self.actionRevoke.setObjectName("actionRevoke")
        self.actionRedo = QtWidgets.QAction(MainWindow)
        self.actionRedo.setObjectName("actionRedo")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionCut = QtWidgets.QAction(MainWindow)
        self.actionCut.setObjectName("actionCut")
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu_2.addAction(self.actionRevoke)
        self.menu_2.addAction(self.actionRedo)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.actionCut)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        self.horizontalSliderHue.sliderPressed.connect(MainWindow.on_horizontalSliderHue_pressed) # type: ignore
        self.horizontalSliderSaturation.sliderPressed.connect(MainWindow.on_horizontalSliderSaturation_pressed) # type: ignore
        self.horizontalSliderValue.sliderPressed.connect(MainWindow.on_horizontalSliderValue_pressed) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButtonBox.setText(_translate("MainWindow", "方框模糊"))
        self.pushButtonBlur.setText(_translate("MainWindow", "均值模糊"))
        self.pushButtonShowRaw.setText(_translate("MainWindow", "查看原图"))
        self.pushButtonReturnRaw.setText(_translate("MainWindow", "恢复原图"))
        self.label.setText(_translate("MainWindow", "色相  "))
        self.label_2.setText(_translate("MainWindow", "饱和度"))
        self.label_3.setText(_translate("MainWindow", "明度  "))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "编辑"))
        self.actionRevoke.setText(_translate("MainWindow", "撤销"))
        self.actionRedo.setText(_translate("MainWindow", "重做"))
        self.actionOpen.setText(_translate("MainWindow", "打开文件"))
        self.actionSave.setText(_translate("MainWindow", "保存文件"))
        self.actionCut.setText(_translate("MainWindow", "裁剪"))
import pic_ui_rc