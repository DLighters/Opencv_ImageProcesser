import sys
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMainWindow
from PyQt5.QtCore import QDir, pyqtSlot

from main_window import Ui_MainWindow


class Image_Viewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename = ""
        self.cv_img = np.ndarray(())
        self.raw_img = np.ndarray(())
        self.copy_img = np.ndarray(())
        self.last_img = []
        self.next_img = []

    @pyqtSlot()
    def on_actionOpen_triggered(self):
        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        self.filename, filtUsed = QFileDialog.getOpenFileName(self, title, curPath, filt)
        if self.filename == "":
            return
        pixmap = QPixmap(self.filename)
        if pixmap.width() > 400:
            pixRatio = pixmap.width() / 400
            pixmap.setDevicePixelRatio(pixRatio)
        self.ui.labelImg.setPixmap(pixmap)
        self.ui.labelImg.setScaledContents(True)
        self.cv_img = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), 1)
        self.raw_img = self.cv_img
        self.copy_img = self.cv_img

    @pyqtSlot()
    def on_actionSave_triggered(self):
        curPath = QDir.currentPath()
        title = "保存图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        savename, filtUsed = QFileDialog.getSaveFileName(self, title, curPath, filt)
        if savename == "":
            return
        cv2.imwrite(savename, self.cv_img)

    def on_horizontalSliderHue_pressed(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

    def on_horizontalSliderHue_valueChanged(self, value):
        if self.filename == "":
            return

        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        self.cv_img[:, :, 0] = np.int32(self.cv_img[:, :, 0] + value * 0.08) % 180
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_HSV2BGR)
        self.refreshShow(self.cv_img)

    def on_horizontalSliderSaturation_pressed(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

    def on_horizontalSliderSaturation_valueChanged(self, value):
        if self.filename == "":
            return

        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        self.copy_img = cv2.cvtColor(self.copy_img, cv2.COLOR_BGR2HSV)

        self.cv_img[:, :, 1] = self.copy_img[:, :, 1] * (value / 75)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_HSV2BGR)
        self.copy_img = cv2.cvtColor(self.copy_img, cv2.COLOR_HSV2BGR)
        self.refreshShow(self.cv_img)

    def on_horizontalSliderValue_pressed(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

    def on_horizontalSliderValue_valueChanged(self, value):
        if self.filename == "":
            return

        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        self.copy_img = cv2.cvtColor(self.copy_img, cv2.COLOR_BGR2HSV)

        self.cv_img[:, :, 2] = self.copy_img[:, :, 2] * (value / 75)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_HSV2BGR)
        self.copy_img = cv2.cvtColor(self.copy_img, cv2.COLOR_HSV2BGR)
        self.refreshShow(self.cv_img)

    # 高斯滤波
    @pyqtSlot()
    def on_pushButtonBlur_clicked(self):
        if self.filename == "":
            return
        value = self.ui.horizontalSliderAverage.value() + 1

        self.last_img.append(self.cv_img)

        self.cv_img = cv2.blur(self.cv_img, ksize=(value, value))
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    @pyqtSlot()
    def on_pushButtonBox_clicked(self):
        if self.filename == "":
            return

        self.last_img.append(self.cv_img)

        value = self.ui.horizontalSliderBox.value() + 1
        self.cv_img = cv2.boxFilter(self.cv_img, ksize=(value, value), ddepth=-1)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    @pyqtSlot()
    def on_pushButtonReturnRaw_clicked(self):
        if self.filename == "":
            return
        self.cv_img = self.raw_img
        self.refreshShow(self.cv_img)

    @pyqtSlot()
    def on_pushButtonShowRaw_pressed(self):
        if self.filename == "":
            return
        self.refreshShow(self.raw_img)

    @pyqtSlot()
    def on_pushButtonShowRaw_released(self):
        if self.filename == "":
            return
        self.refreshShow(self.cv_img)

    def on_actionRevoke_triggered(self):
        if len(self.last_img) == 0:
            return
        self.next_img.append(self.cv_img)
        self.copy_img = self.cv_img = self.last_img.pop()
        self.refreshShow(self.cv_img)

    def on_actionRedo_triggered(self):
        if len(self.next_img) == 0:
            return
        self.last_img.append(self.cv_img)
        self.copy_img = self.cv_img = self.next_img.pop()
        self.refreshShow(self.cv_img)

    def on_actionCut_triggered(self):
        if self.filename == "":
            return

        self.last_img.append(self.cv_img)

        roi = cv2.selectROI(windowName="Cropping", img=self.cv_img, showCrosshair=True, fromCenter=False)

        x, y, w, h = roi
        if roi != (0, 0, 0, 0):
            self.cv_img = self.cv_img[y:y + h, x:x + w].copy()
        self.copy_img = self.cv_img

        cv2.destroyAllWindows()
        self.refreshShow(self.cv_img)

    def refreshShow(self, img):
        # 提取图像的通道和尺寸，用于将OpenCV下的image转换成Qimage
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img.data, width, height, bytesPerline, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        # 将QImage显示出来
        if pixmap.width() > 400:
            pixRatio = pixmap.width() / 400
            pixmap.setDevicePixelRatio(pixRatio)
        self.ui.labelImg.setPixmap(pixmap)
        self.ui.labelImg.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = Image_Viewer()
    main_widget.show()
    sys.exit(app.exec_())
