import sys

import cv2
import numpy as np
from PyQt5.QtCore import QDir, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import Filter
import Image
import Subject
import Text
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
        self.cv_img = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)
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

        # Point 1: 生成与黑色部分对应的mask图像
        mask = np.all(self.cv_img[:, :, :] == [0, 0, 0], axis=-1)

        # Point 2: 将图片从三通道转为四通道
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2BGRA)

        # Point3:  以mask图像为基础，使黑色部分透明化
        self.cv_img[mask, 3] = 0

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

    def on_actionSelectForeground_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)
        self.cv_img = Subject.selectForeground(self.cv_img)

        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionRectSelect_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        # self.cv_img = Subject.rectSelect(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionAffineTrans_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.affineTrans(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionPerspectiveTrans_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.perspectiveTrans(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionConvexLens_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.convexLens(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionConcaveLens_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.concaveLens(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionSinTrans_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.sinTrans(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionVortexFilter_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.vortexFilter(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionVignetting_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.vignetting(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionRelief_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.relief(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionRegionBlur_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.regionBlur(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionMosaic_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.mosaic(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionSketchFilter_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.sketchFilter(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionStyleConversion_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.styleConversion(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionMergeImg_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        bg_filename, bg_filtUsed = QFileDialog.getOpenFileName(self, title, curPath, filt)
        if bg_filename == "":
            return
        bg_img = cv2.imdecode(np.fromfile(bg_filename, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)

        self.cv_img = Image.mergeImg(bg_img, self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionRegionBlur_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.regionBlur(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)


    def on_actionMosaic_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.mosaic(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionSketchFilter_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Filter.sketchFilter(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionHideImg_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        filename_2, filtUsed = QFileDialog.getOpenFileName(self, title, curPath, filt)
        if filename_2 == "":
            return
        img_2 = cv2.imdecode(np.fromfile(filename_2, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)

        self.cv_img = Image.hideImg(self.cv_img, img_2)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionColorSeparation_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Image.colorDivide(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionThreshold_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Image.threshold(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionEditBackground_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        curPath = QDir.currentPath()
        title = "选择图片"
        filt = "所有文件(*.*);;图片文件(*.jpg *.png *.gif)"
        bg_filename, bg_filtUsed = QFileDialog.getOpenFileName(self, title, curPath, filt)

        if bg_filename == "":
            return

        background_img = cv2.imdecode(np.fromfile(bg_filename, dtype=np.uint8), flags=cv2.IMREAD_UNCHANGED)
        self.cv_img = Image.changeBackground(background_img, self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionIDPhoto_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Image.makeIDcard(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionDeleteObject_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Subject.object_removal(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionZoomInOutObject_triggered(self):
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Subject.object_resize(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)


    def on_actionInsertText_triggered(self):
        text = self.ui.textEdit.toPlainText()
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Text.insertText(self.cv_img, text)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def on_actionDeleteText_triggered(self):
        text = self.ui.textEdit.toPlainText()
        if self.filename == "":
            return
        self.last_img.append(self.cv_img)

        self.cv_img = Text.deleteText(self.cv_img)
        self.copy_img = self.cv_img
        self.refreshShow(self.cv_img)

    def refreshShow(self, img):
        # 提取图像的通道和尺寸，用于将OpenCV下的image转换成Qimage
        if img is None:
            return
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img.data, width, height, bytesPerline, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        # 将QImage显示出来
        self.ui.labelImg.setScaledContents(True)
        self.ui.labelImg.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_widget = Image_Viewer()
    main_widget.show()
    sys.exit(app.exec_())
