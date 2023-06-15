import argparse
import random
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class My_Ui(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(My_Ui, self).__init__(parent)
        self.setupUi(self)
        self.timer_video = QtCore.QTimer()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放

        # 权重初始文件名
        self.openfile_name_model = None

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.pushButton_weights, '选择weights文件',
                                                                  'pt/', "*.pt;")
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"权重打开失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.label_2.setText('所选weights文件地址为：' + str(self.openfile_name_model))

    # 模型初始化
    def model_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                            help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--img-size', nargs='+', type=int, default=640,
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用'--weights'中的权重来进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        # 若openfile_name_model不为空，则使用openfile_name_model权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("model initial done")
        QtWidgets.QMessageBox.information(self, u"！", u"模型初始化成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1139, 786)
        MainWindow.setStyleSheet("background-color:")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_img.setGeometry(QtCore.QRect(50, 220, 111, 61))
        self.pushButton_img.setStyleSheet("background-color:")
        self.pushButton_img.setObjectName("pushButton_img")
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.setGeometry(QtCore.QRect(50, 290, 111, 61))
        self.pushButton_video.setStyleSheet("background-color:")
        self.pushButton_video.setObjectName("pushButton_video")
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.setGeometry(QtCore.QRect(50, 360, 111, 61))
        self.pushButton_camera.setStyleSheet("background-color:")
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 30, 821, 521))
        self.label.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton_weights = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_weights.setGeometry(QtCore.QRect(50, 80, 111, 61))
        self.pushButton_weights.setStyleSheet("background-color:")
        self.pushButton_weights.setObjectName("pushButton_weights")
        self.pushButton_init = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_init.setGeometry(QtCore.QRect(50, 150, 111, 61))
        self.pushButton_init.setStyleSheet("background-color:")
        self.pushButton_init.setObjectName("pushButton_init")
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setGeometry(QtCore.QRect(170, 293, 61, 31))
        self.pushButton_stop.setStyleSheet("")
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.pushButton_finish = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_finish.setGeometry(QtCore.QRect(170, 320, 61, 31))
        self.pushButton_finish.setStyleSheet("")
        self.pushButton_finish.setObjectName("pushButton_finish")
        self.label_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 591, 1071, 131))
        self.label_2.setAutoFillBackground(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1139, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLO检测器"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_weights.setText(_translate("MainWindow", "选择权重"))
        self.pushButton_init.setText(_translate("MainWindow", "模型初始化"))
        self.pushButton_stop.setText(_translate("MainWindow", "暂停"))
        self.pushButton_finish.setText(_translate("MainWindow", "结束"))

    # 绑定信号与槽
    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.pushButton_weights.clicked.connect(self.open_model)
        self.pushButton_init.clicked.connect(self.model_init)
        self.pushButton_stop.clicked.connect(self.button_video_stop)
        self.pushButton_finish.clicked.connect(self.finish_detect)

    # 打开图片
    def button_image_open(self):
        self.label_2.setText('图片打开成功')
        name_list = []
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.label_2.setText("图片路径：" + img_name)
        if not img_name:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            print(pred)

            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite('prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(
            self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label.setScaledContents(True)  # 自适应界面大小

    # 打开视频
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        self.label_2.setText("视频地址：" + video_name)
        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            # 进行视频识别时，关闭其他按键点击功能
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)
            self.pushButton_init.setDisabled(True)
            self.pushButton_weights.setDisabled(True)
            self.label.setScaledContents(True)  # 自适应界面大小

    # 显示视频帧
    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            self.label_2.setText(label)  # PyQT页面打印类别和置信度
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()

            # 视频帧显示期间，禁用其他检测按键功能
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.pushButton_init.setDisabled(False)
            self.pushButton_weights.setDisabled(False)

    # 打开摄像头
    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_init.setDisabled(True)
                self.pushButton_weights.setDisabled(True)
                self.pushButton_stop.setDisabled(True)
                self.pushButton_finish.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_init.setDisabled(False)
            self.pushButton_weights.setDisabled(False)
            self.pushButton_stop.setDisabled(False)
            self.pushButton_finish.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    # 暂停/继续 视频
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.pushButton_stop.setText(u'继续')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.pushButton_stop.setText(u'暂停')

    # 结束视频检测
    def finish_detect(self):
        self.cap.release()  # 释放video_capture资源
        self.out.release()  # 释放video_writer资源
        self.label.clear()  # 清空label画布
        # 启动其他检测按键功能
        self.pushButton_video.setDisabled(False)
        self.pushButton_img.setDisabled(False)
        self.pushButton_camera.setDisabled(False)
        self.pushButton_init.setDisabled(False)
        self.pushButton_weights.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            self.pushButton_stop.setText(u'暂停')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = My_Ui()
    ui.show()
    sys.exit(app.exec())

