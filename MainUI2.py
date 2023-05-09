import argparse
import random
import sys
import configparser

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import win32api
import win32con
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog

from action import *
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, checkWay, checkCD
from utils.plots import plot_one_box
from utils.torch_utils import select_device

from ui_setting import Ui_Setting

'''class Movement:
    def __init__(self, play_cd=2.5):
        self.mouse_locate = []
        self.mouse_history = []
        self.gesture_locate = []
        self.history_pop = [(310, 155)]
        self.mouse_point = (0, 0)
        self.play_cd = play_cd
        self.qImg = None

        
    def action(self, label, det):
        # self.label = label
        if label == '1':
            self.mouseMovement(det)
        elif label == 'OK':
            if checkCD(time.time(), self.play_cd):
                self.gestureAction()
        elif label == 'OK':
            if checkCD(time.time(), 0.2):
                self.fourWayAction(det)
        

    def mouseMovement(self, det):
        if len(self.mouse_locate) == 0 and det.shape[0] != 0:
            mask = (det[:, -1] == 0)
            if mask.any():
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
        elif len(self.mouse_locate) == 2:
            if det.shape[0] != 0:
                mask = (det[:, -1] == 0)
                if mask.any():
                    self.mouse_locate.append(
                        xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
                self.mouse_locate.pop(0)
        elif det.shape[0] != 0:
            mask = (det[:, -1] == 0)
            if mask.any():
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])

        if len(self.mouse_locate) == 2:
            x1, y1, w1, h1 = self.mouse_locate[0]
            x2, y2, w2, h2 = self.mouse_locate[1]
            mouse_shift = [x2 - x1, y2 - y1]
            self.mouse_point = (x2, int(y2 - h2 // 2))

            if len(self.mouse_history) == 0:
                self.mouse_history.append((x1, int(y1 - h1 // 2)))
                self.mouse_history.append((x2, int(y2 - h2 // 2)))
            elif len(self.mouse_history) > 10:
                self.history_pop.append(self.mouse_history[0])
                setCursor = self.mouse_history[8]
                x, y = setCursor
                win32api.SetCursorPos((4 * (640 - x), 4 * y))  # 設置滑鼠座標
                self.mouse_history.pop(0)
            else:
                self.mouse_history.append((x2, int(y2 - h2 // 2)))


    def gestureAction(self):
        win32api.keybd_event(179, 0)
        win32api.keybd_event(179, 0, win32con.KEYEVENTF_KEYUP)

    def fourWayAction(self, det):
        if det.shape[0] != 0:
            mask = (det[:, -1] == 1)
            if mask.any():
                self.gesture_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
        if len(self.gesture_locate) == 2:
            x1, y1, w1, h1 = self.gesture_locate[0]
            x2, y2, w2, h2 = self.gesture_locate[1]
            self.gesture_locate.pop()
            self.gesture_locate.pop()
            way = checkWay(x1, x2, y1, y2)
            if way == 1:
                n = 5
                while n:
                    win32api.keybd_event(win32con.VK_VOLUME_UP, 0)
                    win32api.keybd_event(win32con.VK_VOLUME_UP, 0, win32con.KEYEVENTF_KEYUP)
                    n = n - 1
            elif way == 2:
                n = 5
                while n:
                    win32api.keybd_event(win32con.VK_VOLUME_DOWN, 0)
                    win32api.keybd_event(win32con.VK_VOLUME_DOWN, 0, win32con.KEYEVENTF_KEYUP)
                    n = n - 1
'''


class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None, play_cd=2.5):
        super(Main, self).__init__(parent)
        self.setting = Setting()

        self.setWindowIcon(QIcon("images/UI/kk.jpg"))
        self.setupUi(self)
        self.timer_video = QtCore.QTimer()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.mouse_locate = []
        self.mouse_history = []
        self.gesture_locate = []
        self.history_pop = [(310, 155)]
        self.mouse_point = (0, 0)
        self.play_cd = play_cd

        # 权重初始文件名
        self.openfile_name_model = None

    def action(self, label, det):
        # '1', '2', '5', '0', 'OK', 'Good' .fourWayAction "mask"
        if label == '1':
            self.mouseMovement(det)
        elif label == '2':
            if checkCD(time.time(), 1.5):
                mouseRightClk()
        elif label == '0':
            if checkCD(time.time(), self.play_cd):
                self.resetAllGes()
        elif label == '5':
            if checkCD(time.time(), 0.2):
                self.fourWayAction(det)

    def resetAllGes(self):
        self.mouse_locate = []
        self.mouse_history = []
        self.gesture_locate = []

    def mouseMovement(self, det):
        if len(self.mouse_locate) == 0 and det.shape[0] != 0:
            mask = (det[:, -1] == 0)
            if mask.any():
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
        elif len(self.mouse_locate) == 2:
            if det.shape[0] != 0:
                mask = (det[:, -1] == 0)
                if mask.any():
                    self.mouse_locate.append(
                        xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
                self.mouse_locate.pop(0)
        elif det.shape[0] != 0:
            mask = (det[:, -1] == 0)
            if mask.any():
                self.mouse_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])

        if len(self.mouse_locate) == 2:
            x1, y1, w1, h1 = self.mouse_locate[0]
            x2, y2, w2, h2 = self.mouse_locate[1]
            mouse_shift = [x2 - x1, y2 - y1]
            self.mouse_point = (x2, int(y2 - h2 // 2))

            if len(self.mouse_history) == 0:
                self.mouse_history.append((x1, int(y1 - h1 // 2)))
                self.mouse_history.append((x2, int(y2 - h2 // 2)))
            elif len(self.mouse_history) > 10:
                self.history_pop.append(self.mouse_history[0])
                setCursor = self.mouse_history[8]
                x, y = setCursor
                win32api.SetCursorPos((4 * (640 - x), 4 * y))  # 設置滑鼠座標
                self.mouse_history.pop(0)
            else:
                self.mouse_history.append((x2, int(y2 - h2 // 2)))

    def fourWayAction(self, det):
        # '1', '2', '5', '0', 'OK', 'Good' .fourWayAction "mask"
        if det.shape[0] != 0:
            mask = (det[:, -1] == 2)
            if mask.any():
                self.gesture_locate.append(
                    xyxy2xywh(det[mask][-1, :4].view(1, 4)).cpu().numpy().astype(int)[0])
        if len(self.gesture_locate) == 2:
            x1, y1, w1, h1 = self.gesture_locate[0]
            x2, y2, w2, h2 = self.gesture_locate[1]
            self.gesture_locate.pop()
            self.gesture_locate.pop()
            way = checkWay(x1, x2, y1, y2)
            if way == 1:
                volumeUp()
            elif way == 2:
                volumeDown()

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.pushButton_pt, 'Select weights',
                                                                  'pt/', "*.pt;")
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open weights", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.label_2.setText('Weights path：' + str(self.openfile_name_model))

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
        QtWidgets.QMessageBox.information(self, u"ok", u"Model initialize success")

    # ui.py文件的函数
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1092, 697)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 190, 640, 360))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(770, 180, 171, 381))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 30, 671, 151))
        self.layoutWidget.setMinimumSize(QtCore.QSize(40, 60))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_pt = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_pt.setMinimumSize(QtCore.QSize(40, 60))
        self.pushButton_pt.setObjectName("pushButton_pt")
        self.horizontalLayout.addWidget(self.pushButton_pt)
        self.pushButton_init = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_init.setMinimumSize(QtCore.QSize(40, 60))
        self.pushButton_init.setObjectName("pushButton_init")
        self.horizontalLayout.addWidget(self.pushButton_init)
        self.pushButton_sht = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_sht.setMinimumSize(QtCore.QSize(40, 60))
        self.pushButton_sht.setObjectName("pushButton_sht")
        self.horizontalLayout.addWidget(self.pushButton_sht)
        self.pushButton_exit = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_exit.setMinimumSize(QtCore.QSize(40, 60))
        self.pushButton_exit.setObjectName("pushButton_exit")
        self.horizontalLayout.addWidget(self.pushButton_exit)
        self.pushButton_setting = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_setting.setGeometry(QtCore.QRect(770, 70, 162, 60))
        self.pushButton_setting.setMinimumSize(QtCore.QSize(40, 60))
        self.pushButton_setting.setObjectName("pushButton_setting")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1092, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton_setting.clicked.connect(self.setting.show)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "yolov7手勢操作電腦系統"))
        self.pushButton_pt.setText(_translate("MainWindow", "Select weights"))
        self.pushButton_init.setText(_translate("MainWindow", "Model initialize"))
        # self.pushButton_openimg.setText(_translate("MainWindow", "Open picture"))
        # self.pushButton_video.setText(_translate("MainWindow", "Open video"))
        self.pushButton_sht.setText(_translate("MainWindow", "Open Camera"))
        # self.pushButton_stop.setText(_translate("MainWindow", "Pause(video)"))
        self.pushButton_exit.setText(_translate("MainWindow", "Stop(video)"))
        self.label.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", ""))
        self.pushButton_setting.setText(_translate("MainWindow", "Setting"))

    # 绑定信号与槽
    def init_slots(self):
        #        self.pushButton_openimg.clicked.connect(self.button_image_open)
        #       self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_sht.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.pushButton_pt.clicked.connect(self.open_model)
        self.pushButton_init.clicked.connect(self.model_init)

    #        self.pushButton_stop.clicked.connect(self.button_video_stop)

    # 打开图片
    def button_image_open(self):
        self.label_2.setText('图片打开成功')
        name_list = []
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.label_2.setText("图片路径：" + img_name)
        if not img_name:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open picture", buttons=QtWidgets.QMessageBox.Ok,
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
            self, "Open video", "", "*.mp4;;*.avi;;All Files(*)")
        self.label_2.setText("Video path：" + video_name)
        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"Failed to open video", buttons=QtWidgets.QMessageBox.Ok,
                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            # 进行视频识别时，关闭其他按键点击功能
            #    self.pushButton_video.setDisabled(True)
            #   self.pushButton_openimg.setDisabled(True)
            self.pushButton_sht.setDisabled(True)
            self.pushButton_init.setDisabled(True)
            self.pushButton_pt.setDisabled(True)
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
                            L = self.names[int(cls)]
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            self.label_2.setText(label)  # PyQT页面打印类别和置信度
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                            self.action(L, det)
                        if L == '1':
                            cv2.circle(showimg, self.mouse_point, 2, (0, 0, 255), -1)
                            for m_id in range(1, len(self.mouse_history)):
                                cv2.line(showimg, self.mouse_history[m_id], self.mouse_history[m_id - 1], (0, 0, 255),
                                         2)

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
            #      self.pushButton_video.setDisabled(False)
            #      self.pushButton_openimg.setDisabled(False)
            self.pushButton_sht.setDisabled(False)
            self.pushButton_init.setDisabled(False)
            self.pushButton_pt.setDisabled(False)

    # 打开摄像头
    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"Failed to open camera", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                #                self.pushButton_video.setDisabled(True)
                #                self.pushButton_openimg.setDisabled(True)
                self.pushButton_init.setDisabled(True)
                self.pushButton_pt.setDisabled(True)
                #                self.pushButton_stop.setDisabled(True)
                self.pushButton_exit.setDisabled(True)
                self.pushButton_sht.setText(u"Turn off camera")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            #            self.pushButton_video.setDisabled(False)
            #            self.pushButton_openimg.setDisabled(False)
            self.pushButton_init.setDisabled(False)
            self.pushButton_pt.setDisabled(False)
            #           self.pushButton_stop.setDisabled(False)
            self.pushButton_exit.setDisabled(False)
            self.pushButton_sht.setText(u"Detect by Camera")

    # 暂停/继续 视频
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            #         self.pushButton_stop.setText(u'Continue')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
        # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            #         self.pushButton_stop.setText(u'Pause')

    # 结束视频检测
    def finish_detect(self):
        self.cap.release()  # 释放video_capture资源
        self.out.release()  # 释放video_writer资源
        self.label.clear()  # 清空label画布
        # 启动其他检测按键功能
        #       self.pushButton_video.setDisabled(False)
        #       self.pushButton_openimg.setDisabled(False)
        self.pushButton_sht.setDisabled(False)
        self.pushButton_init.setDisabled(False)
        self.pushButton_pt.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            #         self.pushButton_stop.setText(u'Pause')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)


class Setting(QtWidgets.QMainWindow):
    def __init__(self):
        super(Setting, self).__init__()
        self.comboList = None
        self.optionsDefault = ['volumeUp', 'volumeDown', 'mediaPause', 'volumeMute', 'mediaPrevTrack',
                               'mediaNextTrack', 'windowMinimize', 'windowMaximize', 'windowBackToscreen']
        self.config = configparser.ConfigParser()
        self.options = None
        self.gestureActionsgestureActions = None
        self.ui = Ui_Setting()
        self.ui.setupUi(self)


        #self.setupSetting(self, Setting)
        self.options = []
        self.getDefault()
        self.setDefault()

    '''def setupSetting(self, Setting):
        Setting.setObjectName("Setting")
        Setting.resize(1005, 1004)
        Setting.setBaseSize(QtCore.QSize(0, 2))
        self.centralwidget = QtWidgets.QWidget(Setting)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 250, 451, 261))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 40, 411, 211))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.formLayoutWidget.setFont(font)
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.comboBox = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.comboBox_2 = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_2)
        self.comboBox_4 = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.comboBox_4.setFont(font)
        self.comboBox_4.setObjectName("comboBox_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_4)
        self.comboBox_5 = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.comboBox_5.setFont(font)
        self.comboBox_5.setObjectName("comboBox_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.comboBox_5)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.comboBox_3 = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(60, 20, 431, 221))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_2)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(20, 40, 391, 151))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.formLayoutWidget_2.setFont(font)
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.horizontalSlider = QtWidgets.QSlider(self.formLayoutWidget_2)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setProperty("value", 5)
        self.horizontalSlider.setSliderPosition(5)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_8)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.formLayoutWidget_2)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(10)
        self.horizontalSlider_2.setProperty("value", 5)
        self.horizontalSlider_2.setSliderPosition(5)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.horizontalSlider_2)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_9)
        Setting.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Setting)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1005, 25))
        self.menubar.setObjectName("menubar")
        Setting.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Setting)
        self.statusbar.setObjectName("statusbar")
        Setting.setStatusBar(self.statusbar)

        # set combobox
        self.comboList = [self.comboBox, self.comboBox_2, self.comboBox_3, self.comboBox_4, self.comboBox_5]
        for combobox in self.comboList:
            combobox.addItems(self.optionsDefault)

        self.retranslateUi(Setting)
        QtCore.QMetaObject.connectSlotsByName(Setting)
        
        '''


    def retranslateUi(self, Setting):
        _translate = QtCore.QCoreApplication.translate
        Setting.setWindowTitle(_translate("Setting", "Setting"))
        self.groupBox.setTitle(_translate("Setting", " 5 "))
        self.groupBox_2.setTitle(_translate("Setting", "Common"))

        self.label.setText(_translate("Setting", "上"))
        self.label_2.setText(_translate("Setting", "下"))
        self.label_3.setText(_translate("Setting", "左"))
        self.label_4.setText(_translate("Setting", "右"))
        self.label_5.setText(_translate("Setting", "Hold"))
        self.label_6.setText(_translate("Setting", "Hold Time"))
        self.label_7.setText(_translate("Setting", "Travel"))
        self.label_commonHoldtime.setText(_translate("Setting", "0"))
        self.label_travel.setText(_translate("Setting", "0"))

        self.comboList = [self.comboBox, self.comboBox_2, self.comboBox_3, self.comboBox_4, self.comboBox_5]
        for combobox in self.comboList:
            combobox.addItems(self.optionsDefault)

    def getDefault(self):
        self.config.read('config.ini', encoding="utf-8")
        self.gestureActions = self.config.items('gestureAction')
        for key, value in self.gestureActions:
            self.options.append(key)

    def setDefault(self):
        self.comboList = [self.ui.comboBox, self.ui.comboBox_2, self.ui.comboBox_3, self.ui.comboBox_4, self.ui.comboBox_5]
        for combobox in self.comboList:
            combobox.addItems(self.optionsDefault)
        _i = 1
        for combobox in self.comboList:
            current_selection = self.config.get('gestureAction', self.options[_i])
            combobox.setCurrentIndex(self.optionsDefault.index(current_selection))
            combobox.currentIndexChanged.connect(self.saveChange)
            _i += 1

    def saveChange(self):
        _i = 1
        for combobox in self.comboList:
            self.config.set('gestureAction', self.options[_i], combobox.currentText())
            with open('config.ini', 'w', encoding="utf-8") as configfile:
                self.config.write(configfile)
            _i += 1




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Main()
    ui.show()
    sys.exit(app.exec())
