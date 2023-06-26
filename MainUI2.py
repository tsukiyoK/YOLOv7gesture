import argparse
import ctypes
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
import action
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, checkWay, checkCD, check2Hold, set2Hold
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from action import *
from ui_setting import Ui_Setting
from ui_2 import Ui_MainWindow


class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None, play_cd=0.8):
        super(Main, self).__init__(parent)
        self.gestureActions = None
        self.config = configparser.ConfigParser()
        self.travelDefault = 0
        self.holdDefault = 0
        self.setting = Setting()
        self.uiMain = Ui_MainWindow()
        self.uiMain.setupUi(self)
        self.setWindowIcon(QIcon("images/UI/kk.png"))
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
        self.is2hold = False
        # 权重初始文件名
        self.openfile_name_model = None
        self.getDefault()

    def action(self, label, det):
        # '1', '2', '5', '0', 'OK', 'Good' .fourWayAction "mask"
        if label == '1':
            self.mouseMovement(det)
        elif label == '2':
            if checkCD(time.time(), 0.5, 2):
                if not self.is2hold:
                    mouseLeftHold()
                    self.is2hold = True

        elif label == '0':
            if checkCD(time.time(), self.play_cd, 0):
                self.resetAllGes()
                self.is2hold = False

        elif label == '5':
            if checkCD(time.time(), 0.1, 5):
                self.fourWayAction(det, self.travelSens)


        elif label == 'Good':
            if checkCD(time.time(), self.holdSens, 6):
                mouseRightClk()


        elif label == 'OK':
            if checkCD(time.time(), self.holdSens, 7):
                mouseLeftClk()



    def resetAllGes(self):
        self.mouse_locate = []
        self.mouse_history = []
        self.gesture_locate = []
        mouseReset()

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
                x, y = self.mouse_history[7]
                new_x, new_y = self.mouse_history[9]
                val_x, val_y = new_x-x, new_y-y
                #current_x, current_y = win32api.GetCursorPos()
                current_pos = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(current_pos))
                ctypes.windll.user32.SetCursorPos(
                    ctypes.c_int(current_pos.x - (int(self.cursorSens) * val_x)),
                    ctypes.c_int(current_pos.y + (int(self.cursorSens) * val_y))
                )
                #win32api.SetCursorPos((current_x-(int(self.cursorSens)*val_x), current_y+(int(self.cursorSens)*val_y)))  # 設置滑鼠座標
                self.mouse_history.pop(0)
            else:
                self.mouse_history.append((x2, int(y2 - h2 // 2)))

    def fourWayAction(self, det, sens):
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
            way = checkWay(x1, x2, y1, y2, sens)
            if way == 1:
                self.gestureFunction5U()
            elif way == 2:
                self.gestureFunction5D()
            elif way == 3:
                self.gestureFunction5L()
            elif way == 4:
                self.gestureFunction5R()


    def getDefault(self):
        self.config.read('config.ini', encoding="utf-8")
        gestureAction5U = self.config.get('gestureAction', '5u')
        gestureAction5D = self.config.get('gestureAction', '5d')
        gestureAction5L = self.config.get('gestureAction', '5l')
        gestureAction5R = self.config.get('gestureAction', '5r')
        self.gestureFunction5U = getattr(action, gestureAction5U)
        self.gestureFunction5D = getattr(action, gestureAction5D)
        self.gestureFunction5L = getattr(action, gestureAction5L)
        self.gestureFunction5R = getattr(action, gestureAction5R)
        self.travelDefault = self.config['slider']['travel']
        self.holdDefault = self.config['slider']['commonHoldtime']
        self.cursorsensDefault = self.config['slider']['cursorsens']
        self.travelSens = 80+(5-int(self.travelDefault))*10
        self.holdSens = 1.5+((int(self.holdDefault)-5)*0.2)
        self.cursorSens = 4+((int(self.cursorsensDefault)-5)*0.2)

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.uiMain.pushButton_pt, 'Select weights',
                                                                  'pt/', "*.pt;")
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open weights", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        #else:
        #    self.uiMain.label_2.setText('Weights path：' + str(self.openfile_name_model))

    # 模型初始化
    def model_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                            help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--img-size', nargs='+', type=int, default=640,
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
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
        QtWidgets.QMessageBox.information(self, u"ok", u"Model initialize done")

    # ui.py文件的函数

    # 绑定信号与槽
    def init_slots(self):
        self.uiMain.pushButton_sht.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.uiMain.pushButton_pt.clicked.connect(self.open_model)
        self.uiMain.pushButton_init.clicked.connect(self.model_init)
        self.uiMain.pushButton_setting.clicked.connect(self.setting.show)

    # 打开图片
    def button_image_open(self):
        self.uiMain.label_2.setText('图片打开成功')
        name_list = []
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.uiMain.label_2.setText("图片路径：" + img_name)
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
        self.uiMain.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.uiMain.label.setScaledContents(True)  # 自适应界面大小

    # 打开视频
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "*.mp4;;*.avi;;All Files(*)")
        self.uiMain.label_2.setText("Video path：" + video_name)
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
            self.uiMain.label.setScaledContents(True)  # 自适应界面大小

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
                            #self.uiMain.label_2.setText(label)  # PyQT页面打印类别和置信度
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
            self.uiMain.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.uiMain.label.clear()

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
            flag = self.cap.open(1)
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
                self.uiMain.pushButton_init.setDisabled(True)
                self.uiMain.pushButton_pt.setDisabled(True)
                #                self.pushButton_stop.setDisabled(True)
                #self.uiMain.pushButton_exit.setDisabled(True)
                self.uiMain.pushButton_sht.setText(u"Turn off")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.uiMain.label.clear()
            #            self.pushButton_video.setDisabled(False)
            #            self.pushButton_openimg.setDisabled(False)
            self.uiMain.pushButton_init.setDisabled(False)
            self.uiMain.pushButton_pt.setDisabled(False)
            #           self.pushButton_stop.setDisabled(False)
            #self.uiMain.pushButton_exit.setDisabled(False)
            self.uiMain.pushButton_sht.setText(u"Camera")

        self.getDefault()

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
        self.uiMain.label.clear()  # 清空label画布
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
                               'mediaNextTrack', 'windowMinimize', 'windowMaximize', 'windowBackToscreen',
                               'pageUp', 'pageDown']
        self.config = configparser.ConfigParser()
        self.setWindowIcon(QIcon("images/UI/kk.png"))
        self.options = None
        self.gestureActionsgestureActions = None
        self.ui = Ui_Setting()
        self.ui.setupUi(self)

        self.options = []
        self.travelDefault = 0
        self.holdDefault = 0
        self.getDefault()
        self.setDefault()
        self.setSliderDefault()

    def getDefault(self):
        self.config.read('config.ini', encoding="utf-8")
        self.gestureActions = self.config.items('gestureAction')
        for key, value in self.gestureActions:
            self.options.append(key)
        self.travelDefault = self.config['slider']['travel']
        self.holdDefault = self.config['slider']['commonHoldtime']

    def setDefault(self):
        self.comboList = [self.ui.comboBox, self.ui.comboBox_2, self.ui.comboBox_3, self.ui.comboBox_4, self.ui.comboBox_5]
        for combobox in self.comboList:
            combobox.addItems(self.optionsDefault)
        _i = 0
        for combobox in self.comboList:
            current_selection = self.config.get('gestureAction', self.options[_i])
            combobox.setCurrentIndex(self.optionsDefault.index(current_selection))
            combobox.currentIndexChanged.connect(self.saveChange)
            _i += 1

        self.ui.horizontalSlider_travel.valueChanged.connect(lambda: self.setSliderValue(self.ui.horizontalSlider_travel
                                                                                         , self.ui.label_travel))
        self.ui.horizontalSlider_commonHoldtime.valueChanged.connect(lambda:
                                                                     self.setSliderValue(self.ui.horizontalSlider_commonHoldtime, self.ui.label_commonHoldtime))
        self.ui.comboBox_11.addItem("Drag")
        self.ui.comboBox_12.addItem("Enter")
        self.ui.comboBox_13.addItem("RightClick")
    def setSliderDefault(self):
        self.ui.horizontalSlider_travel.setValue(int(self.travelDefault))
        self.ui.horizontalSlider_commonHoldtime.setValue(int(self.holdDefault))
        self.ui.label_travel.setText(f"{int(self.travelDefault)}")
        self.ui.label_commonHoldtime.setText(f"{int(self.holdDefault)}")

    def setSliderValue(self, slider, label):
        label.setText(f"{slider.value()}")

    def saveChange(self):
        _i = 0
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
