# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.custom_utils import CustomPlotBox, DivideImg
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.plots import plot_one_box

class Detector():
    def __init__(self, source='test.mp4', weights='weights.pt', imgsz=640, webcam=False):
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.webcam = webcam
        
        # Initiallize
        set_logging()
        if(torch.cuda.is_available()):
            self.device = select_device('0')
        else:
            self.device = select_device('cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride : kernel移動步伐大小
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.nightMode = True
        if self.half:
            self.model.half()  # to FP16

    def AdjustGamma(self, img, gamma: float = 1.0):
        # build a lookup table mapping the pixel values [0, 255] to
	    # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
	    # apply gamma correction using the lookup table
        return cv2.LUT(img, table)
    
    '''
    Returns: 偵測資料    
    '''
    def predict(self, _img, nightmode=False):
        if nightmode:
            _img = NightPreifix(_img)
            
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        # Padded resize
        img = letterbox(_img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)     
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=False)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        # 0.5: confidence thres
        # 0.45: iou_thres
        pred = non_max_suppression(pred, 0.5, 0.45, classes=None, agnostic=False)   
        return self.postProcessing(pred, _img, img)
        

    '''
    Return: tuple -> (img, [])
    Ex: (圖像, [left, center, right] => 0為警示, 1為不警示)
    '''
    def postProcessing(self, pred, ori_img, img):
        DivideImg(ori_img)
        position_arr = [1, 1, 1]
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', ori_img
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # im0 = self.AdjustGamma(im0, 0.2)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{self.names[int(cls)]} {conf:.2f}'
                    label = f'{self.names[int(cls)]}'
                    CustomPlotBox(pos_arr=position_arr, x=xyxy, img=im0, label=label, 
                                  box_color=self.colors[int(cls)], 
                                  line_thickness=2)
        return (im0, position_arr)

def NightPreifix(img):
    # img = AdjustGamma(img, 2)
    # img = cv2.medianBlur(img, 7) 
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_scaled = cv2.normalize(laplacian, None, 0, 255, 
                                     cv2.NORM_MINMAX).astype("uint8")
    return cv2.add(img, laplacian_scaled)

def AdjustGamma(img, gamma: float = 1.0):
        # build a lookup table mapping the pixel values [0, 255] to
	    # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
	    # apply gamma correction using the lookup table
        return cv2.LUT(img, table)  
  
def detect(save_img=False):
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # view_img: 使用Webcam時需要
    source = './DistanceMeasure/16-remake.jpg'
    weights = 'weights.pt'
    imgsz = 640
    save_txt = ''
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    # device = select_device(opt.device)
    if(torch.cuda.is_available()):
        device = select_device('0')
    else:
        device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride : kernel移動步伐大小
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    '''
    使用不到
    if trace:
        model = TracedModel(model, device, opt.img_size)
    '''

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # 意義不明
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                # model(img, augment=opt.augment)[0]
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            # pred = model(img, augment=opt.augment)[0]
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = non_max_suppression(pred, 0.5, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    label = f'{names[int(cls)]}'
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    CustomPlotBox(pos_arr=[1, 1, 1], x=xyxy, img=im0, label=label, 
                                  box_color=colors[int(cls)], 
                                  line_thickness=2)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.imshow('Object Detection Window', im0)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break
            continue
            '''
            *以下為使用不到的部分
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            '''
    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.waitKey(0)

if __name__ == '__main__':
    detect()
