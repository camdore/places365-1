from threading import Thread # library for multi-threading
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

import threading
import queue
import cv2

import torch

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import json

import datetime

classes_to_filter = ['train'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt  = {
    
    "weights": "weights/yolov7.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter,  # list of classes to filter or None
    "single_cls" : False

}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class VideoReaderThread(threading.Thread):
    def __init__(self, video_path, queue, division):
        threading.Thread.__init__(self)
        self.video = cv2.VideoCapture(video_path)
        self.queue = queue
        self.division = division
    
    # def run(self):
    #     while True:
    #         ret, img0 = self.video.read()
    #         if not ret:
    #             break
    #         self.queue.put(img0)
    def run(self):
        count = 0
        fps = round(float(self.video.get(cv2.CAP_PROP_FPS)))
        while True:
            ret, img0 = self.video.read()
            if not ret:
                break
            # incrémenter le compteur de frames
            
            if count % (fps // self.division) == 0: 
                self.queue.put(img0)

            count += 1
    
    def stop(self):
        self.video.release()

def batch_frames(video_path, batch_size, device, half, imgsz, stride, division):
    video_queue_yolo = queue.Queue(maxsize=batch_size*2)
    video_reader_thread_yolo = VideoReaderThread(video_path, video_queue_yolo, division)
    video_reader_thread_yolo.start()

    video_queue_places = queue.Queue(maxsize=batch_size*2)
    video_reader_thread_places = VideoReaderThread(video_path, video_queue_places, division)
    video_reader_thread_places.start()
    
    batch_yolo = []
    batch_places = []
    count_frame = 0
    
    while True:
        img0_yolo = video_queue_yolo.get()
        img0_places = video_queue_places.get()

        #pr yolov7
        img0_yolo = letterbox(img0_yolo, imgsz, stride)[0]
        img0_yolo = img0_yolo[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img0_yolo = np.ascontiguousarray(img0_yolo)
        img0_yolo = torch.from_numpy(img0_yolo).to(device)
        img0_yolo = img0_yolo.half() if half else img0_yolo.float()  # uint8 to fp16/32
        img0_yolo /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img0_yolo.ndimension() == 3:
            img0_yolo = img0_yolo.unsqueeze(0)

        #pr places365
        # img0_places = returnTF()(torch.from_numpy(np.transpose(img0_places,(2, 0, 1)))).unsqueeze(0)

        batch_yolo.append(img0_yolo)
        batch_places.append(img0_places)
        count_frame += 1
        
        if count_frame == batch_size or not video_reader_thread_yolo.is_alive() and video_queue_yolo.empty():
            count_frame = 0
            yield batch_yolo, batch_places
            batch_yolo = []
            batch_places = []
        
        if not video_reader_thread_yolo.is_alive() and video_queue_yolo.empty() and video_queue_places.empty():
            break
    
    video_reader_thread_yolo.stop()
    video_reader_thread_places.stop()


def run_yolov7_videos(paths, batch_size):
  videos = dict()
  for video_path in paths:
    title = video_path.split('/')[-1]
    videos[title] = run_yolov7_thread(video_path, 32)
  return videos

def run_yolov7_thread(paths, batch_size, division):

  torch.cuda.empty_cache()
  
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
  

    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names

    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:
        classes.append(names.index(class_name))

    if classes:
      classes = [i for i in range(len(names)) if i not in classes]

    cudnn.benchmark = True

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    videos = dict()
    t0 = time.time()
    for video_path in paths:
      title = video_path.split('/')[-1]

      video = cv2.VideoCapture(video_path)
      ret, img0 = video.read()
      fps = video.get(cv2.CAP_PROP_FPS)//division
      nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)//division + 1
      duration = nframes / fps * 1000

      # dataset = LoadImages(video_path, img_size=imgsz, stride=stride)
      
      
      if device.type != 'cpu':
          model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

      count_frame = 0

      t0 = time.time()
      obj = []


      #for path, img, im0s, vid_cap in dataset:
      batches = batch_frames(video_path, batch_size, device, half, imgsz, stride, division)

      for batch, batch_places in batches:
        # batch = torch.Tensor(batch)
        batch = torch.stack(batch)
        # batch = batch.permute(0, 1, 4, 2, 3)
        # batch = np.transpose(batch)
        b, n, c, h, w = batch.shape
        batch = batch.reshape(b, n*c, h, w)
        pred = model(batch, augment=False)[0]


        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)


        for i, det in enumerate(pred):
          # p, im0, frame = path[i], im0s[i].copy(), dataset.count
          s=""
          labels = dict()
          count_frame+=1

          labels["frame"] = count_frame
          labels["timestamp"] = str(datetime.timedelta(milliseconds=int(count_frame * duration / nframes)))
          print(count_frame, duration ,nframes)

          objects = dict()#création du dictionnaire des positions des objets 

          gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            
          for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() #normalise les données de la position des objets 
            name = names[int(cls)]
            if name not in objects: 
              objects[name] = []
            objects[name].append(dict())
            objects[name][-1]['bndbox'] = dict()
            objects[name][-1]['bndbox']['xmin'] = xywh[0]
            objects[name][-1]['bndbox']['xmax'] = xywh[1]
            objects[name][-1]['bndbox']['ymin'] = xywh[2]
            objects[name][-1]['bndbox']['ymax'] = xywh[3]
            objects[name][-1]['conf'] = conf.item()
          
          labels["objects"] = objects
          obj.append(labels)

        # count_batch+=1
        # batch = []
    videos[title] = obj  

  t3 = time_synchronized()

  print(f'Done. ({time.time() - t0:.3f}s)')
  return(videos)

def main():
  video_path = '/content/gdrive/MyDrive/yolov7/Test.mp4'
  res = run_yolov7_thread(video_path, 32, 1)
  print(len(res))
  print(res[-1])

if __name__ == "__main__":
    main()