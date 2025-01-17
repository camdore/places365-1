# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, TensorDataset 
import glob
import time
import datetime
import json 
import multiprocessing
import threading
import queue

start = time.time()

 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_labels():

    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv, weight_softmax, class_idx):

    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnTF():

# load the image transformer
    tf = trn.Compose([
        trn.ToPILImage(),
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():

    # this model has a last conv feature map as 14x14
    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    return model.cuda()
    # return model


########### CHARGEMENT DE PARAMETRES ###########


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.cpu().numpy()
# weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

## fonctions Trang Anh


class VideoReaderThread(threading.Thread):
    def __init__(self, video_path, queue):
        threading.Thread.__init__(self)
        self.video = cv2.VideoCapture(video_path)
        self.queue = queue
    
    def run(self):
        count = 0
        fps = round(float(self.video.get(cv2.CAP_PROP_FPS)))
        division = 1 
        while True:
            ret, img0 = self.video.read()
            if not ret:
                break
            # incrémenter le compteur de frames
            
            if count % (fps // division) == 0: 
                self.queue.put(img0)

            count += 1
    def stop(self):
        self.video.release()

def batch_frames(video_path, batch_size): #, device, half, imgsz, stride
    # video_queue_yolo = queue.Queue(maxsize=batch_size*2)
    # video_reader_thread_yolo = VideoReaderThread(video_path, video_queue_yolo)
    # video_reader_thread_yolo.start()

    video_queue = queue.Queue(maxsize=batch_size*10)
    video_reader_thread = VideoReaderThread(video_path, video_queue)
    video_reader_thread.start()
    time.sleep(1)
    
    # batch_yolo = []
    batch = []
    count_frame = 0
    
    while True:
        # img0_yolo = video_queue.get()
        img0 = video_queue.get()

        #pr yolov7
        # img0_yolo = letterbox(img0_yolo, imgsz, stride)[0]
        # img0_yolo = img0_yolo[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img0_yolo = np.ascontiguousarray(img0_yolo)
        # img0_yolo = torch.from_numpy(img0_yolo).to(device)
        # img0_yolo = img0_yolo.half() if half else img0_yolo.float()  # uint8 to fp16/32
        # img0_yolo /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img0_yolo.ndimension() == 3:
        #     img0_yolo = img0_yolo.unsqueeze(0)
      

        # batch_yolo.append(img0_yolo)
        batch.append(img0)
        count_frame += 1
        
        if count_frame == batch_size or not video_reader_thread.is_alive() and video_queue.empty():
            count_frame = 0
            yield batch 
            batch = []
            # batch_places = []
        
        if not video_reader_thread.is_alive() and video_queue.empty() and video_queue.empty():
            break
    
    video_reader_thread.stop()
    # video_reader_thread_places.stop()

################### PLAYLIST VIDEOS ###################
@profile
def main():
    video_dir = 'videos/'
    
    # list all files in the directory
    all_files = os.listdir(video_dir)

    # filter by video files
    video_files = [f for f in all_files if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:      

        # # Définir la taille du batch

        batch_size = 128
        video_path = os.path.join(video_dir, video_file)
        print(video_file)
        count_frame = 1

        batches = batch_frames(video_path, batch_size)
        
        list_1_video = []
        dict_1_video = {}
        dict_1_video["idVideo"] = video_file

        for batch in batches:
            batch_idx = 0
            # CHARGEMENT DE L'IMAGE

            batch = [returnTF()(torch.from_numpy(np.transpose(frame,(2, 0, 1)))).unsqueeze(0) for frame in batch]
            # print(np.shape(batch))
            batch = torch.cat(batch,dim=0)
            print(np.shape(batch))
            batch = batch.cuda()
            # car data est une liste de 1 seul élement tensor
            
            
            # forward pass sur le batch d'images
            logit = model.forward(batch)
            h_x = F.softmax(logit.cpu(), 1).data.squeeze()
            # h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(1, True)
            probs = probs.numpy()
            idx = idx.numpy()
            
            # affichage des résultats pour le batch en cours
            print(f"BATCH {batch_idx} traité. Nombre d'images dans le batch : .")

            ########## OUTPUT ###########

            print('RESULT ON BATCH ')

            # output the IO prediction
            # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
            # if io_image < 0.5:
            #     print('\n --TYPE OF ENVIRONMENT: indoor')
            # else:
            #     print('\n--TYPE OF ENVIRONMENT: outdoor')

            ########### SCENE CATEGORIES ###########
            batch_idx+=1
            # output the prediction of scene category
            for j in range(batch_size):
                try :  
                    scene_categories_dict = {}
                    for i in range(365):
                        scene_categories_dict[classes[idx[j,i]]] = float(probs[j,i])
                    # ajouter le dictionnaire pour cette image à la liste
                    dict_1_frame = {}
                    dict_1_frame['frame'] = (count_frame+j)
                    dict_1_frame['timestamps'] = (str(datetime.timedelta(seconds=count_frame+j))) 
                    dict_1_frame["scene_attribute"] = scene_categories_dict 
                    list_1_video.append(dict_1_frame) 
                except IndexError:
                    break 
            
            dict_1_video['features']=(list_1_video)

            count_frame+=batch_size

        ################ JSON FILE ######################

        # convert list_scene_categories to JSON string
        json_str = json.dumps(dict_1_video)

        # save the JSON string to file
        with open(f'dict_{video_file}.json', 'w') as f:
            f.write(json_str)

    end = time.time()
    print("Temps d'exécution : {:.2f} secondes".format(end - start))

if __name__ == "__main__":
    main()