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

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


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
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

# BATCH

folder_path = 'img'
image_dataset = datasets.ImageFolder(root=folder_path, transform=tf)

# Définir la taille du batch
batch_size = 4

# Créer un DataLoader pour charger les images en tant que batchs
image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size)

<<<<<<< HEAD
list_scene_categories = []
# forward pass sur chaque batch d'images
=======
# Créer une liste vide pour stocker les dictionnaires des catégories de scènes
scene_categories_list = []

# Boucle sur chaque batch d'images
>>>>>>> 5e0be0ddca23b700258f0d3b152b0b1c7a3e82c6
for batch_idx, (data, target) in enumerate(image_loader):
    # CHARGEMENT DE L'IMAGE
    input_img = data
    
    # forward pass sur le batch d'images
    logit = model.forward(input_img)
<<<<<<< HEAD
    # print("logit :", logit)
=======
>>>>>>> 5e0be0ddca23b700258f0d3b152b0b1c7a3e82c6
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(1, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # affichage des résultats pour le batch en cours
<<<<<<< HEAD
    print(f"BATCH {batch_idx} traité. Nombre d'images dans le batch : {len(data)}.")
    # for i in range(len(data)):
        # print(f"Classe {idx[i]} avec probabilité {probs[i]}")
        # print(np.sum(probs[i]))
=======
    print(f"Batch {batch_idx} traité. Nombre d'images dans le batch : {len(data)}. Résultats :")
>>>>>>> 5e0be0ddca23b700258f0d3b152b0b1c7a3e82c6

    # ########## OUTPUT ###########


    print('RESULT ON BATCH ')

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        print('\n --TYPE OF ENVIRONMENT: indoor')
    else:
        print('\n--TYPE OF ENVIRONMENT: outdoor')

    # ########### SCENE CATEGORIES ###########

    # Créer une liste vide pour stocker les dictionnaires des catégories de scènes pour le batch actuel
    batch_scene_categories = []

    # output the prediction of scene category
    print('\n--SCENE CATEGORIES:')
    for j in range(batch_size):
        print('Numéro de la frame : ', )
        scene_categories_dict = {}
        for i in range (10):
            print('{:.3f} -> {}'.format(probs[j,i], classes[idx[j,i]]))
            scene_categories_dict[classes[idx[j,i]]] = probs[j,i]
        # ajouter le dictionnaire pour cette image à la liste
        list_scene_categories.append(scene_categories_dict)
    # create a dictionary of scene categories and their probabilities

print(list_scene_categories)
print(len(list_scene_categories))

# ########### SCENE ATTRIBUTES ###########


# # Variables permettant d'obtenir les probabilités des attributs
# responses_attribute = W_attribute.dot(features_blobs[1])
# responses_attribute = torch.from_numpy(responses_attribute)
# responses_attribute = F.softmax(responses_attribute,0)
# responses_attribute = responses_attribute.numpy()
# idx_a = np.argsort(responses_attribute)

# print('\n--SCENE ATTRIBUTES:')

# print(', '.join([f'{labels_attribute[idx_a[i]]}: {np.sort(responses_attribute)[i]}' for i in range(-1,-10,-1)]))


# # create a dictionary of scene attributes and their probabilities
# scene_attributes = {labels_attribute[idx_a[i]]: np.sort(responses_attribute)[i] for i in range(-1, -len(idx_a), -1)}


########### HEATMAP ###########


# # generate class activation mapping categories
# print('\nClass activation map is saved as cam.jpg')
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# print(len(idx))

# # render the CAM and output
# img = cv2.imread('test.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.4 + img * 0.5
# cv2.imwrite('cam.jpg', result)

# # generate class activation mapping attributes
# print('\nClass activation map is saved as cam2.jpg')
# # VERIFIER features blobs
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx_a[0]])

# # render the CAM and output
# img = cv2.imread('test.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.4 + img * 0.5
# cv2.imwrite('cam2.jpg', result)
