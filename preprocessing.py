'''
Written by Riken Patel & Jakub Marek & Lohith Muppala
Date: 4/24/2021 
Pre-processing data
'''
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
import os
import cv2
import warnings

warnings.filterwarnings("ignore") #ignores the warnings

## Take the objects in the image and labels them
def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0 #No mask

## Gathers the box coordinate data from the file as variables
def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

## Create a new matrix that houses box, labels, and image id information
def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data,features= 'lxml')
        objects = soup.find_all('object')
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        return target

img_dir = "c:/Users/Owner/Documents/UIC/ECE 491/Project/images"
ano_dir = "c:/Users/Owner/Documents/UIC/ECE 491/Project/annotations"
save_dir = "c:/Users/Owner/Documents/UIC/ECE 491/Project/box_images"

data_mask = []
num_imgs = len(os.listdir(img_dir))
for i in range(num_imgs):
    image_file = 'maksssksksss'+ str(i) + '.png'
    label_file = 'maksssksksss'+ str(i) + '.xml'
    img_path = os.path.join(img_dir,image_file)
    label_path = os.path.join(ano_dir,label_file)
    target = generate_target(i, label_path)
    data_mask.append(target)
    
images = list(sorted(os.listdir(img_dir)))
annotations = list(sorted(os.listdir(ano_dir)))

## Create new images form the boxes found in the original images, save then in a new folder
def image_class(i, data_mask):
    image_file = 'maksssksksss'+ str(i) + '.png'
    img_path = os.path.join(img_dir,image_file)
    image = cv2.imread(img_path)
    temp_img = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #original image
    for j in range(len(data_mask[i]["boxes"])):
        box = data_mask[i]['boxes'][j]
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(temp_img, (int(xmin-10), int(ymin-10)), (int(xmax+10), int(ymax+10)), (255,255,255) ,2)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) #image with the box
        img_box = image[int(ymin-10):int(ymax+10), int(xmin-10):int(xmax+10)]
        label = int(data_mask[i]['labels'][j])
        label_path = 'img_{}_{}_{}.png'.format(i,j,label) #img, box num, label
        try:
            save_path = os.path.join(save_dir,label_path) #.../box_images/img_0_0_0.png
            img_box = cv2.resize(img_box,(50,50),interpolation = cv2.INTER_AREA)
            img_box = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB) #original image
            cv2.imwrite(save_path,img_box) #saves the image into the location  
        except:
            pass

def preprocessing_images(img_dir = img_dir,data_mask = data_mask):
    num_imgs = len(os.listdir(img_dir))
    for i in range(num_imgs):
        image_class(i,data_mask)

preprocessing_images(img_dir,data_mask)
