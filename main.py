'''
Written by Riken Patel & Jakub Marek
Date: 04/24/2021
ECE 491 project
'''

import numpy as np 
import os
import pandas as pd 
import importlib
import warnings
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import random

warnings.filterwarnings("ignore") #ignores the warnings

save_dir = "c:/Users/Owner/Documents/UIC/ECE 491/Project/box_images"

def create_dict(save_dir):
    images_data = []
    labels = []
    for i in range(len(os.listdir(save_dir))):
        img = cv2.imread(os.path.join(save_dir,os.listdir(save_dir)[i])) #reading in the image
        img = img/255.0 #normalizing
        images_data.append(img)
        #extracting the labels
        label = os.listdir(save_dir)[i]
        label = label.split('_')[3]
        label = label.split('.')[0]
        labels.append(int(label)) #appending to the array
    return images_data,labels
img_data,labels = create_dict(save_dir)

X_train,X_test,y_train,y_test = train_test_split(np.array(img_data),np.array(labels),test_size=0.20, random_state=42)

model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(50,50,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10))
print(model.summary())
model.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
monitor = EarlyStopping(monitor='accuracy',patience = 3 , verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=5, callbacks=[monitor])
loss,acc = model.evaluate(X_test,y_test,verbose = 2)
print('Accuracy of the model: ',round(acc,4)*100,'%')

random_array = np.random.randint(len(y_test), size=(1,5))[0]
for i in random_array:
    print('Actual:',y_test[i])
    test_img =  X_test[i]
    expanded_image = np.expand_dims(test_img, axis=0)
    prediction = model.predict(expanded_image)
    print('Predicted:',np.argmax(prediction))
    if np.argmax(prediction) == 1:
        plt.imshow(X_test[i])
        plt.title('Mask')
        plt.show()
    elif np.argmax(prediction) == 0:
        plt.imshow(X_test[i])
        plt.title('No Mask')
        plt.show()
    else: #2
        plt.imshow(X_test[i])
        plt.title('Incorrect Mask')
        plt.show()
