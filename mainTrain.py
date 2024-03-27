import cv2 #library: open-cv, USED TO PERFORM OPERATIONS ON IMAGES AND VIDEOS
import os # USED TO INTERACT WITH FILE SYSTEM
from PIL import Image # PERFORM OPERATION ON IMAGES
import tensorflow as tf
from tensorflow import keras # LIBRARY FOR NEURAL NETWORKS
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize #Keras 3 is a deep learning framework works with TensorFlow, JAX, and PyTorch interchangeably.
from tensorflow.keras.models import Sequential #SEQUENTIAL IS USEFUL FOR STACKING LAYERS WHERE EACH LAYER HAS ONE INPUT TENSOR AND ONE OUTPUT TENSOR.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#LAYERS ARE FUNCTIONS WITH A KNOWN MATHEMATICAL STRUCTURE THAT CAN BE REUSED AND HAVE TRAINABLE VARIABLES.
from keras.utils import to_categorical

image_directory = 'dataset/'

no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/')
dataset = []
label = []

INPUT_SIZE = 64

# print(no_tumor_images)

for i, images_name in enumerate(no_tumor_images): #CONVERTS INTO ENUMARATE OBJECT
    if(images_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory + 'no/'+ images_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, images_name in enumerate(yes_tumor_images):
    if(images_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory + 'yes/'+ images_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# print(y_train.shape)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)
# print(x_train.shape)

#USING THE METHOD TO_CATEGORICAL(), A NUMPY ARRAY (OR) A VECTOR WHICH HAS INTEGERS THAT REPRESENT DIFFERENT CATEGORIES, CAN BE CONVERTED INTO A NUMPY ARRAY (OR) A MATRIX WHICH HAS BINARY VALUES AND HAS COLUMNS EQUAL TO THE NUMBER OF CATEGORIES IN THE DATA
# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)

#PADDING ADD EXTRA VALUE IN THE MATRIX SO THAT THE FIRST ELEMENT GETS IN THE FRAME MULTIPLE TIMES

model = Sequential() #CNN model
#STRIDE SIKPS THE PIXELS IN THE MATRIX
model.add(Conv2D(32, (3,3),input_shape=(INPUT_SIZE, INPUT_SIZE, 3))) #IT HELPS IN DETECTING EDGES IN THE IMAGE. THE FILTER IS MUL WITH THE PIXELS THAT ALLOWS THIS PROCESS
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #REDUCE THE SIZE OF THE MATRIX

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #FLATTENS THE GENERATED MATRIX TO 1D MATRIX AND IS USED AS INPUT TO FULLY CONNECTED LAYER
model.add(Dense(64)) #<-FULLY CONNECTED LAYER
model.add(Activation('relu'))
model.add(Dropout(0.5)) #PREVENTS OVERFITTING
model.add(Dense(1))
model.add(Activation('sigmoid')) #FOR BINARY CLASSIFICATION USE SIGMOID OTHER WISE SOFTMAX

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle = False)
#AN EPOCH IS A COMPLETE ITERATION THROUGH THE ENTIRE TRAINING DATASET IN ONE CYCLE FOR TRAINING THE MACHINE LEARNING MODEL. 

# model.save('BrainTumor10Epochs.h5')
