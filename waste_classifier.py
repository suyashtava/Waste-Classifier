# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 19:27:08 2018

@author: Suyash
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing CNN

classifier=Sequential()

#Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding another CNN LAYER
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add(Flatten())
# full connection

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=6, activation='sigmoid'))
#compile
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = classifier(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=2310,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=217)

#making a new prediction 

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)

training_set.class_indices


