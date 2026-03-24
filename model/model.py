#! /usr/bin/env python
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Monday, June 26, 2023                                              #
# Description: Neural Network Model for Gesture Recognition                #
# Copyright 2023. All Rights Reserved.                                     #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
#==========================================================================#

import os
from abc import ABC, abstractmethod

import numpy as np

#  Disable tensorflow logs
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras import utils
#from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import save_model
from tensorflow.keras import regularizers, optimizers

from data.formats import GESTURE, GESTURE2

import colorama
from colorama import Fore
colorama.init(autoreset=True)

print(f'Tensorflow Version: {tf.__version__}')
print(f'Device Supported:\n{tf.config.list_physical_devices()}')

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

# Init tf gpu
def set_tensorflow_config(per_process_gpu_memory_fraction=1):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    config.gpu_options.allow_growth=True
    tf.compat.v1.Session(config=config)
    tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
set_tensorflow_config()


class NNModel(ABC):
    def __init__(self, num_classes):
   
        self.model = None
        self.num_classes = num_classes
    
    @abstractmethod
    def create_model(self):
        pass

    def train(self, X_train, y_train, args=''):
        
        self.create_model()
        self.model.summary()
        
        max_epoches = 200
        '''
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_decayed_fn = optimizers.schedules.CosineDecay(learning_rate, max_epoches) 
        reduce_lr = callbacks.LearningRateScheduler(lr_decayed_fn)
        sgd = optimizers.legacy.SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        '''
        if args == 'model1':            
            self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
            #self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        else:   # model2
            self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
        
        y_train = utils.to_categorical(y_train, self.num_classes)
        '''
        datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='nearest')
        #datagen = ImageDataGenerator()
        train_data = datagen.flow(X_train, y_train, batch_size=64, shuffle=True)
        self.history = self.model.fit(train_data,
                                      batch_size=train_data.batch_size,
                                      steps_per_epoch=X_train.shape[0]/train_data.batch_size, epochs=30,
                                      callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                                                 callbacks.ModelCheckpoint(self.model_file, verbose=True, save_best_only=False, save_freq='epoch')])
        '''
        #datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, fill_mode='wrap', cval=0, validation_split=0.3)
        #datagen = ImageDataGenerator(width_shift_range=0.15, validation_split=0.2)
        #datagen.fit(X_train)
        datagen = ImageDataGenerator(validation_split=0.2)
        #train_data = datagen.flow(X_train, y_train, batch_size=32, shuffle=True, subset='training')
        #valid_data = datagen.flow(X_train, y_train, batch_size=16, shuffle=True, subset='validation')
        train_data = datagen.flow(X_train, y_train, batch_size=24, shuffle=True, subset='training')
        valid_data = datagen.flow(X_train, y_train, batch_size=12, shuffle=True, subset='validation')
        #train_data = datagen.flow(X_train, y_train, batch_size=16, shuffle=True, subset='training')
        #valid_data = datagen.flow(X_train, y_train, batch_size=8, shuffle=True, subset='validation')
        self.history = self.model.fit(train_data, 
                                      validation_data=valid_data,
                                      steps_per_epoch=len(train_data)/train_data.batch_size,
                                      validation_steps=len(valid_data)/valid_data.batch_size,
                                      epochs=max_epoches, verbose=1,
                                      batch_size=train_data.batch_size,
                                      validation_batch_size=valid_data.batch_size,
                                      callbacks=[callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                                                 callbacks.ModelCheckpoint(self.model_file, verbose=True, save_best_only=True),])
                                                 #reduce_lr])
            
    def load(self):
        print(f'Loading model...', end='')
        
        self.model = tf.keras.models.load_model(self.model_file)
    
        print(f'{Fore.GREEN}Done.')

    def evaluate(self, X_test, y_test, args = ''):
        
        if args == 'model1':            
            y_test = utils.to_categorical(y_test, self.num_classes)
        else:
            y_test = utils.to_categorical(y_test, 2)
            
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return

        #X, y = self.__prep_data(X, y)
        
        results = self.model.evaluate(X_test, y_test)
        
        print(f'Loss: {round(results[0], 4)}', end=' ')
        print(f'Acc: {round(results[1], 4)}')
        
        return round(results[1], 4)
    
    def predict(self, spectrogram_img, debug=False, model_type = ''):
        
        if self.model is None:
            print(f'{Fore.RED}Model not created.')
            return
        
        range_spec = spectrogram_img[0]
        range_spec = range_spec / np.max(range_spec)

        udoppler = spectrogram_img[1]
        udoppler = udoppler / np.max(udoppler)

        angle_spec = spectrogram_img[2]
        angle_spec = angle_spec / np.max(angle_spec)
        
        X_pred = np.array((range_spec, udoppler, angle_spec))
        X_pred = np.transpose(X_pred, (1, 2, 0))
        X_pred = X_pred.reshape((1,) + X_pred.shape)

        if model_type == 'model1':
            y_pred = self.model.predict(X_pred)
        
            best_guess = [y_pred[0].tolist().index(x) for x in sorted(y_pred[0], reverse=True)]
            best_value = sorted(y_pred[0], reverse=True)

            if debug:
                for guess, val in zip(best_guess, best_value):
                    print(f'{Fore.YELLOW}Best guess: {GESTURE(guess).name.lower()}: {val:.2f}')
                print(f'{Fore.CYAN}------------------------------\n')

            if best_value[0] >= 0.75:
                print(f'{Fore.GREEN}Gesture recognized:',
                      f'{Fore.RED}{GESTURE(best_guess[0]).name.lower()}')
                print(f'{Fore.CYAN}==============================\n')
                return GESTURE(best_guess[0]).name.lower()
        
        if model_type == 'model2':
            y_pred = self.model.predict(X_pred)
            
            best_guess = [y_pred[0].tolist().index(x) for x in sorted(y_pred[0], reverse=True)]
            best_value = sorted(y_pred[0], reverse=True)

            if debug:
                for guess, val in zip(best_guess, best_value):
                    print(f'{Fore.YELLOW}Best guess: {GESTURE2(guess).name.lower()}: {val:.2f}')
                print(f'{Fore.CYAN}------------------------------\n')

            if best_value[0] >= 0.5:
                print(f'{Fore.GREEN}Gesture detected:',
                      f'{Fore.RED}{GESTURE2(best_guess[0]).name.lower()}')
                print(f'{Fore.CYAN}==============================\n')
                return GESTURE2(best_guess[0]).name.lower()

# DNN model to detect gesture
class ConvModel2(NNModel):
    def __init__(self, img_height, img_width, num_classes): # num_classes = 2
    
        super().__init__(num_classes)
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model_file = os.path.join(os.path.dirname(__file__), '.model2')
        self.pretrained_model_file = os.path.join(os.path.dirname(__file__), '.model1')

    def create_model(self):
        
        pretrained_model = tf.keras.models.load_model(self.pretrained_model_file)
        model_clone = tf.keras.models.clone_model(pretrained_model)
        model_clone.set_weights(pretrained_model.get_weights())
        
        img_input = layers.Input(shape=(self.img_height, self.img_width, 3), name='model2_in')
        x = pretrained_model(img_input)
        output = layers.Dense(self.num_classes, activation='softmax', name='model2_out')(x)
        
        self.model = Model(img_input, output)
        
        for layer in self.model.layers[:-3]:
            layer.trainable = False
        
# DNN model to recognize gesture
class ConvModel(NNModel):
    def __init__(self, img_height, img_width, num_classes):
    
        super().__init__(num_classes)
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model_file = os.path.join(os.path.dirname(__file__), '.model1')

    def create_model(self):

        img_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        
        x = layers.Conv2D(32, (3,3), padding='same')(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(64, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(128, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(self.num_classes)(x)
        output = layers.Activation('softmax')(x)
        
        self.model = Model(img_input, output)
        
# DNN model to recognize gesture
class VGG16Model(NNModel):
    def __init__(self, img_height, img_width, num_classes):
    
        super().__init__(num_classes)
        
        self.img_height = img_height
        self.img_width = 32
        self.num_classes = num_classes
        self.model_file = os.path.join(os.path.dirname(__file__), '.model1')
        self.weight_decay = 0.0001

    def create_model(self):

        weight_decay = self.weight_decay
        img_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        
        x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(img_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Flatten()(x)
        
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(self.num_classes, kernel_regularizer=regularizers.l2(weight_decay))(x)
        output = layers.Activation('softmax')(x)
        
        self.model = Model(img_input, output)
