#! /usr/bin/env python
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Monday, May 15, 2023                                               #
# Description: data logger for Gesture Recognition                         #
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
import time
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from data.formats import GESTURE, GESTURE2
from utils.utility_functions import print
import colorama
from colorama import Fore
colorama.init(autoreset=True)


class Logger:
    def __init__(self, gesture=None):
        self.volunteer = None
        self.gesture = gesture
        self.log_file = ''
        self.raw_file = ''
        
    def __set_file(self):
        
        try:
            last_sample, last_raw_sample, is_empty = self.get_last_sample(self.volunteer, self.gesture)
            
        except TypeError:
            return False
            
        if is_empty:
            self.log_file = os.path.join(last_sample, self.volunteer+'_1.npy')
            self.raw_file = os.path.join(last_raw_sample, 'raw_'+self.volunteer+'_1.npy')
            print(f'{Fore.RED}{self.gesture.upper()}{Fore.RESET}: sample no. 1')
            return True
        else:
            save_dir = os.path.dirname(last_sample)
            raw_dir = os.path.dirname(last_raw_sample)
            last_sample_name = os.path.splitext(last_sample)[0]
            num = int(os.path.basename(last_sample_name).split('_')[1]) + 1
            self.log_file = os.path.join(save_dir, self.volunteer+'_'+str(num)+'.npy')
            self.raw_file = os.path.join(raw_dir, 'raw_'+self.volunteer+'_'+str(num)+'.npy')
            print(f'{Fore.RED}{self.gesture.upper()}{Fore.RESET}: sample no. {num}')
            return True

    def set_gesture(self, volunteer, gesture):
        self.volunteer = volunteer.lower()
        self.gesture = gesture

    def log(self, data):
        
        if self.__set_file():
            print('Saving...')
        else:
            return

        np.save(self.log_file, data[0])
        np.save(self.raw_file, data[1])
        
        if os.path.isfile(self.log_file):
            print('Gesture saved.')
        else:
            print('Fail to save gesture.')
        
        if os.path.isfile(self.raw_file):
            print('RAW data saved.\n')
        else:
            print('Fail to save RAW data.\n')
        
    @staticmethod
    def get_last_sample(volunteer, gesture):
        
        if isinstance(gesture, str):
            try:
                gesture = GESTURE[gesture.upper()]
            except KeyError:
                gesture = GESTURE2[gesture.upper()]
                
        save_dir = os.path.join(gesture.get_dir(), volunteer)
        raw_dir = os.path.join(save_dir, '..', '..', '..', 'raw_data', gesture.name.lower(), volunteer)
        
        if not (os.path.isdir(save_dir) and os.path.isdir(raw_dir)):
            print(f'Volunteer {volunteer} does NOT exist.\n')
            return
        elif os.listdir(save_dir) == []:
            return save_dir, raw_dir, True                           # True: save_dir is empty
            
            nums = []
            for f in os.listdir(save_dir):
                num = os.path.splitext(f)[0].split('_')[1]
                nums.append(int(num))
            last_sample = volunteer + '_' + str(max(nums)) + '.npy'
            last_raw_sample = 'raw_' + volunteer + '_' + str(max(nums)) + '.npy'

            return os.path.join(save_dir, last_sample), os.path.join(raw_dir, last_raw_sample), False   # False: save_dir is not empty
            
    def discard_last_sample(self):
        
        try:
            last_sample, last_raw_sample, is_empty = self.get_last_sample(self.volunteer, self.gesture)
            
        except TypeError:
            return
        
        if is_empty:
            print(f'{self.volunteer}\'s Gesture {Fore.RED}{self.gesture.upper()} {Fore.RESET}is empty.')
        else:
            os.remove(last_sample)
            os.remove(last_raw_sample)
            print('File deleted.')

    @staticmethod
    def get_data(gesture):
        if isinstance(gesture, str):
            try:
                gesture = GESTURE[gesture.upper()]
            except KeyError:
                gesture = GESTURE2[gesture.upper()]
                
        save_dir = gesture.get_dir()
        for dname in os.listdir(save_dir):
            volunteer = os.path.join(save_dir, dname)
            for fname in tqdm(os.listdir(volunteer), desc='Files', leave=False):
                spectrogram_img = np.load(os.path.join(volunteer, fname))

                range_spec = spectrogram_img[0]
                range_spec = range_spec / np.max(range_spec)

                udoppler = spectrogram_img[1]
                udoppler = udoppler / np.max(udoppler)

                angle_spec = spectrogram_img[2]
                angle_spec = angle_spec / np.max(angle_spec)
                '''
                # image processing, need to be confirmed
                range_spec = range_spec.reshape(range_spec.shape + (1,))
                udoppler = udoppler.reshape(udoppler.shape + (1,))
                angle_spec = angle_spec.reshape(angle_spec.shape + (1,))
                '''
                sample = np.array((range_spec, udoppler, angle_spec))
                sample = np.transpose(sample, (1, 2, 0))

                yield sample

    @staticmethod
    def get_test_data(gesture):
        if isinstance(gesture, str):
            try:
                gesture = GESTURE[gesture.upper()]
            except KeyError:
                gesture = GESTURE2[gesture.upper()]
                
        save_dir = gesture.get_test_dir()
        for dname in os.listdir(save_dir):
            volunteer = os.path.join(save_dir, dname)
            for fname in tqdm(os.listdir(volunteer), desc='Files', leave=False):
                spectrogram_img = np.load(os.path.join(volunteer, fname))

                range_spec = spectrogram_img[0]
                range_spec = range_spec / np.max(range_spec)

                udoppler = spectrogram_img[1]
                udoppler = udoppler / np.max(udoppler)

                angle_spec = spectrogram_img[2]
                angle_spec = angle_spec / np.max(angle_spec)
                '''
                # image processing, need to be confirmed
                range_spec = range_spec.reshape(range_spec.shape + (1,))
                udoppler = udoppler.reshape(udoppler.shape + (1,))
                angle_spec = angle_spec.reshape(angle_spec.shape + (1,))
                '''
                sample = np.array((range_spec, udoppler, angle_spec))
                sample = np.transpose(sample, (1, 2, 0))

                yield sample
                
    @staticmethod
    def get_data2(gesture):
        if isinstance(gesture, str):
            try:
                gesture = GESTURE[gesture.upper()]
            except KeyError:
                gesture = GESTURE2[gesture.upper()]
                
        save_dir = gesture.get_dir()
        for fname in tqdm(os.listdir(save_dir), desc='Files', leave=False):
            spectrogram_img = np.load(os.path.join(save_dir, fname))

            range_spec = spectrogram_img[0]
            range_spec = range_spec / np.max(range_spec)

            udoppler = spectrogram_img[1]
            udoppler = udoppler / np.max(udoppler)

            angle_spec = spectrogram_img[2]
            angle_spec = angle_spec / np.max(angle_spec)
            '''
            # image processing, need to be confirmed
            range_spec = range_spec.reshape(range_spec.shape + (1,))
            udoppler = udoppler.reshape(udoppler.shape + (1,))
            angle_spec = angle_spec.reshape(angle_spec.shape + (1,))
            '''
            sample = np.array((range_spec, udoppler, angle_spec))
            sample = np.transpose(sample, (1, 2, 0))

            yield sample
            
    @staticmethod
    def get_test_data2(gesture):
        if isinstance(gesture, str):
            try:
                gesture = GESTURE[gesture.upper()]
            except KeyError:
                gesture = GESTURE2[gesture.upper()]
                
        save_dir = gesture.get_test_dir()
        for fname in tqdm(os.listdir(save_dir), desc='Files', leave=False):
            spectrogram_img = np.load(os.path.join(save_dir, fname))

            range_spec = spectrogram_img[0]
            range_spec = range_spec / np.max(range_spec)

            udoppler = spectrogram_img[1]
            udoppler = udoppler / np.max(udoppler)

            angle_spec = spectrogram_img[2]
            angle_spec = angle_spec / np.max(angle_spec)
            '''
            # image processing, need to be confirmed
            range_spec = range_spec.reshape(range_spec.shape + (1,))
            udoppler = udoppler.reshape(udoppler.shape + (1,))
            angle_spec = angle_spec.reshape(angle_spec.shape + (1,))
            '''
            sample = np.array((range_spec, udoppler, angle_spec))
            sample = np.transpose(sample, (1, 2, 0))

            yield sample
            
    @staticmethod
    def get_stats(X, y):
        
        num_samples, img_height, img_width, img_channels = X.shape
        num_classes = len(set(y))
        
        print(f'Number of classes: {num_classes}')
        print(f'Number of range bins (image height): {img_height}')
        print(f'Number of frames (image width): {img_width}')
        print(f'Total number of samples: {num_samples}')
        
        return num_samples, img_height, img_width, img_channels, num_classes

    @staticmethod
    def get_all_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_training_data')
        y_file = os.path.join(os.path.dirname(__file__), '.y_training_data')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE, desc='Gestures'):
                for sample in Logger.get_data(gesture):
                    X.append(sample)
                    y.append(gesture.value)
            
            X = np.array(X)
            y = np.array(y)
            
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        
        return X, y

    @staticmethod
    def get_all_test_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_test_data')
        y_file = os.path.join(os.path.dirname(__file__), '.y_test_data')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE, desc='Gestures'):
                for sample in Logger.get_test_data(gesture):
                    X.append(sample)
                    y.append(gesture.value)
                    
            X = np.array(X)
            y = np.array(y)
            
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        
        return X, y
    
    @staticmethod
    def get_model2_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_training_data2')
        y_file = os.path.join(os.path.dirname(__file__), '.y_training_data2')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE2, desc='Gestures'):
                for sample in Logger.get_data2(gesture):
                    X.append(sample)
                    y.append(gesture.value)
                    
            X = np.array(X)
            y = np.array(y)
            
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        
        return X, y
    
    @staticmethod
    def get_model2_test_data(refresh_data=False):
        X_file = os.path.join(os.path.dirname(__file__), '.X_test_data2')
        y_file = os.path.join(os.path.dirname(__file__), '.y_test_data2')
        if refresh_data:
            X = []
            y = []
            for gesture in tqdm(GESTURE2, desc='Gestures'):
                for sample in Logger.get_test_data2(gesture):
                    X.append(sample)
                    y.append(gesture.value)
                    
            X = np.array(X)
            y = np.array(y)
            
            pickle.dump(X, open(X_file, 'wb'))
            pickle.dump(y, open(y_file, 'wb'))
        else:
            print('Loading cached data...', end='')
            X = pickle.load(open(X_file, 'rb'))
            y = pickle.load(open(y_file, 'rb'))
            print(f'{Fore.GREEN}Done.')
        
        return X, y
    
