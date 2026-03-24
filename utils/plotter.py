#! /usr/bin/env python
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Thursday, June 15, 2023                                            #
# Description: Plotter of FMCW RADAR System for Gesture Recognition        #
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

import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from data.logger import Logger
import colorama
from colorama import Fore
colorama.init(autoreset=True)
'''
# Global plot config
mpl.use('Qt5Agg')
mpl.use('TkAgg')
mpl.rcParams['toolbar'] = 'None'
plt.style.use('seaborn-dark')
'''
class Plotter:
    def __init__(self, queue, x_axis=60, x_tick=0.05, y_axis=32, d_res=0.1, V_res=0.1, angle_res=1):
        
        self.x_axis = x_axis
        self.x_tick = x_tick
        self.y_axis  = y_axis
        #self.y_tick = y_tick
        self.d_res = d_res
        self.V_res = V_res
        self.angle_res = angle_res
        self.spec_img = np.zeros((self.y_axis, self.x_axis), dtype=np.float64)
        self.queue = queue
        
        self.fig = None
        self.ax1 = None
        self.imshow1 = None
        self.ax2 = None
        self.imshow2 = None
        self.ax3 = None
        self.imshow3 = None
        
        self.__set_figure()

        connect = self.fig.canvas.mpl_connect
        self.draw = connect('draw_event', self.__grab_background)
        connect('close_event', self.__fig_close)

    def __set_figure(self):
        
        self.fig = plt.figure()
        
        # Range Spectrogram
        self.ax1 = self.fig.add_subplot(311)
        self.imshow1 =  self.ax1.imshow(self.spec_img, cmap='turbo', origin='lower', 
                                        extent=(0, self.x_axis*self.x_tick, 0, self.y_axis*self.d_res))
        self.ax1.set_title("Range Profile Spectrogram")
        #self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Range (m)")
        
        # micro Doppler Effect
        self.ax2 = self.fig.add_subplot(312)
        self.imshow2 =  self.ax2.imshow(self.spec_img, cmap='turbo', origin='upper', 
                                        extent=(0, self.x_axis*self.x_tick, -1, 1))
        self.ax2.set_title("Micro Doppler Spectrogram")
        #self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Normalized Doppler Frequency")

        # Angle Spectrogram
        self.ax3 = self.fig.add_subplot(313)
        self.imshow3 =  self.ax3.imshow(self.spec_img, cmap='turbo', origin='lower', 
                                        extent=(0, self.x_axis*self.x_tick, 
                                                -np.deg2rad(self.y_axis*self.angle_res/2), np.deg2rad(self.y_axis*self.angle_res/2)))
        self.ax3.set_title("Angle Profile Spectrogram")
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Azimuth (rad)")
        
    def __grab_background(self, event=None):
        
        # Temporarily disconnect the draw_event callback to avoid recursion
        canvas = self.fig.canvas
        canvas.mpl_disconnect(self.draw)
        canvas.draw()

        self.draw = canvas.mpl_connect('draw_event', self.__grab_background)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
    def __fig_close(self, event=None):
        self.queue.put('closed')

    def __blit(self):
        self.fig.canvas.restore_region(self.background)
        self.ax1.draw_artist(self.imshow1)
        self.ax2.draw_artist(self.imshow2)
        self.ax3.draw_artist(self.imshow3)
        self.fig.canvas.blit(self.fig.bbox)

    def __update(self):
        
        self.__blit()

    def __get_previos_data(self, volunteer, gesture):
        
        try:
            last_sample, last_raw_sample, is_empty = Logger.get_last_sample(volunteer, gesture)
            #last_sample, is_empty = Logger.get_last_sample(self.gesture)
        except TypeError:
            return None
        
        if is_empty:
            print(f'{volunteer}\'s Gesture {Fore.RED}{gesture.upper()} {Fore.RESET}is empty.')
            return None
        else:
            return np.load(last_sample)
        
    def draw_last_sample(self, volunteer, gesture):
        
        last_spectrogram_img = self.__get_previos_data(volunteer, gesture)
        print('Redrawing...')
        
        if last_spectrogram_img is None:
            print('Nothing to redraw.')
        else:
            plt.gcf().canvas.flush_events()
            #self.spectrogram_img = np.zeros((3, self.y_axis, self.x_axis), dtype=np.float64)
            #self.__update()
            
            self.plot_spectrogram_img(last_spectrogram_img)
            plt.pause(1.0)
            
        print('Done')

    def plot_spectrogram_img(self, spectrogram_img):
        #self.spectrogram_img = np.zeros((self.y_axis, self.x_axis), dtype=np.float64)

        if (spectrogram_img is not None):

            self.spectrogram_img = spectrogram_img
            
            self.imshow1 = self.ax1.imshow(self.spectrogram_img[0], cmap='turbo', origin='lower', 
                                           extent=(0, self.x_axis*self.x_tick, 0, self.y_axis*self.d_res))
            
            self.imshow2 =  self.ax2.imshow(self.spectrogram_img[1], cmap='turbo', origin='upper', 
                                            extent=(0, self.x_axis*self.x_tick, -1, 1))
            
            self.imshow3 =  self.ax3.imshow(self.spectrogram_img[2], cmap='turbo', origin='lower', 
                                            extent=(0, self.x_axis*self.x_tick, 
                                                    -np.deg2rad(self.y_axis*self.angle_res/2), np.deg2rad(self.y_axis*self.angle_res/2)))
                
            self.ax1.draw_artist(self.imshow1)
            self.ax2.draw_artist(self.imshow2)
            self.ax3.draw_artist(self.imshow3)
        
        self.__update()

    def show(self):
        plt.gcf().canvas.flush_events()
        plt.show(block=False)
        time.sleep(1e-2)
        
    def close(self):
        plt.close()
