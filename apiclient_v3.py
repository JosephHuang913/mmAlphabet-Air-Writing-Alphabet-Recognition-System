# -*- coding: utf-8 -*-
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Thursday, June 08, 2023                                            #
# Description: Infineon FMCW RADAR System for Gesture Recognition          #
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
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.math import confusion_matrix
import time
from cmd import Cmd
from threading import Lock
import queue
from queue import Queue
from sklearn.model_selection import train_test_split
import colorama
from colorama import Fore
colorama.init(autoreset=True)

sys.path.append('.')

# Infineon
from mmwave_processing.ifxRadarSDK import *
from mmwave_processing.fft_spectrum import *

from utils.configure import Configure
from utils.plotter import Plotter
from utils.handlers import SignalHandler
from utils.utility_functions import threaded, error, warning
from utils.logger import get_logger
from utils.checks import folder_check

import mmwave_processing.dsp as dsp
from mmwave_processing.dsp.utils import Window, windowing
from mmwave_processing.dsp.compensation import exponential_smoothing
from model.model import ConvModel, ConvModel2, VGG16Model#, LstmModel, TransModel
from data.formats import GESTURE, GESTURE2
from data.formats import num_gestures
from data.logger import Logger

class Console(Cmd):
    def __init__(self, plotter_queues):
        super().__init__()
        
        # read FMCW radar configuration
        self.configs = Configure(config_file='fmcw_radar.cfg')

        # set radar metric
        self.metric = {
            'sample_rate_Hz':           self.configs.sample_rate_Hz,
            'range_resolution_m':       self.configs.range_resolution_m,
            'max_range_m':              self.configs.max_range_m,
            'max_speed_m_s':            self.configs.max_speed_m_s,
            'speed_resolution_m_s':     self.configs.speed_resolution_m_s,
            'frame_repetition_time_s':  self.configs.frame_repetition_time_s,
            'center_frequency_Hz':      self.configs.center_frequency_Hz,
            'rx_mask':                  self.configs.rx_mask,
            'tx_mask':                  self.configs.tx_mask,
            'tx_power_level':           self.configs.tx_power_level,
            'if_gain_dB':               self.configs.if_gain_dB
        }

        # Initiate the radar device
        self.__mmwave_init(metric=self.metric)
        
        # get radar configuration
        config = self.device.get_config()
        self.configs.start_frequency_Hz = config['start_frequency_Hz']
        self.configs.end_frequency_Hz = config['end_frequency_Hz']
        self.configs.num_chirps_per_frame = config['num_chirps_per_frame']
        self.configs.num_samples_per_chirp = config['num_samples_per_chirp']
        self.configs.chirp_repetition_time_s = config['chirp_repetition_time_s']
        self.configs.mimo_mode = config['mimo_mode']
        self.configs.frame_rate = 1 / self.configs.frame_repetition_time_s
        
        # RADAR performance metrics
        self.configs.d_res, self.configs.d_max = dsp.range_resolution_max_distance(self.configs)
        self.configs.V_res, self.configs.V_max, self.configs.wave_length = dsp.doppler_resolution_max_speed(self.configs)
        self.NUM_FRAMES = int(self.configs.frame_rate*self.configs.win_len) # for the view of spectrogram, 60 = 3 seconds for 20 frames/sec
        #doppler_res = -2*self.configs.V_res/self.configs.wave_length        # Doppler Shift resolution
        
        # DSP processing parameters
        #self.SKIP_SIZE = 4
        self.ANGLE_RES = 4
        self.ANGLE_RANGE = self.configs.max_angle_degrees
        self.ANGLE_BINS = (self.ANGLE_RANGE * 2) // self.ANGLE_RES + 1      # 31
        self.VIRT_ANT = self.configs.NUM_RX                                 # number of virtual antenna = 2

        # deep learning model
        self.logger = Logger()
        self.num_gestures = num_gestures(GESTURE)
        self.model_type = 'CNN'
        #self.model_type = 'VGG16'
        self.__set_model(self.model_type)

        # Catching signals
        self.console_queue = Queue()
        SignalHandler(self.console_queue)

        # Threading stuff
        self.sensing_lock = Lock()
        self.printing_lock = Lock()
        self.plotting_lock = Lock()
        self.predicting_lock = Lock()
        self.logging_lock = Lock()

        self.sensing = False
        self.printing = False
        self.plotting = False
        self.predicting = False
        self.logging = False

        self.logging_queue = Queue()
        self.data_queue = Queue()
        self.model_queue = Queue()
        self.plotter_queues = plotter_queues

        self.__set_prompt()
        print(f'{Fore.GREEN}Initialized.\n')
        print(f'{Fore.MAGENTA}--- Gesture Recognition based on Spectrogram ---')
        warning('Type \'help\' for more information.')
    
    def __mmwave_init(self, metric):
        
        # Initiate a radar device
        self.device = Device()
        
        # radar configuration from metric
        cfg = self.device.metrics_to_config(**metric)
        
        # set the configuration of the radar device
        self.device.set_config(**cfg)
        
    def __is_connected(self):
        try:
            if self.device:
                return True
            else:
                return False
        except:
            return False
    
    def __disconnect(self):
        if self.device is not None:
            del self.device
            
    def __set_prompt(self):
        if self.__is_connected():
            self.prompt = f'{Fore.GREEN}[Gesture Recognition]>>{Fore.RESET} '
        else:
            self.prompt = f'{Fore.RED}[mmWave sensor Not Initialized]{Fore.RESET} >> '

    def __set_model(self, type):
        if type == 'CNN':
            self.model1 = ConvModel(self.configs.RANGE_FFT//2, self.NUM_FRAMES, self.num_gestures)
            self.model2 = ConvModel2(self.configs.RANGE_FFT//2, self.NUM_FRAMES, num_gestures(GESTURE2))
        elif type == 'VGG16':
            self.model1 = VGG16Model(self.configs.RANGE_FFT//2, self.NUM_FRAMES, self.num_gestures)
            self.model2 = ConvModel2(self.configs.RANGE_FFT//2, self.NUM_FRAMES, num_gestures(GESTURE2))
        
    def preloop(self):
        '''
        Initialization before prompting user for commands.
        Despite the claims in the Cmd documentation, Cmd.preloop() is not a
        stub.
        '''
        Cmd.preloop(self)       # sets up command completion
        self._hist = []         # No history yet
        self._locals = {}       # Initialize execution namespace for user
        self._globals = {}

    def postloop(self):
        '''
        Take care of any unfinished business.
        Despite the claims in the Cmd documentation, Cmd.postloop() is not a
        stub.
        '''
        Cmd.postloop(self)      # Clean up command completion
        print('Exiting...')

    def precmd(self, line):
        '''
        This method is called after the line has been input but before
        it has been interpreted. If you want to modify the input line
        before execution (for example, variable substitution) do it here.
        '''
        self._hist += [line.strip()]

        try:
            info = self.plotter_queues['info'].get(False)
            if info == 'closed':
                print(f'{Fore.YELLOW}Plotter closed.\n')
                with self.plotting_lock:
                    if self.plotting:
                        self.plotting = False
        except queue.Empty:
            pass

        return line

    def postcmd(self, stop, line):
        '''
        If you want to stop the console, return something that evaluates to
        true. If you want to do some post command processing, do it here.
        '''
        self.__set_prompt()
        return stop

    def emptyline(self):
        '''Do nothing on empty input line'''
        pass

    def default(self, line):
        '''
        Called on an input line when the command prefix is not recognized.
        In that case we execute the line as Python code.
        '''
        try:
            exec(line) in self._locals, self._globals
        except Exception:
            error('Unknown arguments.')
            return

    def do_help(self, args):
        '''
        Get help on command

        \'help\' or \'?\' with no arguments prints a list of commands for
        which help is available \'help <command>\' or \'? <command>\' gives
        help on <command>
        '''
        # The only reason to define this method is for the help text in
        # the doc string
        Cmd.do_help(self, args)

    def do_history(self, args):
        '''Print a list of commands that have been entered'''
        if args != '':
            error('Unknown arguments.')
            return
        print(self._hist)

    def do_exit(self, args):
        '''Exits from the console'''

        if args != '':
            error('Unknown arguments.')
            return

        self.do_stop('plot')
        self.do_stop('sense')
        self.do_stop('mmwave')
        
        if self.__is_connected():
            self.__disconnect()

        os._exit(0)

    def __complete_from_list(self, complete_list, text, line):
        mline = line.partition(' ')[2]
        offs = len(mline) - len(text)
        return [s[offs:] for s in complete_list if s.startswith(mline)]
    
    def do_autoconnect(self, args=''):
        '''
        Auto connecting to mmWave sensor

        Look \'connect\' command for manual connection.

        Usage:
        >> autoconnect
        '''
        if args != '':
            error('Unknown arguments.')
            return

        if self.__is_connected():
            warning('Already connected.')
            print('Reconnecting...')

        self.__mmwave_init(metric=self.metric)
        
    def __model_loaded(self):
        
        if self.model1.model is not None:
            model1 = True
        else:
            model1 = False
            
        if self.model2.model is not None:
            model2 = True
        else:
            model2 = False
        
        return model1, model2

    def do_connect(self, args=''):
        '''
        Manually connecting to mmWave sensor

        Look \'autoconnect\' command for automatic connection.

        Usage:
        >> connect
        '''

        if args != '':
            error('Unknown arguments.')
            return

        if self.__is_connected():
            self.__disconnect()
            print()

        self.__mmwave_init(metric=self.metric)
        
    def do_set_model(self, args=''):
        '''
        Set model type used for prediction. Available models are
        \'CNN\' (convolutional 2D), \'LSTM\' (long short-term memory) and
        \'Transformer\' (transformer). Default is CNN.

        Usage:
        >> set_model CNN
        >> set_model LSTM (not implemented)
        >> set_model Transformer (not implemented)
        '''

        if len(args.split()) > 1:
            error('Too many arguments.')
            return

        if args not in ['CNN', 'VGG16']:       
            warning(f'Unknown argument: {args}')
            return

        self.model_type = args
        self.__set_model(args)

    def complete_set_model(self, text, line, begidx, endidx):
        return self.__complete_from_list(['CNN'], text, line)
        
    def do_get_model(self, args=''):
        '''
        Get current model type.

        Usage:
        >> get_model
        '''

        if args != '':
            error('Unknown arguments.')
            return

        print(f'Current model type: {self.model_type}')
    
    @threaded
    def __sensing_thread(self):
        
        print(f'{Fore.CYAN}=== Sensing ===\n')
        
        while True:
            with self.sensing_lock:
                if not self.sensing:
                    return

            #------------------------------------------#
            # Get one frame from the FMCW radar module #
            #------------------------------------------#

            dataCube = self.device.get_next_frame()             # (num_rx, num_chirps, num_samples) = (3, 32, 64)
            #print(f'dataCube.shape: {dataCube.shape}')
            
            if dataCube is None:
                time.sleep(1e-2)
                continue

            self.data_queue.put(dataCube)
            time.sleep(1e-2)
            
    @threaded
    def __processing_thread(self):
        
        # Start DSP processing
        BINS_for_Gesture = self.configs.num_samples_per_chirp//2    # 32
        Range_Spec = np.zeros((self.NUM_FRAMES, self.configs.num_samples_per_chirp//2), dtype=np.float64)       # (60, 32)
        uDoppler = np.zeros((self.NUM_FRAMES, self.configs.num_chirps_per_frame), dtype=np.float64)             # (60, 32)
        Angle_Spec = np.zeros((self.NUM_FRAMES, self.ANGLE_BINS+1), dtype=np.float64)                           # (60, 32)
        
        range_azimuth = np.zeros((self.ANGLE_BINS, BINS_for_Gesture), dtype=np.float64)   # (31, 32)
        num_vec, steering_vec = dsp.gen_steering_vec(self.ANGLE_RANGE, self.ANGLE_RES, self.VIRT_ANT)
        
        # initial condition for Exponential Smoothing
        init = True         # for exponential smoothing of static clutter removal
        pre_val = None      # for exponential smoothing of static clutter removal
        
        frame_num = 0
        raw_dataCubes = []
        while True:
            with self.sensing_lock:
                if not self.sensing:
                    return

            dataCube = self.data_queue.get()                    # (num_rx, num_chirps, num_samples) = (3, 32, 64)
            #print(f'dataCube.shape: {dataCube.shape}')
            
            if dataCube is None:
                dataCube = np.zeros((self.configs.NUM_RX+1, self.configs.num_chirps_per_frame, self.configs.num_samples_per_chirp))
                
            #===================================#
            # RADAR Signal Processing per Frame #
            #===================================#

            #---------------------#
            # 1. Range Processing #
            #---------------------#

            # Remove DC bias from ADC raw data before Range FFT
            for i in range(self.configs.NUM_RX+1):
                dataCube[i, :, :] = remove_DC_bias(dataCube[i, :, :])

            rawdataCube = dataCube                              # (num_rx, num_chirps, num_samples) = (3, 32, 64)
            dataCube = np.transpose(dataCube, (1, 0, 2))        # (num_chirps, num_rx, num_samples) = (32, 3, 64)
            dataCube = dsp.range_processing(dataCube, n_fft=self.configs.RANGE_FFT, window_type_1d=Window.BLACKMAN)
            
            radar_cube = dataCube[:, :, :BINS_for_Gesture]      # (num_chirps, num_rx, range bins) = (32, 3, 32)
            #print(f'radar_cube.shape: {radar_cube.shape}')
            
            #---------------------------#
            # 2. Static Clutter Removal #
            #---------------------------#

            # exponential smoothing for static clutter removal
            radar_cube_smoothed, pre_val = exponential_smoothing(radar_cube, pre_val, init, alpha=self.configs.alpha)
            init = False

            # static clutter removal: method 1, exponential smoothing
            radar_cube_clutter_removed = radar_cube - radar_cube_smoothed # (num_chirps, num_rx, range bins) = (32, 3, 32)
            #print(f'radar_cube_clutter_removed.shape: {radar_cube_clutter_removed.shape}')
            
            #-----------------------#
            # 3. Doppler Processing #
            #-----------------------#

            # range_doppler.shape: (range bins, doppler bins) = (32, 32)
            # aoa_input.shape: (range bins, num_rx, doppler bins) = (32, 3, 32)
            range_doppler, aoa_input, _ = dsp.doppler_processing(radar_cube_clutter_removed, num_tx_antennas=self.configs.NUM_TX, n_fft=self.configs.DOPPLER_FFT, window_type_2d=Window.HANNING)
            #print(f"range_doppler.shape: {range_doppler.shape}")
            #print(f"aoa_input.shape: {aoa_input.shape}")
            #print(f"fft_ph.shape: {fft_ph.shape}")

            RD_Map = np.fft.fftshift(range_doppler, axes=1)                 # (range bins, doppler bins) = (32, 32)
            
            ind = np.unravel_index(np.argmax(RD_Map), RD_Map.shape)         # (range bin index, doppler bin index)
            
            #----------------------#
            # 4. Range Spectrogram #
            #----------------------#
            
            Range_Spec[:-1,:] = Range_Spec[1:,:]
            Range_Spec[-1,:] = RD_Map[:,ind[1]]             # (num_frames, range bins) = (60, 32)
            
            range_spec_img = Range_Spec.T                   # (range_bins, num_frames) = (32, 60)
            
            #------------------------------#
            # 5. Micro Doppler Spectrogram #
            #------------------------------#

            uDoppler[:-1,:] = uDoppler[1:,:]                # (NUM_FRAMES, doppler bins) = (60, 32)
            uDoppler[-1,:] = RD_Map[ind[0],:]
            
            uDoppler_img = uDoppler.T                       # (doppler bins, num_frames) = (32, 60)
            
            #---------------------------------------------------------------#
            # 6. Beamforming                                                #
            #    MVDR (Minimum Variance Distortionless Response) beamformer #
            #    Also knows as Capon beamformer                             #
            #---------------------------------------------------------------#

            beamWeights = np.zeros((self.VIRT_ANT, BINS_for_Gesture), dtype=np.complex_)
            aoa_in = np.transpose(aoa_input, (2, 1, 0))     # aoa_in.shape: (doppler bins, num_rx, range bins) = (32, 3, 32)
            for i in range(BINS_for_Gesture):
                range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(aoa_in[:, ::2, i].T, steering_vec, magnitude=True, diag_load = 3.16)
                
            #print(f'range azimuth shape:\t{range_azimuth.shape}')           # (num_beams, range bins) = (31, 32)
            #print(f'beam weights shape:\t{beamWeights.shape}')              # (num_rx, range bins) = (2, 32)
        
            #----------------------#
            # 7. Angle Spectrogram #
            #----------------------#
            
            Angle_Spec[:-1,:] = Angle_Spec[1:,:]        # (NUM_FRAMES, num_beams) = (60, 32)
            Angle_Spec[-1,:-1] = range_azimuth[:,np.unravel_index(np.argmax(range_azimuth), range_azimuth.shape)[1]]
            
            angle_spec_img = Angle_Spec.T               # (num_beams, num_frames) = (32, 60)
            
            #==================== Frame Data Recording ====================#
            
            spectrogram_img = np.array((range_spec_img, uDoppler_img, angle_spec_img))
            
            with self.logging_lock:
                if self.logging:
                    frame_num += 1
                    raw_dataCubes.append(rawdataCube)
                    
                    if frame_num == self.NUM_FRAMES:
                        self.logging_queue.put((spectrogram_img, np.array(raw_dataCubes)))
                        
                        raw_dataCubes = []
                        frame_num = 0
                        self.logging = False
            
            with self.plotting_lock:
                if self.plotting:
                    self.plotter_queues['data'].put(spectrogram_img)

            with self.predicting_lock:
                if self.predicting:
                    frame_num += 1
                    
                    if frame_num == self.NUM_FRAMES//2:                 # recognize gesture every 1.5 seconds
                        self.model_queue.put(spectrogram_img)
                        frame_num = 0

            time.sleep(1e-2)

    @threaded
    def __predict_thread(self):
        
        while True:
            spectrogram_img = self.model_queue.get()

            with self.predicting_lock:
                if not self.predicting:
                    return
                
            if (spectrogram_img is not None):
                print('Predicting...')
                pred = self.model2.predict(spectrogram_img=spectrogram_img, model_type='model2')
                
                if pred == 'character':
                    print('Recognizing character...')
                    _ = self.model1.predict(spectrogram_img=spectrogram_img, model_type='model1')

            time.sleep(5e-1)
            
    @threaded
    def __logging_thread(self):
        
        print('Logging...')
        
        while True:
            
            spectrogram_img, raw_dataCube = self.logging_queue.get()
            
            if spectrogram_img is not None:
                self.logger.log((spectrogram_img, raw_dataCube))
                
                return
            
    def do_sense(self, args=''):
        '''
        Start sensing and signal processing thread

        Look \'connect\' and \'autoconnect\' command for connecting to mmWave sensor.

        Usage:
        >> sense
        '''
        
        if args != '':
            error('Unknown arguments.')
            return
        
        with self.sensing_lock:
            if self.sensing:
                warning('Sensor already started.')
                return

        with self.sensing_lock:
            self.sensing = True

        self.__sensing_thread()
        self.__processing_thread()

    def do_stop(self, args=''):
        '''
        Stopping mmwave, sensor and plotter.

        Possible options: \'mmwave\', \'sense\' and \'plot\'.
            \'mmwave\': Sending \'end_session\' command to mmWave sensor.
            \'sense\': Stopping sensor and signal processing threads.
            \'plot\': Closing plotter.
        
        Usage:
        >> stop sense
        >> stop plot
        >> stop mmwave
        '''
        
        if args == '':
            with self.sensing_lock:
                if self.sensing:
                    self.sensing = False
                    self.data_queue.put(None)
                    print('sensor stopped.')
                    
            if self.__is_connected():
                self.__disconnect()
            print('mmWave sensor disconnected.')
            
            #warning(f'Unknown argument.')
        else:
            opts = args.split()

            if 'plot' in opts:
                with self.plotting_lock:
                    if self.plotting:
                        self.plotting = False
                        print('Plotter stopped.')
                self.plotter_queues['cli'].put('close')
                opts.remove('plot')

            if 'sense' in opts:
                with self.sensing_lock:
                    if self.sensing:
                        self.sensing = False
                        self.data_queue.put(None)
                        print('sensor stopped.')
                opts.remove('sense')

            if 'mmwave' in opts:
                if self.__is_connected():
                    self.__disconnect()
                print('mmWave sensor disconnected.')
                opts.remove('mmwave')

            for opt in opts:
                warning(f'Unknown option: {opt}. Skipped.')

    def complete_stop(self, text, line, begidx, endidx):
        return self.__complete_from_list(['mmwave', 'sense', 'plot'], text, line)

    def do_print(self, args=''):
        '''
        Pretty print

        Printing received frames. Sensor should be started before using
        this command. Use <Ctrl-C> to stop this command.

        Usage:
        >> print
        '''
        if args != '':
            error('Unknown arguments.')
            return

        with self.sensing_lock:
            if not self.sensing:
                error('Sensor not started.')
                return

        with self.printing_lock:
            self.printing = True

        self.console_queue.get()

        with self.printing_lock:
            self.printing = False

    def do_plot(self, args=''):
        '''
        Start plotter

        Plotting received image frames. sensor should be started before using
        this command. Use \'stop plot\' to stop this command.

        Usage:
        >> plot
        '''
        if args != '':
            error('Unknown arguments.')
            return

        self.plotter_queues['cli'].put('init')
        
        with self.plotting_lock:
            self.plotting = True

    def do_predict(self, args=''):
        '''
        Start prediction

        Passing received frames through neural network and printing results.
        Sensor should be started before using this command.
        Use <Ctrl-C> to stop this command.

        Usage:
        >> predict
        '''
        if args != '':
            error('Unknown arguments.')
            return

        with self.sensing_lock:
            if not self.sensing:
                error('Sensor not started.')
                return

        if not self.__model_loaded()[0]:
            self.model1.load()
        if not self.__model_loaded()[1]:
            self.model2.load()
            print()

        with self.predicting_lock:
            self.predicting = True

        self.__predict_thread()

        self.console_queue.get()

        with self.predicting_lock:
            self.predicting = False

        self.model_queue.put(None)

    def do_start(self, args=''):
        '''
        Start sensor, plotter and prediction.

        Use <Ctrl-C> to stop this command.

        Usage:
        >> start
        '''
        if args != '':
            error('Unknown arguments.')
            return

        self.do_autoconnect()
        self.do_sense()
        
    def do_train(self, args=''):
        '''
        Train neural network

        Command will first load cached X and y data located in
        'mmwave/data/.X_data' and 'mmwave/data/.y_data' files. This data will be
        used for the training process. If you want to read raw .npy files,
        provide \'refresh\' (this will take few minutes).

        Usage:
        >> train model1
        >> train model1 refresh
        '''

        model1 = False
        model2 = False
        refresh_data = False
        
        if len(args.split()) > 2 or len(args.split()) == 0:
            error('Unknown arguments.')
            return

        
        if 'model1' in args.split() and ('refresh' in args.split()):
            model1 = True
            refresh_data = True
        elif 'model2' in args.split() and 'refresh' in args.split():
            model2 = True
            refresh_data = True
        elif 'model1' in args.split() and 'model2' in args.split():
            model1 = True
            model2 = True
            refresh_data = True
        elif 'model1' in args.split():
            model1 = True
        elif 'model2' in args.split():
            model2 = True
        else:
            warning(f'Unknown argument: {args}')
            return

        if model1:
            print(refresh_data)
            X, y = Logger.get_all_data(refresh_data=refresh_data)
            Logger.get_stats(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7890)
            #X_train, _, y_train, _ = train_test_split(X, y, test_size=self.num_gestures, stratify=y)
            #X, y = Logger.get_all_test_data(refresh_data=refresh_data)
            #_, X_test, _, y_test = train_test_split(X, y, train_size=self.num_gestures, stratify=y)
            '''
            self.model1.train(X_train, y_train, 'model1')

            train_hist = pd.DataFrame(self.model1.history.history)
            train_hist['epoch'] = self.model1.history.epoch
            train_hist.to_csv('train_history.csv', encoding='utf-8')
            '''
            results = []
            for _ in range(1):
                self.model1.train(X_train, y_train, 'model1')
                acc = self.model1.evaluate(X_test, y_test, 'model1')
                results.append(acc)
                preds = self.model1.model.predict(X_test)
                y_preds = np.argmax(preds, axis=1)

                conf_matrix = confusion_matrix(y_test, y_preds, num_classes=self.num_gestures)
                np.save('confusion_matrix_'+str(acc)+'.npy', conf_matrix)
                print(f'confusion matrix of model1:\n{conf_matrix}')
                print()
            
            print()
            print('--------------------------------------------------------------------------------')
            print(f'Test Accuracies: {results}')
            
        if model2:
            X, y = Logger.get_model2_data(refresh_data=refresh_data)
            Logger.get_stats(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_gestures(GESTURE2), stratify=y)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_gestures(GESTURE2), stratify=y, random_state=1234)
            
            self.model2.train(X_train, y_train, 'model2')

            train_hist = pd.DataFrame(self.model2.history.history)
            train_hist['epoch'] = self.model2.history.epoch
            train_hist.to_csv('train_model2_history.csv', encoding='utf-8')
            
    def complete_train(self, text, line, begidx, endidx):
        return self.__complete_from_list(['refresh'], text, line)

    def do_eval(self, args=''):
        '''
        Evaluate neural network

        Command will first load cached X and y data located in
        \'mmwave/data/.X_data\' and \'mmwave/data/.y_data\' files. This data
        will be used for the evaluating process. If you want to read raw .npy
        files, provide \'refresh\' (this will take few minutes).

        Usage:
        >> eval model1
        >> eval model1 refresh
        '''

        model1 = False
        model2 = False
        refresh_data = False
        
        if len(args.split()) > 2 or len(args.split()) == 0:
            error('Unknown arguments.')
            return

        if 'model1' in args.split() and ('refresh' in args.split()):
            model1 = True
            refresh_data = True
        elif 'model2' in args.split() and 'refresh' in args.split():
            model2 = True
            refresh_data = True
        elif 'model1' in args.split() and 'model2' in args.split():
            model1 = True
            model2 = True
            refresh_data = True
        elif 'model1' in args.split():
            model1 = True
        elif 'model2' in args.split():
            model2 = True
        else:
            warning(f'Unknown argument: {args}')
            return
        
        if model1:
            #X, y = Logger.get_all_data(refresh_data=refresh_data)
            X, y = Logger.get_all_test_data(refresh_data=refresh_data)
            Logger.get_stats(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.num_gestures, stratify=y)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7890)
        
            if not self.__model_loaded()[0]:
                self.model1.load()
            '''
            print('Eval validation dataset:')
            self.model.evaluate(X_val, y_val)
            print()

            print('Eval train dataset:')
            self.model.evaluate(X_train, y_train)
            print()
            '''
            print('Evaluate test dataset:')
            self.model1.evaluate(X_test, y_test, 'model1')
        
            #y_preds = self.model.model.predict_on_batch(X)
            preds = self.model1.model.predict(X)
            y_preds = np.argmax(preds, axis=1)

            conf_matrix = confusion_matrix(y, y_preds, num_classes=self.num_gestures)
            np.save('confusion_matrix.npy', conf_matrix)
            print(f'confusion matrix of model1:\n{conf_matrix}')
            print()
            
        if model2:
            #X, y = Logger.get_model2_data(refresh_data=refresh_data)
            X, y = Logger.get_model2_test_data(refresh_data=refresh_data)
            Logger.get_stats(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_gestures(GESTURE2), stratify=y)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_gestures(GESTURE2), stratify=y, random_state=1234)
        
            if not self.__model_loaded()[1]:
                self.model2.load()
            
            print('Evaluate test dataset:')
            self.model2.evaluate(X_test, y_test)
        
            #y_preds = self.model.model.predict_on_batch(X)
            preds = self.model2.model.predict(X)
            y_preds = np.argmax(preds, axis=1)

            conf_matrix = confusion_matrix(y, y_preds, num_classes=num_gestures(GESTURE2))
            np.save('confusion_matrix_model2.npy', conf_matrix)
            print(f'confusion matrix of model2:\n{conf_matrix}')
            print()

    def complete_eval(self, text, line, begidx, endidx):
        return self.__complete_from_list(['refresh'], text, line)

    def do_log(self, args=''):
        '''
        Log data

        Logging specified gesture. Data will be saved in
        \'data/training_data/gesture_folder/volunteer_folder/volunteer_num.npy\' file.
        Possible gesture options: \'a\', \'b\', \'c\', \'d\', \'e\', ...
                          \'v\', \'w\', \'x\', \'y\', \'z\', \'nac\'
        Usage:
        >> log joseph a
        >> log eva b
        >> log esther nac      # not a character
        '''
        
        with self.sensing_lock:
            if not self.sensing:
                error('sensor not started.')
                return

        if args == '':
            error('too few arguments.')
            return
        elif len(args.split()) > 2:
            error('Unknown arguments.')
            return
        else:
            opts = args.split()
        
        if not (GESTURE.check(opts[1]) or GESTURE2.check(opts[1])):
            warning(f'Unknown gesture: {opts[1]}')
            return

        self.logger.set_gesture(opts[0], opts[1])

        with self.logging_lock:
            self.logging = True

        self.__logging_thread().join()

        with self.logging_lock:
            self.logging = False

    def __complete_gestures(self, text, line):
        completions = []
        for gesture in GESTURE:
            completions.append(gesture.name.lower())
        return self.__complete_from_list(completions, text, line)

    def complete_log(self, text, line, begidx, endidx):
        return self.__complete_gestures(text, line)

    def do_remove(self, args=''):
        '''
        Remove last sample

        Removing last gesture sample. Data will be removed from
        \'data/training_data/gesture_folder/volunteer_folder/'.
        Possible gesture options: \'a\', \'b\', \'c\', \'d\', \'e\', ...
                          \'v\', \'w\', \'x\', \'y\', \'z\', \'nac\'
        Usage:
        >> remove joseph a
        >> remove eva b
        >> remove esther nac
        '''
        if args == '':
            error('too few arguments.')
            return
        elif len(args.split()) > 2:
            error('Unknown arguments.')
            return
        else:
            opts = args.split()
            
        if not (GESTURE.check(opts[1]) or GESTURE2.check(opts[1])):
            warning(f'Unknown gesture: {opts[1]}')
            return

        self.logger.set_gesture(opts[0], opts[1])
        self.logger.discard_last_sample()

    def complete_remove(self, text, line, begidx, endidx):
        return self.__complete_gestures(text, line)

    def do_redraw(self, args=''):
        '''
        Redraw sample

        Redrawing last captured gesture file.
        Possible gesture options: \'a\', \'b\', \'c\', \'d\', \'e\', ...
                          \'v\', \'w\', \'x\', \'y\', \'z\', \'nac\'
        Usage:
        >> redraw joseph a
        >> redraw eva b
        >> redraw esther nac
        '''
        
        with self.plotting_lock:
            if not self.plotting:
                error('Plotter not started.')
                return

        if args == '':
            error('too few arguments.')
            return
        elif len(args.split()) > 2:
            error('Unknown arguments.')
            return
        else:
            opts = args.split()
            
        if not (GESTURE.check(opts[1]) or GESTURE2.check(opts[1])):
            warning(f'Unknown argument: {opts[1]}')
            return

        self.plotter_queues['cli'].put('redraw')
        self.plotter_queues['cli'].put((opts[0],opts[1]))

    def complete_redraw(self, text, line, begidx, endidx):
        return self.__complete_gestures(text, line)

@threaded
def console_thread(console):
    while True:
        console.cmdloop()

def init_plotter(plotter_queues, x_axis=60, x_tick=0.05, y_axis=32, d_res=0.1, V_res=0.1, angle_res=1):
    while True:
        try:
            cmd = plotter_queues['cli'].get(False)
            if cmd == 'init':
                plt.close('all')
                plotter = Plotter(plotter_queues['info'], x_axis=x_axis, x_tick=x_tick, y_axis=y_axis, d_res=d_res, V_res=V_res, angle_res=angle_res)
                plotter.show()
                plt.gcf().canvas.flush_events()
                
                return plotter
            
        except queue.Empty:
            pass
        time.sleep(1e-2)

def set_plotter(plotter, command):
    try:
        cmd = command.get(False)
        if cmd == 'close':
            if plotter is not None:
                plotter.close()
            return None
        elif cmd == 'redraw':
            (volunteer, gesture) = command.get()
            
            if plotter is not None:
                plotter.draw_last_sample(volunteer, gesture)
                    
    except queue.Empty:
        pass
    return plotter

def plotting(plotter_queues, x_axis=60, x_tick=0.05, y_axis=32, d_res=0.1, V_res=0.1, angle_res=1):
    plotter = None
    while True:
        if plotter is None:
            plotter = init_plotter(plotter_queues, x_axis=x_axis, x_tick=x_tick, y_axis=y_axis, d_res=d_res, V_res=V_res, angle_res=angle_res)
        else:
            plotter = set_plotter(plotter, plotter_queues['cli'])

        # Plot data
        if plotter is not None:
            try:
                spectrogram_img = plotter_queues['data'].get(False)
                plt.gcf().canvas.flush_events()
                plotter.plot_spectrogram_img(spectrogram_img)
            except queue.Empty:
                pass

        time.sleep(1e-2)

def plot_spectrogram(plotter_queues, x_axis=60, x_tick=0.05, y_axis=32, d_res=0.1, V_res=0.1, angle_res=1):
    global exit
    
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax = fig.add_subplot(111)
    exit = False
    
    while not exit:
    
        try:
            spectrogram_img = plotter_queues['data'].get(False)
            plt.imshow(np.transpose(spectrogram_img, (1, 2, 0)))
            
            plt.title("Spectrogram")
            #plt.ylabel("Range (m)")
            #plt.xlabel("Time (seconds)")
            plt.pause(1e-2)
            plt.clf()
        except queue.Empty:
            pass

def on_press(event):
    global exit
    
    print('press key: ', event.key)
    sys.stdout.flush()
    if event.key == 'q':
        plt.close('all')
        exit = True
    
if __name__ == "__main__":
    
    x_axis = 60                     # NUM_FRAMES
    x_tick = 0.05                   # frame_repetition_time_s
    y_axis = 32                     # num_samples_per_chirp/2 or Bins_for_Gesture or chirps per frame or angle_bins+1
    #y_tick = None
    d_res = 0.020979695091159697    # y_tick
    V_res = 0.09387279183334979     # y_tick
    ANGLE_RES = 4                   # y_tick
    
    plotter_queues = {'data': Queue(), 'cli': Queue(), 'info': Queue()}
    console_thread(Console(plotter_queues))

    # Plotter has to be located in the main thread
    plotting(plotter_queues, x_axis=x_axis, x_tick=x_tick, y_axis=y_axis, d_res=d_res, V_res=V_res, angle_res=ANGLE_RES)
    