#! /usr/bin/env python
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Monday, May 15, 2023                                               #
# Description: gesture definition for Gesture Recognition                  #
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

from enum import Enum, auto
import os

def num_gestures(GESTURE):
    num = 0

    for _ in GESTURE:
        num += 1
    
    return num

class GESTURE(Enum):
    A = 0
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()
    
    @staticmethod
    def check(name):
        for gesture in GESTURE:
            if name.upper() == gesture.name:
                return True
        return False

    def get_dir(self):
        return os.path.join(os.path.dirname(__file__), 'training_data', self.name.lower())
    
    def get_test_dir(self):
        return os.path.join(os.path.dirname(__file__), 'test_data', self.name.lower())

class GESTURE2(Enum):
    NAC = 0
    CHARACTER = auto()
    
    @staticmethod
    def check(name):
        for gesture in GESTURE2:
            if name.upper() == gesture.name:
                return True
        return False

    def get_dir(self):
        return os.path.join(os.path.dirname(__file__), 'training_data', self.name.lower())
    
    def get_test_dir(self):
        return os.path.join(os.path.dirname(__file__), 'test_data', self.name.lower())
