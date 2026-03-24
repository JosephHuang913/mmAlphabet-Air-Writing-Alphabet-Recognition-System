# -*- coding: utf-8 -*-
#==========================================================================#
# Author: Joseph Huang                                                     #
# E-mail: huangcw913@gmail.com                                             #
# Date: Tuesday, August 10, 2021                                           #
# Description: Check if required folders and files exist                   #
# Copyright 2021. All Rights Reserved.                                     #
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


def folder_check(configs):

    if not os.path.exists(configs.log_dir):
        print('log directory not found, creating...')
        os.mkdir(configs.log_dir)
    else:
        print('log directory found.')
