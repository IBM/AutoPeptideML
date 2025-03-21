#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /root/CAMP/std_logger.py
# Project: /home/richard/projects/CAMP
# Created Date: Saturday, July 30th 2022, 3:51:24 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Aug 28 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 Silexon Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import logging
import sys
import os

class StdLogger():
    def __init__(self, file_path = "", stream = False, level=logging.INFO):
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                      datefmt="%H:%M:%S")
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.logger.setLevel(level)
        self.file_path = file_path

        if file_path:
            file_hander = logging.FileHandler(file_path)
            file_hander.setFormatter(formatter)
            self.logger.addHandler(file_hander)

        if stream:
            stream_handler = logging.StreamHandler(stream=sys.stderr)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

# cfg = OmegaConf.load(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_conf.yaml'))
# log_dir = os.path.join(cfg.logger.log_dir, str(int(time())))
# os.makedirs(log_dir)
# file_path = os.path.join(log_dir, "train.log")

Logger = StdLogger("", level=logging.INFO).logger