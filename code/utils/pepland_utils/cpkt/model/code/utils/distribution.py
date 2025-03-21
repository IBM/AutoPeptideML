#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /root/CAMP/distribution.py
# Project: /home/richard/projects/pepland/inference/cpkt/model/code/utils
# Created Date: Saturday, July 30th 2022, 3:50:55 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sat Dec 23 2023
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 HILab Ltd
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

import os
import torch.distributed as dist
import torch
# from .std_logger import Logger


def setup_multinodes(local_rank, world_size):

    torch.cuda.set_device(local_rank)
    # initialize the process group
    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]

    # Logger.info("init_address: tcp://{}:{} | world_size: {}".format(
    #     ip, port, world_size))

    dist.init_process_group("nccl",
                            init_method='tcp://{}:{}'.format(ip, port),
                            rank=int(os.environ['RANK']),
                            world_size=world_size)

    # torch.multiprocessing.set_sharing_strategy("file_system")

    # Logger.info(
    #     f"[init] == local rank: {local_rank}, global rank: {os.environ['RANK']} =="
    # )


def cleanup_multinodes():
    dist.destroy_process_group()