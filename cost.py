# Copyright 2021 Faranak Shamsafar
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Requirements:
    pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
    pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
    pip install onnx
"""

from __future__ import print_function, division
from ptflops import get_model_complexity_info
import torch.nn.parallel
import torch.utils.data
from models import __models__
from utils import *
from thop import profile

C = 3
H = 256
W = 512
# F = 40

"""
For input size: (3, 256, 512):

    Feature size: (320, 64, 128)

    [2D model] Feature size after channel reduction: (32, 64, 128)

    [2D model] Cost size: (48, 64, 128)

    [3D model] Cost size: (40, 48, 64, 128)
        For this, change (1, C, H, W) to (1, F, C, H, W).
"""


def input_constructor(input_shape):
    # For Flops-Counter method
    # Notice the input naming
    inputs = {'L': torch.ones(input_shape), 'R': torch.ones(input_shape)}
    return inputs


with torch.cuda.device(0):

    ################# Using Flops-Counter #################

    # net = __models__['MSNet2D'](192)
    # macs2D, params2D = get_model_complexity_info(net, (1, C, H, W), as_strings=True,
    #                                              print_per_layer_stat=False, verbose=False,
    #                                              input_constructor=input_constructor)
    #
    # net = __models__['MSNet3D'](192)
    # macs3D, params3D = get_model_complexity_info(net, (1, C, H, W), as_strings=True,
    #                                              print_per_layer_stat=False, verbose=False,
    #                                              input_constructor=input_constructor)
    #
    # print("==========================\n", '2D-MobileStereoNet', "\n==========================")
    # print('{:<30}  {:<8}'.format('Number of operations: ', macs2D))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params2D))
    #
    # print("==========================\n", '3D-MobileStereoNet', "\n==========================")
    # print('{:<30}  {:<8}'.format('Number of operations: ', macs3D))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params3D))

    ################# Using THOP (OpCounter) #################

    L = torch.randn(1, C, H, W)
    R = L

    macs2D, params2D = profile(__models__['MSNet2D'](192), inputs=(L, R))
    macs3D, params3D = profile(__models__['MSNet3D'](192), inputs=(L, R))

    print("==========================\n", '2D-MobileStereoNet', "\n==========================")
    print('{:<30}  {:<8}'.format('Number of operations: ', np.round(macs2D / 1000000000), 5))
    print('{:<30}  {:<8}'.format('Number of parameters: ', np.round(params2D / 1000000, 5)))

    print("==========================\n", '3D-MobileStereoNet', "\n==========================")
    print('{:<30}  {:<8}'.format('Number of operations: ', np.round(macs3D / 1000000000), 2))
    print('{:<30}  {:<8}'.format('Number of parameters: ', np.round(params3D / 1000000, 2)))
