# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.core import setup
import setuptools
import os

long_desc = """
A python library for building autoencoders with tabular data.
Currently in development.
"""

reqs= [
    'torch',
    'numpy',
    'pandas>=1.0,<1.4.0dev0',
    'tqdm',
    'scikit-learn==0.23.1',
    'tensorboardX',
    'matplotlib', 
    'wheel',
    'dill'
]
version = '0.0.37'

setup(
    name='dfencoder',
    version=f'{version}',
    description='Autoencoder Library for Tabular Data',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael.r.klear@gmail.com',
    url='https://github.com/alliedtoasters/dfencoder',
    download_url=f'https://github.com/alliedtoasters/dfencoder/archive/v{version}.tar.gz',
    install_requires=reqs,
    setup_requires=reqs,
    packages=['dfencoder']
)
