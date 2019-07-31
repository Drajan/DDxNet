#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 IBM Corporation and others
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Deepta Rajan"

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder


class Heartbeat_Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.X = np.loadtxt(os.path.join(self.data_dir, 'mitbih_'+self.mode+'.csv'), delimiter=',')
        self.labels = np.asarray([int(self.X[i][-1]) for i in range(self.X.shape[0])])
        self.X = np.expand_dims(self.X[:,0:-1], 1) # last column is labels
        enc = OneHotEncoder(sparse=False, categories='auto')
        self.labels_onehot = enc.fit_transform(np.expand_dims(self.labels,1))
        print(self.mode + " dataset: " + "X.shape: ", self.X.shape, " Y.shape", self.labels_onehot.shape)


    def __getitem__(self, index):
        ecg = self.X[index]
        label = self.labels_onehot[index]
        label = torch.from_numpy(label)
        ecg = torch.from_numpy(ecg)

        return ecg, label


    def __len__(self):
        return len(self.labels)



def get_loader(data_dir, mode, batch_size, shuffle):
    heartbeat = Heartbeat_Dataset(data_dir, mode)
    data_loader = DataLoader(dataset=heartbeat, batch_size=batch_size, shuffle=shuffle)

    return data_loader
