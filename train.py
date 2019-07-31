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

__author__ = "Deepta Rajan, Jayaraman J. Thiagarajan"

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import os
import argparse
import datetime
from utils.data_loader import get_loader
from utils.log import ResultsLog
from model.ddxnet_model import DDxNet, DDx_block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using ', device)

def main(args):
    args.model_path = os.path.join(args.model_path, str(datetime.date.today()))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    # Build DDxNet with 4 DDx blocks of convolutions
    model = DDxNet(args.num_channels, args.num_timesteps, DDx_block, [2,6,8,4],
                   args.output_dim, causal=True, use_dilation=True).to(device)

    # multi-gpu training if available
    if(torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1**0.75, verbose=True)

    train_loader = get_loader(args.data_dir, 'train', args.batch_size, args.shuffle)
    test_loader = get_loader(args.data_dir, 'test', args.batch_size, shuffle=False)
    best_val_acc = 0.

    if not os.path.exists(os.path.join('./logs', str(datetime.date.today()))):
        os.makedirs(os.path.join('./logs', str(datetime.date.today())))

    results_file = os.path.join('./logs', str(datetime.date.today()), args.results_file)
    results = ResultsLog(results_file)

    for epoch in range(args.num_epochs):
        avg_loss = 0.
        total_predlabs = []
        total_truelabs = []
        total_probs =[]

        for itr, (X, y_true) in enumerate(train_loader):
            model.train()
            X = X.to(device).float()
            y_true = y_true.to(device).long()

            y_pred = model(X)
            loss = loss_fn(y_pred, torch.max(y_true,1)[1])
            avg_loss += loss.item()/len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            probs = softmax(y_pred)
            _, predlabs = torch.max(probs.data, 1)
            total_probs.extend(probs.data.cpu().numpy())
            total_predlabs.extend(predlabs.data.cpu().numpy())
            total_truelabs.extend(torch.max(y_true,1)[1].data.cpu().numpy())
            batch_acc = accuracy_score(torch.max(y_true,1)[1].data.cpu().numpy(), predlabs.data.cpu().numpy())

            if (itr+1) % 50 == 0:
                print(('Epoch: {} Iter: {}/{} Loss: {} Acc: {}').format(epoch,itr+1,len(train_loader), loss.item(), batch_acc))

            if((itr+1) % len(train_loader)) == 0:
                total_truelabs = np.array(total_truelabs)
                total_predlabs = np.array(total_predlabs)
                total_probs = np.array(total_probs)
                total_train_acc = accuracy_score(total_truelabs, total_predlabs)

                f1 = f1_score(total_truelabs, total_predlabs, average='macro')
                res = {'epoch': epoch + (itr*1.0+1.0)/len(train_loader),
                   'steps': epoch*len(train_loader) + itr+1,
                   'train_loss': avg_loss,
                   'train_f1': f1,
                   'train_acc': total_train_acc}

                model.eval()

                with torch.no_grad():
                    total_predlabs = []
                    total_probs = []
                    total_truelabs = []
                    total_val_loss = 0.

                    for i, (dat, labs) in enumerate(test_loader):
                        dat = dat.to(device).float()
                        labs =  labs.to(device).long()
                        y_pred = model(dat)
                        val_loss = loss_fn(y_pred, torch.max(labs,1)[1])

                        probs = softmax(y_pred)
                        _, predlabs = torch.max(probs.data, 1)
                        total_probs.extend(probs.data.cpu().numpy())
                        total_predlabs.extend(predlabs.data.cpu().numpy())
                        total_truelabs.extend(torch.max(labs,1)[1].data.cpu().numpy())
                        total_val_loss += (val_loss.item()/len(test_loader))

                    total_truelabs = np.array(total_truelabs)
                    total_predlabs = np.array(total_predlabs)
                    total_probs = np.array(total_probs)
                    total_val_acc = accuracy_score(total_truelabs, total_predlabs)

                    total_val_f1 = f1_score(total_truelabs, total_predlabs, average='macro')
                    print("At Epoch: {}, Iter: {}, val_loss: {}, val_acc: {}".format(epoch, itr+1, total_val_loss, total_val_acc))
                    print("Confusion Matrix: ")
                    print(confusion_matrix(total_truelabs, total_predlabs))
                    if(total_val_acc > best_val_acc):
                        best_val_acc = total_val_acc
                        print("saving model")
                        torch.save(model.state_dict(), os.path.join(args.model_path, args.results_file+ '_model.pth'))
                        np.savetxt(os.path.join(args.model_path, args.results_file+ '_prob.txt'),total_probs,delimiter=',')
                        np.savetxt(os.path.join(args.model_path, args.results_file+ '_pred.txt'),total_predlabs,delimiter=',')
                        np.savetxt(os.path.join(args.model_path, args.results_file+ '_true.txt'),total_truelabs,delimiter=',')

                    res['val_loss'] = total_val_loss
                    res['val_acc'] = total_val_acc
                    res['val_f1'] = total_val_f1

                    plot_loss = ['train_loss']
                    plot_acc = ['train_acc']
                    plot_f1 = ['train_f1']

                    plot_loss += ['val_loss']
                    plot_acc += ['val_acc']
                    plot_f1 += ['val_f1']

                    results.add(**res)
                    results.plot(x='epoch', y=plot_loss,
                          title='Multi-Class Loss', ylabel='CE Loss')
                    results.plot(x='epoch', y=plot_acc,
                          title='Accuracy', ylabel='Accuracy')
                    results.plot(x='epoch', y=plot_f1,
                          title='F1-Score (Macro)', ylabel='F1-Score')
                    results.save()

        scheduler.step(total_val_loss, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default = 'results_ecg', help = 'Filename for plot results')
    parser.add_argument('--model_path', type=str, default = './ckpts',
                        help = 'Path for storing   model checkpoints')
    parser.add_argument('--data_dir', type=str, default = './data', help = 'Directory of pre-processed timeseries data and labels')
    parser.add_argument('--batch_size', type=int, default = 128,
                        help = 'batch size')
    parser.add_argument('--shuffle', type=bool, default = True,
                        help = 'shuffle')
    parser.add_argument('--learning_rate', type=float, default = 1e-4,
                        help = 'learning rate')
    parser.add_argument('--clip_grad', type=float, default = -1,
                        help = 'gradient clipping (-1 means no clip)')
    parser.add_argument('--num_epochs', type=int, default = 1,#25,
                        help = 'number of epochs')
    parser.add_argument('--num_channels', type=int, default = 1,
                        help = 'feature dimension')
    parser.add_argument('--num_timesteps', type=int, default = 187,
                        help = 'number of timesteps in every sequence sample')
    parser.add_argument('--output_dim', type=int, default = 5,
                        help = 'dimension of output')


    main(parser.parse_args())
