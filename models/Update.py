#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os

import torch
import torchvision
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.test import test_img
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs,noise_rate):
        self.noise_rate=noise_rate
        self.dataset = dataset
        self.idxs = list(idxs)
        self.imageList=[]
        self.LabelList = []

        for i in range(len(self.idxs)):
            image, label, _ = self.dataset[self.idxs[i]]
            self.imageList.append(image)
            self.LabelList.append(label)
        for idx in range(len(self.LabelList)):
            # print(len(self.targets))
            if random.random() < self.noise_rate:
                a=random.randint(0, 9)
                if a!=self.LabelList[idx]:
                    self.LabelList[idx] =a

                    # image, label, _ = self.dataset[self.idxs[0]]
        # print(image, label, _)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        # image,label,_= self.dataset[self.idxs[index]]
        image=self.imageList[index]
        label = self.LabelList[index]
        # print(label)
        return image, label, index

class DatasetSplit2(Dataset):
    def __init__(self, dataset, idxs,noise_rate):
        self.noise_rate=noise_rate
        self.dataset = dataset
        self.idxs = list(idxs)
        self.imageList=[]
        self.LabelList = []

        for i in range(len(self.idxs)):
            image, label, _ = self.dataset[self.idxs[i]]
            self.imageList.append(image)
            self.LabelList.append(label)
        for idx in range(len(self.LabelList)):
            # print(len(self.targets))
            if random.random() < self.noise_rate:
                self.LabelList[idx] = 1


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        # image,label,_= self.dataset[self.idxs[index]]
        image=self.imageList[index]
        label = self.LabelList[index]
        # print(label)
        return image, label, index


class DatasetSplit1(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image,label= self.dataset[self.idxs[index]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,iter=iter,idx=None):
        self.userId=idx
        self.args = args
        self.epoch=iter
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if self.userId==0:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs,noise_rate=), batch_size=self.args.local_bs, shuffle=True)

        if self.userId==1:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs,noise_rate=), batch_size=self.args.local_bs, shuffle=True)

        if self.userId==2:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs,noise_rate=), batch_size=self.args.local_bs, shuffle=True)

        if self.userId==3:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs,noise_rate=), batch_size=self.args.local_bs, shuffle=True)

        if self.userId==4:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs,noise_rate=), batch_size=self.args.local_bs, shuffle=True)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate"""
        if epoch < args.stage2:
            lr = args.lr
        elif epoch < (args.epochs - args.stage2) // 3 + args.stage2:
            lr = args.lr2
        elif epoch < 2 * (args.epochs - args.stage2) // 3 + args.stage2:
            lr = args.lr2 // 10
        else:
            lr = args.lr2 // 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def test(self,dataset,net,idxs=None,idx=None):
        self.ldr_test = DataLoader(DatasetSplit1(dataset, idxs), batch_size=200, shuffle=False)
        net.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (images, labels1) in enumerate(self.ldr_test):
            images = images.to(args.device)
            labels = labels1.to(args.device)
            # loss = self.loss_func(images,labels)
            output = net(images)
            test_loss += self.loss_func(output, labels).item()*100
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        accuracy = 100.00 * correct / len(self.ldr_test.dataset)
        test_loss /= len(self.ldr_test.dataset)
        # print(len(self.ldr_test.dataset),idx)
        return accuracy, output,test_loss

    def test_output(self,dataset,net,idxs=None,idx=None):
        self.ldr_test = DataLoader(DatasetSplit1(dataset, idxs), batch_size=200, shuffle=False)
        net.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (images, labels1) in enumerate(self.ldr_test):
            images = images.to(args.device)
            labels = labels1.to(args.device)
            # loss = self.loss_func(images,labels)
            output = net(images)
            test_loss += self.loss_func(output, labels).item()*100
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        accuracy = 100.00 * correct / len(self.ldr_test.dataset)
        test_loss /= len(self.ldr_test.dataset)
        # print(len(self.ldr_test.dataset),idx)
        return  output

    def train(self, net):

        if self.userId==0:
            y_file0 = args.dir + "y%03d.npy"% self.userId
            if os.path.isfile(y_file0):
                y = np.load(y_file0)
            else:
                y = []
        if self.userId==1:
            y_file1 = args.dir + "y%03d.npy"% self.userId
            if os.path.isfile(y_file1):
                y = np.load(y_file1)
            else:
                y = []
        if self.userId==2:
            y_file2 = args.dir + "y%03d.npy"% self.userId
            if os.path.isfile(y_file2):
                y = np.load(y_file2)
            else:
                y = []
        if self.userId==3:
            y_file3 = args.dir + "y%03d.npy"% self.userId
            if os.path.isfile(y_file3):
                y = np.load(y_file3)
            else:
                y = []
        if self.userId==4:
            y_file4 = args.dir + "y%03d.npy"% self.userId
            if os.path.isfile(y_file4):
                y = np.load(y_file4)
            else:
                y = []
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=args.weight_decay)
        self.adjust_learning_rate(optimizer, self.epoch)
        epoch_loss = []
        onehot_copy = np.zeros([args.datanum, args.num_classes])
        for iter1 in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels1,index) in enumerate(self.ldr_train):
                images = images.to(args.device)
                labels = labels1.to(args.device)
                index = index.numpy()
                net.zero_grad()
                output = net(images)
                logsoftmax = nn.LogSoftmax(dim=1).to(args.device)
                softmax = nn.Softmax(dim=1).to(args.device)
                if self.epoch < args.stage1:
                    loss = self.loss_func(output, labels)
                    onehot = torch.zeros(labels1.size(0), 10).scatter_(1, labels1.view(-1, 1), 1.0)
                    onehot = onehot.numpy()
                    onehot_copy[index, :] = onehot
                else:
                    update_y = y
                    update_y = update_y[index, :]
                    update_y = torch.FloatTensor(update_y)
                    update_y = update_y.to(args.device)
                    update_y = torch.autograd.Variable(update_y, requires_grad=True)
                    last_y_var = softmax(update_y)
                    # p1+p2
                    P3 = torch.log(0.5 * (softmax(output) + last_y_var)) - torch.log((last_y_var))
                    kl1 = torch.mean(softmax(output) * P3)
                    P4 = torch.log(0.5 * (softmax(output) + last_y_var)) - logsoftmax(output)
                    kl2 = torch.mean(last_y_var * P4)
                    lc = kl1 + kl2
                    lo = self.loss_func(last_y_var, labels)
                le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))
                if self.epoch< args.stage1:
                    loss = loss
                elif self.epoch < args.stage2:
                    loss = lc + args.alpha * lo + args.beta * le
                else:
                    loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()
                if self.epoch >= args.stage1 and self.epoch < args.stage2:
                    lambda1 = args.lambda1
                    # update y_tilde by back-propagation
                    update_y.data.sub_(lambda1 * update_y.grad.data)
                    onehot_copy[index, :] = update_y.data.cpu().numpy()
                # if batch_idx % 10 == 0:
                if   self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter1, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            if self.epoch < args.stage2:
                # save y_tilde
                if self.userId==0:
                    y = onehot_copy
                    y_file0 = args.dir + "y%03d.npy"% self.userId
                    np.save(y_file0, y)
                    y_record = args.dirrecord +"/" +"userId0/y_%03d.npy" % self.epoch
                    np.save(y_record, y)
                if self.userId==1:
                    y = onehot_copy
                    y_file1 = args.dir + "y%03d.npy"% self.userId
                    np.save(y_file1, y)
                    y_record = args.dirrecord +"/"+ "userId1/y_%03d.npy" % self.epoch
                    np.save(y_record, y)
                if self.userId==2:
                    y = onehot_copy
                    y_file2 = args.dir + "y%03d.npy"% self.userId
                    np.save(y_file2, y)
                    y_record = args.dirrecord +"/"+ "userId2/y_%03d.npy" % self.epoch
                    np.save(y_record, y)
                if self.userId==3:
                    y = onehot_copy
                    y_file3 = args.dir + "y%03d.npy"% self.userId
                    np.save(y_file3, y)
                    y_record = args.dirrecord +"/"+ "userId3/y_%03d.npy" % self.epoch
                    np.save(y_record, y)
                if self.userId==4:
                    y = onehot_copy
                    y_file4 = args.dir + "y%03d.npy"% self.userId
                    np.save(y_file4, y)
                    y_record = args.dirrecord +"/"+ "userId4/y_%03d.npy" % self.epoch
                    np.save(y_record, y)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),net



