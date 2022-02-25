import os
import sys
import pickle
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import copy

from models.Update import LocalUpdate
from utils.options import args_parser

from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, client_agg, \
    output_distribution, client_wight_agg

from models.test import test_img, test_img1
import matplotlib
matplotlib.use('Agg')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10_sym(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, noise_rate=0.2):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.ids = list(range(len(self.targets)))

        self._load_meta()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, idx = self.data[index], self.targets[index], self.ids[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # if self.train == True:
        #     return img, target, idx
        # else:
        #     return img, target
        return img, target, idx


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_test(dataset, num_users):
    """
    Sample I.I.D. client test data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = 200
    dict_users_test, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users_test[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users_test[i])
    return dict_users_test


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    noise_rate = 0.0

    # trainloader, testloader = load_data_sym(noise_rate=noise_rate)
    print(args.device)
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    root = './data/cafir'

    trainset = CIFAR10_sym(root, train=True, download=False, transform=transform_train, noise_rate=noise_rate)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=0)
    #测试数据，上传分布
    dict_users_test = cifar_test(testloader, args.num_users)
    #独立同分布数据，每个user持有数据
    dict_users = cifar_iid(trainloader, args.num_users)
    net_glob = CNNCifar(args='cifar').to(args.device)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    w_locals=[]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(5)]
    client_net = {}
    output_locals = {}
    Loss = {}
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        for idx in range(5):
            local = LocalUpdate(args=args, dataset=trainset, idxs=dict_users[idx],iter=iter,idx=idx)
            w, loss, net_local = local.train(net=copy.deepcopy(net_glob).to(args.device))
            client_net[idx] = net_local
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))


        #p2p计算度量
        for idx in range(5):
            test_local, output0, test_loss = local.test(dataset=testset, idxs=dict_users_test[idx], net=client_net[idx], idx=idx)
            output_locals[idx] = output0
            for i in range(5):
                if idx==i:
                    continue
                other_client_output = local.test_output(dataset=testset, idxs=dict_users_test[idx], net=client_net[i],idx=i)
                output_locals[i] = other_client_output
            torch.set_printoptions(edgeitems=768)
            # print(output_locals)
            # 选择性聚合权重参数i
            acc_test1, loss_test = test_img(client_net[idx], testset, args)
            print("总体数据聚合前 LocalTesting accuracy: {:.2f},UserId:{:.0f}".format(acc_test1, idx))
            data,data1,dict3,dict_no_norm = output_distribution(output_locals, idx,iter)
            w_client = client_agg(w_locals,data,idx,data1,dict3,dict_no_norm)

            net_glob.load_state_dict(w_client)
            acc_test, loss_test = test_img(net_glob, testset, args)

            print("总体数据聚合后 Testing accuracy: {:.2f}".format(acc_test))
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)





