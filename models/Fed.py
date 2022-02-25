#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from sympy import *
import math
import sympy
import copy
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.options import args_parser
from scipy import stats

args = args_parser()

dict3 = {}
dict_no_norm = {}

# 计算积分面积
def pdf(x):
  return math.exp(-(x) ** 2 / (2)) / (math.sqrt(2 * math.pi))

def sum_fun_xk(xk, func):
  return sum([func(each) for each in xk])

def integral(a, b, n, func):
  h = (b - a)/float(n)
  xk = [a + i*h for i in range(1, n)]
  return h/2 * (func(a) + 2 * sum_fun_xk(xk, func) + func(b))

def cdfd(a,b,u,o):
  return integral((a-u)/o,(b-u)/o,10000,pdf)

# 计算曲线交点
x=symbols('x')
# 传入均值 标准差 方差
def sl_a(u1, sig1,var1, u2, sig2,var2):

    a=1/var2-1/var1
    b=2*(u1/var1-u2/var2)
    c=u2*u2/var2-u1*u1/var1+2*sympy.log(sig2/sig1)
    y=a*x**2+b*x+c
    return sympy.sympify(y)

def output_distribution(output_locals,idx,iter):
    global dict3
    global dict_no_norm
    data={}
    data1={}
    output={}
    for key in output_locals.keys():
        base1 = [*np.array(output_locals[key].data.cpu().numpy()).flat]
        output[key]=base1
    np.save("11.npy",output)
    _dict = np.load("11.npy", allow_pickle=True)
    dict2 = _dict.item()
    if iter==0:
        dict4 = {}
        dict_no_norm1 = {}
        for key in dict2.keys():
            a_mean = np.mean(dict2[key])
            arr_std = np.std(dict2[key])
            c = stats.kstest(dict2[key], 'norm', (a_mean, arr_std))
            if c[1] > 1E-10:
                dict4[key] = c[1]

            else:
                dict_no_norm1[key] = key

        dict3 = copy.deepcopy(dict4)
        dict_no_norm = copy.deepcopy(dict_no_norm1)
    print(dict3)
    print(dict_no_norm)
    list = [i for i in range(5)]
    if idx in dict3:
        base =[*np.array(output_locals[idx].data.cpu().numpy()).flat]
        base_var = np.var(base)
        base_mean = np.mean(base)
        base_std = np.std(base)
        base_cdfd = cdfd(-1, 1, base_mean, base_std)
        for i in list:
            if i==idx:
                continue
            other_client = [*np.array(output_locals[i].data.cpu().numpy()).flat]
            other_client_var = np.var(other_client)
            other_client_mean = np.mean(other_client)
            other_client_std = np.std(other_client)
            c = sympy.solve( sl_a(base_mean,base_std,base_var,other_client_mean,other_client_std,other_client_var), x)
            if base_std>other_client_std:
                a1 = cdfd(-1, c[0], other_client_mean, other_client_std)
                a2 = cdfd(c[0], c[1], base_mean, base_std)
                a3 = cdfd(c[1], 1, other_client_mean, other_client_std)
                s_chongdie=a1+a2+a3
            else:
                a1 = cdfd(-1, c[0], base_mean, base_std)
                a2 = cdfd(c[0], c[1], other_client_mean, other_client_std)
                a3 = cdfd(c[1], 1, base_mean, base_std)
                s_chongdie = a1 + a2 + a3
            # 存储每个客户端与当前客户端重叠面积占比
            if s_chongdie/base_cdfd>0.85:
                data[i] = s_chongdie / base_cdfd
                # data[i]=i
            else:
                data1[i]=i

    return data,data1,dict3,dict_no_norm

def client_agg(w,data,idx,data1,dict3,dict_no_norm):

    if len(data.keys())>=2:
        w_avg = copy.deepcopy(w[idx])
        # w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(0, len(w)):
                for key in data:
                    if i==key:
                        w_avg[k]+=w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(data)+1)

    elif dict_no_norm:
        # a = []
        # for i in range(5):
        #     if i not in dict_no_norm.keys():
        #         a.append(i)
        a = list(dict3.keys())
        w_avg = copy.deepcopy(w[a[0]])
        # w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(0, len(w)):
                if i == a[0]:
                    continue
                for key in dict3:
                    if i == key:
                        w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(a))
    else:
        a=list(data1.keys())
        w_avg = copy.deepcopy(w[a[0]])
        # w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(0, len(w)):
                if i == a[0]:
                    continue
                for key in data1:
                    if i == key:
                        w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(data1))
    return w_avg

def client_wight_agg(w,data,idx):
    print(data)
    b = []
    for key in data:
        b.append(data[key])
    b = np.array(b,dtype=np.float64)
    d=softmax(b)
    c=[]
    for i in range(len(data)+1):
        if i == idx:
            c.append(0)
        else:
            c.append(d[i - 1])
    a = torch.tensor(c)
    print(a)
    w_avg = copy.deepcopy(w[idx])
    for k in w_avg.keys():
        # w_avg[k] = w_avg[k] * a[0]
        for i in range(0, len(w)):
            for key in data:
                if i==key:
                    w_avg[k] += w[i][k]*a[key]
        w_avg[k] = torch.div(w_avg[k], 2)
    return w_avg


def softmax(x):
    # 计算每行的最大值
    row_max = max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = sum(x_exp)
    s = x_exp / x_sum
    return s



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg




