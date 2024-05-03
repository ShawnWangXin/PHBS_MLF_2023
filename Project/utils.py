import torch
import pandas as pd
import numpy as np

import torch.optim as optim
import math


def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)

def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)

def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


def cal_convariance(x, y): # the 2nd dimension of x and y are the same
    e_x = torch.mean(x, dim = 1).reshape(-1, 1)
    e_y = torch.mean(y, dim = 1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend*y_extend, dim = 2)
    return e_xy - e_x_e_y


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))  # 降序排列，score是预测列，label是真实列
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:  # 
        precision[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()
    # 精确度是指在前k个最高得分的预测中，真正为正的样本所占的比例
    # 召回率是指在所有真正为正的样本中，被预测为正的样本所占的比例
    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()  # 这个用的是pandas库的corr，是可以有nan的
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_result(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)
    # 创建并排的子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图上绘制 rank_ic 数据
    axs[0].plot(df['train_rank_ic'], label='Train')
    axs[0].plot(df['valid_rank_ic'], label='Valid')
    axs[0].plot(df['test_rank_ic'], label='Test')
    axs[0].set_title('Rank IC')
    axs[0].legend()

    # 在第二个子图上绘制 loss 数据
    axs[1].plot(df['train_loss'], label='Train')
    axs[1].plot(df['valid_loss'], label='Valid')
    axs[1].plot(df['test_loss'], label='Test')
    axs[1].set_title('Loss')
    axs[1].legend()

    # 显示图
    plt.show()
