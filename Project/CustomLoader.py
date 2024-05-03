
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import time

class CustomLoader(object):
    def __init__(self, dataset, batch_size, shuffle, flag):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flag = flag
        if flag == 'train':
            idx_list = np.arange(len(self.dataset))[::10]
            if shuffle == True:
                self.idx_list = np.random.permutation(idx_list)
            else:
                self.idx_list = idx_list

        else:
            if shuffle == False:
                self.idx_list = np.arange(len(self.dataset))
            else:
                self.idx_list = np.random.permutation(np.arange(len(self.dataset)))

        self.index = 0
    # 必须要返回一个实现了 __next__ 方法的对象，否则后面无法 for 遍历
    # 因为本类自身实现了 __next__，所以通常都是返回 self 对象即可
    def __iter__(self):
        # 一定要记得重置迭代器
        self.index = 0
        return self
    

    def __next__(self):
        # print(self.dayindex)
        if self.index <= len(self.idx_list) - 1:
            result = self.dataset[self.idx_list[self.index]]
            self.index += 1
            return result
        else:
            # 抛异常，for 内部会自动捕获，表示迭代完成
            raise StopIteration("遍历完了")
        
    def __len__(self):  # 为了进度条
        return len(self.idx_list)
