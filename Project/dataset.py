import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import h5py
import bisect
import pickle
import sys
import os


# tradingdates = pickle.load(open('/home/featurize/data/tradingdates.pkl', 'rb'))
file = h5py.File('./data/tradingdates.h5', 'r')
data = file['dataset'][:]
tradingdates = list(data)
file.close()
class Dataset:
    def __init__(self, start, end, model, matrix_dir,flag):
        self.start = start
        self.end = end
        self.flag = flag
        self.model = model # GCN和HAT需要的是邻接矩阵，HIST需要的是股票概念分类，MTGNN什么都不要
        self.tradedate = [i.decode('utf-8') for i in tradingdates]
        self.start_idx = bisect.bisect_left(self.tradedate, start)
        # self.end_idx = tradingdates['tradingdate'].tolist().index(end)
        self.end_idx = bisect.bisect_left(self.tradedate, end) 
        self.matrix_dir = matrix_dir
        if self.tradedate[self.end_idx] != end:
            self.end_idx -= 1
        self.filename = self.tradedate[self.start_idx-35:self.end_idx+1] # 保险起见，本来是-29
        self.data, self.label, self.stockcode = self.load()
        if self.model == 'GAT' or self.model == 'GCN' or self.model == 'GRU':
            self.adj_data, self.adj_date = self.adj_matrix()
        elif self.model == 'HIST':
            self.concept_data, self.concept_stockcode, self.concept_date = self.stock_concept()
        self.mktcp = self.market_value()

    def load(self):
        Data= []
        label = []
        stockcode = []
        idx = self.start_idx - 35
        
        for file in tqdm(self.filename) if sys.stdout.isatty() else self.filename:
            data = pd.read_hdf(f'./data/data/{file}.h5')
            data = data.reset_index(level=0, drop=True) # stockcode作为索引
            col = [int(i) for i in range(1001,1062)]
            if len(data.columns)!=61:
                data = data.reindex(columns=col, fill_value=0)
            Data.append(data)
            stocks = data.index.unique()
            stockcode.append(stocks)
            close = pd.read_hdf(f'./data/close/{file}.h5')
            close10 = pd.read_hdf(f"./data/close/{self.tradedate[idx+10]}.h5")
            close = close.reindex(stocks)
            close10 = close10.reindex(stocks)
            rr = (close10['adjclose'] - close['adjclose']) / close['adjclose']
            label.append(rr)
            idx += 1
        return Data, label, stockcode

    def all_stock(self,start,end):
        all_stock = set()
        start_idx = bisect.bisect_left(self.tradedate, start)
        end_idx = bisect.bisect_left(self.tradedate, end) - 1
        filename = self.tradedate[start_idx-29-6:end_idx+1]
        for file in filename:
            data = pd.read_hdf(f'./data/data/{file}.h5')
            data = data.reset_index(level=0, drop=True) # stockcode作为索引
            stocks = data.index.unique()
            all_stock = all_stock.union(stocks)
        self.all_stock_list = sorted(list(all_stock))
        return self.all_stock_list
    
    def market_value(self):
        Data= []
        
        dir_path = r"./data/mktcap"
        for file in tqdm(self.filename):
            file_path = os.path.join(dir_path, f'{file}.h5')
            h5_file = pd.read_hdf(file_path, key='df')
            Data.append(h5_file['mktcap'])
        return Data
    
    def adj_matrix(self): # 存邻接矩阵
        Data= []
        date = []

        dir_path = self.matrix_dir
        files = os.listdir(dir_path)
        for file in tqdm(files) if sys.stdout.isatty() else files :
            file_path = os.path.join(dir_path, file)
            data = pd.read_parquet(file_path)
            Data.append(data)
            date.append(file.split('.')[0])
        return Data, date

    def stock_concept(self):  # 存股票概念分类
        Data= []
        stockcode = []
        date = [] 
        
        dir_path = r"./data/preclass"
        files = os.listdir(dir_path)
        for file in tqdm(files) if sys.stdout.isatty() else files :
            file_path = os.path.join(dir_path, file)
            h5_file = h5py.File(file_path, 'r')
            ll = [i.decode('utf-8') for i in list(h5_file['stocks'][:])]
            stockcode.append([i.split('.')[0] for i in ll])
            Data.append(h5_file['classes'][:])
            date.append(file.split('.')[0])
            h5_file.close()
        return Data, stockcode, date

    def __len__(self):
        return len(self.data)-29-6
    

    def __getitem__(self, idx):   
        # 根据日期选择返回的邻接矩阵或者concept矩阵或者None,记得对齐股票代码
        # 你要用当前数据的上一个日期的邻接矩阵
        date = self.filename[idx+29+6]  

        data_file = f'./data/{self.flag}/{self.start}_{self.end}/{date}.h5'
        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as hf:
                Data, label, plus_data, mktcp_data, index_bytes = hf['Data'][:], hf['label'][:], hf['plus_data'][:], hf['mktcp_data'][:], hf['index'][:]    
                index = [i.decode('utf-8') for i in index_bytes]
   
        else:  
            stocks = self.data[idx+29+6].index.unique()
            date = self.filename[idx+29+6]  
            # 获取market_value的数据
            mktcp_data = self.mktcp[idx+29+6]
            Data = []
            count = 0
            i = idx+29+6 # +29是因为要有30天数据，+6是因为本来是从self.start_idx-29开始的，但是因为担心数据缺失，所以多保存了点数据，现在是-35
            while count < 30:
                data = self.data[i]
                data = data.reindex(self.all_stock_list)
                data_arr = data.values
                # if sum(np.any(np.isnan(data.values), axis=1)) == len(data): # 因为20231018这一天缺了因子数据，所以要跳过
                #     data_arr[:,0] = 0
                count += 1
                i -= 1
                Data.append(data_arr)
            # 再反过来
            Data = Data[::-1]
            Data = np.stack(Data, axis=-1) # (stocks, factors, days)
            label = self.label[idx+29+6]
            label = label.reindex(self.all_stock_list)
            label = label.values

            mask1 = np.isnan(Data).any(axis=(1, 2))  # 这里Data和label的行数一定相同，因为load函数里面已经对齐了
            mask3 = np.isinf(Data).any(axis=(1, 2))
            mask2 = np.isnan(label.reshape(-1, 1)).any(axis=1)
            mask = mask1 | mask2 | mask3
             # 二维的 不需要去掉NAN，因为在计算损失的时候会自动去掉

            if self.model == 'GAT' or self.model == 'GCN' or self.model == 'GRU':
                idx1 = bisect.bisect_left(self.adj_date, date) - 1 # 如果刚好是同一天也最好用前一个矩阵的
                plus_data = self.adj_data[idx1]
                plus_data.index = [i[:6] for i in plus_data.index]
                plus_data.columns = [i[:6] for i in plus_data.columns]
                plus_data = plus_data.reindex(index = self.all_stock_list, columns = self.all_stock_list).values
                # 这边把factor数据和adj数据对齐
            # elif self.model == 'HIST':
            #     idx1 = bisect.bisect_left(self.concept_date, date) - 1
            #     plus_data = self.concept_data[idx1]
            #     plus_stockcode = self.concept_stockcode[idx1]
            #     stock = list(set(stock) & set(plus_stockcode))
            #     plus_data = plus_data[[plus_stockcode.index(item) for item in stock], :]
            else: # 比如时空图网络，或者简单的MLP
                plus_data = None
                plus_stockcode = None
            mktcp_data = mktcp_data.reindex(self.all_stock_list).values
            mask3 =  np.isnan(mktcp_data.reshape(-1, 1)).any(axis=1)
            mask = mask | mask3 
            Data = Data[~mask]
            label = label[~mask]
            plus_data = plus_data[~mask,~mask]
            plus_data[np.isnan(plus_data)] = 0
            mktcp_data = mktcp_data[~mask]

            stds = np.std(Data, axis=0)
            Data = (Data - np.mean(Data, axis=0)) / stds
            Data[:, stds==0] = 0# 20231018这一天全为0，要特别设置一下
            label = (label - np.mean(label)) / np.std(label)

            datetime = [self.filename[idx+29+6]] * len(list(np.array(self.all_stock_list)[~mask]))
            # index = pd.MultiIndex.from_tuples(list(zip(datetime, list(np.array(self.all_stock_list)[~mask]))), names=['datetime', 'stock'])  # 这个是为了后面计算指标
            # stock_emb_idx = [self.all_stock.index(item) for item in stock if item in self.all_stock] # 如果是用了图编码的，就会需要这个数据
            index = np.array(self.all_stock_list)[~mask]
            index_bytes = np.array([s.encode('utf-8') for s in index])
            os.makedirs(f'./data/{self.flag}/{self.start}_{self.end}', exist_ok=True)

            with h5py.File(data_file, 'w') as hf:
                hf.create_dataset('Data', data=Data)
                hf.create_dataset('label', data=label)
                hf.create_dataset('plus_data', data=plus_data)
                hf.create_dataset('mktcp_data', data=mktcp_data)
                hf.create_dataset('index', data=index_bytes)

        return Data, label, plus_data, mktcp_data, index, date # 返回一天的feature数据，和一天的标签

