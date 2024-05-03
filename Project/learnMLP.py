import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasetMLP import Dataset
from CustomLoader import CustomLoader



# regiodatetimeG_CN, REG_US]
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# provider_uri = "D:/.qlib/.qlib/qlib_data/cn_data"  # target_dir
# qlib.init(provider_uri=provider_uri, region=REG_CN)
from torch.utils.tensorboard import SummaryWriter

from GCN import GCN
from utils import metric_fn, mse
from models import MLP

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    if model_name.upper() == 'GATS':
        return GATModel
    
    if model_name.upper() == 'GAT':
        return GAT

    if model_name.upper() == 'SFM':
        return SFM_Model

    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'GCN':
        return GCN

    raise ValueError('unknown model name `%s`'%model_name)


def average_params(params_list):  # 计算模型参数的平均值
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:  # 只有一个元素，均值就是其本身
        return params_list[0]
    new_params = collections.OrderedDict()  # 有序字典，记录的是各个参数的平均值，它能够保持插入顺序，所以它里面存储的顺序和params一致
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():  # 这里的k就是不同的参数名
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])


global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args):

    global global_step

    model.train()

    for data in train_loader:
        feature, label, concept_matrix , market_value, _, _ = data
        global_step += 1
        # feature改成两维的输入，才好通用
        feature = feature.reshape(feature.shape[0],-1)
        feature = torch.tensor(feature).to(device).float()
        label = torch.tensor(label).to(device).float()
        if concept_matrix is not None:
            concept_matrix = torch.tensor(concept_matrix).to(device).float()
        market_value = torch.tensor(market_value).to(device).float()
        if args.model_name == 'GCN':
            pred = model(feature, concept_matrix)
        elif args.model_name == 'GAT':
            edges = torch.nonzero(concept_matrix, as_tuple=False).t()
            pred = model(feature,edges)
        else:
            pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)  # 梯度裁剪
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for data in test_loader:
        feature, label, concept_matrix , market_value, index, datetime= data
        feature = feature.reshape(feature.shape[0],-1)
        feature = torch.tensor(feature).to(device).float()
        label = torch.tensor(label).to(device).float()
        if concept_matrix is not None:
            concept_matrix = torch.tensor(concept_matrix).to(device).float()
        market_value = torch.tensor(market_value).to(device).float()

        index = pd.MultiIndex.from_tuples(list(zip([datetime]*len(index), index)), names=['datetime', 'stock'])
        with torch.no_grad():
            if args.model_name == 'GCN':
                pred = model(feature, concept_matrix)
            elif args.model_name == 'GAT':
                edges = torch.nonzero(concept_matrix, as_tuple=False).t()
                pred = model(feature,edges)
            else:
                pred = model(feature)
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = (precision[3] + precision[5] +
            precision[10] + precision[30])/4.0

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/IC', ic, epoch)
    writer.add_scalar(prefix+'/rankIC', rank_ic , epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(model, data_loader, args):

    model.eval()

    preds = []
    for data in data_loader:
        feature, label, concept_matrix , market_value, index, datetime = data
        feature = feature.reshape(feature.shape[0],-1)
        feature = torch.tensor(feature).to(args.device).float()
        label = torch.tensor(label).to(args.device).float()
        if concept_matrix is not None:
            concept_matrix = torch.tensor(concept_matrix).to(device).float()
        market_value = torch.tensor(market_value).to(device).float()

        index = pd.MultiIndex.from_tuples(list(zip([datetime]*len(index), index)), names=['datetime', 'stock'])
        with torch.no_grad():
            if args.model_name == 'GCN':
                pred = model(feature, concept_matrix)
            elif args.model_name == 'GAT':
                edges = torch.nonzero(concept_matrix, as_tuple=False).t()
                pred = model(feature,edges)
            else:
                pred = model(feature)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )  # 只有%s是占位符，其他的都是字符串

    output_path = os.path.join(args.outdir,f'{args.model_name}')
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
    #     print('already runned, exit.')
    #     return

    writer = SummaryWriter(log_dir=output_path)  # Tesnorboard日志存储

    global global_log_file  
    global_log_file = output_path + '/' + '_run.log'

    pprint('create loaders...')
    # train_dataset = Dataset('20180101', '20221231', args.model_name, args.matrix_dir,'train')
    # valid_dataset = Dataset('20230101', '20230630', args.model_name, args.matrix_dir,'valid')
    # test_dataset = Dataset('20230701', '20240315', args.model_name, args.matrix_dir,'test')

    train_dataset = Dataset('20221201', '20221202', args.model_name, args.matrix_dir,'train')
    valid_dataset = Dataset('20221201', '20221202', args.model_name, args.matrix_dir,'valid')
    test_dataset = Dataset('20221201', '20221202', args.model_name, args.matrix_dir,'test')

    train_dataset.all_stock('20180101','20221231')  # 维护一个全市场股票列表
    valid_dataset.all_stock('20180101','20221231')
    test_dataset.all_stock('20180101','20221231')
    
    # 创建 DataLoader
    train_loader = CustomLoader(train_dataset, batch_size=1, shuffle=True, flag = 'train') # 不同天的股票数不同，如果想要batch_size>1，就要自己构造一个
    valid_loader = CustomLoader(valid_dataset, batch_size=1, shuffle=False, flag = 'valid')
    test_loader = CustomLoader(test_dataset, batch_size=1, shuffle=False, flag = 'test')

    # stock2concept_matrix = np.load(args.stock2concept_matrix) 
    # if args.model_name == 'HIST':
        # stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(args.repeat):  # 重复多次
        pprint('create model...')
        if args.model_name == 'SFM':
            model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
        elif args.model_name == 'ALSTM':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
        elif args.model_name == 'Transformer':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
        elif args.model_name == 'HIST':
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
        elif args.model_name == 'MLP':
            model = MLP(args.d_feat)
        else:
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_ic = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        # params_list = collections.deque(maxlen=args.smooth_steps)
        metrics = []  # Collect metrics for CSV

        epoch = 0
        train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
        test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')
        epoch_metrics = [train_loss, train_score, train_ic, train_rank_ic, val_loss, val_score,val_ic, val_rank_ic,test_loss, test_score, test_ic, test_rank_ic]
        metrics.append(epoch_metrics)
        
        for epoch in range(args.n_epochs):
            pprint('Running', times,'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args)  # 训练
            torch.save(model, output_path + '/model_last_epoch.pth')
            # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))  # 含优化器的所有参数的字典，包括当前的学习率、动量等，以及上一步的梯度等

            params_ckpt = copy.deepcopy(model.state_dict())
            # params_list.append(params_ckpt)
            # avg_params = average_params(params_list)  # 最近的几次求了平均
            # model.load_state_dict(avg_params)

            pprint('evaluating...') # evaluate用的是最近几次的平均参数诶
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
            pprint('Train Precision: ', train_precision)
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            pprint('Train Recall: ', train_recall)
            pprint('Valid Recall: ', val_recall)
            pprint('Test Recall: ', test_recall)
            model.load_state_dict(params_ckpt)

            epoch_metrics = [train_loss, train_score, train_ic, train_rank_ic, val_loss, val_score,val_ic, val_rank_ic,test_loss, test_score, test_ic, test_rank_ic]
            metrics.append(epoch_metrics)

            if val_rank_ic > best_ic:
                best_ic = val_rank_ic
                stop_round = 0
                best_epoch = epoch
                # best_param = copy.deepcopy(avg_params)  # 存储的最优模型也存的是平均参数
                torch.save(model, output_path + '/model_best.pth')
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break
            
        pprint('best score:', best_ic, '@', best_epoch)
        # model.load_state_dict(best_param)
        # torch.save(best_param, output_path+'/model.bin')

        # 全部训练已经结束
        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:

            pred= inference(model, eval(name+'_loader'))
            pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            precision, recall, ic, rank_ic = metric_fn(pred)

            pprint(('%s: IC %.6f Rank IC %.6f')%(
                        name, ic.mean(), rank_ic.mean()))
            pprint(name, ': Precision ', precision)
            pprint(name, ': Recall ', recall)
            res[name+'-IC'] = ic
            # res[name+'-ICIR'] = ic.mean() / ic.std() # 那个里面已经求了均值了
            res[name+'-RankIC'] = rank_ic
            # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()
        
        all_precision.append(list(precision.values()))  # 本来是一个字典，现在变成了一个列表
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)

        metrics = pd.DataFrame(metrics, columns=['train_loss', 'train_score',  
                                                 'train_ic', 'train_rank_ic', 'valid_loss', 'valid_score',  'valid_ic', 'valid_rank_ic', 'test_loss', 'test_score', 
                                                   'test_ic', 'test_rank_ic'])
        metrics.to_csv(output_path+'/metrics.csv')
    pprint(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis= 0)  # 同样的k的求均值
    precision_std = np.array(all_precision).std(axis= 0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='MLP')
    parser.add_argument('--d_feat', type=int, default=101)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=1)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')
    parser.add_argument('--matrix_dir', type=str, default ='/root/autodl-tmp/data/matrix_sac')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')


    # input for csi 300

    parser.add_argument('--outdir', default='/home/featurize/work/myGNN/output')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
