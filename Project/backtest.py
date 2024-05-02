from tqdm import tqdm

import pandas as pd
import numpy as np
# import statsmodels.api as sm
from scipy import stats

import seaborn as sns
from matplotlib import pyplot as plt
from pylab import *

# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 1000)


def align(df1, df2):
    date_lst = sorted(list(set(df1.index).intersection(set(df2.index))))
    stock_lst = sorted(list(set(df1.columns).intersection(set(df2.columns))))
    df1 = df1.loc[date_lst, stock_lst]
    df2 = df2.loc[date_lst, stock_lst]
    return df1, df2


def calc_ic(df_factor, df_ret, method='spearman'):
    """
    计算ic
    输入：
        df_factor：因子矩阵(T *N)
        df_ret：价格序列(T *N)
        method：ic的计算方法：'spearman', 'pearson'
    输出：
        ic_by_year：年度ic表现，含ic和rankic
        ic：ic时间序列数据
    """
    # ic = df_factor.corrwith(df_ret,axis=1,method=method)
    # rank_ic = df_factor.rank(pct=True).corrwith(df_ret.rank(pct=True), axis=1, method=method)

    # 对于基本面因子，覆盖度问题，所以需要筛选覆盖的股票
    ic = pd.Series(index=df_factor.index)
    rank_ic = pd.Series(index=df_factor.index)
    for i in df_factor.index:
        fac = df_factor.loc[i, :]
        ret = df_ret.loc[i, :]
        if (fac.notna() & ret.notna()).sum() < 20:  # 覆盖少于20只股票时，不计算IC
            continue
        else:
            ic.loc[i] = fac[(fac.notna()) & (ret.notna())].corr(ret[(fac.notna()) & (ret.notna())])
            rank_ic.loc[i] = fac[(fac.notna()) & (ret.notna())].rank(pct=True).corr(
                ret[(fac.notna()) & (ret.notna())].rank(pct=True), method="spearman")

    # 计算全样本的相关统计数据
    ic_mean, ic_std = ic.mean(), ic.std()
    icir = ic_mean / ic_std
    ic_winrate = len(ic[ic * np.sign(ic_mean) > 0]) / (len(ic[~pd.isna(ic)]) + 1)  # 胜率的计算，即IC方向与其均值方向相同的比例
    rank_ic_mean, rank_ic_std = rank_ic.mean(), rank_ic.std()
    rank_ic_ir = rank_ic_mean / rank_ic_std  # rank ic均值除以标准差

    # 计算分年度的相关统计数据
    ic_mean_ = ic.groupby(ic.index.year).mean()
    ic_std_ = ic.groupby(ic.index.year).std()
    icir_ = ic_mean_ / ic_std_
    ic_winrate_ = ic.groupby(ic.index.year).apply(lambda x: (x * np.sign(ic_mean) > 0).sum() / (len(x) + 1))
    rank_ic_mean_ = rank_ic.groupby(ic.index.year).mean()
    rank_ic_std_ = rank_ic.groupby(ic.index.year).std()
    rank_ic_ir_ = rank_ic_mean_ / rank_ic_std_

    ic_by_year = pd.concat([ic_mean_, ic_std_, icir_, ic_winrate_, rank_ic_mean_, rank_ic_std_, rank_ic_ir_], axis=1)
    ic_by_year.columns = ['ic_mean', 'ic_std', 'icir', 'ic_winrate', 'rank_ic_mean', 'rank_ic_std', 'rank_ic_ir']

    # 返回分年度和总计的表现
    ic_all = pd.DataFrame([ic_mean, ic_std, icir, ic_winrate, rank_ic_mean, rank_ic_std, rank_ic_ir], columns=['all'],
                          index=['ic_mean', 'ic_std', 'icir', 'ic_winrate', 'rank_ic_mean', 'rank_ic_std',
                                 'rank_ic_ir'])
    ic_by_year = pd.concat([ic_by_year, ic_all.T], axis=0)

    return ic_by_year, ic, rank_ic


def factor2daily(factor, daily_dates, shift_days, start_date, end_date):
    '''因子频率细化到天
    因为有些地方没有因子值，是nan，直接ffill会覆盖掉，所以要建一个日期索引列fill_dates，引导ffill

    param
    --------
    factor: T*N, T为因子日期，N为因子个数
    trade_dates: 交易日期列表
    shift_days: 滞后天数

    return
    --------
    daily_fac: T*N, T为交易日期，N为因子个数
    '''

    # start_date, end_date = factor.index[0], factor.index[-1]
    daily_dates = list(filter(lambda x: (x >= start_date) & (x <= end_date), daily_dates))
    fill_dates = pd.Series(data=daily_dates)
    fill_dates[~fill_dates.isin(factor.index)] = np.nan
    fill_dates = fill_dates.ffill()
    factor_daily = pd.DataFrame(index=daily_dates, columns=factor.columns)
    factor_daily.loc[factor.index] = factor
    factor_daily = factor_daily.groupby(fill_dates.tolist()).ffill()
    factor_daily = factor_daily.shift(shift_days)  # 滞后一天
    factor_daily = factor_daily.iloc[shift_days:, :]

    return factor_daily


def factor_coverage(factor, idx_mem):
    '''因子覆盖度，覆盖股票数，覆盖天数
    '''

    fig, ax = plt.subplots(2, 1, figsize=(20, 2 * 5))
    factor[factor.notna()].count(axis=1).plot(ax=ax[0])
    if type(idx_mem) == pd.DataFrame:
        idx_mem[idx_mem == 1].count(axis=1).plot(ax=ax[0])
    ax[0].set_title('Number of Stocks', fontsize=13)
    factor[factor.notna()].count(axis=0).sort_values(ascending=True).plot(ax=ax[1])
    ax[1].set_title('Number of Dates', fontsize=13)

    return fig


def cal_pnl(factor_mtx, factor_direction, idx,
            t_hold, t_beg, t_end, price_mtx, tradeTime, interest, group,
            dates_1y, tx_cost, stockNumLimit):
    # 计算pnl，存到summary中
    factor_mtx = factor_mtx.astype('float')
    return_num = dates_1y // t_hold  # 向下取整求每年的换仓次数

    if factor_direction == -1:
        factor_mtx = factor_mtx * factor_direction  # 调整因子方向

    # 剔除前一天ST的、不交易的个股，涨跌停需要在因子值处处理
    factor_mtx[pd.isna(price_mtx)] = np.nan
    # 排名相同时，取名次的均值
    factor_rank_mtx = factor_mtx.rank(axis=1, pct=True, method='average', na_option='keep').copy()

    # 回测日期序列
    date_all_list = factor_mtx.loc[t_beg:t_end, :].index.values
    date_trade_list = date_all_list[::t_hold]  # 换仓日（计算因子值所用的日期）

    summary = pd.DataFrame(columns=[f'pnl_{i}' for i in range(1, group + 1)], index=date_trade_list)
    summary.loc[:, f'turnover_{group}'] = 0
    summary.loc[:, 'cost'] = 0
    dict_port_list = [dict() for _ in range(group)]
    dict_port_pre = dict()

    if tradeTime == 'close':
        date_opening_lst = date_all_list[::t_hold]
        range_int = -1
    elif tradeTime == 'nextOpen':  # 使用开盘价时,为简化收益率计算,先确认日期
        date_opening_lst = date_all_list[1:][::t_hold]  # 取用后一个交易日的开盘价
        range_int = -2  # 若为次日开盘买入，则倒数第二的换仓日，有可能来不及卖出（倒数第一个换仓日的第二天卖出），故也不选择

    print("trading dates: ", len(date_trade_list[:range_int]))
    for idx_date, date in tqdm(enumerate(date_trade_list[:range_int])):
        # 符合要求的股票池
        idx_stocktransac = np.where(~np.isnan(factor_mtx.loc[date, :]))[0]
        if len(idx_stocktransac) == 0:
            continue
        stocktransac = factor_mtx.columns[idx_stocktransac]
        # 分档计算组合
        if len(stocktransac) >= stockNumLimit:
            # 如果停牌股票很多(可交易股票小于stockNumLimit)，则直接沿用上期持仓
            dict_port_list = [dict() for _ in range(group)]
            s_rank = factor_rank_mtx.loc[date, stocktransac]
            for decile in range(1, group + 1):
                lb = (decile - 1) * (1 / group) if decile > 1 else -0.0000001  # 下界用大于
                ub = decile * (1 / group)  # 上界用小于等于
                stock_in_port = s_rank[(s_rank > lb) & (s_rank <= ub)].index.values  # 属于decile组内的股票
                try:
                    # 等权组合
                    dict_port_list[decile - 1] = dict(zip(stock_in_port, np.zeros(len(stock_in_port)) + 1.0 / len(
                        stock_in_port)))  # zip将stock名和股票权重打包为元组
                except ZeroDivisionError:
                    print(str(len(stock_in_port)), "stock_in_port")
        # 获取开仓平仓日期
        date_op = date_opening_lst[idx_date]
        date_cp = date_opening_lst[idx_date + 1]
        # 计算该期组合收益率
        for idx_decile, dict_port in enumerate(dict_port_list):
            if not dict_port:
                continue
            decile = idx_decile + 1  # 分档，按照1~group来排序号，group档最大
            # 组合内股票代码和权重
            stock_ = dict_port.keys()
            weight_port = np.array([dict_port[s] for s in stock_])
            price_op_port = price_mtx.loc[date_op, stock_]
            price_cp_port = price_mtx.loc[date_cp, stock_]
            ret_port = price_cp_port / price_op_port - 1
            ret_port[pd.isna(ret_port)] = 0

            # 由于loc指定的日期有问题，导致次日open换仓时，会对出一些日期（尤其非日频调仓时）
            # 原为 summary.loc[date_cp,f'pnl_{decile}'] = np.sum(ret_port*weight_port)
            summary.loc[date, f'pnl_{decile}'] = np.sum(
                ret_port * weight_port)  # 是否应该修改为： summary.loc[date,f'pnl_{decile}'] = np.sum(ret_port*weight_port)
            if type(idx) != type(None):
                ret_idx = idx.loc[date_cp, idx.columns[0]] / idx.loc[date_op, idx.columns[0]] - 1
                summary.loc[date, 'pnl_idx'] = ret_idx

            if idx_date > 0 and decile == group:
                # D{group}组计算换手率
                turnover = 0
                # stock_new = set(stock_) - set(dict_port_pre.keys())
                # for k in stock_new:
                #     turnover += dict_port[k]  # 新买入股票
                stock_keep = set(stock_) & set(dict_port_pre.keys())
                for k in stock_keep:
                    # 仅考虑单边换手，即仅考虑买入的所有股票在卖出时产生的交易费用
                    # turnover += max(0, dict_port[k] - dict_port_pre[k])  # 此前已有股票，计算一共需要买入多少权重
                    turnover += max(0, dict_port_pre[k] - dict_port[k])

                stock_old = set(dict_port_pre.keys()) - set(stock_)
                for k in stock_old:
                    turnover += dict_port_pre[k]  # 新买入股票

                # 原为 summary.loc[date_cp,f'turnover_{group}'] = turnover
                # 原为 summary.loc[date_cp,'cost'] = tx_cost*turnover
                summary.loc[date, f'turnover_{group}'] = turnover  # 最大组的换手
                summary.loc[date, 'cost'] = tx_cost * turnover  # 最大组的交易费用
                summary.loc[date, 'num_L'] = len(dict_port)

            if decile == group:
                # 更新期末D{group}组的组合权重
                weigth_port_cp = weight_port * (ret_port + 1.0)
                weigth_port_cp /= np.sum(weigth_port_cp)
                dict_port_pre = dict(zip(stock_, weigth_port_cp))

    summary = summary.iloc[:range_int, :]  # 去掉最后一个换仓日

    # 用summary中的pnl，计算nav和各种指标
    # ---------------------------------------------------------------------------#
    # 单利模式，s为日频收益率
    annRT = lambda s: s.mean() * return_num  # 单利的平均年化
    DD = lambda s: 1 - (1 + s.cumsum()) / (1 + s.cumsum()).expanding().max()  # t时刻时的历史最大值减t时刻的值
    maxDD = lambda s: DD(s).max()
    volatility = lambda s: s.std() * np.sqrt(return_num)  # 单利的年化波动率
    sharpe = lambda s: annRT(s) / volatility(s)  # 单利的夏普比率
    calmar = lambda s: annRT(s) / maxDD(s)  # 单利的卡玛比率

    # 股票复利模式，s为日频收益率pd.Series
    annRT2 = lambda s: (s + 1).cumprod().iat[-1] ** (return_num / len(s)) - 1
    DD2 = lambda s: 1 - (s + 1).cumprod() / (s + 1).cumprod().expanding().max()
    maxDD2 = lambda s: DD2(s).max()

    winRateDay = lambda s: (s > 0).sum() / len(s)
    winRateDay_adj = lambda s: (s > 0).sum() / (len(s) - sum(s.isna()))
    annRT_adj = lambda s: ((s + 1).cumprod().iat[-1] - 1) * (return_num / len(s))

    # 数据统计
    # ---------------------------------------------------------------------------#
    # 收益曲线计算
    pnl_lst = [f'pnl_{i}' for i in range(1, group + 1)]
    if interest == 'simple':

        # 全市场平均收益率
        summary['pnl_allstock'] = summary.loc[:, pnl_lst].mean(axis=1)  # 市场平均收益

        # summary['nav_long_excess'] = 1.0+(summary[f'pnl_{group}']-summary['pnl_allstock']).cumsum()  # 无交易成本的多头超额
        # summary['nav_short_excess'] = 1.0+(summary['pnl_1']-summary['pnl_allstock']).cumsum()  # 无交易成本的空头超额
        # summary['nav_long_excess_fee'] = 1.0+(summary[f'pnl_{group}']-summary['pnl_allstock']-summary['cost']).cumsum()  # 考虑交易成本的超额
        # summary['nav_long_fee'] = 1.0+(summary[f'pnl_{group}']-summary['cost']).cumsum()  # 考虑交易成本的多头收益
        # summary['nav_long_excess_fee'].iloc[0] = 1
        # summary['nav_long_fee'].iloc[0] = 1

        # 画图
        # summary['nav_LS'] = 1.0 + (summary[f'pnl_{group}'] - summary['pnl_1']).cumsum()  # 多空收益率
        # # 首日数据填充
        # summary['nav_LS'].iloc[0] = 1
        # for decile in range(1,group+1):
        #     # 累计收益
        #     summary[f'nav_{decile}'] = 1.0 + (summary[f'pnl_{decile}']).cumsum() # - summary['pnl_allstock']
        # ---------------------------------------------------------------------------#
        # 因子表现统计
        dict_statistics = dict()
        dict_statistics['AnnRet_L'] = annRT(summary[f'pnl_{group}'])
        dict_statistics['AnnRet_S'] = annRT(summary['pnl_1'])
        dict_statistics['AnnRet_LS'] = annRT(summary[f'pnl_{group}'] - summary['pnl_1'])
        dict_statistics['AnnRet_L_ex'] = annRT(summary[f'pnl_{group}'] - summary['pnl_allstock'])
        # dict_statistics['AnnRet_S_excess'] = annRT(summary['pnl_allstock']-summary['pnl_1'])
        # dict_statistics['annualret_allstock'] = annRT(summary['pnl_allstock'])

        dict_statistics['AnnRet_L_fee'] = annRT(summary[f'pnl_{group}'] - summary['cost'])
        dict_statistics['TO_L'] = summary[f'turnover_{group}'].mean() * dates_1y * 2  # 单边换手*2
        # for decile in range(1, group+1):
        #     # 分档年化收益
        #     dict_statistics[f'annualret_{decile}'] = annRT(summary[f'pnl_{decile}'])
        # ---------------------------------------------------------------------------#
        # 考虑交易成本的回测结果

        dict_statistics['SR_L'] = sharpe(summary[f'pnl_{group}'])
        dict_statistics['SR_L_ex'] = sharpe(summary[f'pnl_{group}'] - summary['pnl_allstock'])
        # dict_statistics['Calmar_L'] = calmar(summary[f'pnl_{group}'])
        # dict_statistics['Calmar_L_excess'] = calmar(summary[f'pnl_{group}'] - summary['pnl_allstock'])
        dict_statistics['MDD_L'] = maxDD(summary[f'pnl_{group}'])
        # dict_statistics['mdd_long_excess'] = maxDD(summary[f'pnl_{group}']-summary['pnl_allstock'])
        # dict_statistics['MDD_LS'] = maxDD(summary[f'pnl_{group}']-summary['pnl_1'])
        dict_statistics['num_L'] = summary["num_L"].mean()

    elif interest == 'compound':
        for decile in range(1, group + 1):
            # 累计收益
            summary[f'nav{decile}'] = (1.0 + summary[f'pnl_{decile}']).cumprod()
        # 全市场平均收益率
        summary['pnl_allstock'] = summary.loc[:, pnl_lst].mean(axis=1)
        summary['nav_LS'] = (1.0 + summary[f'pnl_{group}'] - summary['pnl_1']).cumprod()  # 多空收益率
        summary['nav_hedge_wofee'] = (1.0 + summary[f'pnl_{group}'] - summary['pnl_allstock']).cumprod()  # 无交易成本的多头超额
        summary['nav_hedge_short_wofee'] = (1.0 + summary['pnl_1'] - summary['pnl_allstock']).cumprod()  # 无交易成本的空头超额
        summary['nav_hedge_withfee'] = (
                    1.0 + summary[f'pnl_{group}'] - summary['pnl_allstock'] - summary['cost']).cumprod()  # 考虑交易成本的超额
        summary['nav_long_withfee'] = (1.0 + summary[f'pnl_{group}'] - summary['cost']).cumprod()  # 考虑交易成本的多头收益
        # 首日数据填充
        summary['nav_hedge_withfee'].iloc[0] = 1
        summary['nav_long_withfee'].iloc[0] = 1
        summary['nav_LS'].iloc[0] = 1

        # ---------------------------------------------------------------------------#
        # 因子表现统计
        dict_statistics = dict()
        dict_statistics['annualret_long_excess'] = annRT2(summary[f'pnl_{group}'] - summary['pnl_allstock'])
        dict_statistics['annualret_short_excess'] = annRT2(summary['pnl_1'] - summary['pnl_allstock'])
        dict_statistics['annualret_LS'] = annRT2(summary[f'pnl_{group}'] - summary['pnl_1'])
        dict_statistics['annual_turnover'] = summary[f'turnover_{group}'].mean() * dates_1y * 2
        for decile in range(1, group + 1):
            # 分档年化收益
            dict_statistics[f'nav{decile}'] = annRT2(summary[f'pnl_{decile}'])
        # ---------------------------------------------------------------------------#
        # 考虑交易成本的回测结果
        dict_statistics['annualret_long_fee'] = annRT2(summary[f'pnl_{group}'] - summary['cost'])
        dict_statistics['annualret_long_excess_fee'] = annRT2(
            summary[f'pnl_{group}'] - summary['pnl_allstock'] - summary['cost'])
        dict_statistics['mdd_long_fee'] = maxDD2(summary[f'pnl_{group}'] - summary['cost'])
        dict_statistics['mdd_long_excess_fee'] = maxDD2(
            summary[f'pnl_{group}'] - summary['pnl_allstock'] - summary['cost'])
        dict_statistics['mdd_LS'] = maxDD2(summary[f'pnl_{group}'] - summary['pnl_1'])
    else:
        print('Error: interest is wrong!')

    # 多空胜率、以及多头胜率、空头胜率
    dict_statistics['winrate_L'] = winRateDay_adj(summary[f'pnl_{group}'])
    # dict_statistics['winrate_S'] = winRateDay_adj(- summary['pnl_1'])
    # dict_statistics['winrate_LS'] = winRateDay_adj(summary[f'pnl_{group}'] - summary['pnl_1'])

    df_statistics = pd.DataFrame(data=dict_statistics.values(), index=dict_statistics.keys())
    df_statistics = df_statistics.rename(columns={0: "分{}组, 用{}价格回测的年化指标".format(group, tradeTime)})

    if type(idx) != type(None):
        # pnl_by_year = summary[[f"pnl_{group}", "pnl_allstock", "pnl_idx"]].astype("float").groupby(
        #     summary.index.year).sum()
        # pnl_by_year["excess_all"] = pnl_by_year[f"pnl_{group}"] - pnl_by_year["pnl_allstock"]
        # pnl_by_year["excess_idx"] = pnl_by_year[f"pnl_{group}"] - pnl_by_year["pnl_idx"]
        pnl_by_year = summary[[f"pnl_{group}", "pnl_allstock"]].astype("float").groupby(
            summary.index.year).sum()
        pnl_by_year["excess_all"] = pnl_by_year[f"pnl_{group}"] - pnl_by_year["pnl_allstock"]
        # pnl_by_year["excess_idx"] = pnl_by_year[f"pnl_{group}"] - pnl_by_year["pnl_idx"]

    elif type(idx) == type(None):
        pnl_by_year = summary[[f"pnl_{group}", "pnl_allstock"]].astype("float").groupby(
            summary.index.year).sum()
        pnl_by_year["excess_all"] = pnl_by_year[f"pnl_{group}"] - pnl_by_year["pnl_allstock"]

    pnl_by_year.columns = ["Long", "Market", "Excess"]
    # pnl_by_year.columns = ["Long", "Market", "Index", "Market Excess", "Index Excess"]
    # pnl_by_month = summary[[f"pnl_{group}"]].astype("float").groupby([summary.index.year, summary.index.month]).sum()
    # pnl_by_month = pnl_by_month.unstack(level=1)
    # pnl_by_month.columns = pnl_by_month.columns.droplevel(level=0)

    return summary, df_statistics, pnl_by_year  # , pnl_by_month


# if __name__ == "__main__":
from os import path
from config import Config

cfg = Config()

    # start_date, end_date = pd.to_datetime("20230701"), pd.to_datetime("20231231")
    # start_date, end_date = pd.to_datetime("20160101"), pd.to_datetime("20221231")
    # start_date, end_date = pd.to_datetime("20230101"), pd.to_datetime("20231231")
    # start_date, end_date = pd.to_datetime("20180101"), pd.to_datetime("20231231")
    # start_date, end_date = pd.to_datetime("20100331"), pd.to_datetime("20240131")



def bt_plot(factor_path, factor_name, start_date, end_date, fontsize):
    idx_name_lst = ["all"]
    # idx_name_lst = ["all", "hs300", "zz500", "zz1000"]

    for idx_name in idx_name_lst:
        print(factor_name, idx_name, start_date, end_date)

        if idx_name != "all":
            idx_price = pd.read_csv(path.join(cfg.DATA_PATH, f"stock/index_{idx_name}_close.csv"), index_col=0)
            idx_price.index = pd.to_datetime(idx_price.index.astype(str))
            idx_mem = pd.read_csv(path.join(cfg.DATA_PATH, f"stock/index_{idx_name}_mem.csv"), index_col=0) == 1
            idx_mem.index = pd.to_datetime(idx_mem.index.astype("str"))
        elif idx_name == "all":
            # idx_price = pd.read_csv(path.join(cfg.DATA_PATH, f"stock/index_zz500_close.csv"), index_col=0)
            # idx_price.index = pd.to_datetime(idx_price.index.astype(str))
            idx_price = None
            idx_mem = None

            # idx_mem = zz500_mem | hs300_mem #  | zz1000_mem # None
        # print(idx_mem, "idx_mem")
        # factor = pd.read_csv(path.join(cfg.factor_path, f"{factor_name}.csv"), index_col=0)
        # factor.index = pd.to_datetime(factor.index.astype("str"))


        df = pd.read_pickle(factor_path)
        # print(df)
        factor = df["score"].unstack()
        # df = pd.read_pickle("RF_pred.pkl")
        # factor = df["y_pred"].unstack()
        def ticker_format(x):
            if x[0] == "0" or x[0] == "3":
                return x + '.SZ'
            elif x[0] == "6":
                return x + '.SH'
            elif x[0] == "8":
                return x + '.BJ'
            else:
                return x

        factor.columns = [ticker_format(x) for x in factor.columns]
        factor.index = pd.to_datetime(factor.index)

        factor = factor.astype("float")
        factor = factor[(factor.index >= start_date) & (factor.index <= end_date)]
        factor = factor.iloc[::10, :]
        price_df = pd.read_csv(path.join(cfg.DATA_PATH, "stock_close_post.csv"), index_col=0)
        price_df.index = pd.to_datetime(price_df.index.astype("str"))
        daily_dates = pd.Series(price_df.index, index=price_df.index)
        daily_dates = daily_dates.index

        factor = factor[list(set(factor.columns).intersection(set(price_df.columns)))]

        # # 算IC时，使用指数股票池
        if type(idx_mem) == pd.DataFrame:
            idx_mem_ic = idx_mem.copy()
            factor, idx_mem_ic = align(factor, idx_mem_ic)
            factor[~idx_mem_ic] = np.nan

        price_head = price_df.loc[factor.index[:-1], :]
        price_tail = price_df.loc[factor.index[1:], :]
        price_tail.index = price_head.index
        ret_mtx = price_tail / price_head - 1

        factor_ic, ret_mtx = align(factor, ret_mtx)
        ic_by_year, ic, rank_ic = calc_ic(factor_ic, ret_mtx)

        # 因子频率细化到天后，使用指数股票池
        factor_daily = factor2daily(factor, daily_dates, shift_days=1,
                                    start_date=start_date, end_date=end_date)
        if type(idx_mem) == pd.DataFrame:
            factor_daily, idx_mem = align(factor_daily, idx_mem)
            factor_daily[~idx_mem] = np.nan

        factor_direction = np.sign(rank_ic.mean())
        t_hold = 1
        # t_beg = pd.to_datetime("2017-12-01") #factor_daily.index[0] #
        # t_end = factor_daily.index[-1]
        price_mtx = price_df.copy()
        idx_price=None
        # factor_mtx = factor_mtx.loc[:price_mtx.index[-1]]
        tradeTime = 'close'
        interest = 'simple'  # 'compound'
        group = 10
        # top_n = 50

        dates_1y = 252
        tx_cost = 0.002
        stockNumLimit = 0  # 50    # 可交易股票小于多少时，直接沿用上期持仓

        summary, df_statistics, pnl_by_year = cal_pnl(factor_daily, factor_direction, idx_price,
                                                      t_hold, start_date, end_date, price_mtx,
                                                      tradeTime, interest, group,
                                                      dates_1y, tx_cost, stockNumLimit)
        # , pnl_by_month

        # 作图

        fig = plt.figure(figsize=(6, 8), dpi=200, constrained_layout=True)
        fig.suptitle("factor: " + factor_name + ", universe: " + str(idx_name) + ", date: " + start_date.strftime(
            "%Y%m%d") + "~" + end_date.strftime("%Y%m%d"), fontsize=fontsize)
        # gs = fig.add_gridspec(7, 2)
        gs = fig.add_gridspec(2, 2)

        # # 因子覆盖度
        # ax_cover = fig.add_subplot(gs[0, 0])
        # factor_daily[factor_daily.notna()].count(axis=1).plot()
        # if type(idx_mem) == pd.DataFrame:
        #     idx_mem[idx_mem == 1].count(axis=1).plot(ax=ax_cover)
        # ax_cover.set_title("Stock Coverage", fontsize=fontsize)
        # ax_cover.grid(False) # 消除网格线
        #
        # # 因子值分布
        # ax_dist = fig.add_subplot(gs[0, 1])
        # facor_dist = factor_daily.iloc[::dates_1y, :].T
        # facor_dist.columns = facor_dist.columns.strftime("%Y%m")
        # sns.violinplot(data=facor_dist, ax=ax_dist)
        # ax_dist.set_title("Factor Distribution", fontsize=fontsize)
        # ax_dist.grid(False)

        # # ic、rank ic
        # ax_icbyyear = fig.add_subplot(gs[1, :])
        # the_table = ax_icbyyear.table(cellText=ic_by_year.round(4).values,
        #                               rowLabels=ic_by_year.index,
        #                               colLabels=ic_by_year.columns,
        #                               cellLoc='center',
        #                               loc='center')
        # the_table.set_fontsize(13)
        # ax_icbyyear.get_xaxis().set_visible(False)
        # ax_icbyyear.get_yaxis().set_visible(False)
        # the_table.scale(1, 1.5)
        # plt.box(on=None)
        # ax_icbyyear.set_title("IC和Rank IC", fontsize=fontsize)
        # ax_icbyyear.grid(False)

        # ax_ic = fig.add_subplot(gs[2, 0])
        # ax_ic = fig.add_subplot(gs[0, 0])
        # # 设置刻度，每6个月展示"%Y-%m"
        # xtick = np.arange(0, ic.shape[0], 50)
        # xticklabel = pd.Series(ic.index.strftime("%Y")[xtick])
        # # 绘制柱状图， 横轴是数据个数， 纵轴是因子IC的值
        # ax_ic.bar(np.arange(ic.shape[0]), ic.values, alpha=0.7, lw=0.7, color='tab:blue')
        # ax_ic.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        # ax_ic.axhline(ic.mean(), linestyle=':', color='black', lw=2, alpha=0.8)
        # # 设置刻度
        # ax_ic.set_xticks(xtick)
        # ax_ic.set_xticklabels(xticklabel, fontsize=fontsize)
        # ax_ic.set_title(f"IC: {ic.mean():.4f}", fontsize=fontsize)
        # ax_ic.grid(False)
        #
        # ax_ict = ax_ic.twinx()
        # ax_ict.plot(np.arange(ic.shape[0]), ic.cumsum(), color='tab:green', lw=2, alpha=0.8)

        # ax_rank_ic = fig.add_subplot(gs[2, 1])
        ax_rank_ic = fig.add_subplot(gs[0, 0])
        # 设置刻度，每6个月展示"%Y-%m"
        xtick = np.arange(0, rank_ic.shape[0], 50)
        xticklabel = pd.Series(rank_ic.index.strftime("%Y")[xtick])
        # 绘制柱状图， 横轴是数据个数， 纵轴是因子IC的值
        ax_rank_ic.bar(np.arange(rank_ic.shape[0]), rank_ic.values, alpha=0.7, lw=0.7, color='tab:blue')
        ax_rank_ic.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        ax_rank_ic.axhline(rank_ic.mean(), linestyle=':', color='black', lw=2, alpha=0.8)
        # 设置刻度
        ax_rank_ic.set_xticks(xtick)
        ax_rank_ic.set_xticklabels(xticklabel, fontsize=fontsize)
        ax_rank_ic.set_title(f"Rank IC: {rank_ic.mean():.4f}", fontsize=fontsize)
        ax_rank_ic.grid(False)

        ax_rank_ict = ax_rank_ic.twinx()
        ax_rank_ict.plot(np.arange(rank_ic.shape[0]), rank_ic.cumsum(), color='tab:green', lw=2, alpha=0.8)

        # group ret, pnl, indicator

        # ax_indicator = fig.add_subplot(gs[3, 0])
        ax_indicator = fig.add_subplot(gs[0, 1])
        the_table = ax_indicator.table(cellText=df_statistics.round(4).values,
                                       rowLabels=df_statistics.index,
                                       # colLabels=df_statistics.columns,
                                       cellLoc='center',
                                       loc='center right')
        the_table.set_fontsize(fontsize)
        ax_indicator.get_xaxis().set_visible(False)
        ax_indicator.get_yaxis().set_visible(False)
        the_table.scale(0.5, 1.5)
        plt.box(on=None)
        ax_indicator.set_title("Annual Indicator", fontsize=fontsize, loc="center")
        ax_indicator.grid(False)

        # ax_pnl = fig.add_subplot(gs[3, 1])
        # ax_pnl = fig.add_subplot(gs[2, 1])
        # the_table = ax_pnl.table(cellText=pnl_by_year.round(4).values,
        #                          rowLabels=pnl_by_year.index,
        #                          colLabels=pnl_by_year.columns,
        #                          cellLoc='center',
        #                          loc='center right')
        # the_table.set_fontsize(fontsize)
        # ax_pnl.get_xaxis().set_visible(False)
        # ax_pnl.get_yaxis().set_visible(False)
        # the_table.scale(1.2, 1.6)
        # plt.box(on=None)
        # ax_pnl.set_title("Annual Return", fontsize=fontsize, loc="left")
        # ax_pnl.grid(False)

        # ax_pnl_m = fig.add_subplot(gs[4, :])
        # the_table = ax_pnl_m.table(cellText=pnl_by_month.round(4).values,
        #                     rowLabels=pnl_by_month.index,
        #                     colLabels=pnl_by_month.columns,
        #                     cellLoc='center',
        #                     loc='center')
        # the_table.set_fontsize(fontsize)
        # ax_pnl_m.get_xaxis().set_visible(False)
        # ax_pnl_m.get_yaxis().set_visible(False)
        # the_table.scale(1, 1.5)
        # plt.box(on=None)
        # # ax_pnl.set_title("nav by year and excess return", fontsize=fontsize)
        # ax_pnl_m.grid(False)

        # ax_group = fig.add_subplot(gs[4, :])
        ax_group = fig.add_subplot(gs[1, :])
        summary['nav_LS'] = (1.0 + summary[f'pnl_{group}'] - summary['pnl_1']).cumprod()  # 多空收益率
        # 首日数据填充
        summary['nav_LS'].iloc[0] = 1
        for decile in range(1, group + 1):
            # 累计收益
            summary[f'nav_{decile}'] = (1.0 + summary[f'pnl_{decile}']).cumprod()  # - summary['pnl_allstock'])

        nav_lst = ['nav_LS'] + [f'nav_{i}' for i in range(1, group + 1)]
        # summary.loc[:, nav_lst].plot(lw=2, cmap=cm.coolwarm)
        # plt.show()
        summary.loc[:, nav_lst].plot(lw=2, ax=ax_group, cmap=cm.coolwarm)
        legend = ['long_short', 'short'] + list(range(2, group)) + ['long']
        ax_group.legend(legend, loc="upper left") #, fontsize=fontsize
        ax_group.set_title("Group Return", fontsize=fontsize)
        ax_group.grid(False)

        # ax_rct = fig.add_subplot(gs[6, :])
        # # summary.iloc[-250:,:].loc[:, [f"pnl_{group}", "pnl_allstock", "pnl_idx"]].cumsum().plot(ax=ax_rct)
        # (summary.iloc[-250:, :].loc[:, [f"pnl_{group}", "pnl_allstock", "pnl_idx"]]+1).cumprod().plot(ax=ax_rct)
        #
        # ax_rct.legend(loc='upper left')
        # ax_rct.set_title("long return in recent 1 year")
        # ax_rct.grid(False)

        fig.show()
        fig.savefig(path.join(cfg.DATA_PATH,
                              f"factor_{factor_name}, universe_{str(idx_name)}, date_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.png"))

