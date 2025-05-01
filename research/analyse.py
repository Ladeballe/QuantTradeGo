import numpy as np
import pandas as pd

from .backtest import _calc_rtn_by_signal


def _calc_pos_phase_stat(df_signal, df_pos_net_rtn):
    df_pos_chg_flag = (df_signal.unstack() != df_signal.unstack().shift(-1)).astype(int)  # 与上一个时间戳状态不同的点
    df_pos_phase = df_pos_chg_flag.cumsum().stack()
    df_pos_phase_stat = pd.concat([df_pos_net_rtn, df_signal, df_pos_phase], axis=1,
                                  keys=['net_rtn', 'signal', 'pos_phase'])
    df_pos_phase_stat = df_pos_phase_stat.groupby(['symbol', 'pos_phase']).agg({'net_rtn': 'sum', 'signal': 'first'})
    return df_pos_phase_stat


def _calc_win_ratio_by_signal(df_signal, df_fwd_rtn, cost_ratio):
    df_pos_net_rtn = _calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio)
    df_pos_phase_stat = _calc_pos_phase_stat(df_signal, df_pos_net_rtn)
    df_pos_phase_stat['win_flag'] = (df_pos_phase_stat['net_rtn'] > 0).astype(int)
    df_win_ratio = df_pos_phase_stat[df_pos_phase_stat['signal'] != 0].groupby(['symbol', 'win_flag'])['signal'].count()
    df_win_ratio = df_win_ratio.unstack()
    df_win_ratio = df_win_ratio[1] / (df_win_ratio[0] + df_win_ratio[1])
    return df_win_ratio


def calc_win_ratio_by_signal(df_signal, df_fwd_rtn, cost_ratio=0.0003):
    df_win_ratio = _calc_win_ratio_by_signal(df_signal, df_fwd_rtn, cost_ratio)
    df_win_ratio_long = _calc_win_ratio_by_signal(df_signal.mask(df_signal < 0, 0), df_fwd_rtn, cost_ratio)
    df_win_ratio_short = _calc_win_ratio_by_signal(df_signal.mask(df_signal > 0, 0), df_fwd_rtn, cost_ratio)
    df_pos_win_ratio_merge = pd.concat([df_win_ratio, df_win_ratio_long, df_win_ratio_short],
                                       keys=['all', 'long', 'short'], axis=1)
    return df_pos_win_ratio_merge


def _calc_plr_by_signal(df_signal, df_rtn, cost_ratio):
    """ 计算profit_loss_ratio """
    df_pos_net_rtn = _calc_rtn_by_signal(df_signal, df_rtn, cost_ratio)
    df_pos_phase_stat = _calc_pos_phase_stat(df_signal, df_pos_net_rtn)
    df_pos_phase_stat['win_flag'] = (df_pos_phase_stat['net_rtn'] > 0).astype(int)
    df_plr = df_pos_phase_stat[df_pos_phase_stat['signal'] != 0].groupby(['symbol', 'win_flag'])['net_rtn'].mean()
    df_plr = df_plr.unstack()
    df_plr = df_plr[1] / df_plr[0].abs()
    return df_plr


def calc_plr_by_signal(df_signal, df_rtn, cost_ratio=0.0003):
    df_plr = _calc_plr_by_signal(df_signal, df_rtn, cost_ratio)
    df_plr_long = _calc_plr_by_signal(df_signal.mask(df_signal < 0, 0), df_rtn, cost_ratio)
    df_plr_short = _calc_plr_by_signal(df_signal.mask(df_signal > 0, 0), df_rtn, cost_ratio)
    df_plr_merge = pd.concat([df_plr, df_plr_long, df_plr_short],
                              keys=['all', 'long', 'short'], axis=1)
    return df_plr_merge


def analyse_trade_stat(df_signal):
    def count_days(s_signal):
        product = s_signal.name
        s_signal = pd.concat([s_signal, (s_signal != s_signal.shift(1)).astype(int).cumsum()], axis=1,
                             keys=['sig', 'nth'])
        s_signal_count = s_signal.groupby('nth')\
            .agg(['count', 'first'])\
            .droplevel(0, axis=1)\
            .rename({'count': product}, axis=1)
        df_product_stat = s_signal_count.groupby('first').agg(['mean', 'count', 'median']).stack()[product]
        return df_product_stat
    
    def get_begin_end_date(sr):
        if sr.empty or sr.isna().all():
            return pd.Series([np.nan, np.nan], index=['begin_date', 'end_date'])
        else:
            return pd.Series([sr.dropna().index[0], sr.dropna().index[-1]], index=['begin_date', 'end_date'])

    df_signal = df_signal.unstack()
    df_trade_stat = df_signal.apply(count_days).T
    df_trade_stat = df_trade_stat[[1, -1]]
    df_trade_stat.columns = ['long_trade_days_mean', 'long_trade_num', 'long_trade_days_median',
                             'short_trade_days_mean', 'short_trade_num', 'short_trade_days_median']
    
    df_date_stat = df_signal.apply(get_begin_end_date).T
    df_date_stat['tradeday_count'] = (~df_signal.isna()).sum()
    df_date_stat['years'] = (df_date_stat['end_date'] - df_date_stat['begin_date']).dt.days / 365
    
    df_trade_stat = pd.concat([df_trade_stat, df_date_stat], axis=1)
    df_trade_stat['long_trade_ann_num'] = df_trade_stat['long_trade_num'] / df_trade_stat['years']
    df_trade_stat['short_trade_ann_num'] = df_trade_stat['short_trade_num'] / df_trade_stat['years']
    return df_trade_stat


def analyse_trade_phase(df_signal):
    def phase_begin_end(s_signal):
        product = s_signal.name
        s_signal = pd.concat([s_signal, (s_signal != s_signal.shift(1)).astype(int).cumsum()], axis=1,
                             keys=['sig', 'nth'])
        s_signal_date = s_signal.reset_index().groupby('nth').agg({'sig': 'first', 'date': ['first', 'last']})
        s_signal_date.columns = ['sig', 'begin_date', 'end_date']
        s_signal_date['end_date'] += pd.Timedelta(days=1)
        s_signal_date = s_signal_date[(s_signal_date['sig']!=0) & (~s_signal_date['sig'].isna())]
        dict_signal_date = pd.Series([s_signal_date.to_dict()], name=product)
        return dict_signal_date

    df_signal = df_signal.unstack()
    df_trade_phase = df_signal.apply(phase_begin_end).loc[0]
    return df_trade_phase


def analyse_trade_time(df_signal):
    """统计交易时间起始与终止"""
    df_date_stat = df_signal.unstack().apply(
        lambda sr: pd.Series([sr.dropna().index[0], sr.dropna().index[-1]], index=['begin_date', 'end_date'])).T
    df_date_stat['tradeday_count'] = (~df_signal.unstack().isna()).sum()
    df_date_stat['trade_duration'] = (df_date_stat['end_date'] - df_date_stat['begin_date']).dt.days / 365
    return df_date_stat


def analyse_rtn(df_rtn):
    """分析交易的收益"""
    df_total_rtn = df_rtn.unstack().sum().unstack(0)
    df_daily_rtn = df_rtn['all'].unstack().resample('D').sum()
    return df_total_rtn, df_daily_rtn


def analyse_test_res(df_rtn, df_trade_stat):
    df_rtn = df_rtn.mean(axis=1).resample('D').sum()
    sum_metric = _calc_metric(df_rtn, df_rtn.shape[0])
    sum_metric['long_days_median'] = df_trade_stat['long_trade_days_median'].mean()
    sum_metric['short_days_median'] = df_trade_stat['short_trade_days_median'].mean()
    return sum_metric


def analyse_product_test_res(df_rtn):
    def _transfer_metric(sr):
        sr = sr.dropna().resample('D').sum()
        sr = _calc_metric(sr, sr.shape[0])
        sr['begin_date'], sr['end_date'] = sr.pop('date_range')
        return pd.Series(sr)

    df_metric = df_rtn.apply(_transfer_metric).T
    return df_metric


def _calc_metric(df_rtn, trade_days):
    max_drawdown = (df_rtn.cumsum().cummin() - df_rtn.cumsum()).min()
    total_rtn = df_rtn.sum()
    ann_rtn = total_rtn / trade_days * 365 if trade_days > 0 else np.nan
    calmar = abs(ann_rtn / max_drawdown) if max_drawdown != 0 else np.nan
    daily_rtn, daily_std = df_rtn.mean(), df_rtn.std()
    sharpe_ratio = np.sqrt(252) * daily_rtn / daily_std if daily_std > 0 else np.nan
    metric_dict = {
        'total_rtn': total_rtn,
        'ann_rtn': ann_rtn,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'sharpe_ratio': sharpe_ratio,
        'daily_rtn': daily_rtn,
        'daily_std': daily_std,
        'trade_days': trade_days,
        'date_range': [df_rtn.index[0], df_rtn.index[-1]] if not df_rtn.empty else [None, None],
    }
    return metric_dict
