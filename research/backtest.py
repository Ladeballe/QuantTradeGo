import pandas as pd


# 回测部分
def _calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio):
    df_pos_rtn = df_signal * df_fwd_rtn
    df_pos_chg = df_signal.unstack().diff(1).stack()
    df_pos_cost = (df_pos_chg.abs() * cost_ratio).reindex(df_pos_rtn.index)
    df_pos_net_rtn = df_pos_rtn - df_pos_cost
    return df_pos_net_rtn


def calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio=0.0003):
    df_pos_rtn = _calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio)
    df_pos_rtn_long = _calc_rtn_by_signal(df_signal.mask(df_signal < 0, 0), df_fwd_rtn, cost_ratio)
    df_pos_rtn_short = _calc_rtn_by_signal(df_signal.mask(df_signal > 0, 0), df_fwd_rtn, cost_ratio)
    df_pos_rtn_merge = pd.concat([df_pos_rtn, df_pos_rtn_long, df_pos_rtn_short],
                                 keys=['all', 'long', 'short'], axis=1)
    return df_pos_rtn_merge