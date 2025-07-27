import datetime

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt, gridspec as gs
import seaborn as sns

from test_func import data
from test_func.util import convert_frame_to_stack
from .base_plot import plot_seg_bar


idx = pd.IndexSlice

""" cross_test_plot 因子截面测试绘图部分 """


def calc_turn(sr):
    df = pd.concat([sr, sr.shift(1)], axis=1)

    def _func(sr):
        try:
            ratio = 1 - len(set(sr.iloc[0]) & set(sr.iloc[1])) / len(sr.iloc[1])
            return ratio
        except:
            return np.nan

    return df.apply(_func, axis=1)


def calc_ic(sr_ic, ic_stats_rolling_days):
    idx = sr_ic.index
    ma = sr_ic.rolling(ic_stats_rolling_days).mean()
    std = sr_ic.rolling(ic_stats_rolling_days).std()
    return idx, ma - 2 * std, ma + 2 * std


def cross_test_plot(
        df_fac, fac_name, begin_date, end_date,
        bins=20, demean_bool=True, ic_stats_rolling_days=10,
        time_str_list=[f'{h:0>2}:00' for h in range(0, 24, 4)],
        fwd_rtn_names=['fwd_rtn_1d', 'fwd_rtn_12h', 'fwd_rtn_6h'],
        res_path=r'D:\research\CRYPTO_vp_fac\results'
):
    plot_num = 6

    fig = plt.figure(figsize=(len(time_str_list) * 5, len(fwd_rtn_names) * (plot_num * 5)))
    gs0 = gs.GridSpec(len(fwd_rtn_names) * plot_num, len(time_str_list))

    for gs0_xloc, fwd_rtn_name in enumerate(fwd_rtn_names):
        for gs0_yloc, time_str in enumerate(time_str_list):
            t0 = pd.Timestamp.now()
            df_group_rtn = df_fac.set_index('date').at_time(time_str).groupby('factor_group')[fwd_rtn_name].mean()
            if demean_bool:
                df_group_rtn = df_group_rtn - df_group_rtn.mean()
            ax0 = fig.add_subplot(gs0[gs0_xloc * plot_num + 0, gs0_yloc])
            ax0.set_title(
                f'{fwd_rtn_name} {time_str}\n'
                f'{0:>2d}: {df_group_rtn[0] * 1000:.2f}‰, {bins - 1:>2d}: {df_group_rtn[bins - 1] * 1000:.2f}‰'
            )
            df_group_rtn.plot(kind='bar', ax=ax0)
            t1 = pd.Timestamp.now()
            print(t1 - t0)

            df_pnl = df_fac.set_index('date').at_time(time_str).groupby(['date', 'factor_group'])[
                fwd_rtn_name].mean().unstack().cumsum()
            if demean_bool:
                df_pnl = df_pnl.sub(df_pnl.mean(axis=1), axis=0)
            ax1 = fig.add_subplot(gs0[gs0_xloc * plot_num + 1, gs0_yloc])
            ax1.set_title(
                f'{0:>2d}: {df_pnl.dropna().iloc[-1, 0] * 100:.2f}%, {bins - 1:>2d}: {df_pnl.dropna().iloc[-1, -1] * 100:.2f}%')
            df_pnl[[0, 1, bins - 2, bins - 1]].plot(ax=ax1)
            if df_group_rtn[0] > df_group_rtn[bins - 1]:
                (df_pnl[0] - df_pnl[bins - 1]).plot(ax=ax1, linestyle='--', label=f'{0}-{bins - 1}')
            else:
                (df_pnl[bins - 1] - df_pnl[0]).plot(ax=ax1, linestyle='--', label=f'{bins - 1}-{0}')
            ax1.legend()
            t2 = pd.Timestamp.now()
            print(t2 - t1)

            df_symbol_count = df_fac.set_index('date').at_time(time_str).groupby(['date', 'factor_group'])[
                'symbol'].count().unstack()
            ax2 = fig.add_subplot(gs0[gs0_xloc * plot_num + 2, gs0_yloc])
            df_symbol_count[[0, 1, bins - 2, bins - 1]].plot(ax=ax2)
            t3 = pd.Timestamp.now()
            print(t3 - t2)

            # sr_ic = df_fac.set_index('date').at_time(time_str).groupby('date')[
            #             ['raw_factor', fwd_rtn_name]].corr().unstack().iloc[:, 1]
            # sr_rank_ic = df_fac.set_index('date').at_time(time_str).groupby('date')[['raw_factor', fwd_rtn_name]].corr(
            #     method='spearman').unstack().iloc[:, 1]
            # ax3 = fig.add_subplot(gs0[gs0_xloc * plot_num + 3, gs0_yloc])
            # ax3.fill_between(*calc_ic(sr_ic, ic_stats_rolling_days), color='b', alpha=0.1)
            # ax3.fill_between(*calc_ic(sr_rank_ic, ic_stats_rolling_days), color='r', alpha=0.1)
            # sr_ic.rolling(ic_stats_rolling_days).mean().plot(ax=ax3, c='b', label='ic')
            # sr_rank_ic.rolling(ic_stats_rolling_days).mean().plot(ax=ax3, c='r', label='ic')
            # ic_mean = sr_ic.mean()
            # ic_std = sr_ic.std()
            # rank_ic_mean = sr_rank_ic.mean()
            # ir = ic_mean / ic_std
            # ic_t_stat, ic_t_pval = sp.stats.ttest_1samp(sr_ic.dropna(), 0)
            # ax3.set_title(
            #     f"ic_mean:{ic_mean:.2f}, ic_std:{ic_std:.2f}, rank_ic_mean:{rank_ic_mean:.2f}\n"
            #     f"ir:{ir:.2f}, ic_t_stat:{ic_t_stat:.2f}, ic_t_pval:{ic_t_pval:.2f}"
            # )
            # t4 = pd.Timestamp.now()
            # print(t4 - t3)

            ax4 = fig.add_subplot(gs0[gs0_xloc * plot_num + 4, gs0_yloc])
            df_turn = df_fac.set_index('date').at_time(time_str) \
                .groupby(['date', 'factor_group'])['symbol'].apply(lambda sr: list(sr)) \
                .unstack()[[0, bins - 1]] \
                .apply(calc_turn)
            df_turn.plot(ax=ax4)
            ax4.set_title(f'{0}:{df_turn[0].mean():.2f}, {bins - 1}:{df_turn[bins - 1].mean():.2f}')
            t5 = pd.Timestamp.now()
            print(t5 - t4)

            ax5 = fig.add_subplot(gs0[gs0_xloc * plot_num + 5, gs0_yloc])
            df_pnl = df_fac.set_index('date').at_time(time_str).groupby(['date', 'factor_group'])[
                fwd_rtn_name].mean().unstack().iloc[-20:].cumsum()
            if demean_bool:
                df_pnl = df_pnl.sub(df_pnl.mean(axis=1), axis=0)
            df_pnl[[0, 1, bins - 2, bins - 1]].plot(ax=ax5)
            if df_group_rtn[0] > df_group_rtn[bins - 1]:
                (df_pnl[0] - df_pnl[bins - 1]).plot(ax=ax5, linestyle='--', label=f'{0}-{bins - 1}')
            else:
                (df_pnl[bins - 1] - df_pnl[0]).plot(ax=ax5, linestyle='--', label=f'{bins - 1}-{0}')
            ax5.legend()
            t6 = pd.Timestamp.now()
            print(t6 - t5)

    fig.tight_layout()

    file_name = fr"{res_path}\{fac_name}_{begin_date}-{end_date}.png"
    print(f"- {pd.Timestamp.now()} save fig: {file_name}")
    fig.savefig(file_name)
    plt.close(fig)


def cross_test_plot_20(
        df_fac, fac_name, begin_date, end_date,
        bins=20, demean_bool=True, ic_stats_rolling_days=10,
        time_str_list=[f'{h:0>2}:00' for h in range(0, 24, 4)],
        fwd_rtn_names=['fwd_rtn_1d', 'fwd_rtn_12h', 'fwd_rtn_6h'],
        res_path=r'D:\research\CRYPTO_vp_fac\results'
):
    fig = plt.figure(figsize=(
        (len(fwd_rtn_names) + 1 + 1) * 5,
        (1 + len(time_str_list)) * 5
    ))
    gs0 = gs.GridSpec(1 + len(time_str_list), (len(fwd_rtn_names) + 1 + 1), figure=fig)

    df_fac = df_fac.take(np.concatenate([pd.DatetimeIndex(df_fac['date']).indexer_at_time(time_str) for time_str in time_str_list]))
    df_fac['datetime'] = df_fac['date']
    df_fac['time'] = df_fac['date'].dt.time
    df_fac['date'] = df_fac['date'].dt.date
    if demean_bool:
        df_fac[fwd_rtn_names] = df_fac[fwd_rtn_names] - df_fac[fwd_rtn_names].groupby(df_fac['datetime']).transform('mean')

    # 1 * len(fwd_rtn_names)
    df_group_rtn = df_fac.groupby(['time', 'factor_group'])[fwd_rtn_names].mean().unstack()
    gs00 = gs.GridSpecFromSubplotSpec(1, len(fwd_rtn_names), subplot_spec=gs0[0, 0:len(fwd_rtn_names)])
    for i, fwd_rtn_name in enumerate(fwd_rtn_names):
        ax = fig.add_subplot(gs00[0, i])
        plot_seg_bar(df_group_rtn[fwd_rtn_name], ax)

    # len(time_str_list) * len(fwd_rtn_names)
    df_pnl = df_fac.groupby(['time', 'factor_group', 'date'])[fwd_rtn_names].mean()
    df_pnl = df_pnl.groupby(['time', 'factor_group']).transform('cumsum')
    gs01 = gs.GridSpecFromSubplotSpec(
        len(time_str_list), len(fwd_rtn_names),
        subplot_spec=gs0[1:1 + len(time_str_list), 0:len(fwd_rtn_names)])
    for i, time_str in enumerate(time_str_list):
        for j, fwd_rtn_name in enumerate(fwd_rtn_names):
            ax = fig.add_subplot(gs01[i, j])
            df_pnl.loc[
                idx[
                    datetime.time(*[int(ts) for ts in time_str.split(':')]),
                    [0, 1, 2, bins - 3, bins - 2, bins - 1]
                ],
                fwd_rtn_name
            ].unstack(level=1).droplevel(0).plot(
                ax=ax, title=f"{time_str}-{fwd_rtn_name} pnl"
            )

    # 1 * len(time_str_list)
    df_symbol_count = df_fac.groupby(['date', 'time', 'factor_group'])['symbol'].count().unstack(1).unstack()
    gs02 = gs.GridSpecFromSubplotSpec(
        len(time_str_list), 1,
        subplot_spec=gs0[1:1 + len(time_str_list), len(fwd_rtn_names)])
    for i, time_str in enumerate(time_str_list):
        ax = fig.add_subplot(gs02[i, 0])
        df_symbol_count[datetime.time(*[int(ts) for ts in time_str.split(':')])]\
            [[0, 1, 2, bins - 3, bins - 2, bins - 1]]\
            .plot(title=time_str, ax=ax)

    # 1 * len(time_str_list)
    df_turn = df_fac.groupby(['time', 'factor_group', 'date'])['symbol'].apply(lambda sr: list(sr))\
        .loc[idx[:, [0, 1, 2, bins - 3, bins - 2, bins - 1], :]
    ].groupby('factor_group').apply(calc_turn).droplevel(0).unstack(0).unstack(0)
    gs03 = gs.GridSpecFromSubplotSpec(
        len(time_str_list), 1,
        subplot_spec=gs0[1:1 + len(time_str_list), len(fwd_rtn_names) + 1])
    for i, time_str in enumerate(time_str_list):
        ax = fig.add_subplot(gs03[i, 0])
        ax.set_ylim([0, 1])
        df_turn[datetime.time(*[int(ts) for ts in time_str.split(':')])]\
            .plot(title=time_str, ax=ax)

    fig.tight_layout()

    file_name = fr"{res_path}\{fac_name}_{begin_date}-{end_date}.png"
    print(f"- {pd.Timestamp.now()} save fig: {file_name}")
    fig.savefig(file_name)
    plt.close(fig)
