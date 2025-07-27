import traceback
from typing import List, Dict, Callable, Union

import pandas as pd
import numpy as np
from matplotlib import gridspec as gs, pyplot as plt
from lightweight_charts import Chart

from ..data import get_factor, load_sub_type
from ..enums import SUB_TYPE_DICT


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SigTestPlotter:
    def __init__(self) -> None:
        self.load_price_data_flag = False
        self.load_figure_config_flag = False

        self.df_rtn = None

    def load_price_data(self, price_fac_name='wgt_close', price_fac_lib_name='qiugang.factor_UNI_daily') -> None:
        if self.load_price_data_flag:
            pass
        else:
            self.price_fac_name = price_fac_name
            self.price_fac_lib_name = price_fac_lib_name
            self.load_price_data_flag = True

    def load_param_data(
            self,
            products: Union[List[str], None] = None, begin_date: Union[str, None] = None,
            end_date: Union[str, None] = None,
            **kwargs: Union[pd.DataFrame, Dict]
    ) -> None:
        """加载数据

        Parameters
        ----------
        products
        begin_date
        end_date
        kwargs

        Examples
        --------
        An example implemented to get variables: 'df_signal', 'df_factor', 'df_rtn'.
        The method is a more general implemented format.

        >>> def load_param_data(
        ...         self,
        ...         products: Union[List[str], None]=None, begin_date: Union[str, None]=None, end_date: Union[str, None]=None,
        ...         df_signal: pd.DataFrame, df_factor: pd.DataFrame, df_rtn: pd.DataFrame
        ... ) -> None:
        ...     self.products = products if products is not None else self.df_rtn.columns.tolist()
        ...     self.products = pd.Index(get_product_list()).intersection(self.products).tolist()
        ...     self.begin_date = begin_date if begin_date is not None else self.df_rtn.index[0]
        ...     self.end_date = end_date if end_date is not None else self.df_rtn.index[-1]
        ...
        ...     self.df_signal = df_signal
        ...     self.df_factor = df_factor
        ...     self.df_rtn = df_rtn
        ...
        ...     self.product_num = len(self.products)

        Returns
        -------

        """
        self.df_name_list = list(kwargs.keys())
        if "df_rtn" not in self.df_name_list:
            raise ValueError("Missing important parameter: 'df_rtn'")
        for attr_name, df in kwargs.items():  # 用以设置dataframe或其他变量
            setattr(self, attr_name, df)

        self.products = products if products is not None else self.df_rtn.columns.tolist()
        self.product_num = len(self.products)
        self.begin_date = begin_date if begin_date is not None else self.df_rtn.index[0]
        self.end_date = end_date if end_date is not None else self.df_rtn.index[-1]

    def load_figure_config_v0(
            self,
            ax_length0=6, ax_length1=6, ax_width=10,
            font_size=15, color='red', fontweight='bold'
    ) -> None:
        if self.load_figure_config_flag:
            pass
        else:
            self.ax_length0, self.ax_length1, self.ax_width = ax_length0, ax_length1, ax_width
            self.title_fontdict = {'size': font_size, 'color': color, 'fontweight': fontweight}
            self.load_figure_config_flag = True

    def load_figure_config_v1(
            self, ax_length=6, ax_width=10,
            font_size=15, color='red', fontweight='bold'
    ):
        if self.load_figure_config_flag:
            pass
        else:
            self.ax_length, self.ax_width = ax_length, ax_width
            self.title_fontdict = {'size': font_size, 'color': color, 'fontweight': fontweight}
            self.load_figure_config_flag = True

    def load_plotter_config(self, *args: Callable[..., None]) -> None:
        self.callable_list = list(args)
        self.callable_num = len(self.callable_list)

    def load_title_output_path_config(self, title_text, output_path):
        self.title_text = title_text
        self.output_path = output_path

    def config_v0(self):  # FIXME: 设置图像配置
        """
        Examples
        --------
        历史实现方式
        >>> def config(self):
        ...     # 加载各项预加载项
        ...     self.load_price_data()
        ...     self.load_figure_config()
        ...     # 构建figure和axes
        ...     self.axes_num = self.product_num + (self.callable_num - 1)
        ...     self.y = int(np.min([np.sqrt(self.axes_num), 7]))
        ...     self.x = np.ceil(self.axes_num / self.y).astype(int)
        ...     y_pos, x_pos = np.meshgrid(np.arange(0, self.y, 1), np.arange(0, self.x, 1))
        ...
        ...     self.gs_pos_iter = iter(zip(x_pos.reshape((-1,)), y_pos.reshape((-1,))))
        ...
        ...     self.fig = plt.figure(figsize=(self.ax_width * self.y, self.ax_length * self.x))
        ...     self.gs = gs.GridSpec(self.x, self.y)
        ...     # 加载数据: 价格, 因子, 波动率等
        ...     self.df_price = read_factor_to_cache(self.price_fac_name, self.begin_date, self.end_date,
        ...                                          self.price_fac_lib_name, is_return_multi_index=False)

        Returns
        -------

        """
        # 加载各项预加载项
        self.load_price_data()
        self.load_figure_config_v0()
        # 构建figure和axes
        self.y = int(np.min([np.sqrt(self.product_num + self.callable_num), 9]))
        self.x0 = np.ceil(self.product_num / self.y).astype(int)
        self.x1 = np.ceil((self.callable_num - 1) / self.y).astype(int)
        self.fig = plt.figure(
            figsize=(self.ax_width * self.y,
                     self.ax_length0 * self.x0 + self.ax_length1 * self.x1 + min(1, int(0.2 * self.ax_length0))))

        org_gs = gs.GridSpec(self.ax_length0 * self.x0 + self.ax_length1 * self.x1, 1,
                             left=0.015, right=0.985, bottom=0.025, top=0.975)
        self.gs0 = gs.GridSpecFromSubplotSpec(
            self.x0, self.y, org_gs[:self.ax_length0 * self.x0], wspace=0.1, hspace=0.15)
        y_pos0, x_pos0 = np.meshgrid(np.arange(0, self.y, 1), np.arange(0, self.x0, 1))
        self.gs_pos_iter0 = iter(zip(x_pos0.reshape((-1,)), y_pos0.reshape((-1,))))
        self.gs1 = gs.GridSpecFromSubplotSpec(
            self.x1, self.y, org_gs[self.ax_length0 * self.x0 + min(1, int(0.2 * self.ax_length0)):],
            wspace=0.1, hspace=0.15)
        y_pos1, x_pos1 = np.meshgrid(np.arange(0, self.y, 1), np.arange(0, self.x1, 1))
        self.gs_pos_iter1 = iter(zip(x_pos1.reshape((-1,)), y_pos1.reshape((-1,))))

    def config_v1(self):  # FIXME: 设置图像配置
        """
        Returns
        -------
        """
        # 加载各项预加载项
        self.load_price_data()
        self.load_figure_config_v1()
        # 构建figure和axes
        self.y = int(np.min([np.sqrt(self.callable_num), 9]))
        self.x = np.ceil(self.callable_num / self.y).astype(int)
        self.fig = plt.figure(
            figsize=(self.ax_width * self.y, self.ax_length * self.x / 0.95))

        self.gs = gs.GridSpec(self.x, self.y, left=0.015, right=0.985, bottom=0.025, top=0.95, figure=self.fig)
        y_pos, x_pos = np.meshgrid(np.arange(0, self.y, 1), np.arange(0, self.x, 1))
        self.gs_pos_iter = iter(zip(x_pos.reshape((-1,)), y_pos.reshape((-1,))))

    def process_data(self) -> None:
        for df_name in self.df_name_list:
            # 设置数据
            setattr(self, df_name, getattr(self, df_name).loc[pd.IndexSlice[self.begin_date: self.end_date]])

    def plot_v0(self) -> None:
        price_plot_func = self.callable_list[0]
        for i, product in enumerate(self.products):
            gs_pos = next(self.gs_pos_iter0)
            gs_sub = self.gs0[gs_pos]  # 遍历grid_spec
            ax = price_plot_func(self, gs_sub, product)
            if i == self.y // 2 + self.y % 2 - 1:  # 获取位于中间的ax
                self.title_ax = ax

        for plot_func in self.callable_list[1:]:
            gs_pos = next(self.gs_pos_iter1)
            gs_sub = self.gs1[gs_pos]  # 遍历grid_spec
            plot_func(self, gs_sub)

    def plot_v1(self) -> None:
        for plot_func in self.callable_list:
            gs_pos = next(self.gs_pos_iter)
            gs_sub = self.gs[gs_pos]  # 遍历grid_spec
            plot_func(self, gs_sub)

    def set_title(self, title) -> None:
        #title = title + f"\n{self.title_ax.get_title()}"
        self.fig.suptitle(title)

    def save_fig(self, fig_filename):
        # self.fig.tight_layout()
        self.fig.savefig(fig_filename)
        plt.close(self.fig)


def plot_price_pnl_fac(plotter, gs_sub, product):
    ax = plotter.fig.add_subplot(gs_sub)

    s_price = plotter.df_price[product].ffill().bfill()
    s_rtn = s_price / s_price.iloc[0]
    s_rtn.plot(ax=ax, rot=30, fontsize=8)

    s_cum_rtn = plotter.df_rtn['all'] \
                    .unstack()[product] \
                    .cumsum() + 1
    s_cum_rtn.plot(ax=ax.twinx(), color='red', style='--', rot=30, grid=True, fontsize=8)

    s_trade_phase = pd.DataFrame(plotter.df_trade_phase[product])
    s_trade_phase.apply(lambda sr: ax.axvspan(
        sr['begin_date'], sr['end_date'],
        facecolor={1: 'r', -1: 'b'}[sr['sig']], alpha=0.2
    ), axis=1)

    ax.set_title(f"{product} {100 * plotter.df_total_rtn.loc[product, 'all']:.1f}%", fontdict=plotter.title_fontdict)
    return ax


def plot_price_rtn_fac(plotter, gs_sub, product):
    ax = plotter.fig.add_subplot(gs_sub)

    s_price = plotter.df_price[product].ffill().bfill()
    s_rtn = s_price / s_price.iloc[0]
    s_rtn.plot(ax=ax, rot=30, fontsize=8)

    s_cum_rtn = plotter.df_rtn['all'] \
                    .unstack()[product] \
                    .cumsum() + 1
    s_cum_rtn.plot(ax=ax, color='red', style='--', rot=30, grid=True, fontsize=8)

    s_trade_phase = pd.DataFrame(plotter.df_trade_phase[product])
    s_trade_phase.apply(lambda sr: ax.axvspan(
        sr['begin_date'], sr['end_date'],
        facecolor={1: 'r', -1: 'b'}[int(sr['sig'] > 0) - int(sr['sig'] < 0)], alpha=0.2
    ), axis=1)

    ax.set_title(f"{product} {100 * plotter.df_total_rtn.loc[product, 'all']:.1f}%", fontdict=plotter.title_fontdict)
    return ax


def plot_price_rtn_pos(plotter, gs_sub, product):  # FIXME: 绘制价格+pos
    gs_sub_sub = gs.GridSpecFromSubplotSpec(4, 1, gs_sub, hspace=0)  # 设置hspace=0, 使两张图之间没有纵向间距
    ax0 = plotter.fig.add_subplot(gs_sub_sub[:3])
    ax1 = plotter.fig.add_subplot(gs_sub_sub[3])
    try:
        s_price = plotter.df_price[product].ffill().bfill()
        s_rtn = s_price / s_price.iloc[0]
        s_rtn.plot(ax=ax0, rot=30, fontsize=8)
    
        s_cum_rtn = plotter.df_rtn['all'] \
                        .unstack()[product] \
                        .cumsum() + 1
        s_cum_rtn.plot(ax=ax0.twinx(), color='red', style='--', rot=30, grid=True, fontsize=8)
        xlim0, xlim1 = s_cum_rtn.index[0], s_cum_rtn.index[-1]
    
        s_trade_phase = pd.DataFrame(plotter.df_trade_phase[product])
        s_trade_phase.apply(lambda sr: ax0.axvspan(
            sr['begin_date'], sr['end_date'],
            facecolor={1: 'r', -1: 'b'}[int(sr['sig'] > 0) - int(sr['sig'] < 0)], alpha=0.2
        ), axis=1)
        s_pos = plotter.df_pos[product]
        s_pos.index -= pd.Timedelta(hours=21)
        s_pos.reindex(s_cum_rtn.index).ffill().plot(ax=ax1, rot=30, fontsize=8, color='orange', grid=True)
        # ax0.set_xticklabels([])
        ax0.set_xlim(xlim0, xlim1)
        ax1.set_xlim(xlim0, xlim1)
    
        ax0.set_title(f"{product} {100 * plotter.df_total_rtn.loc[product, 'all']:.1f}%", fontdict=plotter.title_fontdict)
    except:
        traceback.print_exc()
    return ax0


def plot_ann_trades(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_trade_stat['long_trade_ann_num'] \
        .rename('long') \
        .plot(ax=ax, kind='bar', rot=30, title='年均交易次数', fontsize=8)
    (plotter.df_trade_stat['short_trade_ann_num'] * -1) \
        .rename('short') \
        .plot(ax=ax, kind='bar', color='green', grid=True, rot=30, fontsize=8)


def plot_trade_days(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_trade_stat['long_trade_days_mean'] \
        .rename('long') \
        .plot(ax=ax, kind='bar', rot=30, title='持仓天数', fontsize=8)
    (plotter.df_trade_stat['short_trade_days_mean'] * -1) \
        .rename('short') \
        .plot(ax=ax, kind='bar', color='green', grid=True, rot=30, fontsize=8)


def plot_rtn_line(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_rtn['all'] \
        .unstack() \
        .sum(axis=1) \
        .cumsum() \
        .plot(ax=ax, label='long+short', rot=30, grid=True, legend=True, title='全品种费后PnL', fontsize=8)


def plot_total_rtn_by_product(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_total_rtn['all'].plot(
        ax=ax, kind='bar', rot=30, grid=True, title='费后,单边按万3计算', fontsize=8, legend=False)


def plot_annual_rtn_by_product(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    (plotter.df_total_rtn['all'] / plotter.df_trade_stat['years']).plot(
        ax=ax, kind='bar', rot=30, grid=True, title='费后年化', fontsize=8, legend=False)


def plot_annual_rtn(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_rtn = plotter.df_rtn['all'] \
        .unstack() \
        .sum(axis=1)
    df_rtn.groupby(df_rtn.index.year) \
        .sum() \
        .rename('年度费后收益') \
        .plot(ax=ax, kind='bar', rot=30, title="分年度费后收益", grid=True, legend=True, fontsize=8)


def plot_long_short_rtn(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_rtn['long'] \
        .unstack() \
        .sum(axis=1) \
        .cumsum() \
        .plot(ax=ax, rot=30, grid=True, legend=True, label='long', title='全品种费后 long/short', fontsize=8)
    plotter.df_rtn['short'] \
        .unstack() \
        .sum(axis=1) \
        .cumsum() \
        .plot(ax=ax, rot=30, grid=True, legend=True, label='short', fontsize=8)


def plot_sector_total_rtn(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_rtn = plotter.df_rtn['all'].reset_index()
    df_rtn['sector'] = df_rtn['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    # df_rtn = df_rtn.groupby(['trade_day', 'sector'])
    df_rtn = df_rtn.groupby(['date', 'sector'])['all'] \
        .sum() \
        .unstack() \
        .cumsum()
    df_rtn.plot(ax=ax, rot=30, grid=True, legend=True, title='全品种费后 分板块', fontsize=8)


def plot_sector_long_rtn(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_rtn = plotter.df_rtn['long'].reset_index()
    df_rtn['sector'] = df_rtn['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    # df_rtn = df_rtn.groupby(['trade_day', 'sector']) \
    df_rtn = df_rtn.groupby(['date', 'sector'])['long'] \
        .sum() \
        .unstack() \
        .cumsum()
    df_rtn.plot(ax=ax, rot=30, grid=True, legend=True, title='全品种费后 分板块long', fontsize=8)


def plot_sector_short_rtn(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_rtn = plotter.df_rtn['short'].reset_index()
    df_rtn['sector'] = df_rtn['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    # df_rtn = df_rtn.groupby(['trade_day', 'sector']) \
    df_rtn = df_rtn.groupby(['date', 'sector'])['short'] \
        .sum() \
        .unstack() \
        .cumsum()
    df_rtn.plot(ax=ax, rot=30, grid=True, legend=True, title='全品种费后 分板块short', fontsize=8)


def plot_sector_long_num(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_signal = plotter.df_signal.reset_index()
    df_signal['sector'] = df_signal['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    df_sector_signal_count = df_signal \
        .groupby(['date', 'signal', 'sector']) \
        .count()
    df_sector_signal_count = df_sector_signal_count.loc[pd.IndexSlice[:, 1, :]] \
        .unstack() \
        .fillna(0) \
        .droplevel(0, axis=1)
    df_sector_signal_count.plot(ax=ax, kind='area', title=f'多头分板块品种数量', fontsize=8)


def plot_sector_short_num(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_signal = plotter.df_signal.reset_index()
    df_signal['sector'] = df_signal['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    df_sector_signal_count = df_signal \
        .groupby(['date', 'signal', 'sector']) \
        .count()
    df_sector_signal_count = df_sector_signal_count.loc[pd.IndexSlice[:, -1, :]] \
        .unstack() \
        .fillna(0) \
        .droplevel(0, axis=1)
    df_sector_signal_count.plot(ax=ax, kind='area', title=f'空头分板块品种数量', fontsize=8)


def plot_long_short_num(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_signal = plotter.df_signal.reset_index()
    df_signal['sector'] = df_signal['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    df_signal_num = plotter.df_signal \
        .reset_index() \
        .groupby(['date', 'signal']) \
        .count() \
        .unstack() \
        .droplevel(0, axis=1) \
        .fillna(0)
    df_signal_num['diff'] = df_signal_num[1] - df_signal_num[-1]
    df_signal_num[[-1, 1, 'diff']].plot(ax=ax, title=f'多空品种数量', grid=True, fontsize=8)


def plot_sector_net_num(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    df_signal = plotter.df_signal.reset_index()
    df_signal['sector'] = df_signal['symbol'].apply(lambda x: SUB_TYPE_DICT[x])
    df_signal_sector = df_signal \
        .groupby(['date', 'sector', 'signal']) \
        .count() \
        .unstack() \
        .droplevel(0, axis=1) \
        .fillna(0)
    df_signal_sector_diff = df_signal_sector[1] - df_signal_sector[-1]
    df_signal_sector_diff \
        .unstack() \
        .plot(ax=ax, title=f'多空品种数量 分行业', grid=True, legend=True, fontsize=8)


def plot_total_trade_num(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_win_ratio['trade_num_total'].plot(ax=ax, title='总交易次数', kind='bar', rot=30, fontsize=8)


def plot_win_ratio(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_win_ratio['all'].plot(ax=ax, title='总胜率', kind='bar', rot=30, fontsize=8)


def plot_gain_loss_ratio(plotter, gs_sub):
    ax = plotter.fig.add_subplot(gs_sub)
    plotter.df_plr_ratio['all'].plot(ax=ax, title='总胜率', kind='bar', rot=30, fontsize=8)


func_list = [
    plot_ann_trades, plot_trade_days, plot_total_rtn_by_product, plot_annual_rtn_by_product,
    plot_rtn_line, plot_annual_rtn, plot_long_short_rtn, plot_sector_total_rtn,
    plot_sector_long_rtn, plot_sector_short_rtn, plot_sector_long_num, plot_sector_short_num, plot_long_short_num,
    plot_sector_net_num, plot_total_trade_num, plot_win_ratio, plot_gain_loss_ratio
]
func_list2 = [
    plot_ann_trades, plot_trade_days, plot_total_rtn_by_product, plot_annual_rtn_by_product,
    plot_rtn_line, plot_annual_rtn, plot_long_short_rtn, plot_sector_total_rtn,
    plot_sector_long_rtn, plot_sector_short_rtn
]
func_list3 = [
    plot_ann_trades, plot_trade_days, plot_total_rtn_by_product, plot_annual_rtn_by_product,
    plot_rtn_line, plot_annual_rtn, plot_long_short_rtn, plot_sector_total_rtn,
    plot_sector_long_rtn, plot_sector_short_rtn, plot_sector_long_num, plot_sector_short_num, plot_long_short_num,
    plot_sector_net_num, plot_win_ratio, plot_gain_loss_ratio
]


""" lightweight_charts 动态绘图部分 """


def lwc_marker_buy_sell_close(chart, df_signal):
    df_signal_draw = pd.concat([
        df_signal,
        df_signal.diff().abs().cumsum().fillna(0).rename('signal_group')],
        axis=1
    )
    df_signal_marker = df_signal_draw.reset_index().groupby('signal_group').agg(
        {'t': ['first', 'last'], 'signal': 'first'})
    df_signal_marker.columns = ['begin_date', 'end_date', 'signal']

    for i, sr in df_signal_marker[df_signal_marker['signal']!=0].iterrows():
        if sr['signal'] == 1:
            chart.marker(sr['begin_date'].to_pydatetime(), "below", "arrow_up", "red", "l")
            chart.marker(sr['end_date'].to_pydatetime(), "above", "arrow_down", "purple", "lc")
        elif sr['signal'] == -1:
            chart.marker(sr['begin_date'].to_pydatetime(), "above", "arrow_down", "blue", "s")
            chart.marker(sr['end_date'].to_pydatetime(), "below", "arrow_up", "purple", "sc")
