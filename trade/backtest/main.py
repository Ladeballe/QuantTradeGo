import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
import sqlalchemy

from v1.strategy_1 import CrossSecStrategy


engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')


def read_history_data(engine, begin_date, end_date, symbols):
    symbols = ','.join([f"'{symbol}'" for symbol in symbols])
    ts_0 = int(begin_date.timestamp() * 1000)
    ts_1 = int(end_date.timestamp() * 1000)
    df = pd.read_sql(
        f"select * from bnc_kline_15m where ts0 > {ts_0} and ts0 <= {ts_1} and symbol in ({symbols})",  # 使用ts0 < {ts1}确保策略也只使用已有的数据
        engine
    )
    df = df.rename({'t': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'q': 'amt', 'bq': 'amt_buy',
                    'n': 'trade_num'}, axis=1)
    return df


def set_dict_of_data(df, list_basic_fac_names):
    dict_of_data = dict()
    df.index = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '_' + df['symbol']
    for name in list_basic_fac_names:
        dict_of_data[name] = df[['date', 'symbol', name]].rename({name: 'raw_factor'}, axis=1)
    return dict_of_data


def backtest():
    engine = sqlalchemy.create_engine("mysql+pymysql://root:444666@localhost:3306/market_data")
    list_basic_fac_names = ['open', 'high', 'low', 'close', 'amt', 'amt_buy', 'trade_num']
    symbols = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']['bnc-future-universe'] \
        .find_one({'name': 'universe_bnc_perp_future_usdt'}, projection={'symbols': True})['symbols']

    begin_date, end_date = '2024-08-01', '2024-09-10'
    # strategy_config_dict = {
    #     'strategy_name': 'strate
    #     gy0000',
    #     'model_fpath': r'D:\research\cross_sec_strategies\strategy_v10\models\st_1.0.0_fac_1_0000.json',
    #     'fac_fpath': r'D:\research\cross_sec_strategies\strategy_v10\fac_list\fac_v1.0.0_1.xlsx',
    #     'open_time': '00:00',
    #     'close_time': '06:00',
    #     'top_portion': 0.03
    # }
    strategy_config_dict = {
        'strategy_name': 'strategy1200',
        'model_fpath': r'D:\research\cross_sec_strategies\strategy_v10\models\st_1.0.1_fac_1_1200.json',
        'fac_fpath': r'D:\research\cross_sec_strategies\strategy_v10\fac_list\fac_v1.0.1_1.xlsx',
        'open_time': '12:00',
        'close_time': '18:00',
        'top_portion': 0.03
    }

    strategy_name = strategy_config_dict['strategy_name']
    model_fpath, fac_fpath = strategy_config_dict['model_fpath'], strategy_config_dict['fac_fpath']
    open_time, close_time = strategy_config_dict['open_time'], strategy_config_dict['close_time']
    top_portion = strategy_config_dict['top_portion']

    strategy = CrossSecStrategy(strategy_name, model_fpath, fac_fpath, open_time, close_time)
    dict_excess_rtn = dict()
    list_pos_rtn = list()
    list_pos_symbol = list()
    list_df_fac = list()
    list_df_sig = list()
    for date in pd.date_range(begin_date, end_date, freq='D'):
        datetime = pd.Timestamp(date.strftime('%Y-%m-%d ') + strategy.open_time)
        df_hist = read_history_data(engine, datetime - pd.Timedelta(days=3), datetime, symbols)
        dict_of_data = set_dict_of_data(df_hist, list_basic_fac_names)

        # t0 = pd.Timestamp.now()
        strategy.calc_fac(dict_of_data)
        # strategy.calc_fac_mp(dict_of_data)
        # print('time consumed', pd.Timestamp.now() - t0)
        df_sig = strategy.calc_open_sig().to_frame()

        list_df_fac.append(strategy.df_fac)
        list_df_sig.append(df_sig)

        df_sig_sort = df_sig.sort_values(0, ascending=False)
        long_symbols, short_symbols = df_sig_sort.index[:8], df_sig_sort.index[-8:]

        df_future = read_history_data(engine, datetime, datetime + pd.Timedelta(hours=6, minutes=1), symbols)
        df_fwd_rtn = df_future.pivot(index='date', columns='symbol', values='open').agg('log').diff().sum()

        df_excess_rtn = df_fwd_rtn - df_fwd_rtn.mean()

        daily_rtn = df_excess_rtn[long_symbols].mean() - df_excess_rtn[short_symbols].mean()
        daily_pos_excess_rtn = df_excess_rtn[long_symbols.to_list() + short_symbols.to_list()]
        dict_excess_rtn[date] = daily_rtn
        list_pos_rtn.append(daily_pos_excess_rtn.to_list())
        list_pos_symbol.append(daily_pos_excess_rtn.index.to_series().str.strip('10').str[:-4].to_list())
        print(f"{date}, {daily_rtn * 100:.4f}%")
    df_excess_rtn = pd.Series(dict_excess_rtn)

    df_pos_rtn = pd.DataFrame(list_pos_rtn).T * 100
    df_pos_symbol = pd.DataFrame(list_pos_symbol).T
    df_pos_symbol = df_pos_symbol + '\n' + df_pos_rtn.round(2).astype(str) + '%'
    rtn_mean, rtn_std = np.mean(df_pos_rtn.values.reshape(-1,)), np.std(df_pos_rtn.values.reshape(-1,))

    fig, axes = plt.subplots(2, 1, figsize=(20, 20))
    df_excess_rtn.cumsum().plot(ax=axes[0])
    sns.heatmap(data=df_pos_rtn, annot=df_pos_symbol, ax=axes[1], fmt='', cbar=False,
                vmax=rtn_mean + 2 * rtn_std, vmin=rtn_mean - 2 * rtn_std)
    plt.show()

    filepath = f'backtest_{strategy.name}_{begin_date}_{end_date}_{int(pd.Timestamp.now().timestamp())}'
    df_fac = pd.concat(list_df_fac, axis=1)
    df_sig = pd.concat(list_df_sig, axis=1)
    os.mkdir(filepath)
    fig.savefig(f'{filepath}/test.png')
    df_fac.to_excel(f'{filepath}/df_fac.xlsx')
    df_sig.to_excel(f'{filepath}/df_sig.xlsx')


if __name__ == '__main__':
    backtest()
