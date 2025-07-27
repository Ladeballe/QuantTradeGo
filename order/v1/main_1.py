import time
import traceback
from functools import partial

import pandas as pd
import pymongo
import sqlalchemy
from apscheduler.schedulers.background import BackgroundScheduler
import binance_scripts

from test_func.main import iter_flatten_df_fac_info
from v1.strategy_1 import CrossSecStrategy, SimpleCrossSecStrategy


engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')


def read_data(engine, hours, symbols, sheet='bnc_kline_15m'):
    symbols = ','.join([f"'{symbol}'" for symbol in symbols])
    ts_now = int(time.time() * 1000)
    ts_0 = int(ts_now / 1000 // (15 * 60) * (15 * 60) * 1000 - hours * 3600 * 1000)
    df = pd.read_sql(
        f"select * from {sheet} where ts0 > {ts_0} and ts0 <= {ts_now} and "
        f"symbol in ({symbols})",
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


def strategy_open(strategy, client, top_portion, df_exchange_info):
    print(f'{pd.Timestamp.now()}, {strategy.name} open position')
    engine = sqlalchemy.create_engine("mysql+pymysql://root:444666@localhost:3306/market_data")
    list_basic_fac_names = ['open', 'high', 'low', 'close', 'amt', 'amt_buy', 'trade_num']
    symbols = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']['bnc-future-universe']\
                  .find_one({'name': 'universe_bnc_perp_future_usdt'}, projection={'symbols': True})['symbols']
    df = read_data(engine, 72, symbols)
    dict_of_data = set_dict_of_data(df, list_basic_fac_names)

    strategy.calc_fac(dict_of_data)
    df_sig = strategy.calc_open_sig().to_frame()

    long_symbols = df_sig[df_sig[0] > 1 - top_portion].index
    short_symbols = df_sig[df_sig[0] < top_portion].index

    df_ticker = pd.DataFrame(client.futures_ticker()).set_index('symbol')
    sr_quantity = 15 / df_ticker['lastPrice'].astype('float')

    list_order_res = list()
    for long_symbol in long_symbols:
        try:
            quantity = round(sr_quantity[long_symbol], df_exchange_info.loc[long_symbol, 'quantityPrecision'])
            order_res = client.futures_create_order(
                symbol=long_symbol, side='BUY',
                quantity=quantity,
                type='MARKET')
            list_order_res.append(order_res)
            print(order_res)
        except:
            traceback.print_exc()
            print(long_symbol)
    for short_symbol in short_symbols:
        try:
            quantity = round(sr_quantity[short_symbol], df_exchange_info.loc[short_symbol, 'quantityPrecision'])
            order_res = client.futures_create_order(
                symbol=short_symbol, side='SELL',
                quantity=quantity,
                type='MARKET')
            list_order_res.append(order_res)
            print(order_res)
        except:
            traceback.print_exc()
            print(short_symbol)

    strategy.pos = pd.DataFrame(list_order_res)
    strategy.pos_open_bool = True


def strategy_open2(strategy, client, top_portion, df_exchange_info, dict_universe_info, dict_open_info):
    print(f'{pd.Timestamp.now()}, {strategy.name} open position')
    engine = sqlalchemy.create_engine("mysql+pymysql://root:444666@localhost:3306/market_data")
    list_basic_fac_names = ['open', 'high', 'low', 'close', 'amt', 'amt_buy', 'trade_num']
    symbols = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']['bnc-future-universe']\
                  .find_one(dict_universe_info, projection={'symbols': True})['symbols']
    df = read_data(engine, 240, symbols, 'bnc_kline_5m')
    dict_of_data = set_dict_of_data(df, list_basic_fac_names)

    strategy.calc_fac(dict_of_data)
    df_sig = strategy.calc_open_sig().to_frame()

    long_symbols = df_sig[eval(dict_open_info['long'])].index
    short_symbols = df_sig[eval(dict_open_info['short'])].index

    df_ticker = pd.DataFrame(client.futures_ticker()).set_index('symbol')
    sr_quantity = 20 / df_ticker['lastPrice'].astype('float')

    list_order_res = list()
    for long_symbol in long_symbols:
        try:
            quantity = round(sr_quantity[long_symbol], df_exchange_info.loc[long_symbol, 'quantityPrecision'])
            print('long:', long_symbol)
            order_res = client.futures_create_order(
                symbol=long_symbol, side='BUY',
                quantity=quantity,
                type='MARKET')
            list_order_res.append(order_res)
            print(order_res)
        except:
            traceback.print_exc()
            print(long_symbol)
    for short_symbol in short_symbols:
        try:
            quantity = round(sr_quantity[short_symbol], df_exchange_info.loc[short_symbol, 'quantityPrecision'])
            print('short:', short_symbol)
            order_res = client.futures_create_order(
                symbol=short_symbol, side='SELL',
                quantity=quantity,
                type='MARKET')
            list_order_res.append(order_res)
            print(order_res)
        except:
            traceback.print_exc()
            print(short_symbol)

    strategy.pos = pd.DataFrame(list_order_res)
    strategy.pos_open_bool = True


def strategy_close(strategy, client):
    print(f'{pd.Timestamp.now()}, {strategy.name} close position')
    if strategy.pos_open_bool:
        for i, sr_pos in strategy.pos.iterrows():
            symbol = sr_pos['symbol']
            quantity = sr_pos['origQty']
            side = 'BUY' if sr_pos['side'] == 'SELL' else 'SELL'
            client.futures_create_order(
                symbol=symbol, side=side,
                quantity=quantity,
                type='MARKET')
        strategy.pos_open_bool = False


def strategy_scheduler(
        trade_api, list_strategy_config_dict
):
    scheduler = BackgroundScheduler()
    client = binance_scripts.Client(trade_api['api_key'], trade_api['secret_key'])
    dict_exchange_info = client.futures_exchange_info()
    df_exchange_info = pd.DataFrame(dict_exchange_info['symbols']).set_index('symbol')

    for strategy_config_dict in list_strategy_config_dict:
        strategy_name = strategy_config_dict['strategy_name']
        model_fpath, fac_fpath = strategy_config_dict['model_fpath'], strategy_config_dict['fac_fpath']
        open_time, close_time = strategy_config_dict['open_time'], strategy_config_dict['close_time']
        top_portion = strategy_config_dict['top_portion']
        strategy = CrossSecStrategy(strategy_name, model_fpath, fac_fpath, open_time, close_time)

        scheduler.add_job(
            partial(strategy_open, strategy=strategy, client=client, top_portion=top_portion,
                    df_exchange_info=df_exchange_info),
            trigger='cron', hour=strategy.open_time[:2], minute=strategy.open_time[3:], misfire_grace_time=10,
            id=str(hash(strategy_name + '_open')), name=strategy_name + '_open',
        )
        scheduler.add_job(
            partial(strategy_close, strategy=strategy, client=client),
            trigger='cron', hour=strategy.close_time[:2], minute=strategy.open_time[3:], misfire_grace_time=10,
            id=str(hash(strategy_name + '_close')), name=strategy_name + '_close',
        )

    try:
        scheduler.start()
        print(f'{pd.Timestamp.now()}, scheduler started...')
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print('Shutting down.')
    print('done')


if __name__ == "__main__":
    trade_api1 = {
        'api_key': 'YVCzWRztzDUYYfdSXLJBmvzEXFOVSB35RXFz9afEu4zK6kAXDzW4UDWEljbyMCGZ',
        'secret_key': 'o5nOfps70bxlD30X45J4X6wyKjxSM9iw9wNSCeoFRFqSQHRe8Vjx0wXRMUWDPZcM'
    }

    is_run = False
    if is_run:
        # list_strategy_config_dict = [
            # {
            #     'strategy_name': 'strategy1400',
            #     'model_fpath': r'D:\research\cross_sec_strategies\strategy_v10\models\st_1.0.0_fac_1_1400.json',
            #     'fac_fpath': r'D:\research\cross_sec_strategies\strategy_v10\fac_list\fac_v1.0.0_1.xlsx',
            #     'open_time': '14:00',
            #     'close_time': '20:00',
            #     'top_portion': 0.03
            # },
            # {
            #     'strategy_name': 'strategy0200',
            #     'model_fpath': r'D:\research\cross_sec_strategies\strategy_v10\models\st_1.0.0_fac_1_0200.json',
            #     'fac_fpath': r'D:\research\cross_sec_strategies\strategy_v10\fac_list\fac_v1.0.0_1.xlsx',
            #     'open_time': '02:02',
            #     'close_time': '08:00',
            #     'top_portion': 0.03
            # },
        #     {
        #         'strategy_name': 'strategy0800',
        #         'model_fpath': r'D:\research\cross_sec_strategies\strategy_v10\models\st_1.0.0_fac_1_0800.json',
        #         'fac_fpath': r'D:\research\cross_sec_strategies\strategy_v10\fac_list\fac_v1.0.0_1.xlsx',
        #         'open_time': '08:03',
        #         'close_time': '14:00',
        #         'fac_status': ['basic', 'statistics', 'tech'],
        #         'top_portion': 0.03
        #     }
        # ]
        # strategy_scheduler(trade_api1, list_strategy_config_dict)
        strategy_name = 'strategy0800'
        model_fpath = r'D:\research\cross_sec_strategies\Strategy_v20\models\xgbrank_fac_set_v10_0800.json'
        fac_fpath = r'D:\research\cross_sec_strategies\factor_set\strategy_fac_set_v10.xlsx'
        open_time, close_time = '07:56', '14:00'
        top_portion = 0.2
        dict_universe_info = {
            'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k'
        }
        dict_fac_info = {
            'trade': [4]
        }
        dict_open_info = {
            'long': 'df_sig.rank(pct=True)[0] < 0.05',
            'short': 'df_sig.rank(pct=True)[0] > 0.95'
        }
        df_fac_info = pd.read_excel(fac_fpath)
        for key, value in dict_fac_info.items():
            if isinstance(value, list):
                df_fac_info = df_fac_info[df_fac_info[key].isin(value)]
            else:
                df_fac_info = df_fac_info[df_fac_info[key] == value]
        df_fac_info = iter_flatten_df_fac_info(df_fac_info)

        strategy = CrossSecStrategy(strategy_name, model_fpath, df_fac_info, open_time, close_time)
        client = binance_scripts.Client(trade_api1['api_key'], trade_api1['secret_key'])
        dict_exchange_info = client.futures_exchange_info()
        df_exchange_info = pd.DataFrame(dict_exchange_info['symbols']).set_index('symbol')

        strategy_open2(strategy, client, top_portion, df_exchange_info, dict_universe_info, dict_open_info)

    is_run = True
    if is_run:
        # strategy_scheduler(trade_api1, list_strategy_config_dict)
        strategy_name = 'strategy0800'
        model_fpath = r'D:\research\cross_sec_strategies\Strategy_v20\models\xgbrank_fac_set_v10_0800.json'
        fac_fpath = r'D:\research\cross_sec_strategies\factor_set\strategy_fac_set_v1.1.xlsx'
        open_time, close_time = '06:00', '12:00'
        top_portion = 0.2
        dict_universe_info = {
            'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k'
        }
        dict_fac_info = {
            'create': [1]
        }
        dict_open_info = {
            'long': 'df_sig.rank(pct=True)[0] < 0.05',
            'short': 'df_sig.rank(pct=True)[0] > 0.95'
        }
        df_fac_info = pd.read_excel(fac_fpath)
        for key, value in dict_fac_info.items():
            if isinstance(value, list):
                df_fac_info = df_fac_info[df_fac_info[key].isin(value)]
            else:
                df_fac_info = df_fac_info[df_fac_info[key] == value]

        df_fac_info = iter_flatten_df_fac_info(df_fac_info)

        strategy = SimpleCrossSecStrategy(strategy_name, model_fpath, df_fac_info, open_time, close_time)
        client = binance_scripts.Client(trade_api1['api_key'], trade_api1['secret_key'])
        dict_exchange_info = client.futures_exchange_info()
        df_exchange_info = pd.DataFrame(dict_exchange_info['symbols']).set_index('symbol')

        strategy_open2(strategy, client, top_portion, df_exchange_info, dict_universe_info, dict_open_info)
