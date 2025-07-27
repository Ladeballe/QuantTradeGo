import json
import time
import datetime
import traceback
from queue import Empty
from threading import Thread
from multiprocessing import Queue, Lock, Process, Event
import asyncio
import websockets

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pymongo
import binance_scripts as bnc
import aiomysql

from test_func import data
from quant_trade_go.data_handler import DataHandlerV1
from quant_trade_go.factor_handler import FactorHandlerV2
from quant_trade_go.strategy_handler import StrategyHandlerV2
from quant_trade_go.order_handler import OrderHandlerV1
from quant_trade_go.userdata_handler import UserDataHandlerV1
from quant_trade_go.strategies.strategy_v2_0_0 import TestStrategy, StrategyV2


def data_process_func(dh, queue_data, lock_data, stop_event):
    def _update_queue():
        ratio = dh.update_data()
        time.sleep(1)
        print(pd.Timestamp.now(), 'data process:', ratio)  # 一个检测指标，已经有多少行数据被更新了
        if ratio == 1:
            dict_df_data = dh.get_dict_df()
            # 分离data_queue的生产者和消费者
            # print(pd.Timestamp.now(), 'data process try to acquire lock ...')
            # while not queue_data.empty():
            #     try:
            #         print(pd.Timestamp.now(), 'data process: clear data_queue')
            #         queue_data.get(timeout=0.01)
            #     except Empty:
            #         print(pd.Timestamp.now(), 'data process: clear data_queue - data_queue empty')
            #         break
            # print(pd.Timestamp.now(), 'data process: start putting data...')
            queue_data.put(dict_df_data, timeout=0.1)
            # print(pd.Timestamp.now(), 'data process: data putted')

    dh.init_engine()
    print(f"{pd.Timestamp.now()} data_process: market data load begin...")
    dh.read_data()
    print(f"{pd.Timestamp.now()} data_process: market data load completed.")

    while not stop_event.is_set():
        # print(f"{pd.Timestamp.now()} data_process: update ...")
        _update_queue()
    print(f"{pd.Timestamp.now()} data_process: end...")


def factor_process_func(fh, queue_fac, queue_data, lock_fac, stop_event):
    _queue_fac_calc_task = Queue()
    ts = 0

    def _init_queue_task():
        nonlocal ts
        dict_df_data = queue_data.get()
        ts = dict_df_data['open']['date'].max().timestamp()
        print(f"{pd.Timestamp.now()} factor_process: new factor_calc_task initiated, {pd.Timestamp(ts * 1e9)}")
        _queue_fac_calc_task.put((ts, dict_df_data))

    def _update_queue_task():
        nonlocal ts
        while not stop_event.is_set():
            try:
                dict_df_data = queue_data.get(timeout=10)
                ts1 = dict_df_data['open']['date'].max().timestamp()
                if ts1 > ts:
                    ts = ts1
                    print(f"{pd.Timestamp.now()} factor_process: new factor_calc_task initiated, {pd.Timestamp(ts * 1e9)}")
                    _queue_fac_calc_task.put((ts, dict_df_data))
            except Empty:
                continue

    def _calc_fac():
        while not stop_event.is_set():
            try:
                fac_ts, dict_df_data = _queue_fac_calc_task.get(timeout=10)
            except Empty:
                continue

            print(f"{pd.Timestamp.now()}, factor_process: factor calculation begin...")
            fh.calc_fac(dict_df_data)
            print(f"{pd.Timestamp.now()}, factor_process: factor calculation completed.")
            df_fac = fh.get_df_fac()

            with lock_fac:
                while not queue_fac.empty():
                    queue_fac.get()
                queue_fac.put((fac_ts, df_fac))

    fh.init_factor_tasks()
    _init_queue_task()  # 初始化阶段, 先计算一遍

    calc_fac_thread = Thread(target=_calc_fac)
    calc_fac_thread.start()
    update_queue_thread = Thread(target=_update_queue_task)
    update_queue_thread.start()

    calc_fac_thread.join()
    update_queue_thread.join()


def factor_process_func2(fh, queue_fac, queue_data, queue_fac_task, lock_fac, stop_event):
    class DataRecorder:
        dict_df_data = None

    dr = DataRecorder()
    _fac_calculate_task_queue = Queue()

    def _data_listener():
        while not stop_event.is_set():
            try:
                dr.dict_df_data = queue_data.get(timeout=10)
            except Empty:
                continue

    def _strategy_listener():
        while not stop_event.is_set():
            try:
                strategy, time_trigger = queue_fac_task.get(timeout=10)
                print(f"{pd.Timestamp.now()} factor_process: new factor calculation initiated on {strategy} {time_trigger}")
                while not stop_event.is_set():
                    if dr.dict_df_data is not None:
                        date = dr.dict_df_data['open']['date'].max()
                        ts = date.timestamp()
                        time_trigger_bool = date.time() == datetime.time(int(time_trigger[:2]), int(time_trigger[3:]))
                        is_last_bar_finished_bool = dr.dict_df_data['is_finished'].groupby('date')['raw_factor'].sum().iloc[-2] == dr.dict_df_data['is_finished']['symbol'].unique().shape[0]
                        if time_trigger_bool and is_last_bar_finished_bool:
                            print(f"{pd.Timestamp.now()}, factor_process put fac calc task ", ts, strategy)
                            _fac_calculate_task_queue.put((ts, strategy, dr.dict_df_data))
                            break
                        else:
                            # print(f"{pd.Timestamp.now()}, factor_process new data got not suitable ", date, time_trigger, time_trigger_bool, is_last_bar_finished_bool)
                            continue
            except Empty:
                continue

    def _calc_fac():
        while not stop_event.is_set():
            try:
                fac_ts, strategy, dict_df_data = _fac_calculate_task_queue.get(timeout=10)
            except Empty:
                continue

            df_fac_task_list = fh.filter_df_fac_task_list(strategy.name)
            print(f"{pd.Timestamp.now()}, factor_process: factor calculation begin...")
            fh.calc_fac(dict_df_data, df_fac_task_list)
            # dfs = []  # 用于存储实盘数据
            print(f"{pd.Timestamp.now()}, factor_process: factor calculation completed.")
            df_fac = fh.get_df_fac()
            # df_fac.to_excel(r'D:\python_projects\quant_trade_go\test\fac_test.xlsx')
            # for key, df in dict_df_data.items():  # 用于存储实盘数据
            #     df = df.reset_index()
            #     df.columns = ['index', 'date', 'symbol', 'raw_factor']
            #     df = df.set_index(['index', 'date', 'symbol'])['raw_factor']
            #     dfs.append(df)
            # df = pd.concat(dfs, axis=1, keys=list(dict_df_data.keys())).droplevel(0, axis=0).drop(['date'], axis=1)
            # data.save_factor(df, 'fac_basic_0', 'fac_5m.fac_temp')

            with lock_fac:
                while not queue_fac.empty():
                    queue_fac.get()
                queue_fac.put((fac_ts, strategy, df_fac))

    fh.init_factor_tasks()

    data_listener_thread = Thread(target=_data_listener)
    data_listener_thread.start()
    calc_fac_thread = Thread(target=_strategy_listener)
    calc_fac_thread.start()
    update_queue_thread = Thread(target=_calc_fac)
    update_queue_thread.start()

    data_listener_thread.join()
    calc_fac_thread.join()
    update_queue_thread.join()


def strategy_process_func(sh, trade_event, stop_event):
    while not stop_event.is_set():
        if trade_event.is_set():
            time.sleep(1)
            try:
                fac_ts, df_fac = queue_fac.get(timeout=10)
                sh.update_factor(fac_ts, df_fac)
                for strategy_name, strategy in sh.strategies.items():
                    sh.run_strategy(strategy)
            except Empty:
                continue
        else:
            time.sleep(10)


def strategy_process_func2(sh, queue_order_res, trade_event, stop_event):
    def _order_res_listener():
        while not stop_event.is_set():
            try:
                order_res = queue_order_res.get(timeout=10)
                sh.update_strategy(order_res)
            except Empty:
                continue
            except ValueError:
                traceback.print_exc()

    def _strategy_runner():
        while not stop_event.is_set():
            if trade_event.is_set():
                for strategy_name, strategy in sh.strategies.items():
                    sh.run_strategy(strategy)
                time.sleep(1)
            else:
                time.sleep(10)

    sh.connect_to_mysql()
    order_listener_thread = Thread(target=_order_res_listener)
    order_listener_thread.start()
    strategy_runner_thread = Thread(target=_strategy_runner)
    strategy_runner_thread.start()

    order_listener_thread.join()
    strategy_runner_thread.join()


def order_process_func(oh, queue_order, queue_order_res, stop_event):
    while not stop_event.is_set():
        try:
            orders_info = queue_order.get(timeout=10)
            orders_res = oh.trade(orders_info)
            for order_res in orders_res:
                queue_order_res.put(order_res)
        except Empty:
            continue


# def user_data_process_func(oh, queue_order_res, stop_event):
#     event_loop = oh.event_loop
#     listen_key = event_loop.run_until_complete(oh.get_listen_key())
#
#     async def _user_data_wss():
#         wss_uri = 'wss://fstream.binance_scripts.com/ws/{listenKey}'
#         wss_client = await websockets.connect(wss_uri.format(listenKey=listen_key))
#         print(f"{pd.Timestamp.now()} user data process, start listening.")
#         while not stop_event.is_set():
#             message = await wss_client.recv()
#             print(f"{pd.Timestamp.now()} user data process, {message}")
#             message = json.loads(message)
#             if message['e'] == 'ORDER_TRADE_UPDATE':
#                 message['info_type'] = 'update'
#                 queue_order_res.put(message)
#
#     async def _keep_lk_alive_func():
#         ts0 = pd.Timestamp.now().timestamp()
#         while not stop_event.is_set():
#             await asyncio.sleep(60)
#             if pd.Timestamp.now().timestamp() - ts0 > 45 * 60:
#                 await oh.keep_alive_listen_key(listen_key)
#                 ts0 = pd.Timestamp.now().timestamp()
#
#     async def _main():
#         task0 = event_loop.create_task(_user_data_wss())
#         task1 = event_loop.create_task(_keep_lk_alive_func())
#
#         await task0
#         await task1
#
#     event_loop.run_until_complete(_main())


def user_data_process_func(uh, queue_order_res, stop_event):
    event_loop = uh.event_loop
    event_loop.run_until_complete(uh.main(queue_order_res, stop_event))


if __name__ == '__main__':
    fac_file_path = r'D:\research\CRYPTO_cross_sec_strategies\factor_set\strategy_fac_set_v1.3.xlsx'
    dict_universe_info = {
        'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k'
    }
    test_mode = False
    hours = 240
    symbols = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']['bnc-future-universe'] \
        .find_one(dict_universe_info, projection={'symbols': True})['symbols']
    sheet = 'bnc_kline_5m'

    queue_data = Queue()
    queue_fac_task = Queue()
    queue_fac = Queue()
    queue_order = Queue()
    queue_order_res = Queue()
    lock_data = Lock()
    lock_fac = Lock()
    lock_sig = Lock()
    stop_event = Event()
    trade_event = Event()

    xgbrank_strategy_0600_0800 = StrategyV2(
        'xgbrank_0600_0800',
        r'D:\research\CRYPTO_cross_sec_strategies\Strategy_v20\models\xgbrank_v2.1_fac_set_v1.3_0615_0815.json',
        {
            'open_pos': {'time': '06:05', 'type': 'MARKET', 'BUY': 'df_sig>0.95', 'SELL': 'df_sig<0.05'},
            'stop_loss': {'time': '06:00', 'type': 'STOP', 'long': 0.95, 'short': 1.05},
            'take_profit': {'time': '11:00', 'type': 'TAKE_PROFIT', 'long': 1.01, 'short': 0.99},
            'force_close_pos': {'time': '08:15', 'type': 'MARKET', 'long': 'close', 'short': 'close'}
        },
        400
    )
    xgbrank_strategy_0800_1000 = StrategyV2(
        'xgbrank_0800_1000',
        r'D:\research\CRYPTO_cross_sec_strategies\Strategy_v20\models\xgbrank_v2.1_fac_set_v1.3_0800_1000.json',
        {
            'open_pos': {'time': '08:00', 'type': 'MARKET', 'BUY': 'df_sig>0.95', 'SELL': 'df_sig<0.05'},
            'stop_loss': {'time': '08:00', 'type': 'STOP', 'long': 0.95, 'short': 1.05},
            'take_profit': {'time': '11:00', 'type': 'TAKE_PROFIT', 'long': 1.01, 'short': 0.99},
            'force_close_pos': {'time': '10:00', 'type': 'MARKET', 'long': 'close', 'short': 'close'}
        },
        400
    )
    xgbrank_strategy_1200_1400 = StrategyV2(
        'xgbrank_1200_1400',
        r'D:\research\CRYPTO_cross_sec_strategies\Strategy_v20\models\xgbrank_v2.1_fac_set_v1.3_1200_1400.json',
        {
            'open_pos': {'time': '12:00', 'type': 'MARKET', 'BUY': 'df_sig>0.95', 'SELL': 'df_sig<0.05'},
            'stop_loss': {'time': '12:00', 'type': 'STOP', 'long': 0.95, 'short': 1.05},
            'take_profit': {'time': '12:00', 'type': 'TAKE_PROFIT', 'long': 1.01, 'short': 0.99},
            'force_close_pos': {'time': '14:00', 'type': 'MARKET', 'long': 'close', 'short': 'close'}
        },
        200
    )
    xgbrank_user_data_test_strategy = StrategyV2(
        'xgbrank_test',
        r'D:\research\CRYPTO_cross_sec_strategies\Strategy_v20\models\xgbrank_v2.1_fac_set_v1.3_test.json',
        {
            'open_pos': {'time': '00:00', 'type': 'MARKET', 'BUY': '[df_sig.index[0]]', 'SELL': '[df_sig.index[-1]]'},
            'stop_loss': {'time': '00:00', 'type': 'STOP', 'loss_ratio': 0.05},
            'take_profit': {'time': '05:00', 'type': 'TAKE_PROFIT', 'profit_ratio': 0.01},
            'force_close_pos': {'time': '12:00', 'type': 'MARKET'}
        },
        12
    )
    xgbrank_fac_test_strategy = StrategyV2(
        'xgbrank_test2',
        r'D:\research\CRYPTO_cross_sec_strategies\Strategy_v20\models\xgbrank_v2.1_fac_set_v1.3_0800_1000.json',
        {
            'open_pos': {'time': '05:00', 'type': 'MARKET', 'BUY': '[df_sig.sort_values().index[0]]', 'SELL': '[df_sig.sort_values().index[-1]]'},
            'stop_loss': {'time': '04:29', 'type': 'STOP', 'long': 0.95, 'short': 1.05},
            'take_profit': {'time': '11:00', 'type': 'TAKE_PROFIT', 'long': 1.01, 'short': 0.99},
            'force_close_pos': {'time': '05:45', 'type': 'MARKET', 'long': 'close', 'short': 'close'}
        },
        12
    )

    data_handler = DataHandlerV1(hours, symbols, sheet)
    factor_handler = FactorHandlerV2(fac_file_path)
    strategy_handler = StrategyHandlerV2({
        xgbrank_strategy_0600_0800.name: xgbrank_strategy_0600_0800,
        # xgbrank_strategy_0800_1000.name: xgbrank_strategy_0800_1000,
        # xgbrank_user_data_test_strategy.name: xgbrank_user_data_test_strategy,
        # xgbrank_fac_test_strategy.name: xgbrank_fac_test_strategy,
    }, queue_fac, queue_fac_task, queue_order)
    order_handler = OrderHandlerV1(test_mode)
    user_data_handler = UserDataHandlerV1()

    data_process = Process(target=data_process_func,
                           args=(data_handler, queue_data, lock_data, stop_event))
    factor_process = Process(target=factor_process_func2,
                             args=(factor_handler, queue_fac, queue_data, queue_fac_task, lock_fac, stop_event))
    strategy_thread = Thread(target=strategy_process_func2,
                             args=(strategy_handler, queue_order_res, trade_event, stop_event))
    order_thread = Thread(target=order_process_func,
                          args=(order_handler, queue_order, queue_order_res, stop_event))
    user_data_thread = Thread(target=user_data_process_func,
                              args=(user_data_handler, queue_order_res, stop_event))

    data_process.start()
    factor_process.start()
    strategy_thread.start()
    order_thread.start()
    user_data_thread.start()

    trade_event.set()

    while True:
        time.sleep(3)

    data_process.join()
    factor_process.join()
    strategy_thread.join()
    order_thread.join()
    user_data_thread.join()

    data_process.close()
    factor_process.close()

    queue_data.close()
    queue_fac_task.close()
    queue_fac.close()
    queue_order.close()
    queue_order_res.close()
