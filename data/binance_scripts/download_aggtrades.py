import traceback
import time
import json
import asyncio
import aiohttp
from aiohttp.client_exceptions import ClientProxyConnectionError
from queue import Empty

import numpy as np
import pandas as pd
from sqlalchemy.dialects.mysql import insert
import aiomysql


proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}
base_url = "https://fapi.binance.com"


class Proxy:
    def __init__(self, url):
        if url.startswith("http"):
            self.url = url
        else:
            self.url = "http://" + url
        self.last_called_ts = pd.Timestamp.now().timestamp()
        self.last_reset_ts = pd.Timestamp.now().timestamp()
        self.used_weight = 0
        self.last_status_code = 200
        self.success_time = 0
        self.failure_time = 0
        self.success_prob = 0.5

    def __str__(self):
        return self.url

    def __repr__(self):
        return self.url

    def update_last_called_ts(self):
        self.last_called_ts = pd.Timestamp.now().timestamp()

    def update_success_time(self, is_succeed=True):
        if is_succeed:
            self.success_time += 1
        else:
            self.failure_time += 1
        self.success_prob = self.success_time / (self.success_time + self.failure_time)

    def update(self, status, headers):
        self.last_called_ts = pd.Timestamp(headers['Date']).timestamp()
        self.last_status_code = status
        if 'X-MBX-USED-WEIGHT-1M' in headers.keys():
            self.used_weight = int(headers['X-MBX-USED-WEIGHT-1M'])
            if self.used_weight <= 21:
                self.last_reset_ts = pd.Timestamp(headers['Date']).timestamp()

    def flatten(self):
        sr_res = pd.Series(
            [self, self.url, self.last_called_ts, self.last_reset_ts, self.used_weight, self.last_status_code,
             self.success_time, self.failure_time, self.success_prob],
            index=['Proxy', 'proxy', 'last_called_ts', 'last_reset_ts', 'used_weight', 'last_status_code',
                   'success_time', 'failure_time', 'success_prob']
        )
        return sr_res


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(symbol=stmt.inserted.symbol, a=stmt.inserted.a)
    result = conn.execute(stmt)
    return result.rowcount


async def aggtrades_task(session, side, params, proxy, loop):
    symbol = params['symbol']
    async with session.get(
            "https://fapi.binance.com/fapi/v1/aggTrades",
            params=params, proxy=str(proxy), ssl=False
    ) as res:
        proxy.update(res.status, res.headers)
        # print(f"{pd.Timestamp.now()}, aggtrades_task: {params}, {side}, {proxy.flatten().to_dict()}")

        if res.status != 200:
            raise ConnectionError(f'Connection Error: {res.status}, {await res.text()}')

        df_data = pd.DataFrame(await res.json())
        df_data['symbol'] = symbol
        df_data = df_data.rename({'T': 'ts'}, axis=1)
        df_data['t'] = pd.to_datetime(df_data['ts'] * 1e6)
        df_data = df_data.astype({'p': float, 'q': float})
        df_data['v'] = df_data['p'] * df_data['q']
        df_data['n'] = df_data['l'] - df_data['f']

        data = df_data.values
        data = [tuple(d) for d in data]

        mysql_conn = await aiomysql.connect(
            host='127.0.0.1', port=3306, user='root', password='444666',
            db='market_data', loop=loop, autocommit=True)
        cursor0 = await mysql_conn.cursor()

        table_name = 'bnc_aggtrades'
        columns = ', '.join(df_data.columns)
        placeholders = ', '.join(['%s'] * df_data.shape[1])
        replace_query = f"REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"
        await cursor0.executemany(replace_query, data)
        await cursor0.close()

        # cursor1 = await mysql_conn.cursor()
        if side == 'forward':
            update_str = "SET a1=%s, ts1=%s "
            params = (df_data['a'].iloc[-1], df_data['ts'].iloc[-1], symbol)
        elif side == 'backward':
            update_str = "SET a0=%s, ts0=%s "
            params = (df_data['a'].iloc[0], df_data['ts'].iloc[0], symbol)
        else:
            raise ValueError("Side must be 'forward' or 'backward'")
        sql_code = "UPDATE bnc_aggtrades_record " + \
                   update_str + \
                   "WHERE symbol = %s"
        # mysql_res = await cursor1.execute(sql_code, params)
        print(f"{pd.Timestamp.now()}, handle_task, {symbol}, "
              f"{pd.Timestamp(df_data['ts'].iloc[0] * 1e6)}, {pd.Timestamp(df_data['ts'].iloc[-1] * 1e6)} "
              f"{df_data['a'].iloc[0]}, {df_data['a'].iloc[-1]} "
              # f"{mysql_res}, {params}")
              f"{params}, {proxy.flatten().to_dict()}")
        # await cursor1.close()
        mysql_conn.close()
        return sql_code, params


async def task_worker(queue_task_given, queue_task_res, queue_proxy, stopped_event, side, loop):
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=5, ssl=False))
    while not stopped_event.is_set():
        params = await queue_task_given.get()
        while not stopped_event.is_set():
            proxy = await queue_proxy.get()
            try:
                proxy.update_last_called_ts()
                res = await aggtrades_task(session, side, params, proxy, loop)
                proxy.update_success_time()
                await queue_task_res.put(res)
                break
            except ConnectionError:
                traceback.print_exc()
                proxy.update_success_time(False)
            except ClientProxyConnectionError as e:
                print(f"{pd.Timestamp.now()}, task_worker, ", params, e, queue_proxy.qsize())
                proxy.update_success_time(False)


async def task_generator(queue_task_give, queue_task_res, stopped_event, side, end_ts, loop):
    mysql_conn = await aiomysql.connect(
        host='127.0.0.1', port=3306, user='root', password='444666',
        db='market_data', loop=loop, autocommit=True)
    sql_code = "SELECT *, a1 - a0 num FROM bnc_aggtrades_record;"
    cursor = await mysql_conn.cursor()
    await cursor.execute(sql_code)
    df_task = pd.DataFrame(await cursor.fetchall(), columns=np.array(cursor.description)[:, 0])
    await cursor.close()
    df_task = df_task[df_task['symbol'].str[-4:] == 'USDT']
    df_task['is_ready'] = True
    df_task = df_task.set_index('symbol')
    steps = 0
    while not stopped_event.is_set():
        steps += 1
        if side == 'forward':
            df_task = df_task.sort_values(by='ts1', ascending=True)
            df_task = df_task[df_task['ts1'] <= end_ts]
        elif side == 'backward':
            df_task = df_task.sort_values(by='ts0', ascending=False)
            df_task = df_task[(df_task['ts0'] >= end_ts) & (df_task['a0'] >= 0)]
        else:
            raise ValueError("Side must be 'forward' or 'backward'.")
        if df_task.empty:
            print('All tasks are done')
            stopped_event.set()
        while not queue_task_res.empty():
            try:
                res = queue_task_res.get_nowait()
                cursor1 = await mysql_conn.cursor()
                if side == 'forward':
                    sql_code, params = res
                    a1, ts1, symbol_res = params
                    df_task.loc[symbol_res, 'is_ready'] = True
                    df_task.loc[symbol_res, 'a1'] = a1
                    df_task.loc[symbol_res, 'ts1'] = ts1
                elif side == 'backward':
                    sql_code, params = res
                    a0, ts0, symbol_res = params
                    df_task.loc[symbol_res, 'is_ready'] = True
                    df_task.loc[symbol_res, 'a0'] = a0
                    df_task.loc[symbol_res, 'ts0'] = ts0
                else:
                    raise ValueError("Side must be 'forward' or 'backward'.")
                await cursor1.execute(sql_code, params)
                await cursor1.close()
            except Empty:
                print('Queue is empty')
                break
        df_task_ready = df_task[df_task['is_ready']]
        if df_task_ready.empty:
            print(f"{pd.Timestamp.now()}, add_task: task_empty")
            continue
        else:
            row = df_task_ready.iloc[0]
            symbol = row.name
            if side == 'forward':
                fromId = row['a1'] + 1
            else:
                fromId = row['a0'] - 1000
            params = {
                'symbol': symbol,
                'limit': 1000,
                'fromId': int(fromId)
            }
            df_task.loc[symbol, 'is_ready'] = False
            await queue_task_give.put(params)
            print(f"{pd.Timestamp.now()}, add_task: {symbol}, {row['a0']}, {fromId}, {df_task[df_task['is_ready']].shape[0]}")
            if steps % 100 == 0:
                steps = 0
                print(df_task)


async def get_proxy_pool(stopped_event, queue_proxy):
    sr_proxies = pd.read_table(r'D:\python_projects\data_downloader\proxy\proxyscrape_premium_http_proxies.txt', header=None)[0]
    sr_proxies = sr_proxies.apply(Proxy).rename('Proxy')
    while not stopped_event.is_set():
        df_proxies = sr_proxies.apply(lambda x: x.flatten())
        df_proxies_usable = df_proxies[
            (df_proxies['last_status_code']==200)&((pd.Timestamp.now().timestamp() - df_proxies['last_called_ts']) > 5)]
        df_proxies_usable = df_proxies_usable.sort_values(['success_prob', 'last_called_ts'], ascending=[False, True])
        df_proxies_usable.to_excel('df_proxies.xlsx')
        print(
            f"{pd.Timestamp.now()}, proxy_pool: \n{df_proxies_usable[['used_weight', 'success_time', 'success_prob']]}")
        for i in range(min(200, df_proxies_usable.shape[0])):
            await queue_proxy.put(df_proxies_usable.iloc[i, 0])
        for i in range(min(20, df_proxies_usable.shape[0])):
            await queue_proxy.put(df_proxies_usable.iloc[-i, 0])
        sr_proxies = df_proxies['Proxy']


def main(side='backward', end_ts='2024-01-01'):
    queue_task_give = asyncio.Queue(maxsize=50)
    queue_task_return = asyncio.Queue()
    queue_proxy = asyncio.Queue(maxsize=200)
    stopped_event = asyncio.Event()
    end_ts = int(pd.Timestamp(end_ts).timestamp() * 1e3)

    loop = asyncio.get_event_loop()

    async def _main():
        task_generator_coro = asyncio.create_task(
            task_generator(queue_task_give, queue_task_return, stopped_event, side, end_ts, loop))
        proxy_pool_coro = asyncio.create_task(get_proxy_pool(stopped_event, queue_proxy))
        list_coros = [task_generator_coro, proxy_pool_coro]
        for i in range(200):
            list_coros.append(asyncio.create_task(
                task_worker(queue_task_give, queue_task_return, queue_proxy, stopped_event, side, loop)
                )
            )

        await asyncio.gather(*list_coros)

    asyncio.run(_main())


if __name__ == '__main__':
    main('backward', '2024-11-30 23:59:59')
