import aiohttp
import asyncio
import aiomysql
import pandas as pd
import zipfile
import io
import dolphindb as ddb
from jsonpath import jsonpath

from test_func import data


sr_price_precision = None


async def task_generator(queue_task, loop):
    mysql_conn = await aiomysql.connect(
        host='127.0.0.1', port=3306, user='root', password='444666',
        db='market_data', loop=loop, autocommit=True
    )
    session = aiohttp.ClientSession(loop=loop)
    async with session.get(
            "https://fapi.binance.com/fapi/v1/exchangeInfo",
    ) as res:
        exchange_info_dict = await res.json()

    data_price_precision = jsonpath(exchange_info_dict, '$.symbols.[symbol,pricePrecision]')
    data_price_precision = [data_price_precision[0::2], data_price_precision[1::2]]
    global sr_price_precision
    sr_price_precision = pd.DataFrame(data_price_precision, index=['symbol', 'pricePrecision']).T.set_index('symbol')['pricePrecision']
    cursor0 = await mysql_conn.cursor()
    sr_symbols = pd.Series(data.load_symbols_from_exchange_info())
    sr_symbols = sr_symbols[sr_symbols.str[-4:] == 'USDT']
    for date in pd.date_range('2024-07-01', '2025-01-12', freq='1D')[::-1]:
        for _, symbol in sr_symbols.items():
            query = f"SELECT * FROM bnc_aggtrades_from_website_record WHERE date=%s AND symbol=%s"
            params = (date, symbol)
            list_res = await cursor0.execute(query, params)
            if list_res == 1:
                print(f"{pd.Timestamp.now()}, task_generator, {symbol}, {date}, already done")
                continue
            else:
                await queue_task.put((symbol, date))
                print(f"{pd.Timestamp.now()}, task_generator, {symbol}, {date}, {queue_task.qsize()}")


async def request_worker(queue_task, queue_content, stop_event):
    session = aiohttp.ClientSession()
    while not stop_event.is_set():
        symbol, date = await queue_task.get()
        while not stop_event.is_set():
            try:
                res = await session.get(f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date.strftime('%Y-%m-%d')}.zip")
                print(f"{pd.Timestamp.now()}, request_worker, {symbol}, {date}, {res.status}")
                if res.status != 200:
                    break
                content = await res.content.read()
            except TimeoutError:
                print(f"{pd.Timestamp.now()}, request_worker, TimeoutError, {symbol}, {date}, {res.status}, wait for 60s...")
                await asyncio.sleep(60)
                continue
            await queue_content.put((symbol, date, content))
            break


async def mysql_worker(queue_content, stop_event, loop):
    mysql_conn = await aiomysql.connect(
            host='127.0.0.1', port=3306, user='root', password='444666',
            db='market_data', loop=loop, autocommit=True
    )
    session = ddb.session()
    session.connect("localhost", 8848, "admin", "123456")
    appender = ddb.TableAppender(dbPath="dfs://market_data", tableName="bnc_aggtrades", ddbSession=session)
    while not stop_event.is_set():
        cursor0 = await mysql_conn.cursor()

        symbol, date, content = await queue_content.get()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            with zip_ref.open(zip_ref.namelist()[0]) as file_bookdepth:
                df_bookdepth = pd.read_csv(file_bookdepth)

        df_bookdepth = df_bookdepth.rename({'timestamp': 't', 'percentage': 'depth', 'depth': 'q', 'notional': 'v'},
                                           axis=1)
        df_bookdepth = df_bookdepth.pivot(index='timestamp', columns='percentage')
        df_bookdepth['p'] = (df_bookdepth['v'] / df_bookdepth['q']).round(sr_price_precision[symbol])
        df_bookdepth = df_bookdepth.pivot(index='t', columns='depth')
        df_bookdepth.columns = ['a', 'p', 'q', 'f', 'l', 'ts', 'm']
        df_bookdepth['symbol'] = symbol
        df_bookdepth['t'] = pd.to_datetime(df_bookdepth['ts'] * 1e6)
        df_bookdepth = df_bookdepth.astype({'p': float, 'q': float})
        df_bookdepth['v'] = df_bookdepth['p'] * df_bookdepth['q']
        df_bookdepth['n'] = df_bookdepth['l'] - df_bookdepth['f']
        df_bookdepth = df_bookdepth[['ts', 'symbol', 'a', 't', 'f', 'l', 'p', 'q', 'v', 'n', 'm']]

        print(f"{pd.Timestamp.now()}, mysql_worker, {symbol}, {date}, executing...")
        appender.append(df_bookdepth)
        print(f"{pd.Timestamp.now()}, mysql_worker, {symbol}, {date}, executed.")
        ts = int(pd.Timestamp.now().timestamp() * 1e3)
        params = (date, symbol, ts)
        insert_query = f"INSERT INTO bnc_bookdepth5_from_website_record (date, symbol, ts) VALUES (%s, %s, %s)"
        await cursor0.execute(insert_query, params)
        await cursor0.close()
        print(f"{pd.Timestamp.now()}, mysql_worker, {symbol}, {date}, record saved.")


async def main():
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    queue_task = asyncio.Queue(maxsize=5)
    queue_content = asyncio.Queue(maxsize=5)

    tasks = list()

    tasks.append(task_generator(queue_task, loop))
    for i in range(5):
        tasks.append(request_worker(queue_task, queue_content, stop_event))
    for i in range(1):
        tasks.append(mysql_worker(queue_content, stop_event, loop))

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
