import time

import requests as rq
import pandas as pd
import pymongo
import sqlalchemy
from sqlalchemy.dialects.mysql import insert
from apscheduler.schedulers.background import BackgroundScheduler

from binance_scripts.enums import *


def get_res_trans_to_json(*url_comps, **url_params):
    url = ''.join(url_comps)
    url = url + '?' + '&'.join([f'{k}={v}' for k, v in url_params.items()])

    res = rq.get(url)
    if res.status_code == 200:
        data = res.json()
        return data
    else:
        raise Exception(f'Status Code: {res.status_code}. Error occurs! Message: {res.text}')


def process_ohlcv_json2df(data, symbol):
    df = pd.DataFrame(data, columns=['ts0', 'o', 'h', 'l', 'c', 'v', 'ts1', 'q', 'n', 'bq', 'bv', ' ig'])
    df['symbol'] = symbol
    df['t'] = pd.to_datetime(df['ts0'] * 1e6)
    df = df[['ts0', 'symbol', 'ts1', 't', 'o', 'h', 'l', 'c', 'v', 'q', 'bv', 'bq', 'n']]
    df = df.astype({'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'q': float, 'bv': float, 'bq': float})
    return df


def save_to_mongodb(data, db, coll, filter=dict()):
    client = pymongo.MongoClient('localhost:27017')
    db = client[db]
    coll = db[coll]

    coll.update_many(filter=filter, update={'$set': data}, upsert=True)


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(ts0=stmt.inserted.ts0, symbol=stmt.inserted.symbol)
    result = conn.execute(stmt)
    return result.rowcount


def save_ohlcv_to_mysql(data, db, table):
    mysql_conn = sqlalchemy.create_engine(f'mysql+pymysql://root:444666@localhost:3306/{db}')
    data.to_sql(name=table, con=mysql_conn, if_exists="append", index=False, method=insert_on_conflict_update)


def read_mongodb_exchange_info():
    coll = pymongo.MongoClient('localhost', 27017)['crypto_data']['bnc-fapi-exchangeInfo']
    data = coll.find_one(sort=[('serverTime', -1)], limit=1)
    data = pd.DataFrame(data['symbols'])
    return data


def job_download_exchange_info():
    data = get_res_trans_to_json(FUTURE_BASE_URL, FUTURE_API_V1_URL, EXCHANGE_INFO_URL)
    save_to_mongodb(data, 'crypto_data', 'bnc-fapi-exchangeInfo', {'serverTime': data['serverTime']})


def job_download_ohlcv():
    # 将读取symbols的工作放置于每次运行开始时，这只是权宜之策，因为apscheduler的变量传递功能目前并没有想到比较好的解决方式
    # df_symbols = read_mongodb_exchange_info()
    # df_symbols = df_symbols[df_symbols['status'] == 'TRADING']
    # symbols = df_symbols['symbols'].to_list()
    print(f'{pd.Timestamp.now()} - Start downloading ohlcv data...')
    symbols = ['BTCUSDT', 'ETHUSDT']

    ts0 = int(time.time() // 60 * 60) * 1000
    ts1 = int(time.time() + 60) * 1000
    print(f"ts0: {pd.Timestamp(ts0 * 1e6)} ts1: {pd.Timestamp(ts1 * 1e6)}")
    for symbol in symbols:
        data = get_res_trans_to_json(
            FUTURE_BASE_URL, FUTURE_API_V1_URL, KLINE_URL, symbol=symbol, interval='1m',
            startTime=ts0, endTime=ts1)
        data = process_ohlcv_json2df(data, symbol)
        print(data, pd.Timestamp(data.loc[0, 'ts0'] * 1e6))
        save_ohlcv_to_mysql(data, 'market_data', 'test_bnc_kline_1m')


if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        job_download_ohlcv,
        trigger='cron', second='5',
        id=str(hash('job_download_ohlcv')), name='job_download_ohlcv'
    )

    try:
        scheduler.start()
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print('Shutting down.')
    # job_download_ohlcv()
    print('done')
