import traceback
import time
import json
import logging

import numpy as np
import pandas as pd
import requests as rq
import sqlalchemy
from sqlalchemy.dialects.mysql import insert
from test_func import data


proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}
base_url = "https://fapi.binance.com"
mysql_conn = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(symbol=stmt.inserted.symbol, ts0=stmt.inserted.ts0)
    result = conn.execute(stmt)
    return result.rowcount


def main(begin_date):
    symbols = data.load_symbols_from_exchange_info()
    print(begin_date)
    periods = pd.period_range(begin_date, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    for period in periods:
        for symbol in symbols:
            try:
                logging.info(f'{symbol}, {period}, begin')
                res = rq.get(
                    base_url + f'/futures/data/openInterestHist?' + '&'.join([f'{k}={v}' for k, v in {
                        'symbol': symbol, 'period': '5m',
                        'startTime': int(period.start_time.timestamp() * 1e3),
                        'endTime': int(period.end_time.timestamp() * 1e3),
                        'limit': 288
                    }.items()]),
                    proxies=proxies
                )
                df = pd.DataFrame(json.loads(res.text))
                if df.empty:
                    logging.warning(f'{symbol}, empty: {res.text}')
                    time.sleep(4 / 3)
                    continue
                df.columns = ['symbol', 'oi', 'oiv', 'ts0']
                df.to_sql(name="bnc_oi_5m", con=mysql_conn, if_exists="append", index=False,
                          method=insert_on_conflict_update)
                time.sleep(0.03)
            except rq.ConnectionError:
                traceback_info = traceback.format_exc()
                logging.warning(f'{symbol}, error \n{traceback_info}')
                continue


if __name__ == '__main__':
    main('2024-12-19')
