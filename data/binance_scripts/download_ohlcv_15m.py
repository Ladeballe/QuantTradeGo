import traceback
import logging

import pandas as pd
import sqlalchemy
from sqlalchemy.dialects.mysql import insert
import binance as bnc
from binance.exceptions import BinanceAPIException

from test_func import data

proxies = {
    'http': 'http://localhost:7890',
    'https': 'http://localhost:7890'
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(ts0=stmt.inserted.ts0, symbol=stmt.inserted.symbol)
    result = conn.execute(stmt)
    return result.rowcount


def main(begin_time, end_time=None):
    client = bnc.Client(requests_params={'proxies': proxies})
    mysql_conn = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    symbols = data.load_symbols_from_exchange_info()

    for i, symbol in enumerate(symbols):
        logging.info(f'{i}, {symbol}, begin')
        try:
            df1 = pd.DataFrame(
                client.get_historical_klines(
                    symbol, "15m",
                    int(pd.Timestamp(begin_time).timestamp() * 1000),
                    int(pd.Timestamp.now().timestamp() * 1000),
                    klines_type=bnc.enums.HistoricalKlinesType.FUTURES
                ),
                columns=['ts0', 'o', 'h', 'l', 'c', 'v', 'ts1', 'q', 'n', 'bq', 'bv',' ig']
            )
            df1['symbol'] = symbol
            df1['t'] = pd.to_datetime(df1['ts0'] * 1e6)
            df1 = df1[['ts0', 'symbol', 'ts1', 't', 'o', 'h', 'l', 'c', 'v', 'q', 'bv', 'bq', 'n']]
            df1 = df1.astype({'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'q': float, 'bv': float, 'bq': float})
            df1.to_sql(name="bnc_kline_15m", con=mysql_conn, if_exists="append", index=False, method=insert_on_conflict_update)
            logging.info(f'{i}, {symbol}, completed')
        except KeyboardInterrupt:
            break
        except (ConnectionError, ConnectionResetError, BinanceAPIException):
            traceback.print_exc()
            continue
    logging.info(f'download_ohlcv_15m completed')


if __name__ == '__main__':
    main('2024-12-14')
