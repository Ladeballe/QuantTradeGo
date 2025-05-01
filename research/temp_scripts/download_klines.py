import pandas as pd
import binance as bnc
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import insert

from test_func.data import load_symbols


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(ts0=stmt.inserted.ts0, symbol=stmt.inserted.symbol)
    result = conn.execute(stmt)
    return result.rowcount


if __name__ == "__main__":
    client = bnc.Client()
    engine = create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')

    symbols = load_symbols(contractType="PERPETUAL", status="TRADING", quoteAsset="USDT")

    for symbol in symbols[105:]:
        # df = pd.DataFrame(
        #     client.get_historical_klines(symbol, "5m",
        #                                  int(pd.Timestamp('2023-01-01').timestamp() * 1e3),
        #                                  int(pd.Timestamp('2024-04-01').timestamp() * 1e3),
        #                                  klines_type=bnc.enums.HistoricalKlinesType.FUTURES),
        #     columns=['ts0', 'o', 'h', 'l', 'c', 'v', 'ts1', 'q', 'n', 'bv', 'bq', 'ig'],
        # )
        df = pd.DataFrame(
            client.get_historical_klines(symbol, "1m",
                                         int(pd.Timestamp('2023-01-01').timestamp() * 1e3),
                                         int(pd.Timestamp('2023-04-01').timestamp() * 1e3),
                                         klines_type=bnc.enums.HistoricalKlinesType.FUTURES),
            columns=['ts0', 'o', 'h', 'l', 'c', 'v', 'ts1', 'q', 'n', 'bv', 'bq', 'ig'],
        )
        df = df.astype(
            {'o': np.float64, 'h': np.float64, 'l': np.float64, 'c': np.float64, 'v': np.float64, 'q': np.float64,
             'n': int, 'bv': np.float64, 'bq': np.float64})
        df = df.iloc[:, :-1]
        df['symbol'] = symbol
        # df.to_sql(name="bnc_kline_5m", con=engine, if_exists="append", index=False, method=insert_on_conflict_update)
        df.to_sql(name="bnc_kline_1m", con=engine, if_exists="append", index=False, method=insert_on_conflict_update)
        print(symbol)
