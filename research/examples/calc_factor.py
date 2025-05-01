import pandas as pd
import binance as bnc
import numpy as np
from sqlalchemy import create_engine
import arctic


store = arctic.Arctic("localhost")
fac_lib = store['vp_fac.factor_UNI_15m']

engine = create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')


def load_data(symbols=None):
    sql_code = 'SELECT t, symbol, o FROM bnc_kline_15m'
    if symbols:
        if isinstance(symbols, str):
            sql_code += f' WHERE symbol = "{symbols}")'
        elif isinstance(symbols, list):
            sql_code += f' WHERE symbol in ({','.join([f'"{symbol}"' for symbol in symbols])}) '
    df = pd.read_sql(sql_code, engine)
    return df


def calc_std(df):
    for t in [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50]:
        df_fac = df.rolling(t).std()
        fac_lib.write(f'std_{t}', df_fac)


if __name__ == '__main__':
    df = load_data()
    df = df.rename({'t': 'date'}, axis=1).pivot(index='date', columns='symbol').droplevel(0, axis=1)
    calc_std(df)
    print('done')
