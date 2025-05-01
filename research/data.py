import logging
from functools import lru_cache

import arctic
import pandas as pd
from arctic import Arctic
import sqlalchemy
import pymongo


store = Arctic("localhost")
engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')
mongo_conn = pymongo.MongoClient('mongodb://localhost:27017')


def get_factor(fac_name, fac_lib, begin_date, end_date):
    lib = store[fac_lib]
    # FIXME: 由于pandas.date_range的性质，导致date_range默认的频率为Day，所以如果进行小时级以下的索引，可能并不能得到想要的结果
    df_factor = lib.read(fac_name, chunk_range=pd.date_range(begin_date, end_date))
    return df_factor


@lru_cache(maxsize=8)
def get_lru_factor(fac_name, fac_lib, begin_date, end_date):
    df_factor = get_factor(fac_name, fac_lib, begin_date, end_date)
    return df_factor


def save_factor(df_factor, fac_name, fac_lib):
    if not store.library_exists(fac_lib):
        store.initialize_library(fac_lib, lib_type=arctic.CHUNK_STORE)
    if df_factor.empty:
        raise ValueError("The dataframe to save is empty for 'df_factor'!")

    lib = store[fac_lib]
    if lib.has_symbol(fac_name):
        lib.update(fac_name, df_factor, chunk_size='M')
    else:
        lib.write(fac_name, df_factor, chunk_size='M')
    # logging.info(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} save factor: {fac_name}, "
    print(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} save factor: {fac_name}, "
                 f"{df_factor.index[0]}, {df_factor.index[-1]}")


def load_symbols(**kwargs):
    sql_code = "SELECT symbol FROM bnc_excg_info"
    where_conditions = list()
    for key, value in kwargs.items():
        if isinstance(value, list):
            value = [f"'{v}'" for v in value]
            where_conditions.append(f"{key} IN ({','.join(value)})")
        else:
            where_conditions.append(f"{key} = '{value}'")
    where_conditions_code = " WHERE " + " AND ".join(where_conditions)
    sql_code = sql_code + where_conditions_code if where_conditions else sql_code
    symbols = pd.read_sql_query(sql_code, engine)['symbol'].to_list()
    return symbols


def load_symbols_from_exchange_info():
    db = mongo_conn['crypto_data']
    df_contract_info = pd.DataFrame(db['bnc-fapi-exchangeInfo'].find().sort('serverTime', -1)[0]['symbols'])
    symbols = df_contract_info['symbol'].to_list()
    return symbols



def load_sub_type(symbols=None):
    sql_code = "SELECT symbol, underlyingSubType FROM bnc_excg_info"
    if symbols is not None:
        symbols = [f"'{symbol}'" for symbol in symbols]
        sql_code += f" WHERE symbol IN ({','.join(symbols)})"
    df_sub_type = pd.read_sql_query(sql_code, engine)
    return df_sub_type

@lru_cache(maxsize=4)
def load_ohlcv_data(symbols=None, sheet="bnc_kline_15m", begin_date=None, end_date=None):
    sql_code = f"SELECT t, symbol, o, h, l, c, q, bq, v, bv, n FROM {sheet}"
    sql_conditions = list()
    if symbols:
        symbols = [f"'{symbol}'" for symbol in symbols]
        sql_conditions.append(f"symbol IN ({','.join(symbols)})")
    if begin_date:
        sql_conditions.append(f"ts0 >= '{int(pd.Timestamp(begin_date).timestamp()*1e3)}'")
    if end_date:
        sql_conditions.append(f"ts0 < '{int(pd.Timestamp(end_date).timestamp()*1e3)}'")
    sql_code += " WHERE " + " AND ".join(sql_conditions)
    df_ohlcv = pd.read_sql_query(sql_code, engine)
    return df_ohlcv


def load_bnc_oi_data(symbols=None, sheet="bnc_global_ls_acct_ratio_5m", begin_date=None, end_date=None):
    sql_code = f"SELECT * FROM {sheet}"
    sql_conditions = list()
    if symbols:
        symbols = [f"'{symbol}'" for symbol in symbols]
        sql_conditions.append(f"symbol IN ({','.join(symbols)})")
    if begin_date:
        begin_ts = int(pd.Timestamp(begin_date).timestamp() * 1e3)
        sql_conditions.append(f"ts0 >= '{begin_ts}'")
    if end_date:
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1e3)
        sql_conditions.append(f"ts0 <= '{end_ts}'")
    sql_code += " WHERE " + " AND ".join(sql_conditions)
    df_oi = pd.read_sql_query(sql_code, engine)
    return df_oi


if __name__ == "__main__":
    # df_sub_type = load_sub_type().fillna('others')
    # sub_type_dict = dict(zip(df_sub_type['symbol'], df_sub_type['underlyingSubType']))
    df_ohlcv = load_ohlcv_data(["ETHUSDT"])
    print('done')
