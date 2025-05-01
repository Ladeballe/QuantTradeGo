import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pymongo

from test_func import data


def create_universe_basic():
    db = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']
    df_contract_info = pd.DataFrame(db['bnc-fapi-exchangeInfo'].find().sort('serverTime', -1)[0]['symbols'])
    df_universe = df_contract_info[
        (df_contract_info['contractType'] == 'PERPETUAL') & (df_contract_info['status'] == 'TRADING') & \
        (df_contract_info['quoteAsset'] == 'USDT') & (
                    (pd.Timestamp.now().timestamp() * 1000 - df_contract_info['onboardDate']) / (
                        1000 * 60 * 60 * 24) >= 90)]
    dict_universe = {
        'name': 'universe_bnc_perp_future_usdt',
        'symbols': df_universe['symbol'].to_list(),
        'universe': df_universe.to_dict('records')
    }
    db['bnc-future-universe'].update_one({'name': 'universe_bnc_perp_future_usdt'}, {'$set': dict_universe},
                                         upsert=True)


def create_universe_basic_top_150(blacklist):
    db = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']
    df_contract_info = pd.DataFrame(db['bnc-fapi-exchangeInfo'].find().sort('serverTime', -1)[0]['symbols'])
    df_contract_info = df_contract_info[~df_contract_info['symbol'].isin(blacklist)]
    df_universe = df_contract_info[
        (df_contract_info['contractType'] == 'PERPETUAL') & (df_contract_info['status'] == 'TRADING') & \
        (df_contract_info['quoteAsset'] == 'USDT')
    ].sort_values("onboardDate").iloc[:150]
    dict_universe = {
        'name': 'universe_bnc_perp_future_usdt_top_150',
        'symbols': df_universe['symbol'].to_list(),
        'universe': df_universe.to_dict('records')
    }
    db['bnc-future-universe'].update_one({'name': 'universe_bnc_perp_future_usdt_top_150'}, {'$set': dict_universe},
                                         upsert=True)


def create_universe_basic_onboard_over_1yr_or_amt_gt_200k(blacklist):
    db = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']
    df_contract_info = pd.DataFrame(db['bnc-fapi-exchangeInfo'].find().sort('serverTime', -1)[0]['symbols'])
    df_contract_info = df_contract_info[~df_contract_info['symbol'].isin(blacklist)]
    df_amt = data.get_factor('amt', 'fac_15m.fac_basic', '2024-01-01', '2024-10-30')
    df_contract_info = df_contract_info.merge(df_amt.mean().to_frame('mean_amt').reset_index(), left_on='symbol', right_on='index')
    df_universe = df_contract_info[
        ((pd.to_datetime(df_contract_info['onboardDate'] * 1e6) < (pd.Timestamp.now() - pd.Timedelta(days=365))
          ) | (df_contract_info['mean_amt'] > 2 * 1e6)
        ) & (df_contract_info['status'] == 'TRADING'
        ) & (df_contract_info['underlyingSubType'] != 'USDC'
        ) & (df_contract_info['contractType'] == 'PERPETUAL'
        ) & (df_contract_info['symbol'].str[-1] != 'C'
        ) & (df_contract_info['underlyingType'] == 'COIN'
        )]
    df_universe = df_universe.drop(['mean_amt'], axis=1)
    dict_universe = {
        'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k',
        'created_time': pd.Timestamp.now(),
        'symbols': df_universe['symbol'].to_list(),
        'universe': df_universe.to_dict('records')
    }
    db['bnc-future-universe'].update_one(
        {'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k'},
        {'$set': dict_universe}, upsert=True)


if __name__ == '__main__':
    blacklist = ['USDCUSDT', 'DAIUSDT']
    # create_universe_basic()
    # create_universe_basic_top_150(blacklist)
    create_universe_basic_onboard_over_1yr_or_amt_gt_200k(blacklist)
