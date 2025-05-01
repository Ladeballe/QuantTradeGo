"""
获取交易所信息，包括交易对、资产等信息

将数据存储于mongodb中，因为交易所信息是结构复杂的键值对
"""
import requests as rq
import pymongo
import sqlalchemy

from enums import *


if __name__ == '__main__':
    url = FUTURE_BASE_URL + FUTURE_API_V1_URL + EXCHANGE_INFO_URL
    res = rq.get(url)
    data = res.json()
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')

    coll = pymongo.MongoClient('localhost:27017')['crypto_data']['bnc-fapi-exchangeInfo']
    coll.update_one(filter={'serverTime': data['serverTime']}, update={'$set': data}, upsert=True)
    conn = engine.connect()
    conn.begin()
    data_symbols = data['symbols']
    for data_symbol in data_symbols:
        data_symbol = {
            'symbol': data_symbol['symbol'],
            'pair': data_symbol['pair'],
            'contractType': data_symbol['contractType'],
            'deliveryDate': data_symbol['deliveryDate'],
            'onboardDate': data_symbol['onboardDate'],
            'status': data_symbol['status'],
            'maintMarginPercent': data_symbol['maintMarginPercent'],
            'requiredMarginPercent': data_symbol['requiredMarginPercent'],
            'baseAsset': data_symbol['baseAsset'],
            'quoteAsset': data_symbol['quoteAsset'],
            'marginAsset': data_symbol['marginAsset'],
            'pricePrecision': data_symbol['pricePrecision'],
            'quantityPrecision': data_symbol['quantityPrecision'],
            'baseAssetPrecision': data_symbol['baseAssetPrecision'],
            'quotePrecision': data_symbol['quotePrecision'],
            'underlyingType': data_symbol['underlyingType'],
            'underlyingSubType': ','.join(data_symbol['underlyingSubType']),
            'triggerProtect': data_symbol['triggerProtect'],
            'liquidationFee': data_symbol['liquidationFee'],
            'marketTakeBound': data_symbol['marketTakeBound'],
            'maxMoveOrderLimit': data_symbol['maxMoveOrderLimit']
        }
        conn.execute(
            sqlalchemy.text("""
                REPLACE INTO bnc_symbols (
                   symbol, pair, contractType, deliveryDate, onboardDate, status,
                   maintMarginPercent, requiredMarginPercent, baseAsset, quoteAsset, marginAsset,
                   pricePrecision, quantityPrecision, baseAssetPrecision, quotePrecision,
                   underlyingType, underlyingSubType, triggerProtect, liquidationFee,
                   marketTakeBound, maxMoveOrderLimit
                ) VALUES (
                   :symbol, :pair, :contractType, :deliveryDate, :onboardDate, :status,
                   :maintMarginPercent, :requiredMarginPercent, :baseAsset, :quoteAsset, :marginAsset,
                   :pricePrecision, :quantityPrecision, :baseAssetPrecision, :quotePrecision,
                   :underlyingType, :underlyingSubType, :triggerProtect, :liquidationFee,
                   :marketTakeBound, :maxMoveOrderLimit
                )
                """),
            data_symbol
        )
    conn.commit()
    conn.close()
    print('done.')
