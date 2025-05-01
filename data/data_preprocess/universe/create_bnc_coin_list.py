import sqlalchemy
import pymongo


if __name__ == '__main__':
    db = pymongo.MongoClient('localhost:27017')['crypto_data']
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    conn = engine.connect()

    data_collection = []
    coll = db['bnc-fapi-exchangeInfo']
    cursor = coll.find()
    for i in range(cursor.count()):
        data_collection.append(cursor.next())

    conn.begin()
    for data in data_collection:
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
