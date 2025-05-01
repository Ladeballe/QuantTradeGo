import aiohttp
import asyncio
import traceback
import logging
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.dialects.mysql import insert
import pymongo


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')


def main():
    asyncio.run(async_main())


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(ts=stmt.inserted.ts, symbol=stmt.inserted.symbol)
    result = conn.execute(stmt)
    return result.rowcount


async def async_main():
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    client = aiohttp.ClientSession(
        headers={
            "accept": "application/json",
            "x-cg-demo-api-key": "CG-QrNB8ySd5ZQ3nooPNT9Zzct1\t"
        })
    conn = engine.connect()
    df_symbols = pd.DataFrame(conn.execute(sqlalchemy.text("SELECT * FROM bnc_symbols WHERE geckoName IS NOT NULL")))
    for i, row in df_symbols.iterrows():
        try:
            conn.begin()
            symbol = row['symbol']
            gecko_name = row['geckoName']
            url = f"https://api.coingecko.com/api/v3/coins/{gecko_name}/market_chart?vs_currency=usd&days=365&interval=daily"
            res = await client.get(url)
            data = await res.json()
            df_data = pd.concat([
                pd.DataFrame(data['prices'], columns=['ts', 'price']).set_index('ts'),
                pd.DataFrame(data['market_caps'], columns=['ts', 'market_cap']).set_index('ts'),
                pd.DataFrame(data['total_volumes'], columns=['ts', 'vol']).set_index('ts')
            ], axis=1)
            df_data['t'] = pd.to_datetime(df_data.index * 1e6)
            df_data['symbol'] = symbol
            df_data = df_data.reset_index()[['ts', 'symbol', 't', 'price', 'market_cap', 'vol']]
            df_data.to_sql(name="gecko_market_cap_1d", con=conn, if_exists="append", index=False,
                           method=insert_on_conflict_update)
            logging.info(f'{i}, {symbol}, {gecko_name}, completed')
            conn.commit()
            await asyncio.sleep(2)
        except:
            traceback.print_exc()
            conn.rollback()
    await client.close()
    logging.info(f'download_market_cap completed')


if __name__ == '__main__':
    main()
