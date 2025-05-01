import pandas as pd
import asyncio
import aiohttp
import pymongo


async def main():
    db = pymongo.MongoClient('localhost:27017')['crypto_data']
    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    client = aiohttp.ClientSession(
        headers={
            "accept": "application/json",
            "x-cg-demo-api-key": "CG-QrNB8ySd5ZQ3nooPNT9Zzct1\t"
    })
    coll = db['gecko-coins-coin_list']
    res = await client.get(url)
    data = await res.json()
    data = {
        'downloadTime': int(pd.Timestamp.now().timestamp() * 1e6),
        'coins': data
    }
    coll.insert_one(data)


if __name__ == '__main__':
    asyncio.run(main())
