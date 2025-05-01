import json

import asyncio
import aiohttp
import websockets
import pandas as pd
import aiomysql

from .strategy import OrderStatus


TRADE_API = {
    'api_key': 'I2UgUwKmF1P4JJvz8oOGzVvuvQmcyYDpak1qMrPQxlDH1Ss4OfWlMqr33Lo3yfKL',
    'secret_key': 'mzWJnHlag4m0Lm7pKlKqOMssxTbm05Ev6GvVoCHLsyXKtCKZQG7gWLmetNtOUpRr'
}


class UserDataHandler:
    BASE_URI = 'wss://fstream.binance.com/ws/{listenKey}'
    BASE_URL = 'https://fapi.binance.com'
    LISTEN_KEY_SUB_URL = '/fapi/v1/listenKey'

    def __init__(self):
        self.headers = {'X-MBX-APIKEY': TRADE_API['api_key']}
        self.session = aiohttp.ClientSession(headers=self.headers)
        self.event_loop = asyncio.get_event_loop()
        self.wss_client = None
        self.mysql_conn = None


class UserDataHandlerV1(UserDataHandler):
    async def connect_to_mysql(self):
        self.mysql_conn = await aiomysql.connect(
            host='127.0.0.1', port=3306, user='root', password='444666',
            db='trader_data', autocommit=True, loop=self.event_loop)

    async def get_listen_key(self):
        res = await self.session.post(self.BASE_URL + self.LISTEN_KEY_SUB_URL)
        listen_key = (await res.json())['listenKey']
        return listen_key

    async def keep_lk_alive_func(self, listen_key, stop_event):
        ts0 = pd.Timestamp.now().timestamp()
        while not stop_event.is_set():
            await asyncio.sleep(60)
            if pd.Timestamp.now().timestamp() - ts0 > 45 * 60:
                await self._keep_alive_listen_key(listen_key)
                ts0 = pd.Timestamp.now().timestamp()

    async def _keep_alive_listen_key(self, listen_key):
        params = {'listenKey': listen_key}
        url = self.BASE_URL + self.LISTEN_KEY_SUB_URL
        res = await self.session.put(url, params=params)
        res_status = res.status
        res_json = await res.json()
        return res_status, res_json

    async def _user_data_wss(self, listen_key, stop_event, queue_order_res):
        await self.connect_to_mysql()
        self.wss_client = await websockets.connect(self.BASE_URI.format(listenKey=listen_key))
        print(f"{pd.Timestamp.now()} user data process, start listening.")
        while not stop_event.is_set():
            message = await self.wss_client.recv()
            print(f"{pd.Timestamp.now()} user data process, {message}")
            message = json.loads(message)
            match message['e']:
                case 'ACCOUNT_UPDATE':
                    pass
                case 'ORDER_TRADE_UPDATE':
                    order_status = OrderStatus.from_wss_json(message)
                    queue_order_res.put(order_status)
                case _:
                    pass

    async def main(self, queue_order_res, stop_event):
        listen_key = await self.get_listen_key()

        task0 = asyncio.create_task(self.keep_lk_alive_func(listen_key, stop_event))
        task1 = asyncio.create_task(self._user_data_wss(listen_key, stop_event, queue_order_res))

        await task0
        await task1
