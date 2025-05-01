import hmac
import hashlib
from queue import Empty
from multiprocessing import Process

import aiohttp
import asyncio
import numpy as np
import pandas as pd
import binance as bnc

TRADE_API = {
    'api_key': 'aaa',  # Your api_key for binance
    'secret_key': 'bbb'  # Your api_secret for binance
}
ORDER_RES_DICT = {
    'updateTime': 'event_time',
    'timeInForce': 'time_in_force', 'origQty': 'order_quantity', 'price': 'order_price',
    'avgPrice': 'average_price', 'stopPrice': 'stop_price', 'status': 'order_status',
    'orderId': 'order_id', 'cumQty': 'cumulative_quantity',
    'reduceOnly': 'reduce_only', 'origType': 'original_order_type', 'positionSide': 'position_side',
    'selfTradePreventionMode': 'self_trade_prevention_mode', 'priceMatch': 'price_match_mode',
    'goodTillDate': 'gtd_time', 'code': 'err_code', 'msg': 'err_msg'
}


class OrderHandler:
    def __init__(self, is_test: bool = False):
        self.is_test = is_test
        self.client = bnc.Client()
        self.headers = {'X-MBX-APIKEY': TRADE_API['api_key']}
        self.session = aiohttp.ClientSession(headers=self.headers)
        self.event_loop = asyncio.get_event_loop()
        self.wss_client = None

    @staticmethod
    def _get_ordered_params(params):
        params.sort(key=lambda x: x[0])
        return params


class OrderHandlerV1(OrderHandler):
    BASE_URL = 'https://fapi.binance.com'
    SERVER_TIME_SUB_URL = '/fapi/v1/time'
    EXCHANGE_INFO_SUB_URL = '/fapi/v1/exchangeInfo'
    BOOK_TICKER_SUB_URL = '/fapi/v1/ticker/bookTicker'
    TICKER_24HR_URL = '/fapi/v1/ticker/24hr'
    ORDER_SUB_URL = '/fapi/v1/order'
    POSITION_SUB_URL = '/fapi/v2/positionRisk'
    LISTEN_KEY_SUB_URL = '/fapi/v1/listenKey'
    WSS_URI = 'wss://fstream.binance_scripts.com/ws/'

    def __init__(self, is_test: bool = False):
        super().__init__(is_test)
        self.df_exchange_info = None
        self.df_ticker = None
        self.df_24hr_ticker = None
        self.event_loop.run_until_complete(self.get_exchange_info())

    def trade(self, orders_info):
        async def _calc_order_params(order):
            order_type = order.data['order_type']
            match order_type:
                case 'BEST_PRICE':
                    order, order_dict = await self.best_price_order(order)
                case 'MARKET':
                    order, order_dict = await self.market_order(order)
                case 'STOP' | 'TAKE_PROFIT':
                    order, order_dict = await self.stop_order(order)
                case _:
                    raise Exception(f'Invalid order type: {order_type}')
            return order, order_dict

        async def _main():
            orders = orders_info['orders']
            tasks = [_calc_order_params(order) for order in orders]
            orders = await asyncio.gather(*tasks)
            # orders = [orders[0]]  # FIXME

            tasks = [self.place_order(*order) for order in orders]
            orders_res = await asyncio.gather(*tasks)
            return orders_res

        return self.event_loop.run_until_complete(_main())

    async def best_price_order(self, order, has_quantity=False):
        await self.get_book_ticker()
        symbol = order.data['symbol']
        side = order.data['side']
        if not has_quantity:
            quantity_precision = self.df_exchange_info.set_index('symbol')['quantityPrecision'].astype(int)[symbol]
        else:
            quantity_precision = None
        if side == 'BUY':
            best_price = self.df_ticker.set_index('symbol')['bidPrice'][symbol]
        elif side == 'SELL':
            best_price = self.df_ticker.set_index('symbol')['askPrice'][symbol]
        else:
            raise ValueError('side must be "BUY" or "SELL"')
        order = self.add_quantity_price(order, quantity_precision, best_price, has_price=True)
        order_dict = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'quantity': order.data['order_quantity'],
            'price': order.data['order_price'],
            'newClientOrderId': order.data['client_order_id'],
            'recvWindow': '5000',
            'timeinforce': 'GTD',
            'goodtilldate': str(int((pd.Timestamp.now() + pd.Timedelta(days=1)).timestamp() * 1e3))  # FIXME: 这里的参数后续应该需要进一步修改
        }
        return order, order_dict

    async def market_order(self, order, has_quantity=False):
        await self.get_24hr_ticker()
        symbol = order.data['symbol']
        if not has_quantity:
            quantity_precision = self.df_exchange_info.set_index('symbol')['quantityPrecision'].astype(int)[symbol]
            last_price = self.df_24hr_ticker.set_index('symbol')['lastPrice'].astype(float)[symbol]
            order = self.add_quantity_price(order, quantity_precision, last_price)
        order_dict = {
            'symbol': symbol,
            'side': order.data['side'],
            'type': order.data['order_type'],
            'quantity': order.data['order_quantity'],
            'newClientOrderId': order.data['client_order_id'],
            'recvWindow': '5000',
        }
        return order, order_dict

    async def stop_order(self, order, has_quantity=False):
        await self.get_book_ticker()
        symbol = order.data['symbol']
        side = order.data['side']
        if not has_quantity:
            quantity_precision = self.df_exchange_info.set_index('symbol')['quantityPrecision'].astype(int)[symbol]
            last_price = self.df_ticker.set_index('symbol')['lastPrice'].astype(float)[symbol]
            stop_price = order.data['stop_price'] * last_price
            order = self.add_quantity_price(order, quantity_precision, stop_price)
        order_dict = {
            'symbol': symbol,
            'side': side,
            'type': order.data['order_type'],
            'quantity': order.data['order_quantity'],
            'price': order.data['order_price'],
            'stopPrice': order.data['stop_price'],
            'newClientOrderId': order.data['client_order_id'],
            'recvWindow': '5000',
        }
        return order, order_dict

    @staticmethod
    def add_quantity_price(order, quantity_precision, price, price_precision=None, has_price=False):
        quantity = order.data['order_quantity']
        if not isinstance(quantity, str):
            order.data['order_quantity'] = str(np.round(quantity / price, quantity_precision))
        if has_price:
            if not isinstance(price, str):
                order.data['order_price'] = str(np.round(price, price_precision))
            else:
                order.data['order_price'] = price
        return order

    def get_signature(self, params):
        signature = hmac.new(
            TRADE_API['secret_key'].encode(),
            "&".join(["=".join(param) for param in self._get_ordered_params(params)]).encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _get(self, url, params=None, try_until_success=False):
        while True:
            try:
                res = await self.session.get(url, params=params)
                return res
            except asyncio.TimeoutError:
                print(f"{pd.Timestamp.now()}, order_handler, TimeoutError occurs when connecting to {url}")
            except Exception as e:
                print(f"{pd.Timestamp.now()}, order_handler, Error occurs when connecting to {url}: {type(e)} {e}")
            if not try_until_success:
                break

    async def _get_with_signature(self, url, params):
        try:
            server_time = await self.get_server_time()
            params.append(('timestamp', server_time))
            signature = self.get_signature(params)
            params.append(('signature', signature))
            res = await self.session.post(url, data=params, timeout=60)
            return res
        except asyncio.TimeoutError:
            print(f"{pd.Timestamp.now()}, order_handler, TimeoutError occurs when connecting to {url}")

    async def _post(self, url, params=None, try_until_success=False):
        while True:
            try:
                res = await self.session.post(url, params=params)
                return res
            except asyncio.TimeoutError:
                print(f"{pd.Timestamp.now()}, order_handler, TimeoutError occurs when connecting to {url}")
            except Exception as e:
                print(f"{pd.Timestamp.now()}, order_handler, Error occurs when connecting to {url}: {type(e)} {e}")
            if not try_until_success:
                break

    async def _put(self, url, params, try_until_success=False):
        while True:
            try:
                res = await self.session.put(url, params=params)
                return res
            except asyncio.TimeoutError:
                print(f"{pd.Timestamp.now()}, order_handler, TimeoutError occurs when connecting to {url}")
            except Exception as e:
                print(f"{pd.Timestamp.now()}, order_handler, Error occurs when connecting to {url}: {type(e)} {e}")
            if not try_until_success:
                break

    async def get_server_time(self):
        res = await self._get(self.BASE_URL + self.SERVER_TIME_SUB_URL, try_until_success=True)
        server_time = str((await res.json())['serverTime'])
        return server_time

    async def get_exchange_info(self):
        res = await self._get(self.BASE_URL + self.EXCHANGE_INFO_SUB_URL, try_until_success=True)
        res_json = await res.json()
        self.df_exchange_info = pd.DataFrame(res_json['symbols'])

    async def get_book_ticker(self):
        res = await self._get(self.BASE_URL + self.BOOK_TICKER_SUB_URL, try_until_success=True)
        res_json = await res.json()
        self.df_ticker = pd.DataFrame(res_json)

    async def get_24hr_ticker(self):
        res = await self._get(self.BASE_URL + self.TICKER_24HR_URL, try_until_success=True)
        res_json = await res.json()
        self.df_24hr_ticker = pd.DataFrame(res_json)

    async def get_listen_key(self):
        res = await self._post(self.BASE_URL + self.LISTEN_KEY_SUB_URL, try_until_success=True)
        listen_key = (await res.json())['listenKey']
        return listen_key

    async def place_order(self, order, order_dict):
        params = list(order_dict.items())
        url = self.BASE_URL + self.ORDER_SUB_URL + ('/test' if self.is_test else '')
        server_time = await self.get_server_time()
        params.append(('timestamp', server_time))
        signature = self.get_signature(params)
        params.append(('signature', signature))
        res = await self._post(url, params=params, try_until_success=True)
        res_status = res.status
        res_json = await res.json()
        res_json['res_status'] = res_status
        sr_res = pd.Series(res_json).rename(ORDER_RES_DICT)
        order.data = order.data.fillna(sr_res)
        if res_status != 200:
            order.data['order_id'] = 0
            order.data['order_status'] = 'ERR'
        else:
            order.data['order_status'] = 'INIT'
        return order

    async def keep_alive_listen_key(self, listen_key):
        params = {'listenKey': listen_key}
        url = self.BASE_URL + self.LISTEN_KEY_SUB_URL
        res = await self._put(url, params)
        res_status = res.status
        res_json = await res.json()
        return res_status, res_json
