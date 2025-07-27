import json

import asyncio

from quant_trade_go.order_handler import OrderHandlerV1


def main():
    oh = OrderHandlerV1(True)
    orders = json.loads(
        """[{'symbol': '1000FLOKIUSDT', 'type': 'MARKET', 'quantity': '820.0', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'DOGEUSDT', 'type': 'MARKET', 'quantity': '453.0', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'FLMUSDT', 'type': 'MARKET', 'quantity': '2038.0', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'OMGUSDT', 'type': 'MARKET', 'quantity': '436.6', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'STMXUSDT', 'type': 'MARKET', 'quantity': '24958.0', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'TLMUSDT', 'type': 'MARKET', 'quantity': '10740.0', 'side': 'BUY', 'recvWindow': '5000'}, {'symbol': 'AGLDUSDT', 'type': 'MARKET', 'quantity': '33.0', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'ALGOUSDT', 'type': 'MARKET', 'quantity': '211.1', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'CRVUSDT', 'type': 'MARKET', 'quantity': '80.9', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'DENTUSDT', 'type': 'MARKET', 'quantity': '53651.0', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'IOTAUSDT', 'type': 'MARKET', 'quantity': '247.3', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'POPCATUSDT', 'type': 'MARKET', 'quantity': '101.0', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'RSRUSDT', 'type': 'MARKET', 'quantity': '5516.0', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'STGUSDT', 'type': 'MARKET', 'quantity': '203.0', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'TAOUSDT', 'type': 'MARKET', 'quantity': '0.158', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'WIFUSDT', 'type': 'MARKET', 'quantity': '42.2', 'side': 'SELL', 'recvWindow': '5000'}, {'symbol': 'XVGUSDT', 'type': 'MARKET', 'quantity': '5593.0', 'side': 'SELL', 'recvWindow': '5000'}]""".replace("'", '"')
    )

    tasks = [oh.place_order(order) for order in orders]

    async def _main():
        list_res = await asyncio.gather(*tasks)
        return list_res

    loop = asyncio.get_event_loop()
    list_res = loop.run_until_complete(_main())
    print(list_res)


if __name__ == '__main__':
    main()