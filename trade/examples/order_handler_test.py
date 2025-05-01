import asyncio

import pandas as pd
import sqlalchemy
from sqlalchemy import text

from quant_trade_go.order_handler import OrderHandlerV1
from quant_trade_go.strategy import OrderStatus


async def main():
    oh = OrderHandlerV1(False)

    params = []
    server_time = await oh.get_server_time()
    params.append(('timestamp', server_time))
    signature = oh.get_signature(params)
    params.append(('signature', signature))
    res = await oh.session.get('https://fapi.binance.com/fapi/v3/positionRisk', params=params)
    df_pos = pd.DataFrame(await res.json())
    df_pos['positionAmtFloat'] = df_pos['positionAmt'].astype(float)
    df_pos = df_pos[df_pos['positionAmtFloat']!=0]

    orders = [
        oh.best_price_order(
            OrderStatus(
                symbol=sr['symbol'], order_quantity=sr['positionAmt'].strip('-'),
                side='BUY' if sr['positionAmt'][0] == '-' else 'SELL',
                client_order_id=f'manual_close_order_{i}', order_type='BEST_PRICE'
            ),
            True
        ) for i, sr in df_pos.iterrows()]
    orders = await asyncio.gather(*orders)
    # res = asyncio.run(oh.place_order(order))
    print(orders)
    orders_res = await asyncio.gather(*[oh.place_order(*order) for order in orders])
    print(orders_res)

    oh.session.close()


async def main2():
    oh = OrderHandlerV1(True)
    await oh.best_price_order()


if __name__ == '__main__':
    asyncio.run(main2())
