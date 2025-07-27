import asyncio
import aiohttp
import aiomysql


async def main():
    loop = asyncio.get_event_loop()
    mysql_conn = await aiomysql.connect(
        host='127.0.0.1', port=3306, user='root', password='444666',
        db='trader_data', autocommit=True, loop=loop)
    cursor = await mysql_conn.cursor()
    sql = """
            REPLACE INTO order_trade_update (
            event_type, event_time, symbol, client_order_id, side, order_type, time_in_force,
                order_quantity, order_price, average_price, stop_price, execution_type, order_status,
                order_id, last_executed_quantity, cumulative_filled_quantity, last_executed_price,
                commission_asset, commission_quantity, trade_time, trade_id, buy_bust_price,
                sell_bust_price, is_maker, reduce_only, trigger_type, original_order_type,
                position_side, is_conditional_order, profit, self_trade_prevention_mode, 
                price_match_mode, gtd_time
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
    data = (
        'ORDER_TRADE_UPDATE', 1736414376903, 'ZRXUSDT', 'ios_C5v1PwMMLLLr9wn5G1Vz', 'BUY', 'LIMIT',
        'GTC', '40.8', '0.4900', '0', '0', 'NEW', 'NEW', 8643028258, '0', '0', '0', 'USDT', '0',
        1736414376903, 0, '19.99200', '0', False, True, 'CONTRACT_PRICE', 'LIMIT', 'BOTH', False,
        '0', 'EXPIRE_MAKER', 'NONE', 0
    )
    await cursor.execute(sql, data)
    await cursor.close()


if __name__ == '__main__':
    asyncio.run(main())
