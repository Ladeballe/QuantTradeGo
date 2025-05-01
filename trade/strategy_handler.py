import datetime
from queue import Empty

import asyncio
import aiomysql
import pandas as pd

from .strategy import OrderStatus


class StrategyHandler:
    def __init__(self, strategies, queue_fac, queue_fac_task, queue_order):
        self.strategies = strategies
        self.queue_fac = queue_fac
        self.queue_fac_task = queue_fac_task
        self.queue_order = queue_order
        self.dict_fac = dict()


class StrategyHandlerV2(StrategyHandler):
    event_loop = None
    mysql_conn = None

    def connect_to_mysql(self):
        self.event_loop = asyncio.get_event_loop()

        async def _main():
            self.mysql_conn = await aiomysql.connect(
                host='127.0.0.1', port=3306, user='root', password='444666',
                db='trader_data', autocommit=True, loop=self.event_loop)

        self.event_loop.run_until_complete(_main())

    @staticmethod
    def _time_trigger(time_str):
        trigger_bool = pd.Timestamp.now().time() > datetime.time.fromisoformat(time_str) > (pd.Timestamp.now() - pd.Timedelta(minutes=5)).time()
        return trigger_bool

    def run_strategy(self, strategy):
        """ 包含策略是否执行的判断，策略执行的步骤 """
        strategy_name = strategy.name
        strategy_status = strategy.status
        pos_rule = strategy.pos_rule
        # print(f"{pd.Timestamp.now()}, strategy_process running... : {strategy_name} {strategy_status}")
        match strategy_status:
            case 0:  # 空仓状态
                if self._time_trigger(pos_rule['open_pos']['time']):
                    print(f"{pd.Timestamp.now()} strategy_process, {strategy_name}, open_pos")
                    strategy.set_status(1)
                    self.queue_fac_task.put((strategy, pos_rule['open_pos']['time']))
                    while True:  # FIXME: hiding dead loop
                        try:
                            fac_ts, strategy1, df_fac = self.queue_fac.get(timeout=1)
                            print(f"{pd.Timestamp.now()}, strategy_process, {strategy_name}, get_df_fac, {fac_ts}, {strategy_name}, {strategy1.name}")
                            if strategy1.name == strategy_name:
                                break
                            else:
                                self.queue_fac.put((fac_ts, strategy1, df_fac))
                        except Empty:
                            continue
                    print(f"{pd.Timestamp.now()}, strategy_process, {strategy_name}, calculate signal...")
                    strategy.calc_sig(df_fac)
                    orders_info = strategy.open_orders()
                    self.queue_order.put(orders_info)
            # case 1:  # 仍然处于交易状态
            case 2:  # 持仓状态
                if self._time_trigger(pos_rule['force_close_pos']['time']):
                    print(f"{pd.Timestamp.now()}, strategy_process, close_pos")
                    orders_info = strategy.close_orders()
                    strategy.set_status(1)
                    self.queue_order.put(orders_info)

    def update_strategy(self, order_res):
        self.event_loop.run_until_complete(order_res.insert_into_sql(self.mysql_conn))

        client_order_id = order_res.data['client_order_id']
        symbol = order_res.data['symbol']
        side = order_res.data['side']
        order_status = order_res.data['order_status']
        quantity = order_res.data['cumulative_quantity']
        price = order_res.data['order_price']
        average_price = order_res.data['average_price']

        strategy_name, act, idx = client_order_id.split('-')
        strategy = self.strategies[strategy_name]
        strategy.orders.append(order_res)

        match order_status:
            case 'INIT':
                # INIT的信息返回的时间很晚，不记录在pos中
                # strategy.pos[symbol] = {
                #     'side': side,
                #     'quantity': '0',
                #     'price': price,
                #     'order_status': order_status,
                # }
                pass
            case 'NEW':
                strategy.pos[symbol] = {
                    'side': side,
                    'quantity': '0',
                    'price': price,
                    'order_status': order_status,
                }
            case 'PARTIALLY_FILLED' | 'FILLED':
                strategy.pos[symbol]['quantity'] = quantity
                strategy.pos[symbol]['average_price'] = average_price
                strategy.pos[symbol]['order_status'] = order_status
        df_pos = pd.DataFrame(strategy.pos).T
        print(df_pos)
        if len(strategy.pos) > 1 and (df_pos['order_status'].isin(['FILLED', 'ERR'])).all():
            match strategy.status_list[-2]:  # 需要前一个状态的信息，避免重复下单
                case 0:
                    strategy.set_status(2)
                case 2:
                    strategy.set_status(0)
