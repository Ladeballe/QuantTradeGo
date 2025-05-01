import datetime
import traceback

import numpy as np
import pandas as pd


class Strategy:
    def __init__(self, name):
        self.name = name
        self.sig = None
        self.status = False
        self.orders = list()
        self.pos = dict()

    def set_status(self, status=None):
        if status is None:
            self.status = not self.status
        else:
            self.status = status

    def calc_sig(self, df_fac):
        pass


class ScheduledStrategy(Strategy):
    def __init__(self, name, trigger):
        super().__init__(self, name)
        self.trigger = datetime.time.fromisoformat(trigger)

    def config_trigger(self, scheduler):
        return

    def change_trigger(self, trigger):
        self.trigger = datetime.time.fromisoformat(trigger)


class ModelStrategy(Strategy):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def calc_sig(self, df_fac):
        return


class OrderStatus:
    def __init__(
            self,
            event_time=0,
            symbol=None,
            client_order_id=None,
            side=None,
            order_type=None,
            time_in_force=None,
            order_quantity=None,
            order_price=None,
            average_price=None,
            stop_price=None,
            execution_type=None,
            order_status=None,
            order_id=None,
            last_executed_quantity=None,
            cumulative_quantity=None,
            last_executed_price=None,
            commission_asset=None,
            commission_quantity=None,
            trade_time=None,
            trade_id=None,
            buy_bust_price=None,
            sell_bust_price=None,
            is_maker=None,
            reduce_only=None,
            trigger_type=None,
            original_order_type=None,
            position_side=None,
            is_activation_price=None,
            activation_price=None,
            callback_rate=None,
            is_conditional_order=None,
            profit=None,
            self_trade_prevention_mode=None,
            price_match_mode=None,
            gtd_time=None,
            res_status=None,
            err_code=None,
            err_msg=None
    ):
        self.data = pd.Series(
            [
                event_time, symbol, client_order_id, side,
                order_type, time_in_force, order_quantity, order_price,
                average_price, stop_price, execution_type, order_status,
                order_id, last_executed_quantity, cumulative_quantity,
                last_executed_price, commission_asset, commission_quantity,
                trade_time, trade_id, buy_bust_price, sell_bust_price, is_maker,
                reduce_only, trigger_type, original_order_type, position_side,
                is_activation_price, activation_price, callback_rate, is_conditional_order,
                profit, self_trade_prevention_mode, price_match_mode, gtd_time,
                res_status, err_code, err_msg
            ],
            index=[
                'event_time', 'symbol', 'client_order_id', 'side',
                'order_type', 'time_in_force', 'order_quantity', 'order_price',
                'average_price', 'stop_price', 'execution_type', 'order_status',
                'order_id', 'last_executed_quantity', 'cumulative_quantity',
                'last_executed_price', 'commission_asset', 'commission_quantity',
                'trade_time', 'trade_id', 'buy_bust_price',
                'sell_bust_price', 'is_maker', 'reduce_only', 'trigger_type',
                'original_order_type', 'position_side', 'is_activation_price',
                'activation_price', 'callback_rate', 'is_conditional_order',
                'profit', 'self_trade_prevention_mode', 'price_match_mode', 'gtd_time',
                'res_status', 'err_code', 'err_msg'
                ],
            name=order_id)

    @classmethod
    def from_wss_json(cls, message):
        kwargs = {
            'event_time': message['E'],
            'symbol': message['o']['s'],
            'client_order_id': message['o']['c'],
            'side': message['o']['S'],
            'order_type': message['o']['o'],
            'time_in_force': message['o']['f'],
            'order_quantity': message['o']['q'],
            'order_price': message['o']['p'],
            'average_price': message['o']['ap'],
            'stop_price': message['o']['sp'],
            'execution_type': message['o']['x'],
            'order_status': message['o']['X'],
            'order_id': message['o']['i'],
            'last_executed_quantity': message['o']['l'],
            'cumulative_quantity': message['o']['z'],
            'last_executed_price': message['o']['L'],
            'commission_asset': message['o']['N'],
            'commission_quantity': message['o']['n'],
            'trade_time': message['o']['T'],
            'trade_id': message['o']['t'],
            'buy_bust_price': message['o']['b'],
            'sell_bust_price': message['o']['a'],
            'is_maker': message['o']['m'],
            'reduce_only': message['o']['R'],
            'trigger_type': message['o']['wt'],
            'original_order_type': message['o']['ot'],
            'position_side': message['o']['ps'],
            'is_conditional_order': message['o']['pP'],
            'profit': message['o']['rp'],
            'self_trade_prevention_mode': message['o']['V'],
            'price_match_mode': message['o']['pm'],
            'gtd_time': message['o']['gtd']
        }
        return cls(**kwargs)

    def __repr__(self):
        return self.data.__repr__()

    async def insert_into_sql(self, mysql_conn):
        cursor = await mysql_conn.cursor()
        sql = """
            INSERT INTO order_trade_update (
                id, event_time, symbol, client_order_id, side,
                order_type, time_in_force, order_quantity, order_price,
                average_price, stop_price, execution_type, order_status,
                order_id, last_executed_quantity, cumulative_quantity,
                last_executed_price, commission_asset, commission_quantity,
                trade_time, trade_id, buy_bust_price, sell_bust_price, is_maker,
                reduce_only, trigger_type, original_order_type, position_side,
                is_activation_price, activation_price, callback_rate, is_conditional_order,
                profit, self_trade_prevention_mode, price_match_mode, gtd_time,
                res_status, err_code, err_msg
            ) VALUES (
                UUID(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        data = list(self.data.replace(np.nan, None))
        print(data)
        await cursor.execute(sql, data)
        await cursor.close()
