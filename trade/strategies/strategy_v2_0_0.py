import numpy as np
import pandas as pd
from apscheduler.triggers.cron import CronTrigger
import xgboost as xgb

from quant_trade_go.strategy import ScheduledStrategy, ModelStrategy, OrderStatus


class TestStrategy(ScheduledStrategy, ModelStrategy):
    def __init__(self, name, trigger, model_path, pos_cond, val_total):
        ScheduledStrategy.__init__(self, name, trigger)
        self.pos_cond = pos_cond
        self.val_total = val_total
        model = xgb.Booster()
        model.load_model(model_path)
        ModelStrategy.__init__(self, name, model)

    def calc_sig(self, df_fac):
        df_fac = (df_fac - df_fac.mean()) / df_fac.std()
        dmat = xgb.DMatrix(df_fac)
        df_sig = pd.Series(
            self.model.predict(dmat), index=df_fac.index)
        self.sig = dict()
        self.sig['ts'] = pd.Timestamp.now().timestamp()
        self.sig['sig'] = df_sig.to_dict()

    def calc_pos(self):
        df_sig = pd.Series(self.sig['sig']).rank(pct=True)
        long_symbols = df_sig[eval(self.pos_cond['long'])].index
        short_symbols = df_sig[eval(self.pos_cond['short'])].index

        sr_pos_long = pd.Series(1 / len(long_symbols), index=long_symbols)
        sr_pos_short = pd.Series(1 / len(short_symbols), index=short_symbols)
        return sr_pos_long, sr_pos_short


class StrategyV2(ModelStrategy):
    def __init__(self, name, model_path, pos_rule, val_total):
        model = xgb.Booster()
        model.load_model(model_path)
        ModelStrategy.__init__(self, name, model)
        self.pos_rule = pos_rule
        self.val_total = val_total
        self.status = 0
        self.status_list = [0]

    def set_status(self, status=None):
        if status is None:
            self.status = not self.status
        else:
            self.status = status
            self.status_list.append(status)

    def calc_sig(self, df_fac):
        df_fac = (df_fac - df_fac.mean()) / df_fac.std()
        dmat = xgb.DMatrix(df_fac)
        df_sig = pd.Series(
            self.model.predict(dmat), index=df_fac.index)
        self.sig = dict()
        self.sig['ts'] = pd.Timestamp.now().timestamp()
        self.sig['sig'] = df_sig.to_dict()

    def open_orders(self):
        df_sig = pd.Series(self.sig['sig']).rank(pct=True)
        buy_symbols = df_sig[eval(self.pos_rule['open_pos']['BUY'])].index
        sell_symbols = df_sig[eval(self.pos_rule['open_pos']['SELL'])].index

        sr_pos_buy = pd.Series(1 / len(buy_symbols), index=buy_symbols) * self.val_total / 2
        sr_pos_sell = pd.Series(1 / len(sell_symbols), index=sell_symbols) * self.val_total / 2

        orders = list()
        for i, (symbol, quantity) in enumerate(sr_pos_buy.items()):
            orders.append(OrderStatus(
                event_time=int(pd.Timestamp.now().timestamp() * 1e3), symbol=symbol,
                client_order_id=f'{self.name}-open-{i}', order_type='MARKET', side='BUY',
                order_quantity=quantity))
        for i, (symbol, quantity) in enumerate(sr_pos_sell.items(), start=sr_pos_buy.shape[0]):
            orders.append(OrderStatus(
                event_time=int(pd.Timestamp.now().timestamp() * 1e3), symbol=symbol,
                client_order_id=f'{self.name}-open-{i}', order_type='MARKET', side='SELL',
                order_quantity=quantity))
        orders_info = {'strategy': self.name, 'orders': orders}
        return orders_info

    def stop_loss_orders(self):
        if self.pos is not None:
            orders = [{
                'symbol': order['symbol'], 'quantity': order['quantity'],
                'price': 1 - order['stop_ratio'] * (2 * (order['side'] == 'BUY') - 1),
                'side': 'SELL' if order['side'] == 'BUY' else 'BUY'
            } for order in self.pos]
            orders_info = {'strategy': self.name, 'type': 'STOP', 'orders': orders}
            return orders_info

    def take_profit_orders(self):
        if self.pos is not None:
            orders = [{
                'symbol': order['symbol'], 'quantity': order['quantity'],
                'price': 1 + order['take_ratio'] * (2 * (order['side'] == 'BUY') - 1),
                'side': 'SELL' if order['side'] == 'BUY' else 'BUY'
            } for order in self.pos]
            orders_info = {'strategy': self.name, 'type': 'LIMIT', 'orders': orders}
            return orders_info

    def close_orders(self):
        orders = list()
        for i, (symbol, pos) in enumerate(self.pos.items()):
            match pos['order_status']:
                case 'ERR':
                    # SKIP the unsuccessful positions
                    pass
                case 'FILLED':
                    orders.append(OrderStatus(
                        event_time=int(pd.Timestamp.now().timestamp() * 1e3), symbol=symbol,
                        client_order_id=f'{self.name}-close-{i}', order_type='BEST_PRICE',
                        side='SELL' if pos['side'] == 'BUY' else 'BUY',
                        order_quantity=pos['quantity']
                    ))
                case _:
                    raise ValueError(f'Unknown order status: {pos["order_status"]}')
        orders_info = {'strategy': self.name, 'orders': orders}
        return orders_info
