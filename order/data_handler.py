import time

import pandas as pd
import sqlalchemy


class DataHandler:
    def __init__(self):
        self.df = None

    def start(self):
        pass


class DataHandlerV1(DataHandler):
    def __init__(self, hours, symbols, sheet):
        super().__init__()
        self.hours = hours
        self.symbols = symbols
        self.sheet = sheet
        self.engine = None
        self.dict_vars = {
            't': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
            'q': 'amt', 'bq': 'amt_buy', 'n': 'trade_num', 'x': 'is_finished'}

    def init_engine(self):
        self.engine = sqlalchemy.create_engine("mysql+pymysql://root:444666@localhost:3306/market_data")

    def read_data(self):
        symbols = ','.join([f"'{symbol}'" for symbol in self.symbols])
        ts_now = int(time.time() * 1000)
        # ts_now = int(pd.Timestamp('2025-01-02 06:00:00').timestamp() * 1000)
        ts_0 = int(ts_now / 1000 // (5 * 60) * (5 * 60) * 1000 - self.hours * 3600 * 1000)
        self.df = pd.read_sql(
            f"select * from {self.sheet} where ts0 > {ts_0} and ts0 <= {ts_now} and "
            f"symbol in ({symbols})",
            self.engine
        ).rename(self.dict_vars, axis=1)

    def update_data(self):
        symbols = ','.join([f"'{symbol}'" for symbol in self.symbols])
        ts_now = int(time.time() // (5 * 60) * (5 * 60) * 1000)
        ts_load_0 = ts_now - 5 * 60 * 1000  # 更新最近2根bar
        # ts_now = int(pd.Timestamp('2025-01-02 06:00:00').timestamp() * 1000)
        ts_0 = ts_now - self.hours * 3600 * 1000
        updated_df = pd.read_sql(
            f"select * from {self.sheet} where ts0 >= {ts_load_0} and "
            f"symbol in ({symbols})",
            self.engine
        ).rename(self.dict_vars, axis=1)
        ratio = updated_df[updated_df['ts0'] == ts_now].shape[0] / len(self.symbols)  # 因为更改了bar的更新策略，所以需要修改该部分
        self.df = pd.concat([self.df, updated_df]).drop_duplicates(subset=['ts0', 'symbol'], keep='last')
        self.df = self.df[self.df['ts0'] > ts_0]
        return ratio

    def get_dict_df(self):
        dict_of_data = dict()
        df = self.df.copy()
        df.index = self.df['date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '_' + df['symbol']
        for name in self.dict_vars.values():
            dict_of_data[name] = df[['date', 'symbol', name]].rename({name: 'raw_factor'}, axis=1)
        return dict_of_data
