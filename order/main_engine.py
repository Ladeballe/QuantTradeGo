import time

from data_handler import DataHandler
from factor_handler import FactorHandler
from strategy_handler import StrategyHandler
from order_handler import OrderHandler


class StrategyEngine:
    def __init__(
            self,
            data_handler: DataHandler,
            factor_handler: FactorHandler,
            strategy_handler: StrategyHandler,
            order_handler: OrderHandler
    ):
        self.dh = None
        self.fh = None
        self.sh = None
        self.oh = None

    def start(self):
        pass

    def debug(self):
        pass
