import multiprocessing as mp

import numpy as np
import pandas as pd
import json
import xgboost

from test_func import formula as tff, util as tfu


class Strategy:
    def __init__(self, name, model_fpath, df_fac_info, *args, **kwargs):
        self.name = name
        self.model_fpath = model_fpath
        self.df_fac_info = df_fac_info
        for k, v in kwargs.items():
            setattr(self, k, v)


def _calc_fac_mp(fac_name, formula, dict_of_data, queue):
    tokens = tff.parse_formula_to_tokens(formula)
    formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens, False)
    formula_root = tff.calc_formula_ast_tree(formula_ast_tree, dict_of_data, False)
    df_fac = tfu.convert_stacked_frame_to_frame(formula_root.value)
    df_fac = df_fac.iloc[-1].rename(fac_name)
    queue.put(df_fac)


def callback(msg):
    print(msg)


class CrossSecStrategy(Strategy):
    def __init__(self, name, model_fpath, df_fac_info, open_time, close_time, *args, **kwargs):
        super().__init__(name, model_fpath, df_fac_info, *args, **kwargs)
        self.model = xgboost.Booster()
        self.model.load_model(self.model_fpath)
        self.open_time = open_time
        self.close_time = close_time
        self.df_fac = None
        self.df_sig = None
        self.pos = None
        self.pos_open_bool = False

    def calc_fac_mp(self, dict_of_data):
        with mp.Manager() as manager:
            list_df_fac = list()
            queue = manager.Queue()
            pool = mp.Pool(10)
            for i, row in self.df_fac_info.iterrows():
                fac_name, formula = row['name'], row['formula']
                pool.apply_async(func=_calc_fac_mp, args=(fac_name, formula, dict_of_data.copy(), queue),
                                 error_callback=callback)

            pool.close()
            pool.join()
            while not queue.empty():
                df_fac = queue.get()
                list_df_fac.append(df_fac)
        self.df_fac = pd.concat( list_df_fac, axis=1)[self.df_fac_info['name']]

    def calc_fac(self, dict_of_data):
        list_df_fac = list()
        for i, row in self.df_fac_info.iterrows():
            fac_name, formula = row['name'], row['formula']
            tokens = tff.parse_formula_to_tokens(formula)
            formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens, False)
            formula_root = tff.calc_formula_ast_tree(formula_ast_tree, dict_of_data, False)
            df_fac = tfu.convert_stacked_frame_to_frame(formula_root.value)
            df_fac = df_fac.iloc[-1].rename(fac_name)
            list_df_fac.append(df_fac)
        self.df_fac = pd.concat(list_df_fac, axis=1)

    def calc_open_sig(self):
        self.df_sig = pd.Series(self.model.predict(xgboost.DMatrix(self.df_fac.replace([np.inf, -np.inf], np.nan))), index=self.df_fac.index)
        return self.df_sig

    def calc_close_sig(self):
        pass


class SimpleCrossSecStrategy(CrossSecStrategy):
    def calc_open_sig(self):
        return self.df_fac['channel_R_30m'].rename(0)

    def calc_close_sig(self):
        pass
