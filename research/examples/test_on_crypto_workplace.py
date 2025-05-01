import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from test_func.main import factor_test, factor_load, iter_fac_names, iter_formula2
from test_func.data import load_symbols


strategy = "crypto_test"
begin_date, end_date = "2023-01-01", "2024-04-01"
file_tag = ''
cost_ratio = 0.00018
skip_finished = False
calc_rtn_method = 'ByRtn'
qty_param = 'value_1000'
initial_capital = 1000
resample_freq = "15T"
between_time = None

symbols = [
    '1000SHIBUSDT', 'ADAUSDT', 'ALGOUSDT', 'ATOMUSDT', 'BAKEUSDT',
     'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCDOMUSDT',
     'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DASHUSDT', 'DEFIUSDT', 'DOGEUSDT',
     'DOTUSDT', 'EGLDUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'GTCUSDT',
     'ICXUSDT', 'IOSTUSDT', 'IOTAUSDT', 'KAVAUSDT', 'KNCUSDT', 'LINKUSDT',
     'LTCUSDT', 'MKRUSDT', 'MTLUSDT', 'NEOUSDT', 'NKNUSDT', 'OGNUSDT',
     'OMGUSDT', 'ONTUSDT', 'QTUMUSDT', 'RLCUSDT', 'RUNEUSDT', 'SNXUSDT',
     'SOLUSDT', 'SRMUSDT', 'STORJUSDT', 'SUSHIUSDT', 'SXPUSDT', 'THETAUSDT',
     'TRBUSDT', 'TRXUSDT', 'VETUSDT', 'WAVESUSDT', 'XLMUSDT', 'XMRUSDT',
     'XRPUSDT', 'XTZUSDT', 'YFIUSDT', 'ZECUSDT', 'ZILUSDT', 'ZRXUSDT'
]

vola_fac_name = ''
fwd_rtn_name = 'fwd_rtn_15m'  # 预先决定
fwd_rtn_lib = 'public.public_factor'  # 预先决定


def backtest_double_bolling():
    version = "0.1.0"
    output_path = f"D:/research/crypto_test/test_results/{version}"

    fac_names = {
        "fac_name0": [
            ("bolling", "bolling_{window}"),
        ]
    }
    fac_libs = {
        "fac_name0": "vp_fac.factor_UNI_15m",
    }

    window = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50, ]
    fac_params_dict = {
        "window": window,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]_AND_{side}{fac_name0}[FFILL,{sig_trans1}]"

    side = ["", "-"]
    sig_trans0 = ["-2_0_0_2"]
    sig_trans1 = ["RBD,0.1_0.5_0.5_0.9"]
    # sig_trans_put_off = ["PUTOFF_1"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
        "sig_trans1": sig_trans1,
        # "sig_trans_put_off": sig_trans_put_off,
    }

    for format_fac_names, format_fac_types, format_fac_params in iter_fac_names(
        fac_names, fac_params_dict
    ):
        df_factor = factor_load(symbols, format_fac_names, fac_libs, begin_date, end_date, between_time)
        for formula, res_params_dict in iter_formula2(
            base_formula, format_fac_names, format_fac_types, formula_params_dict
        ):
            res_params_dict.update(format_fac_params)
            test_params = {
                'params': res_params_dict, 'skip_finished': skip_finished,
                'file_tag': file_tag, 'symbols': symbols,
                'vola_fac_name': vola_fac_name, 'fwd_rtn_name': fwd_rtn_name, 'fwd_rtn_lib': fwd_rtn_lib,
                'cost_ratio': cost_ratio, 'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param,
                'initial_capital': initial_capital, 'resample_freq': resample_freq,
                'fac_names': format_fac_names, 'fac_types': format_fac_types, 'fac_libs': fac_libs
            }
            factor_test(strategy, version, formula, begin_date, end_date, df_factor, output_path, test_params)
    print('done')


def backtest_bolling():
    version = "0.3.0"
    output_path = f"D:/research/crypto_test/test_results/{version}"

    fac_names = {
        "fac_name0": [
            ("bolling", "bolling_{window}"),
        ]
    }
    fac_libs = {
        "fac_name0": "vp_fac.factor_UNI_15m",
    }

    window = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50, ]
    fac_params_dict = {
        "window": window,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]"

    side = ["", "-"]
    sig_trans0 = ["-2_0_0_2", "RBD,0.1_0.5_0.5_0.9"]
    # sig_trans_put_off = ["PUTOFF_1"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
        # "sig_trans_put_off": sig_trans_put_off,
    }

    for format_fac_names, format_fac_types, format_fac_params in iter_fac_names(
        fac_names, fac_params_dict
    ):
        df_factor = factor_load(symbols, format_fac_names, fac_libs, begin_date, end_date, between_time)
        for formula, res_params_dict in iter_formula2(
            base_formula, format_fac_names, format_fac_types, formula_params_dict
        ):
            res_params_dict.update(format_fac_params)
            test_params = {
                'params': res_params_dict, 'skip_finished': skip_finished,
                'file_tag': file_tag, 'symbols': symbols,
                'vola_fac_name': vola_fac_name, 'fwd_rtn_name': fwd_rtn_name, 'fwd_rtn_lib': fwd_rtn_lib,
                'cost_ratio': cost_ratio, 'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param,
                'initial_capital': initial_capital, 'resample_freq': resample_freq,
                'fac_names': format_fac_names, 'fac_types': format_fac_types, 'fac_libs': fac_libs
            }
            factor_test(strategy, version, formula, begin_date, end_date, df_factor, output_path, test_params)
    print('done')


def backtest_3ma():
    version = "0.4.0"
    output_path = f"D:/research/crypto_test/test_results/{version}"

    fac_names = {
        "fac_name0": [
            ("3ma", "2ma_{window0}_{window1}"),
        ],
        "fac_name1": [
            ("3ma", "2ma_{window1}_{window2}"),
        ],
    }
    fac_libs = {
        "fac_name0": "vp_fac.factor_UNI_15m",
        "fac_name1": "vp_fac.factor_UNI_15m",
    }

    window0 = [8, 15, 50, 100, 200, 500, 1000]
    window1 = [32, 60, 200, 400, 800, 2000, 4000]
    window2 = [128, 240, 800, 1600, 3200, 8000, 16000]
    fac_params_dict = {
        "window0": window0,
        "window1": window1,
        "window2": window2,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]_AND_{side}{fac_name1}[FFILL,{sig_trans0}]"

    side = ["", "-"]
    sig_trans0 = ["0"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
    }

    for format_fac_names, format_fac_types, format_fac_params in iter_fac_names(
            fac_names, fac_params_dict, iter_fac_names_type="order", iter_fac_params_type="order"
    ):
        df_factor = factor_load(symbols, format_fac_names, fac_libs, begin_date, end_date, between_time)
        for formula, res_params_dict in iter_formula2(
                base_formula, format_fac_names, format_fac_types, formula_params_dict
        ):
            res_params_dict.update(format_fac_params)
            test_params = {
                'params': res_params_dict, 'skip_finished': skip_finished,
                'file_tag': file_tag, 'symbols': symbols,
                'vola_fac_name': vola_fac_name, 'fwd_rtn_name': fwd_rtn_name, 'fwd_rtn_lib': fwd_rtn_lib,
                'cost_ratio': cost_ratio, 'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param,
                'initial_capital': initial_capital, 'resample_freq': resample_freq,
                'fac_names': format_fac_names, 'fac_types': format_fac_types, 'fac_libs': fac_libs
            }
            factor_test(strategy, version, formula, begin_date, end_date, df_factor, output_path, test_params)
    print('done')


def backtest_bolling_volume():
    version = "0.5.0"
    output_path = f"D:/research/crypto_test/test_results/{version}"

    fac_names = {
        "fac_name0": [
            ("bolling", "bolling_{window}"),
        ],
        "fac_name1": [
            ("volume", "volume"),
        ]
    }
    fac_libs = {
        "fac_name0": "vp_fac.factor_UNI_15m",
        "fac_name1": "vp_fac.factor_UNI_15m",
    }

    window = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50, ]
    fac_params_dict = {
        "window": window,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]_ENTRYFILT_{side}{fac_name1}[FFILL,{sig_trans1}]"

    side = ["", "-"]
    sig_trans0 = ["-2_0_0_2"]
    sig_trans1 = ["RBW_720,-1_0_0.85_0.95"]
    # sig_trans_put_off = ["PUTOFF_1"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
        "sig_trans1": sig_trans1,
        # "sig_trans_put_off": sig_trans_put_off,
    }

    for format_fac_names, format_fac_types, format_fac_params in iter_fac_names(
        fac_names, fac_params_dict
    ):
        df_factor = factor_load(symbols, format_fac_names, fac_libs, begin_date, end_date, between_time)
        for formula, res_params_dict in iter_formula2(
            base_formula, format_fac_names, format_fac_types, formula_params_dict
        ):
            res_params_dict.update(format_fac_params)
            test_params = {
                'params': res_params_dict, 'skip_finished': skip_finished,
                'file_tag': file_tag, 'symbols': symbols,
                'vola_fac_name': vola_fac_name, 'fwd_rtn_name': fwd_rtn_name, 'fwd_rtn_lib': fwd_rtn_lib,
                'cost_ratio': cost_ratio, 'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param,
                'initial_capital': initial_capital, 'resample_freq': resample_freq,
                'fac_names': format_fac_names, 'fac_types': format_fac_types, 'fac_libs': fac_libs
            }
            factor_test(strategy, version, formula, begin_date, end_date, df_factor, output_path, test_params)
    print('done')


if __name__ == "__main__":
    # backtest_double_bolling()
    # backtest_bolling()
    # backtest_3ma()
    backtest_bolling_volume()
