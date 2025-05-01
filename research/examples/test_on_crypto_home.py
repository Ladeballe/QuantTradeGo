import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arctic import Arctic

from test_func.main import factor_test, factor_load, iter_fac_names, iter_formula2
from test_func.data import load_symbols


def main():
    strategy = "crypto_test"
    version = "0.2.0"
    begin_date, end_date = "2023-01-01", "2024-04-01"
    output_path = f"D:/research/crypto_test/test_results/{version}"
    file_tag = ''
    cost_ratio = 0
    skip_finished = False
    calc_rtn_method = 'ByRtn'
    qty_param = 'value_1000'
    initial_capital = 1000
    resample_freq = "15T"
    between_time = None

    symbols = load_symbols(contractType="PERPETUAL", quoteAsset="USDT", status="TRADING")[:50]

    vola_fac_name = ''
    fwd_rtn_name = 'fwd_rtn_15m'  # 预先决定
    fwd_rtn_lib = 'public.public_factor'  # 预先决定

    fac_names = {
        "fac_name0": [
            ("bolling", "bolling_{window}"),
        ]
    }
    fac_libs = {
        "fac_name0": "vp_factor.factor_UNI_15m",
    }

    window = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50, ]
    fac_params_dict = {
        "window": window,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]_AND_{side}{fac_name0}[FFILL,{sig_trans1}]"

    side = ["", "-"]
    sig_trans0 = ["-2_0_0_2"]
    sig_trans1 = ["RBD,0.025_0.5_0.5_0.975"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
        "sig_trans1": sig_trans1,
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
    main()
