import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightweight_charts import Chart

from test_func.main import factor_test, factor_load, iter_fac_names, iter_formula2, public_factor_load
from test_func.formula import expr_trans_sig
from test_func.backtest import calc_rtn_by_signal
from test_func.data import load_ohlcv_data, load_symbols

idx = pd.IndexSlice

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

vola_fac_name = ''
fwd_rtn_name = 'fwd_rtn_15m'  # 预先决定
fwd_rtn_lib = 'public.public_factor'  # 预先决定


if __name__ == "__main__":
    version = "0.1.0"
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
    # symbols = load_symbols()
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'ORDIUSDT', 'DOGEUSDT', 'SOLUSDT', 'AVAXUSDT', 'ARBUSDT']

    window = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10, 20, 50, ]
    fac_params_dict = {
        "window": window,
    }

    base_formula = "{side}{fac_name0}[FFILL,{sig_trans0}]_ENTRYFILT_{side}{fac_name1}[FFILL,{sig_trans1}]"

    side = ["", "-"]
    sig_trans0 = ["-2_0_0_2"]
    sig_trans1 = ["RBW_200,-1_0_0.85_0.95"]
    # sig_trans_put_off = ["PUTOFF_1"]

    formula_params_dict = {
        "side": side,
        "sig_trans0": sig_trans0,
        "sig_trans1": sig_trans1,
        # "sig_trans_put_off": sig_trans_put_off,
    }
    # df_ohlcv = load_ohlcv_data(symbols)
    # df_ohlcv = df_ohlcv.drop('symbol', axis=1).rename(
    #     {'t': 'time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, axis=1)

    df_ohlcv = pd.read_csv('D:/python_projects/test_func/examples/ohlcv.csv', index_col=1)

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

            df_fwd_rtn = public_factor_load(fwd_rtn_name, fwd_rtn_lib, begin_date, end_date).stack().rename("fwd_rtn")
            df_price = public_factor_load("price_15m", "public.public_factor", begin_date, end_date)
            df_factor = df_factor.unstack().truncate(begin_date, end_date).stack()
            df_signal = expr_trans_sig(formula, df_factor)  # FIXME: expr_trans_sig需要改进
            df_fwd_rtn = df_fwd_rtn.reindex(df_signal.index)

            if calc_rtn_method == "ByRtn":
                df_rtn_merge = calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio=cost_ratio)

            symbol = symbols[0]
            chart = Chart(inner_width=1, inner_height=0.5)
            chart1 = chart.create_subchart(width=1, height=0.15, sync=True)
            chart2 = chart.create_subchart(width=1, height=0.15, sync=True)
            chart3 = chart.create_subchart(width=1, height=0.2, sync=True)

            index = df_ohlcv.index
            df_factor_draw = pd.concat([
                df_rtn_merge['all'].cumsum().to_frame('cum_rtn'),
                df_factor,
                expr_trans_sig("bolling_100[FFILL,-2_0_0_2]", df_factor).rename('bolling_signal'),
                expr_trans_sig("volume[FFILL,RBW_720]", df_factor).rename("volume_rbw"),
                expr_trans_sig(f"volume[FFILL,{sig_trans1[0]}]", df_factor).rename("volume_rbw_signal"),
                df_signal * 2
            ], axis=1).loc[idx[:, symbol], :].droplevel(1)

            chart.set(df_ohlcv)

            window = format_fac_params['window']
            mean = df_ohlcv['open'].rolling(window).mean()
            std = df_ohlcv['open'].rolling(window).std()
            upper_band = mean + 2 * std
            lower_band = mean - 2 * std

            ma_line = chart.create_line(f"ma{window}", color="cyan")
            upper_band_line = chart.create_line(f"upper", color="yellow")
            lower_band_line = chart.create_line(f"lower", color="yellow")

            ma_line.set(mean.to_frame(f"ma{window}"))
            upper_band_line.set(upper_band.to_frame("upper"))
            lower_band_line.set(lower_band.to_frame("lower"))

            df_signal_draw = pd.concat([df_signal.loc[idx[:, symbol]],
                                        df_signal.loc[idx[:, symbol]]
                                           .diff().abs().cumsum().fillna(0).rename('signal_group')],
                                       axis=1)
            df_signal_marker = df_signal_draw.reset_index().groupby('signal_group').agg({'date': ['first', 'last'], 'signal': 'first'})
            df_signal_marker.columns = ['begin_date', 'end_date', 'signal']

            for i, sr in df_signal_marker.iterrows():
                if sr['signal'] == 1:
                    chart.marker(sr['begin_date'].to_pydatetime(), "below", "arrow_up", "red", "l")
                    chart.marker(sr['end_date'].to_pydatetime(), "above", "arrow_down", "purple", "lc")
                elif sr['signal'] == -1:
                    chart.marker(sr['begin_date'].to_pydatetime(), "above", "arrow_down", "blue", "s")
                    chart.marker(sr['end_date'].to_pydatetime(), "below", "arrow_up", "purple", "sc")

            fac_line = chart1.create_line(format_fac_names["fac_name0"], color="orange")
            sig_line = chart1.create_line(f"signal", color="white")
            fac_line.set(df_factor_draw)
            sig_line.set(df_factor_draw)

            volume_line = chart2.create_line(format_fac_names["fac_name1"] + "_rbw", color="red")
            volume_line.set(df_factor_draw)

            rtn_line = chart3.create_line("cum_rtn", color="purple")
            rtn_line.set(df_factor_draw)

            chart.show(block=False)
    print("done")
