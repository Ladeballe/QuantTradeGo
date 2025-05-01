import pandas as pd
import arctic

from test_func import data


def ohlcv_save_to_sql(begin_date, end_date, sheet, lib_name):
    df_ohlcv = data.load_ohlcv_data(begin_date=begin_date, end_date=end_date, sheet=sheet)
    df_ohlcv = df_ohlcv.set_index(['t', 'symbol'])
    for fac_name, col_name in zip(
        ['open', 'high', 'low', 'close', 'amt', 'amt_buy', 'vol', 'vol_buy', 'trade_num'],
        ['o', 'h', 'l', 'c', 'q', 'bq', 'v', 'bv', 'n']
    ):
        df_fac = df_ohlcv[col_name].unstack().rename_axis('date')
        data.save_factor(df_fac, fac_name, lib_name)


def config_data_save_to_sql(begin_date, end_date, sheet, lib_name, fac_names, col_names):
    df_data = data.load_bnc_oi_data(begin_date=begin_date, end_date=end_date, sheet=sheet)
    df_data['t'] = pd.to_datetime(df_data['ts0'] * 1e6)
    df_data = df_data.set_index(['t', 'symbol'])
    for fac_name, col_name in zip(fac_names, col_names):
        try:
            df_fac = df_data[col_name].unstack().rename_axis('date')
            data.save_factor(df_fac, fac_name, lib_name)
        except ValueError:
            continue


if __name__ == '__main__':
    is_run = True
    if is_run:
        begin_date, end_date = '2025-02-01', '2025-04-30'
        period_range = pd.period_range(begin_date, end_date, freq='1M')
        for period in period_range:
            begin_date, end_date = period.start_time, period.end_time
            ohlcv_save_to_sql(begin_date, end_date, 'bnc_kline_15m', 'fac_15m.fac_basic')
            ohlcv_save_to_sql(begin_date, end_date, 'bnc_kline_5m', 'fac_5m.fac_basic')

    is_run = False
    if is_run:
        begin_date, end_date = '2024-11-01', '2025-01-31'
        period_range = pd.period_range(begin_date, end_date, freq='1M')
        for period in period_range:
            begin_date, end_date = period.start_time, period.end_time
            config_data_save_to_sql(begin_date, end_date, 'bnc_oi_5m', 'fac_5m.fac_basic', ['oi', 'oiv'], ['oi', 'oiv'])
            config_data_save_to_sql(begin_date, end_date, 'bnc_taker_ls_ratio_5m', 'fac_5m.fac_basic', ['long_taker', 'short_taker', 'long_short_taker_ratio'], ['l', 's', 'lsr'])
            config_data_save_to_sql(begin_date, end_date, 'bnc_top_ls_acct_ratio_5m', 'fac_5m.fac_basic', ['long_acct', 'short_acct', 'long_short_acct_ratio'], ['l', 's', 'lsr'])
            config_data_save_to_sql(begin_date, end_date, 'bnc_top_ls_pos_ratio_5m', 'fac_5m.fac_basic', ['long_pos', 'short_pos', 'long_short_pos_ratio'], ['l', 's', 'lsr'])
            config_data_save_to_sql(begin_date, end_date, 'bnc_global_ls_acct_ratio_5m', 'fac_5m.fac_basic', ['long_gacct', 'short_gacct', 'long_short_gacct_ratio'], ['l', 's', 'lsr'])
