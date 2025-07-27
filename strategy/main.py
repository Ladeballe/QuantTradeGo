import tqdm
import numpy as np
import pandas as pd
import pymongo
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns

from .. import data
from ..main import iter_params, iter_flatten_df_fac_info
from ..util import convert_frame_to_stack


def load_dict_df_fac(df_fac_info, time, begin_date, end_date, symbols):
    dict_df_fac = dict()
    for i, row in tqdm.tqdm(df_fac_info.iterrows(), total=df_fac_info.shape[0]):
        fac_name, lib_name = row['name'], row['save_lib_name']
        df_fac = data.get_factor(fac_name, lib_name, begin_date, end_date)\
            .at_time(time)\
            .rename_axis('symbol', axis=1)
        df_fac = df_fac[df_fac.columns.intersection(symbols)]
        df_fac = convert_frame_to_stack(df_fac)\
            .reset_index()\
            .set_index(['index', 'date', 'symbol'])
        dict_df_fac[fac_name] = df_fac
    df_fac = pd.concat(list(dict_df_fac.values()), keys=dict_df_fac.keys(), axis=1).sort_index().droplevel(1, axis=1)
    return df_fac


def load_df_fwd_rtn(fwd_rtn_name, time, begin_date, end_date, symbols):
    df_fwd_rtn = data.get_factor(fwd_rtn_name, 'fac_15m.util', begin_date, end_date)\
        .at_time(time)\
        .rename_axis('symbol', axis=1)
    df_fwd_rtn = df_fwd_rtn[df_fwd_rtn.columns.intersection(symbols)]
    return df_fwd_rtn


def trans_df_y_rank(df_fwd_rtn):
    df_y = df_fwd_rtn.rank(axis=1, pct=True)
    df_y = convert_frame_to_stack(df_y)\
        .reset_index()\
        .set_index(['index', 'date', 'symbol'])\
        .sort_index()
    return df_y


def strategy_test_fig(df_fac, df_fwd_rtn, model, portion):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.reshape(-1)
    dmatrix = xgb.DMatrix(
        df_fac.drop(['y'], axis=1),
        df_fac['y'])
    y_preds = model.predict(dmatrix)
    df_y_preds = pd.DataFrame(y_preds, index=df_fac.index, columns=['y_preds'])
    df_y_preds['fwd_rtn'] = convert_frame_to_stack(df_fwd_rtn) \
        .reset_index() \
        .set_index(['index', 'date', 'symbol']) \
        .sort_index()
    df_y_preds['group'] = df_y_preds.groupby('date')['y_preds'].rank(pct=True)
    df_y_preds['true_group'] = df_y_preds.groupby('date')['fwd_rtn'].rank(pct=True)

    df_long_portfolio = df_y_preds[df_y_preds['group'] > 1 - portion]
    df_short_portfolio = df_y_preds[df_y_preds['group'] < portion]

    df_long_pnl = df_long_portfolio.groupby('date')['fwd_rtn'].mean().cumsum()
    df_short_pnl = df_short_portfolio.groupby('date')['fwd_rtn'].mean().cumsum()
    df_pnl = pd.concat([
        df_long_pnl, df_short_pnl, df_long_pnl - df_short_pnl
    ], keys=['long', 'short', 'agg'], axis=1)
    df_pnl.plot(color=['red', 'green', 'blue'], ax=axes[0])

    df_long_num = df_long_portfolio.groupby('date')['y_preds'].count()
    df_short_num = df_short_portfolio.groupby('date')['y_preds'].count()
    df_portfolio_num = pd.concat([
        df_long_num, df_short_num
    ], axis=1, keys=['long', 'short'])
    df_portfolio_num.plot(ax=axes[1])

    df_long_symbol_counts = df_long_portfolio.reset_index(2).groupby('symbol')['y_preds'].count().rename('long')
    df_short_symbol_counts = df_short_portfolio.reset_index(2).groupby('symbol')['y_preds'].count().rename('short')
    df_symbol_counts = pd.concat([df_long_symbol_counts, df_short_symbol_counts], axis=1).stack().rename(
        'appear_days').reset_index()
    sns.histplot(df_symbol_counts, x='appear_days', hue='level_1', ax=axes[2])

    df_predict_rank = pd.concat([
        df_long_portfolio.groupby('date')['true_group'].mean().rolling(10).mean() - 0.5,
        df_short_portfolio.groupby('date')['true_group'].mean().rolling(10).mean() - 0.5
    ], axis=1, keys=['long', 'short'])
    df_predict_rank.plot(ax=axes[3])
    return df_y_preds# , df_long_portfolio, df_short_portfolio
