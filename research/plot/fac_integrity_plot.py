import numpy as np
from matplotlib import pyplot as plt


def plot_nan_value_num(df_fac, fac_num=50):
    fig, ax = plt.subplots(figsize=(15, 5))
    df_fac_total_nan_stat = df_fac.isna().sum().sort_values(ascending=False)
    df_fac_total_nan_stat.iloc[:fac_num].plot(ax=ax, kind='bar', figsize=(15, 5))  # 绘制缺失较多的symbol
    return fig


def plot_nan_ratio_line(df_fac, fac_num=200, fac_num_per_ax=5):
    df_fac_total_nan_stat = df_fac.isna().sum().sort_values(ascending=False)
    df_fac_nan_ratio_daily = df_fac.isna().groupby('date').sum() / \
                             df_fac.fillna(0).groupby('date').count()  # 统计每个fac每天的缺失率
    axes_num = np.ceil(fac_num // fac_num_per_ax)
    fig, axes = plt.subplots(axes_num, figsize=(5, 4 * axes_num))
    for i, ax in enumerate(axes):
        for fac_name in df_fac_total_nan_stat.iloc[i * 5: (i + 1) * 5]:
            df_fac_nan_ratio_daily[fac_name].plot(ax=ax, label=fac_name)
    return fig
