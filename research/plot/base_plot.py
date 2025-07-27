from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def plot_seg_bar(df_fac, ax):
    ax.grid()
    twin_ax = ax.twiny()
    # colors = iter(mcolors.TABLEAU_COLORS.values())
    colors = iter(mcolors.XKCD_COLORS.values())
    x_shape = df_fac.shape
    width = 9 / x_shape[1]
    x_tick_lv0 = [0.5 + 10 * i for i in range(x_shape[0])]
    x_tick_lv0_label = [5 + 10 * i for i in range(x_shape[0])]
    x_ticks, x_ticklabels = [], []
    for i, row_name in enumerate(df_fac.index):
        x_tick_lv1 = [x_tick_lv0[i] + (j + 0.5) * width for j in range(x_shape[1])]
        x_ticks.append(x_tick_lv1[-1])
        x_ticklabels.append(df_fac.columns[-1])
        ax.bar(x_tick_lv1, df_fac.loc[row_name], color=next(colors), edgecolor='black', width=width)
    twin_ax.set_xticks(x_tick_lv0_label), twin_ax.set_xticklabels(df_fac.index), twin_ax.set_xlim(*ax.get_xlim())
    ax.set_xticks(x_ticks), ax.set_xticklabels(x_ticklabels)
