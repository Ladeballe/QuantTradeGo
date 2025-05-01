import traceback
import functools

import numpy as np
import pandas as pd
import torch
import numba as nb
from ..util import align_stack_frames, convert_frame_to_stack, convert_stacked_frame_to_frame


def set_func_value(facs=None, window=None):
    def decorator(func):
        if facs is not None:
            setattr(func, 'facs', facs)
        if window is not None:
            setattr(func, 'window', window)
        return func
    return decorator


def _calc_max(*fac):
    df_fac = align_stack_frames(*fac)
    df_fac['raw_factor'] = df_fac.iloc[:, 2:].max(axis=1)
    df_fac = df_fac.pivot(index='date', columns='symbol', values='raw_factor')
    return df_fac


def _calc_min(*fac):
    df_fac = align_stack_frames(*fac)
    df_fac['raw_factor'] = df_fac.iloc[:, 2:].min(axis=1)
    df_fac = df_fac.pivot(index='date', columns='symbol', values='raw_factor')
    return df_fac


def _resize_to_3d_array(*df_facs):
    df_fac = align_stack_frames(*df_facs)
    df_fac = df_fac.pivot(index='date', columns='symbol')
    idx_date, idx_fac, idx_symbol = df_fac.index, df_fac.columns.levels[0], df_fac.columns.levels[1]
    arr_fac = df_fac.values.reshape(df_fac.shape[0], idx_fac.shape[0], -1)
    return arr_fac, idx_date, idx_symbol


# older version of _calc_corr
# @set_func_value(facs=[0, 1], window=2)
# def _calc_corr(df_fac0, df_fac1, window):
#     df_fac = align_stack_frames(df_fac0, df_fac1)
#     df_fac['raw_factor'] = df_fac.groupby('symbol')[['raw_factor0', 'raw_factor1']]\
#         .apply(lambda df: df.rolling(int(window)).corr().unstack().iloc[:, 2]).droplevel(0)
#     df_fac = df_fac.pivot(index='date', columns='symbol', values='raw_factor')
#     return df_fac
@set_func_value(facs=[0, 1], window=2)
def _calc_corr(df_fac0, df_fac1, window):
    window = int(window)

    def _calc_corr_sub_func(arr_fac):
        arr_fac.shape[0] - window
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            arr_fac_window = arr_fac[i:i + window]
            fac_a = arr_fac_window[:, 0, :]
            fac_b = arr_fac_window[:, 1, :]
            res = ((fac_a * fac_b).mean(axis=0) - fac_a.mean(axis=0) * fac_b.mean(axis=0)) / (fac_a.std(axis=0) * fac_b.std(axis=0))
            arr_res[i, :] = res
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac0, df_fac1)
    arr_res = _calc_corr_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


@set_func_value(facs=[1, 2], window=1)
def _calc_if(cond, df_fac0, df_fac1):
    if not isinstance(df_fac0, pd.Series):
        df_fac0 = pd.Series(df_fac0, index=cond.index)
    if not isinstance(df_fac1, pd.Series):
        df_fac1 = pd.Series(df_fac1, index=cond.index)
    df_fac0, df_fac1 = df_fac0.reindex(cond.index), df_fac1.reindex(cond.index)
    df_fac = df_fac0.mask(cond == 1, df_fac1)
    return df_fac


@set_func_value(facs=[1, 2, 3], window=2)
def _calc_switch(cond, df_fac0, df_fac1, df_fac2):
    if not isinstance(df_fac0, pd.Series):
        df_fac0 = pd.Series(df_fac0, index=cond.index)
    if not isinstance(df_fac1, pd.Series):
        df_fac1 = pd.Series(df_fac1, index=cond.index)
    if not isinstance(df_fac2, pd.Series):
        df_fac2 = pd.Series(df_fac2, index=cond.index)
    df_fac0 = df_fac0.reindex(cond.index)
    df_fac1 = df_fac1.reindex(cond.index)
    df_fac2 = df_fac2.reindex(cond.index)
    df_fac = df_fac0.mask(cond < 0, df_fac1).mask(cond == 0, df_fac2)
    return df_fac


# older version of _calc_regbeta
# @set_func_value(window=0)
# def _calc_regbeta(window, *df_facs):
#     def _calc_regbeta_sub_func(arr):
#         y, X = arr[:, 0], arr[:, 1:]
#         beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
#         return beta[-1]
#
#     df_fac = align_stack_frames(*df_facs)
#     columns = df_fac.columns[2:]
#     df_fac['raw_factor'] = df_fac.groupby('symbol') \
#         .apply(lambda df: df[columns].rolling(int(window), method='table') \
#                .apply(_calc_regbeta_sub_func, raw=True, engine='numba')
#                )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac


# newer version of _calc_regbeta, using torch is faster than using two set of for loop with numba acceleration. 3 times faster than older version
@set_func_value(window=0)
def _calc_regbeta(window, *df_facs):
    window = int(window)

    def _calc_regbeta_sub_func(arr_fac):
        tensor_fac = torch.tensor(arr_fac)
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            tensor_fac_window = tensor_fac[i:i + window]
            tensor_fac_y = tensor_fac_window[:, [0], :].permute(2, 0, 1)
            tensor_fac_X = tensor_fac_window[:, 1:, :].permute(2, 0, 1)
            tensor_fac_X = torch.concat([torch.ones(tensor_fac_X.shape), tensor_fac_X], 2)  # add constant
            tensor_beta = torch.bmm(
                torch.bmm(
                    torch.bmm(
                        tensor_fac_X.transpose(1, 2),
                        tensor_fac_X
                    ).inverse(),
                    tensor_fac_X.transpose(1, 2)
                ),
                tensor_fac_y
            )[:, 1, 0]
            arr_res[i] = tensor_beta
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(*df_facs)
    arr_res = _calc_regbeta_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac

# older version of _calc_reg_resid
# @set_func_value(window=0)
# def _calc_reg_resid(window, *df_facs):
#     @nb.jit(nopython=True)
#     def _calc_reg_resid_sub_func(arr):
#         y, X = arr[:, 0], arr[:, 1:]
#         beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
#         e = y - np.dot(X, beta)
#         return e[-1]
#
#     df_fac = align_stack_frames(*df_facs)
#     columns = df_fac.columns[2:]
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[columns].rolling(int(window), method='table')\
#             .apply(_calc_reg_resid_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac


# newer version of _calc_reg_resid
@set_func_value(window=0)
def _calc_reg_resid(window, *df_facs):
    window = int(window)

    def _calc_reg_resid_sub_func(arr_fac):
        tensor_fac = torch.tensor(arr_fac)
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            tensor_fac_window = tensor_fac[i:i + window]
            tensor_fac_y = tensor_fac_window[:, [0], :].permute(2, 0, 1)
            tensor_fac_X = tensor_fac_window[:, 1:, :].permute(2, 0, 1)
            tensor_fac_X = torch.concat([torch.ones(tensor_fac_X.shape), tensor_fac_X], 2)  # add constant
            tensor_beta = torch.bmm(
                torch.bmm(
                    torch.bmm(
                        tensor_fac_X.transpose(1, 2),
                        tensor_fac_X
                    ).inverse(),
                    tensor_fac_X.transpose(1, 2)
                ),
                tensor_fac_y
            )
            tensor_resid = tensor_fac_y - torch.bmm(tensor_fac_X, tensor_beta)
            arr_res[i] = tensor_resid[:, -1, 0]  # only take the last step
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(*df_facs)
    arr_res = _calc_reg_resid_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


# older version of _calc_reg_rsquared
# @set_func_value(window=0)
# def _calc_reg_rsquared(window, *df_facs):
#     @nb.jit(nopython=True)
#     def _calc_reg_rsquared_sub_func(arr):
#         y, X = arr[:, 0], arr[:, 1:]
#         beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
#         e = y - np.dot(X, beta)
#         r_squared = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
#         return r_squared
#
#     df_fac = align_stack_frames(*df_facs)
#     columns = df_fac.columns[2:]
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[columns].rolling(int(window), method='table')\
#             .apply(_calc_reg_rsquared_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac


# newer version of _calc_reg_rsquared
@set_func_value(window=0)
def _calc_reg_rsquared(window, *df_facs):
    window = int(window)

    def _calc_reg_rsquared_sub_func(arr_fac):
        tensor_fac = torch.tensor(arr_fac)
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            tensor_fac_window = tensor_fac[i:i + window]
            tensor_fac_y = tensor_fac_window[:, [0], :].permute(2, 0, 1)
            tensor_fac_X = tensor_fac_window[:, 1:, :].permute(2, 0, 1)
            tensor_fac_X = torch.concat([torch.ones(tensor_fac_X.shape), tensor_fac_X], 2)  # add constant
            tensor_beta = torch.bmm(
                torch.bmm(
                    torch.bmm(
                        tensor_fac_X.transpose(1, 2),
                        tensor_fac_X
                    ).inverse(),
                    tensor_fac_X.transpose(1, 2)
                ),
                tensor_fac_y
            )
            tensor_resid = tensor_fac_y - torch.bmm(tensor_fac_X, tensor_beta)
            r_squared = 1 - (tensor_resid ** 2).sum(axis=1) / ((tensor_fac_y - tensor_fac_y.mean(axis=1).resize(tensor_fac_y.shape[0], 1, 1))**2).sum(axis=1)
            arr_res[i] = r_squared[:, 0]  # only take the last step
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(*df_facs)
    arr_res = _calc_reg_rsquared_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


def _calc_reindex(df_fac_a, df_fac_b):
    if df_fac_a['symbol'].unique().shape[0] == 1:
        df_fac = df_fac_b.merge(df_fac_a[['date', 'raw_factor']], on='date').pivot(index='date', columns='symbol', values='raw_factor_y')
    else:
        raise NotImplementedError('Reindex on  is not implemented!')
    return df_fac


# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_mean(df_fac_a, df_fac_b, window):
#     # @nb.jit(nopython=True)
#     def _calc_wgt_mean_sub_func(arr):
#         try:
#             fac, fac_wgt = arr[:, 0], arr[:, 1]
#             fac_wgt = fac_wgt + fac_wgt.mean() * 1e-3  # FIXME: 这里的处理很有可能带来较大的误差
#             res = np.dot(fac, fac_wgt) / fac_wgt.sum()
#             return res
#         except:
#             return np.nan
#
#     df_fac = align_stack_frames(df_fac_a, df_fac_b)
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[['raw_factor0', 'raw_factor1']].rolling(int(window), method='table')\
#             .apply(_calc_wgt_mean_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac
#
#
# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_sum(df_fac_a, df_fac_b, window):
#     def _calc_wgt_sum_sub_func(arr):
#         try:
#             fac, fac_wgt = arr[:, 0], arr[:, 1]
#             fac_wgt = fac_wgt + fac_wgt.mean()
#             res = np.dot(fac, fac_wgt)
#             return res
#         except:
#             return np.nan
#
#     df_fac = align_stack_frames(df_fac_a, df_fac_b)
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[['raw_factor0', 'raw_factor1']].rolling(int(window), method='table')\
#             .apply(_calc_wgt_sum_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac


@set_func_value(facs=[0, 1], window=2)
def _calc_wgt_sum(df_fac_a, df_fac_b, window):
    window = int(window)

    def _calc_wgt_mean_sub_func(arr_fac):
        arr_fac.shape[0] - window
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            arr_fac_window = arr_fac[i:i + window]
            fac = arr_fac_window[:, 0, :]
            fac_wgt = arr_fac_window[:, 1, :]
            res = np.diag(np.dot(fac.T, fac_wgt))
            arr_res[i, :] = res
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
    arr_res = _calc_wgt_mean_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


@set_func_value(facs=[0, 1], window=2)
def _calc_wgt_mean(df_fac_a, df_fac_b, window):
    window = int(window)

    def _calc_wgt_mean_sub_func(arr_fac):
        arr_fac.shape[0] - window
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            arr_fac_window = arr_fac[i:i + window]
            fac = arr_fac_window[:, 0, :]
            fac_wgt = arr_fac_window[:, 1, :]
            np.diag(np.dot(fac.T, fac_wgt))
            res = np.diag(np.dot(fac.T, fac_wgt)) / np.diag(np.dot(np.ones(fac_wgt.shape).T, fac_wgt))
            arr_res[i, :] = res
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
    arr_res = _calc_wgt_mean_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


# 检查代码
# 再次检查后，发现计算是没有问题的，不过检查时在进行手动对齐时，需要手动设置对齐
# fac_res = []
# symbol = 'BTCUSDT'
# fac_a_coin = df_fac_a[df_fac_a['symbol']==symbol]['raw_factor']
# fac_b_coin = df_fac_b[df_fac_b['symbol']==symbol]['raw_factor'].iloc[1:]
# for i in range(6, fac_a_coin.shape[0]):
#     val = (fac_a_coin.iloc[i-window:i] * fac_b_coin.iloc[i-window:i]).sum() / fac_b_coin.iloc[i-window:i].sum()
#     fac_res.append(val)
# res = pd.concat([pd.Series(fac_res, index=df_fac.index), df_fac[symbol]], axis=1)


def parse_process_unary_oprt(process_unary_oprt):
    unary_oprt_list, unary_oprt_param_list = list(), list()
    for unary_oprt in process_unary_oprt.strip('<>').split(','):
        unary_oprt, unary_oprt_params = unary_oprt.split('_')[0].lower(), unary_oprt.split('_')[1:]
        unary_oprt_list.append(unary_oprt)
        try:
            unary_oprt_param = float(unary_oprt_params[0])
            unary_oprt_param_list.append(unary_oprt_param)
        except IndexError:
            unary_oprt_param_list.append(0.)
    return unary_oprt_list, unary_oprt_param_list


@nb.jit(nopython=True)
def calc_unary_oprt_np(arr, unary_oprt_list, unary_oprt_param_list):
    for unary_oprt, unary_oprt_param in zip(unary_oprt_list, unary_oprt_param_list):
        if unary_oprt == 'rank':
            arr = np.argsort(arr).astype(nb.float64) / arr.shape[0]
        # elif unary_oprt == 'gt':
        #     arr = arr > unary_oprt_param
        # elif unary_oprt == 'lt':
        #     arr = arr < unary_oprt_param
    return arr


# older version of WgtMeanPro
# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_mean_process(df_fac_a, df_fac_b, window, process_unary_oprt):
#     def _calc_wgt_mean_sub_func(arr):
#         try:
#             fac, fac_wgt = arr[:, 0], calc_unary_oprt_np(arr[:, 1], process_unary_oprt)
#             fac_wgt = fac_wgt + fac_wgt.mean() * 1e-3  # FIXME: 这里的处理很有可能带来较大的误差
#             res = np.dot(fac, fac_wgt) / fac_wgt.sum()
#             return res
#         except:
#             return np.nan
#
#     df_fac = align_stack_frames(df_fac_a, df_fac_b)
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[['raw_factor0', 'raw_factor1']].rolling(int(window), method='table')\
#             .apply(_calc_wgt_mean_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac
#
#
# older version of WgtSumPro
# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_sum_process(df_fac_a, df_fac_b, window, process_unary_oprt):
#     unary_oprt, unary_oprt_param = process_unary_oprt[1:-1].split('_')
#     unary_oprt_param = float(unary_oprt_param)
#
#     def _calc_wgt_sum_sub_func(arr):
#         try:
#             fac = arr[:, 0]
#             fac_wgt = arr[:, 1]
#             fac_wgt = fac_wgt.argsort().argsort() / (fac_wgt.shape[0] - 1)
#             if unary_oprt == 'gt':
#                 fac_wgt = fac_wgt > unary_oprt_param
#             elif unary_oprt == 'lt':
#                 fac_wgt = fac_wgt < unary_oprt_param
#             else:
#                 raise KeyboardInterrupt
#             fac_wgt = fac_wgt + np.mean(fac_wgt) * 1e-3  # FIXME: 这里的处理很有可能带来较大的误差
#             res = np.dot(fac, fac_wgt) / fac_wgt.sum()
#             return res
#         except:
#             return np.nan
#
#     df_fac = align_stack_frames(df_fac_a, df_fac_b)
#     df_fac['raw_factor'] = df_fac.groupby('symbol')\
#         .apply(lambda df: df[['raw_factor0', 'raw_factor1']].rolling(int(window), method='table')\
#             .apply(_calc_wgt_sum_sub_func, raw=True, engine='numba')
#         )['raw_factor0'].droplevel(0)
#     df_fac = convert_stacked_frame_to_frame(df_fac[['date', 'symbol', 'raw_factor']])
#     return df_fac

@set_func_value(facs=[0, 1], window=2)
def _calc_wgt_mean_process(df_fac_a, df_fac_b, window, process_unary_oprt):
    unary_oprt, unary_oprt_param = process_unary_oprt[1:-1].split('_')
    unary_oprt_param = float(unary_oprt_param)
    window = int(window)

    @nb.jit(nopython=True)
    def _calc_wgt_sum_sub_func(arr_fac):
        arr_fac.shape[0] - window
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            arr_fac_window = arr_fac[i:i + window]
            fac = arr_fac_window[:, 0, :]
            fac_wgt = arr_fac_window[:, 1, :]
            for j in range(0, fac_wgt.shape[1]):
                if unary_oprt == 'gt':
                    fac_wgt[:, j] = fac_wgt[:, j].argsort().argsort() / (fac_wgt.shape[0] - 1) > unary_oprt_param
                elif unary_oprt == 'lt':
                    fac_wgt[:, j] = fac_wgt[:, j].argsort().argsort() / (fac_wgt.shape[0] - 1) < unary_oprt_param
                else:
                    raise ValueError
            np.diag(np.dot(fac.T, fac_wgt))
            res = np.diag(np.dot(fac.T, fac_wgt)) / np.diag(np.dot(np.ones(fac_wgt.shape).T, fac_wgt))
            arr_res[i, :] = res
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
    arr_res = _calc_wgt_sum_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_mean_process2(df_fac_a, df_fac_b, window, process_unary_oprt):
#     unary_oprt, unary_oprt_param = process_unary_oprt[1:-1].split('_')
#     unary_oprt_param = float(unary_oprt_param)
#     window = int(window)
#
#     def _calc_wgt_sum_pro_sub_func(arr_fac):
#         tensor_fac = torch.tensor(arr_fac)
#         arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
#         for i in range(0, arr_fac.shape[0] - window):
#             tensor_fac_window = tensor_fac[i:i + window]
#             fac = tensor_fac_window[:, 0, :]
#             fac_wgt = tensor_fac_window[:, 1, :]
#             fac_wgt = torch.argsort(torch.argsort(fac_wgt, 0), 0) / fac_wgt.shape[0]
#             if unary_oprt == 'gt':
#                 fac_wgt = (fac_wgt > unary_oprt_param).to(torch.float64)
#             elif unary_oprt == 'lt':
#                 fac_wgt = (fac_wgt < unary_oprt_param).to(torch.float64)
#             arr_res[i] = (fac.T @ fac_wgt).diag() / fac_wgt.sum(axis=0)
#         return arr_res
#
#     arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
#     arr_res = _calc_wgt_sum_pro_sub_func(arr_fac)
#     df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
#     return df_fac


@set_func_value(facs=[0, 1], window=2)
def _calc_wgt_sum_process(df_fac_a, df_fac_b, window, process_unary_oprt):
    unary_oprt, unary_oprt_param = process_unary_oprt[1:-1].split('_')
    unary_oprt_param = float(unary_oprt_param)
    window = int(window)

    @nb.jit(nopython=True)
    def _calc_wgt_sum_sub_func(arr_fac):
        arr_fac.shape[0] - window
        arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
        for i in range(0, arr_fac.shape[0] - window):
            arr_fac_window = arr_fac[i:i + window]
            fac = arr_fac_window[:, 0, :]
            fac_wgt = arr_fac_window[:, 1, :]
            for j in range(0, fac_wgt.shape[1]):
                if unary_oprt == 'gt':
                    fac_wgt[:, j] = fac_wgt[:, j].argsort().argsort() / (fac_wgt.shape[0] - 1) > unary_oprt_param
                elif unary_oprt == 'lt':
                    fac_wgt[:, j] = fac_wgt[:, j].argsort().argsort() / (fac_wgt.shape[0] - 1) < unary_oprt_param
                else:
                    raise ValueError
            res = np.diag(np.dot(fac.T, fac_wgt))
            arr_res[i, :] = res
        return arr_res

    arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
    arr_res = _calc_wgt_sum_sub_func(arr_fac)
    df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
    return df_fac


# torch version is not fast enough
# @set_func_value(facs=[0, 1], window=2)
# def _calc_wgt_sum_process2(df_fac_a, df_fac_b, window, process_unary_oprt):
#     unary_oprt, unary_oprt_param = process_unary_oprt[1:-1].split('_')
#     unary_oprt_param = float(unary_oprt_param)
#     window = int(window)
#
#     def _calc_wgt_sum_pro_sub_func(arr_fac):
#         tensor_fac = torch.tensor(arr_fac)
#         arr_res = np.zeros((arr_fac.shape[0] - window, arr_fac.shape[-1]))
#         for i in range(0, arr_fac.shape[0] - window):
#             tensor_fac_window = tensor_fac[i:i + window]
#             fac = tensor_fac_window[:, 0, :]
#             fac_wgt = tensor_fac_window[:, 1, :]
#             fac_wgt = torch.argsort(torch.argsort(fac_wgt, 0), 0) / fac_wgt.shape[0]
#             if unary_oprt == 'gt':
#                 fac_wgt = (fac_wgt > unary_oprt_param).to(torch.float64)
#             elif unary_oprt == 'lt':
#                 fac_wgt = (fac_wgt < unary_oprt_param).to(torch.float64)
#             arr_res[i] = (fac.T @ fac_wgt).diag() / fac_wgt.sum(axis=0)
#         return arr_res
#
#     arr_fac, idx_date, idx_symbol = _resize_to_3d_array(df_fac_a, df_fac_b)
#     arr_res = _calc_wgt_sum_pro_sub_func(arr_fac)
#     df_fac = pd.DataFrame(arr_res, index=idx_date[window:], columns=idx_symbol)
#     return df_fac


def _calc_INDEX_NCI(df_fac):
    """
    常函数，根据Nasdaq给出的Nasdaq Cryptocurrency Index 计算方式计算的指数
    具体可参考：https://www.nasdaq.com/solutions/crypto-index
    """
    index_wgt = {
        'BTCUSDT': 0.7114,
        'ETHUSDT': 0.2656,
        'LINKUSDT': 0.0082,
        'LTCUSDT': 0.0049,
        'ARBUSDT': 0.0042,
        'UNIUSDT': 0.0032,
        'DOTUSDT': 0.0025,  # XLM刚刚上线, 数据不足, 暂时不算入指数成分, 权重调整到LINK上
    }
    sr_index_wgt = pd.Series(index_wgt).reindex(df_fac.columns).fillna(0)
    df_index = (df_fac * sr_index_wgt).mean(axis=1).to_frame(df_fac.columns[0]).reindex(df_fac.columns, axis=1)
    df_fac = df_index.T.ffill().T
    return df_fac


def _calc_agg_INDEX_NCI(df_fac):
    """
    常函数，根据Nasdaq给出的Nasdaq Cryptocurrency Index 计算方式计算的指数
    具体可参考：https://www.nasdaq.com/solutions/crypto-index
    """
    index_wgt = {
        'BTCUSDT': 0.7114,
        'ETHUSDT': 0.2656,
        'LINKUSDT': 0.0082,
        'LTCUSDT': 0.0049,
        'ARBUSDT': 0.0042,
        'UNIUSDT': 0.0032,
        'DOTUSDT': 0.0025,  # XLM刚刚上线, 数据不足, 暂时不算入指数成分, 权重调整到LINK上
    }
    sr_index_wgt = pd.Series(index_wgt).reindex(df_fac.columns).fillna(0)
    df_index = (df_fac * sr_index_wgt).sum(axis=1).to_frame(df_fac.columns[0]).reindex(df_fac.columns, axis=1)
    df_fac = df_index.T.ffill().T
    return df_fac


def _calc_agg(df_fac, method):
    """
    计算所有因子截面加和
    """
    if method == 'sum':
        df_index = df_fac.sum(axis=1).to_frame(df_fac.columns[0]).reindex(df_fac.columns, axis=1)
        df_fac = df_index.T.ffill().T
    elif method == 'mean':
        df_index = df_fac.mean(axis=1).to_frame(df_fac.columns[0]).reindex(df_fac.columns, axis=1)
        df_fac = df_index.T.ffill().T
    elif method == 'demean':
        df_fac = df_fac.sub(df_fac.mean(axis=1), axis=0)
    elif method == 'zscore':
        df_fac = df_fac.sub(df_fac.mean(axis=1), axis=0).div(df_fac.std(axis=1), axis=0)
    return df_fac
