import itertools
from typing import List, Dict, Any, Union, Tuple
import multiprocessing as mp
import traceback
from functools import lru_cache

import pandas as pd
import pymongo

from utils import get_path

from .data import get_factor
from .formula import expr_trans_sig
from .backtest import calc_rtn_by_signal
from .analyse import (
    calc_win_ratio_by_signal, calc_plr_by_signal,
    analyse_trade_stat, analyse_trade_phase, analyse_trade_time, analyse_rtn, analyse_test_res, analyse_product_test_res
)
from .plot import SigTestPlotter, plot_price_pnl_fac, plot_price_rtn_fac, plot_price_rtn_pos, func_list, func_list2, func_list3


mongo_client = pymongo.MongoClient("localhost:27017")
coll = mongo_client["FS_test"]["test_results"]


def iter_fac_names(base_fac_names: Dict[str, List[Tuple]], params_dict: Dict[str, List[Any]],
                   iter_order: Union[None, List[str]] = None,
                   iter_fac_names_type: str='product', iter_fac_params_type: str='product'):
    """遍历生成因子名称

    Parameters
    ----------
    base_fac_names
    params_dict
    iter_order

    Returns
    -------

    """
    if iter_order is None:
        iter_order = list(params_dict.keys())
    iter_args = [params_dict[param_name] for param_name in iter_order]
    if iter_fac_names_type == 'product':
        iter_fac_names = itertools.product(*base_fac_names.values())  # 遍历因子的组合
    elif iter_fac_names_type == 'order':
        iter_fac_names = iter(zip(*base_fac_names.values()))
    else:
        raise ValueError(f'Wrong param for "iter_fac_names_type": {iter_fac_names_type}!')
    for base_fac_name_group in iter_fac_names:
        base_fac_name_group = dict(zip(base_fac_names.keys(), base_fac_name_group))
        if iter_fac_params_type == 'product':
            iter_res = itertools.product(*iter_args)
        elif iter_fac_params_type == 'order':
            iter_res = iter(zip(*iter_args))
        else:
            raise ValueError(f'Wrong param for "iter_fac_params_type": {iter_fac_params_type}!')
        for params in iter_res:
            params_dict = dict(zip(iter_order, params))  # 根据params的名字构建字典
            format_fac_names = dict(zip(
                base_fac_name_group.keys(),
                list(map(lambda x: x[1].format(**params_dict), base_fac_name_group.values()))
            ))  # 根据params_dict获取因子的名称
            fac_types = dict(zip(
                base_fac_name_group.keys(),
                list(map(lambda x: x[0], base_fac_name_group.values()))
            ))  # 解析因子对应的类型
            yield format_fac_names, fac_types, params_dict


def iter_formula(base_formula: str, fac_names: Dict[str, str],
                 params_dict: Dict[str, Union[List[Any], Tuple[str, Dict[str, List[Any]]]]],
                 iter_mode: str = 'pairwise',
                 iter_order: Union[None, List[str]] = None, **kwargs):
    """遍历以生成表达式的函数

    Parameters
    ----------
    base_formula: str
        基准表达式
        定义为这样的形式： '{fac_name1}[FFILL,{0},{1}]_{2}_{fac_name2}[{3}]'
    fac_names: List[str]
        存储所有的因子名称
    params_dict: Dict[str: List[Any]]
        存储需要遍历的参数列表
    iter_order: List[str]
        参数遍历的顺序

    Returns
    -------

    """
    if iter_order is None:
        iter_order = list(params_dict.keys())
    iter_args = [params_dict[param_name] for param_name in iter_order]
    if iter_mode == 'pairwise':
        iter_res = zip(*iter_args)
    elif iter_mode == 'product':
        iter_res = itertools.product(*iter_args)
    else:
        raise ValueError(f'Unknown param "iter_mode" value: "{iter_mode}"!')
    for params in iter_res:
        format_params_dict = dict()
        params_dict = dict(zip(iter_order, params))  # 根据params的名字构建字典
        format_params_dict.update(fac_names)
        format_params_dict.update(params_dict)
        formula = base_formula.format(**format_params_dict)
        yield formula, params_dict


def iter_formula2(base_formula: str, fac_names: Dict[str, str], fac_types: Dict[str, str],
                  params_dict: Dict[str, Union[List[Any], Tuple[str, Dict[str, List[Any]]]]], 
                  iter_order: Union[None, List[str]] = None, **kwargs):
    """遍历以生成表达式的函数

    Parameters
    ----------
    base_formula: str
        基准表达式
        定义为这样的形式： '{fac_name1}[FFILL,{0},{1}]_{2}_{fac_name2}[{3}]'
    fac_names: List[str]
        存储所有的因子名称
    params_dict: Dict[str: List[Any]]
        存储需要遍历的参数列表
    iter_order: List[str]
        参数遍历的顺序

    Returns
    -------

    """
    params_dict = params_dict.copy()
    if iter_order is None:
        iter_order = list(params_dict.keys())
    for param_name, params in params_dict.items():
        if isinstance(params, tuple):
            param_name_order, param_name_dict = params
            params_dict[param_name] = param_name_dict[fac_types[param_name_order]]
    iter_args = [params_dict[param_name] for param_name in iter_order]
    iter_res = itertools.product(*iter_args)
    for params in iter_res:
        format_params_dict = dict()
        params_dict = dict(zip(iter_order, params))  # 根据params的名字构建字典
        format_params_dict.update(fac_names)
        format_params_dict.update(params_dict)
        formula = base_formula.format(**format_params_dict)
        yield formula, params_dict


def iter_params(base_name, base_formula: str,
                params_dict: Dict[str, Union[List[Any], Tuple[str, Dict[str, List[Any]]]]],
                iter_mode: str = 'pairwise', iter_config: List[Union[str, List[str]]] = None,
                iter_order: Union[None, List[str]] = None, **kwargs):
    """遍历以生成表达式的函数

    Parameters
    ----------
    base_formula: str
        基准表达式
        定义为这样的形式： '{fac_name1}[FFILL,{0},{1}]_{2}_{fac_name2}[{3}]'
    fac_names: List[str]
        存储所有的因子名称
    params_dict: Dict[str: List[Any]]
        存储需要遍历的参数列表
    iter_mode: str
        参数遍历的具体方式
    iter_config: List[str, List[str]]
        存储有'config'模式下的具体匹配方式
    iter_order: List[str]
        参数遍历的顺序

    Returns
    -------

    """
    if iter_order is None:
        iter_order = list(params_dict.keys())
    iter_args = [params_dict[param_name] for param_name in iter_order]
    if iter_mode == 'pairwise':
        iter_res = zip(*iter_args)
    elif iter_mode == 'product':
        iter_res = itertools.product(*iter_args)
    elif iter_mode == 'config':
        iter_order, iter_args = list(), list()
        for params_set in iter_config:
            if isinstance(params_set, str):
                iter_args.append([(arg, ) for arg in params_dict[params_set]])
                iter_order += [params_set]
            elif isinstance(params_set, list):
                iter_args.append(list(zip(
                    *[params_dict[param_name] for param_name in params_set]
                )))
                iter_order += list(params_set)
            else:
                raise TypeError(f"Wrong type for param 'iter_config': {type(params_set)}")
        iter_args, iter_res = list(itertools.product(*iter_args)), list()
        for i, iter_arg in enumerate(iter_args):
            iter_arg_new = list()
            for args in iter_arg:
                iter_arg_new += list(args)
            iter_res.append(iter_arg_new)
    else:
        raise ValueError(f'Unknown param "iter_mode" value: "{iter_mode}"!')
    for params in iter_res:
        try:
            format_params_dict = dict()
            params_dict = dict(zip(iter_order, params))  # 根据params的名字构建字典
            format_params_dict.update(params_dict)
            name = base_name.format(**format_params_dict)
            formula = base_formula.format(**format_params_dict)
            yield name, formula
        except:
            print(base_name, base_formula)
            raise


def iter_flatten_df_fac_info(df_fac_info):
    list_fac_info = list()
    for i, row in df_fac_info.iterrows():
        name, formula, params, status = row['name'], row['formula'], eval(row['params']), row['status']
        param_iter_mode, param_iter_config = row['param_iter_mode'], eval(row['param_iter_config'])
        name_formula_list = iter_params(name, formula, params, param_iter_mode, param_iter_config)

        for name, formula in name_formula_list:
            new_row = row.copy()
            new_row['name'] = name
            new_row['formula'] = formula
            list_fac_info.append(new_row)
    df_fac_info = pd.concat(list_fac_info, axis=1).T
    return df_fac_info


# 加载因子
def factor_load(symbols, fac_names, fac_libs, begin_date, end_date, between_time=None):
    fac_df_list = []
    for fac_name, fac_lib in zip(fac_names.values(), fac_libs.values()):
        df_factor = _load_single_factor((fac_name).strip('-'), fac_lib, begin_date, end_date, symbols, between_time)  # FIXME: strip('-')用来使负向因子也能使用
        fac_df_list.append(df_factor)
    df_factor = pd.concat(fac_df_list, axis=1).sort_index()
    return df_factor


@lru_cache(maxsize=256)
def public_factor_load(fac_name, fac_lib, begin_date, end_date, between_time=None):
    # TODO: 公共因子可能还需要
    df_pub_fac = get_factor(fac_name, fac_lib, begin_date, end_date)
    if between_time is not None:
        df_pub_fac = df_pub_fac.between_time(between_time[0], between_time[1])
    return df_pub_fac


def _load_single_factor(fac_name, fac_lib, begin_date, end_date, symbols, between_time):
    df_factor = get_factor(fac_name, fac_lib, begin_date, end_date)
    df_factor[list(set(symbols) - set(df_factor.columns))] = 0
    df_factor = df_factor[symbols]
    if between_time is not None:
        df_factor = df_factor.between_time(between_time[0], between_time[1])
    df_factor = df_factor.stack().to_frame(fac_name)
    df_factor.index.rename(['date', 'symbol'], inplace=True)
    return df_factor


def factor_test(strategy, version, formula, begin_date, end_date,
                df_factor, output_path, test_params):  # FIXME: 后续支持ByQty检测
    try:
        query = {'strategy': strategy, 'formula': formula,
                 'begin_date': begin_date, 'end_date': end_date, 'version': version}
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), query)
        params, skip_finished, file_tag = test_params['params'], test_params['skip_finished'], test_params['file_tag']
        fwd_rtn_name, fwd_rtn_lib = test_params.get('fwd_rtn_name', 'fwd_rtn_15m'), test_params.get('fwd_rtn_lib', 'public.fwd_rtn')
        fwd_pnl_name, fwd_pnl_lib = test_params.get('fwd_pnl_name', 'fwd_pnl_15m'), test_params.get('fwd_pnl_lib', 'public.fwd_pnl')
        ctt_val_name, ctt_val_lib = test_params.get('ctt_val_name', 'ctt_val_15m'), test_params.get('ctt_val_lib', 'public.ctt_val')
        vola_fac_name, qty_param = test_params['vola_fac_name'], test_params['qty_param']
        cost_ratio, calc_rtn_method = test_params['cost_ratio'], test_params['calc_rtn_method']
        initial_capital, resample_freq = test_params['initial_capital'], test_params['resample_freq']
        fac_names, fac_types, fac_libs = test_params['fac_names'], test_params['fac_types'], test_params['fac_libs']
        symbols = test_params['symbols']

        if skip_finished and coll.find_one(query) is not None:
            return

        df_fwd_rtn = public_factor_load(fwd_rtn_name, fwd_rtn_lib, begin_date, end_date).stack().rename("fwd_rtn")
        df_price = public_factor_load("price_15m", "public.public_factor", begin_date, end_date)
        df_factor = df_factor.unstack().truncate(begin_date, end_date).stack()
        df_signal = expr_trans_sig(formula, df_factor)  # FIXME: expr_trans_sig需要改进
        df_fwd_rtn = df_fwd_rtn.reindex(df_signal.index)

        if calc_rtn_method == "ByRtn":
            df_rtn_merge = calc_rtn_by_signal(df_signal, df_fwd_rtn, cost_ratio=cost_ratio)
        # elif calc_rtn_method == "ByQty":  # FIXME: 后续添加ByQty检测方法
        #     df_rtn_merge, df_pnl = rff.因子表达式_计算收益单边_ByQty(
        #         df_factor, qty_param, initial_capital,
        #         fwd_pnl_name, ctt_val_name, fwd_pnl_lib, ctt_val_lib,
        #         cost_ratio=cost_ratio)
        df_win_ratio = calc_win_ratio_by_signal(df_signal, df_fwd_rtn, cost_ratio=cost_ratio)
        df_plr_ratio = calc_plr_by_signal(df_signal, df_fwd_rtn, cost_ratio=cost_ratio)
        df_trade_stat, df_trade_phase = analyse_trade_stat(df_signal), analyse_trade_phase(df_signal)  # 统计每笔交易的时间
        df_total_rtn, df_daily_rtn = analyse_rtn(df_rtn_merge)  # 分析收益
        sum_metric = analyse_test_res(df_daily_rtn, df_trade_stat)
        df_metric = analyse_product_test_res(df_daily_rtn)

        plotter = SigTestPlotter()
        plotter.load_param_data(
            symbols, begin_date, end_date,
            df_rtn=df_rtn_merge, df_signal=df_signal, df_trade_stat=df_trade_stat, df_trade_phase=df_trade_phase,
            df_total_rtn=df_total_rtn, df_win_ratio=df_win_ratio, df_plr_ratio=df_plr_ratio,
            sum_metric=sum_metric, df_price=df_price
        )
        plotter.load_figure_config_v1()
        plotter.load_plotter_config(*func_list3)
        plotter.config_v1()
        plotter.plot_v1()

        title = f"{formula} {begin_date} {end_date} {file_tag}\n" \
                f"总收益:{sum_metric['total_rtn'] * 100:.2f}% 年化:{sum_metric['ann_rtn'] * 100:.2f}% " \
                f"夏普:{sum_metric['sharpe_ratio']:.3} 回撤:{sum_metric['max_drawdown'] * 100:.2f}% \n" \
                f"持仓天数中值 long_days:{sum_metric['long_days_median']:.2f} " \
                f"short_days:{sum_metric['short_days_median']:.2f}"
        fig_filename = get_path(f"{output_path}\\FS_{file_tag}_{formula}_{begin_date}_{end_date}.png")
        plotter.set_title(title)
        plotter.save_fig(fig_filename)

        plot_signal_result_dict = {
            'formula': formula, 'begin_date': begin_date, 'end_date': end_date,
            'df_stat': df_trade_stat.to_json(),
            'df_daily_rtn': df_daily_rtn.to_json(),
            'df_win_ratio': df_win_ratio.to_json(), 'df_product_metric': df_metric.to_json(),
            'fig_file_name': fig_filename, 'update_time': pd.Timestamp.now().to_pydatetime(),
            'cost_ratio': cost_ratio, 'resample_freq': resample_freq, 'hour_list': '',
            'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param, 'initial_capital': initial_capital,
            'fac_names': fac_names, 'fac_types': fac_types, 'fac_libs': fac_libs, 'symbol': symbols
        }
        plot_signal_result_dict.update(query)
        plot_signal_result_dict.update(sum_metric)
        plot_signal_result_dict["params"] = params
        coll.update_one(query, {'$set': plot_signal_result_dict}, upsert=True)
    except:
        print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), "Error occurs!", query, end=' ')
        traceback.print_exc()
