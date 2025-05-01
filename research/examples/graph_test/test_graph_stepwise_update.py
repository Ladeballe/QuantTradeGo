import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import test_func as tf
from test_func import data as tfd
from test_func import formula as tff
from test_func.formula.calc_formula import init_drawing_graph
from test_func import util as tfu

idx = pd.IndexSlice


def stepwise_wrapper(func):
    def wrapper(*args, **kwargs):

        return func(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    df_fac_task_list = pd.read_excel(r'D:\research\CRYPTO_vp_fac\CRYPTO_create_fac_15m.xlsx')
    df_fac_task_list = df_fac_task_list[df_fac_task_list['create'] == 4]
    df_fac_task_list = tf.main.iter_flatten_df_fac_info(df_fac_task_list)

    begin_date, end_date0, end_date1 = '2024-10-01', '2024-10-25', '2024-10-26'
    period_freq = '1M'
    processes_num = 1
    basic_lib_names_list = [
        'fac_15m.fac_basic'
    ]

    periods = pd.period_range(begin_date, end_date0, freq=period_freq)
    df_factor_tree_dict = dict()
    list_expr_tree_node = list()
    for i, row in df_fac_task_list.iterrows():
        name, formula = row['name'], row['formula']
        save_lib_name = row['save_lib_name']
        load_offset, add_lib_names_list = eval(row['load_offset']), eval(row['add_lib_names_list'])
        fac_lib_dict = tfu.get_fac_lib_dict(basic_lib_names_list + add_lib_names_list)

        tokens = tff.parse_formula_to_tokens(formula)
        formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens)
        list_expr_tree_node.append((name, formula_ast_tree))

    org_nodes, res_nodes = dict(), dict()
    node_dict = tff.NodeDict()
    for fac_name, expr_tree_node in list_expr_tree_node:
        res_nodes[fac_name] = tff.trans_expr_tree_to_graph(fac_name, expr_tree_node, org_nodes, res_nodes, node_dict)

    fac_lib_dict = tfu.get_fac_lib_dict(list(set(
        basic_lib_names_list + df_fac_task_list['add_lib_names_list'].apply(lambda x: eval(x)).sum()
    )))
    tokens = [(org_node.name, org_node.node_type) for org_node in org_nodes.values() if org_node.node_type == 'fac']
    dict_df_data = tff.load_factor_from_tokens(tokens, begin_date, end_date0 + ' 23:59:59', fac_lib_dict)
    t0 = pd.Timestamp.now()
    tff.loop_calc_formula_graph(list(org_nodes.values()), dict_df_data, tff.calc_formula_graph, debug_bool=False)
    print(pd.Timestamp.now() - t0)
    dict_df_data_new = tff.load_factor_from_tokens(tokens, begin_date, end_date1 + ' 23:59:59', fac_lib_dict)
    for key, value in dict_df_data_new.items():
        df_fac = tfu.convert_stacked_frame_to_frame(value).loc[:end_date0 + ' 00:15:00']
        dict_df_data_new[key] = tfu.convert_frame_to_stack(df_fac)
    t0 = pd.Timestamp.now()
    tff.loop_calc_formula_graph(list(org_nodes.values()), dict_df_data_new, tff.calc_formula_graph_stepwise, debug_bool=False, steps=1)
    print(pd.Timestamp.now() - t0)
    print('done')
