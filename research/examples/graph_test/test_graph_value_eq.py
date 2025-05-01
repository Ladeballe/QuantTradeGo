import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt

import test_func as tf
from test_func import data as tfd
from test_func import formula as tff
from test_func.formula.calc_formula import init_drawing_graph
from test_func import util as tfu


if __name__ == '__main__':
    df_fac_task_list = pd.read_excel(r'D:\research\CRYPTO_vp_fac\CRYPTO_create_fac_15m.xlsx')
    df_fac_task_list = df_fac_task_list[df_fac_task_list['create'] == 4]
    df_fac_task_list = tf.main.iter_flatten_df_fac_info(df_fac_task_list)

    begin_date, end_date = '2024-10-21', '2024-10-31 23:59:59'
    period_freq = '1M'
    processes_num = 1
    basic_lib_names_list = [
        'fac_15m.fac_basic'
    ]

    periods = pd.period_range(begin_date, end_date, freq=period_freq)
    df_factor_tree_dict = dict()
    list_expr_tree_node = list()
    for i, row in df_fac_task_list.iterrows():
        name, formula = row['name'], row['formula']
        save_lib_name = row['save_lib_name']
        load_offset, add_lib_names_list = eval(row['load_offset']), eval(row['add_lib_names_list'])
        fac_lib_dict = tfu.get_fac_lib_dict(basic_lib_names_list + add_lib_names_list)

        tokens = tff.parse_formula_to_tokens(formula)
        dict_of_data = tff.load_factor_from_tokens(tokens, begin_date, end_date, fac_lib_dict)
        formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens)
        list_expr_tree_node.append((name, formula_ast_tree))
        df_factor = tfu.convert_stacked_frame_to_frame(
            tff.calc_formula_ast_tree(formula_ast_tree, dict_of_data).value
        ).truncate(begin_date, end_date)
        df_factor_tree_dict[name] = df_factor

    org_nodes, res_nodes = dict(), dict()
    node_dict = tff.NodeDict()
    for fac_name, expr_tree_node in list_expr_tree_node:
        res_nodes[fac_name] = tff.trans_expr_tree_to_graph(fac_name, expr_tree_node, org_nodes, res_nodes, node_dict)

    graph = nx.DiGraph()
    plt.figure(figsize=(50, 15))
    for org_node in org_nodes.values():
        init_drawing_graph(org_node, graph)
    pos = graphviz_layout(graph, 'dot')  # 为图形设置布局
    graph_node_order = dict(zip(
        list(graph.nodes), range(len(list(graph.nodes)))
    ))
    org_nodes_index_list, res_nodes_index_list = list(), list()
    for org_node in org_nodes.values():
        graph_node_order.pop(org_node.node_dict_key)
        org_nodes_index_list.append(org_node.node_dict_key)
    for res_node in res_nodes.values():
        graph_node_order.pop(res_node.node_dict_key)
        res_nodes_index_list.append(res_node.node_dict_key)
    mid_nodes_index_list = list(graph_node_order.keys())
    labels = dict(zip(list(graph.nodes), list(graph.nodes)))
    nx.draw_networkx_nodes(graph, pos, nodelist=org_nodes_index_list, node_color='skyblue', node_size=2000)
    nx.draw_networkx_nodes(graph, pos, nodelist=res_nodes_index_list, node_color='#ffc0cb', node_size=2000)
    nx.draw_networkx_nodes(graph, pos, nodelist=mid_nodes_index_list, node_color='orange', node_size=2000)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True, arrowstyle='wedge')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.savefig('fac.png')
    plt.show()

    fac_lib_dict = tfu.get_fac_lib_dict(list(set(
        basic_lib_names_list + df_fac_task_list['add_lib_names_list'].apply(lambda x: eval(x)).sum()
    )))
    tokens = [(org_node.name, org_node.node_type) for org_node in org_nodes.values() if org_node.node_type == 'fac']
    dict_df_data = tff.load_factor_from_tokens(tokens, begin_date, end_date, fac_lib_dict)
    tff.loop_calc_formula_graph(list(org_nodes.values()), dict_df_data, tff.calc_formula_graph)
    for key in df_factor_tree_dict.keys():  # rank值的差异
        print((df_factor_tree_dict[key].rank(axis=1, pct=True) - tfu.convert_stacked_frame_to_frame(
            res_nodes[key].value).rank(axis=1, pct=True)).abs().sum().sum())  # nan值不会判定相等，可以用sum判断是否有实际值的差异
    for key in df_factor_tree_dict.keys():  # 实际值的差异
        print((df_factor_tree_dict[key] - tfu.convert_stacked_frame_to_frame(
            res_nodes[key].value)).abs().sum().sum())  # nan值不会判定相等，可以用sum判断是否有实际值的差异

    # graph_node = res_nodes['test_fac0']
    # tree_node = list_expr_tree_node[0][1]
    # div_gnode = graph_node.from_nodes[0].from_nodes[0].from_nodes[0].from_nodes[1]
    # div_tnode = tree_node.children[0].children[0].children[0].children[1]
    # graph_node.from_nodes[0].from_nodes[0].value[1] == tree_node.children[0].children[0].value[1]
    # graph_node.from_nodes[0].from_nodes[0].from_nodes[0].value[0] == tree_node.children[0].children[0].children[0].value[0]
    print('done')
