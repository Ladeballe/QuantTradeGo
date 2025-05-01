from typing import Dict

import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt

from test_func import formula as tff
from test_func.formula.calc_formula import (
    NodeDict, trans_expr_tree_to_graph, loop_calc_formula_graph, calc_formula_graph, init_drawing_graph,
    scrutinize_tree_endpoint, scrutinize_tree_middlepoint)
from test_func import util as tfu
import test_func as tf


def test_expr_treenode_type():
    """ 验证所有末梢节点都是 [fac, param, num], 所有中介节点都是[unary_oprt, oprt, func]"""
    df_fac_task_list = pd.read_excel(r'D:\research\CRYPTO_vp_fac\CRYPTO_create_fac.xlsx')
    df_fac_task_list = df_fac_task_list[df_fac_task_list['create'] == 1]
    df_fac_task_list = tf.main.iter_flatten_df_fac_info(df_fac_task_list)

    endpoint_list = list()
    middlepoint_list = list()
    for i, row in df_fac_task_list.iterrows():
        name, formula = row['name'], row['formula']
        tokens = tff.parse_formula_to_tokens(formula)
        formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens)
        scrutinize_tree_endpoint(formula_ast_tree, endpoint_list)
        scrutinize_tree_middlepoint(formula_ast_tree, middlepoint_list)

    df0 = pd.DataFrame([[e.name, e.node_type, e] for e in endpoint_list])
    df1 = pd.DataFrame([[e.name, e.node_type, e] for e in middlepoint_list])
    return df0, df1


if __name__ == '__main__':
    df_fac_task_list = pd.read_excel(r'D:\research\CRYPTO_vp_fac\CRYPTO_create_fac_15m.xlsx')
    df_fac_task_list = df_fac_task_list[df_fac_task_list['create'] == 4]
    df_fac_task_list = tf.main.iter_flatten_df_fac_info(df_fac_task_list)

    list_expr_tree_node = list()
    for i, row in df_fac_task_list.iterrows():
        fac_name, formula = row['name'], row['formula']
        tokens = tff.parse_formula_to_tokens(formula)
        formula_ast_tree = tff.trans_tokens_to_expr_ast_tree(tokens)
        list_expr_tree_node.append((fac_name, formula_ast_tree))

    org_nodes, res_nodes = dict(), dict()
    node_dict = NodeDict()
    for fac_name, expr_tree_node in list_expr_tree_node:
        res_nodes[fac_name] = trans_expr_tree_to_graph(fac_name, expr_tree_node, org_nodes, res_nodes, node_dict)
        # 可视化nodes
        # graph = nx.Graph()
        # plt.figure(figsize=(20, 10))
        # for org_node in org_nodes.values():
        #     init_drawing_graph(org_node, graph)
        # pos = graphviz_layout(graph, 'dot')  # 为图形设置布局
        # nx.draw(graph, pos, with_labels=True, node_color='skyblue',
        #         node_size=2000, edge_color='k', linewidths=1, font_size=8)
        # plt.savefig('fac.png')
        # plt.show()

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

    loop_calc_formula_graph(list(org_nodes.values()), dict(), calc_formula_graph)

    print('done')
