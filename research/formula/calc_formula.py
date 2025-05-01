import os
import re
from typing import List, Dict, Callable
from functools import lru_cache, partial
from multiprocessing import Pool, Lock, Queue, Event, Manager

import numpy as np
import pandas as pd
import torch
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt

from ..data import get_factor, get_lru_factor
from ..util import convert_time_str, align_stack_frames, convert_frame_to_stack, convert_stacked_frame_to_frame, get_fac_lib_dict
from .calc_formula_funcs import (
    _calc_max, _calc_corr, _calc_if, _calc_min, _calc_switch, _calc_regbeta, _calc_reg_resid, _calc_reg_rsquared,
    _calc_reindex, _calc_wgt_mean, _calc_wgt_sum, _calc_wgt_mean_process, _calc_wgt_sum_process,
    _calc_agg_INDEX_NCI, _calc_agg
)


def _parse_node_params(*nodes):
    params = [node.value for node in nodes]
    return params


def _concat_comma(x, y):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]
    return x + y


def _concat_sr(x, y, func):
    """ 这样做是为了避免因子计算后 """
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        df_fac = align_stack_frames(x, y)
        x, y = df_fac['raw_factor0'], df_fac['raw_factor1']
        df_fac['raw_factor'] = func(x, y)
        return df_fac[list(df_fac.columns[:2]) + ['raw_factor']]
    elif isinstance(x, pd.DataFrame):
        df_fac = x.copy()  # 在一些情况下，会导致child的节点值发生变化，这在ExprAstTree中没问题，因为对于树只需要运行一次，但是对于graph而言，这样的设计会导致计算node的to_node时，导致node.value发生变化
        df_fac['raw_factor'] = func(x['raw_factor'], y)
        return x
    elif isinstance(y, pd.DataFrame):
        df_fac = y.copy()  # 在一些情况下，会导致child的节点值发生变化，这在ExprAstTree中没问题，因为对于树只需要运行一次，但是对于graph而言，这样的设计会导致计算node的to_node时，导致node.value发生变化
        df_fac['raw_factor'] = func(x, y['raw_factor'])
        return y
    else:
        return func(x, y)


oprt_dict = {'none': -1, 'func': 0, '(': 0, ')': 0, ',': 1, '+': 2, '-': 2, '*': 3, '/': 3, '^': 4}
oprt_calc_func_dict = {
    '+': partial(_concat_sr, func=lambda x, y: x + y),
    '-': partial(_concat_sr, func=lambda x, y: x - y),
    '*': partial(_concat_sr, func=lambda x, y: x * y),
    '/': partial(_concat_sr, func=lambda x, y: x / y),
    '^': partial(_concat_sr, func=lambda x, y: x ** y),
    ',': _concat_comma
}
func_dict = {'Max': _calc_max, 'Corr': _calc_corr, 'If': _calc_if, 'Min': _calc_min, 'Switch': _calc_switch,
             'RegBeta': _calc_regbeta, 'RegResid': _calc_reg_resid, 'RegRsquared': _calc_reg_rsquared,
             'Reindex': _calc_reindex,
             'WgtMean': _calc_wgt_mean, 'WgtSum': _calc_wgt_sum,
             'WgtMeanPro': _calc_wgt_mean_process, 'WgtSumPro': _calc_wgt_sum_process,
             }


def _calc_fac_stepwise(func, steps, *params):
    params = list(params)
    if hasattr(func, 'window'):
        window = int(params[func.window])
    else:
        window = 1
    if hasattr(func, 'facs'):
        facs = func.facs
    else:
        facs = list(range(len(params)))
        if hasattr(func, 'window'):
            facs.remove(func.window)

    for i in facs:
        if isinstance(params[i], pd.DataFrame):
            params[i] = convert_frame_to_stack(convert_stacked_frame_to_frame(params[i]).iloc[-(window + steps - 1):])
    df_fac = func(*params)
    if df_fac.index.dtype.name == 'datetime64[ns]':
        df_fac = convert_frame_to_stack(df_fac.iloc[-steps:])
    else:
        df_fac = convert_frame_to_stack(convert_stacked_frame_to_frame(df_fac).iloc[-steps:])
    return df_fac


def update_func_dict(new_func_dict: Dict[str, Callable]):
    global func_dict
    func_dict.update(new_func_dict)


class ExprTreeNode:
    """
    将表达式的
    - 因子
    - 数值
    - 一元运算符
    - 二元运算符
    - 括号
    - 函数
    转化为二叉树, 便于后续对二叉树进行后序遍历计算因子
    """
    def __init__(self, name, node_type, value=None, parent=None, children=None):
        self.name = name
        self.value = value
        self.node_type = node_type
        self.parent = parent
        self.children = children if children else list()

    @classmethod
    def empty_node(cls):
        """ 目前已经不再使用 """
        return cls('none', 'none')

    def _indent_repr(self, indent=4):
        """ 格式化输出, 自动计算节点的缩进 """
        lines = [
            "ExprTreeNode{",
            indent * " " + f"name: '{self.name}'",
            indent * " " + "children: [",
        ]
        # if self.unary_oprt:
        #     lines.insert(2, indent * " " + f'unary_oprt: {self.unary_oprt}')
        if len(self.children) > 0:
            for child in self.children:
                child_lines = child._indent_repr(indent)
                child_lines = [" " * indent * 2 + child_line for child_line in child_lines]
                lines += child_lines
            lines += [indent * " " + "]", "}"]
        else:
            lines[-1] += "]"
            lines += ["}"]
        return lines

    def __repr__(self):
        return "\n".join(self._indent_repr())

    def set_node_type(self, node_type):
        self.node_type = node_type

    def set_parent(self, parent_node):
        self.parent = parent_node

    def set_children(self, children):
        self.children += children

    def print_tree(self, method='concise'):
        """ 将树打印出来
        - concise: 只打印节点的名称, 按照缩进绘制树结构
        - detail: 打印节点的名称, 类型, 一元操作符, 以及子节点
        """
        if method == 'concise':
            print_tree(self)
        elif method == 'detail':
            print(self)


"""
整个graph包含2个node endpoint:
- 原始因子端: 设置为一个字典进行存储, {name: node}
- 最终因子端: 设置为一个字典进行存储, {name: node}
- 中间为去重之后的计算过程
在转换的过程中可以去重
"""


class NodeDict(dict):
    def set_value(self, key, value):
        if key in self.keys():
            self[key][key + f"#{len(self[key])}"] = value
            return key + f"#{len(self[key])}"
        else:
            self[key] = dict()
            self[key][key + "#0"] = value
            return key + "#0"

    def to_dict(self):
        flatten_dict = dict()
        for value in self.values():
            flatten_dict.update(value)
        return flatten_dict


class ExprGraphNode:
    """
    Attributes
    ----------
    name : str
        node的名称, 也是formula和ExprNodeTree的名称
    value : dataframe
        node的值, 即pandas.DataFrame存储着具体的因子值
    node_type : str
        node的类型, 与ExprTreeNode的类型相对应
    node_dict_key : str
        node的唯一表示key, 通过NodeDict添加新值时赋予, 是对node的唯一标识符
    act_flag : int
        用于逐步更新而设计, 初始值为0, 每次更新, flag += 1
    from_nodes : List[ExprGraphNode]
        上游节点的List
    to_nodes : List[ExprGraphNode]
        下游节点的List

    Methods
    -------
    structure_hash
    short_repr
    set_from_node
    set_to_node
    get_to_node
    delete
    save_to_node_dict
    """
    def __init__(self, name, node_type, value=None, from_nodes=None, to_nodes=None):
        self.name = name
        self.value = value
        self.node_type = node_type
        self.node_dict_key = None
        self.act_flag = 0
        self.from_nodes = from_nodes if from_nodes else list()
        self.to_nodes = to_nodes if to_nodes else list()

    def __repr__(self):
        repr = f"{self.name}"
        # if self.to_nodes:  # 更加复杂的版本，便于debug
        #     repr += "{" + f"to_nodes:{[node for node in self.to_nodes]}" + "}"
        if self.from_nodes:
            repr += "{" + f"from_nodes:{[node for node in self.from_nodes]}"
        # return f"{self.name}" + "{" + f"to_nodes:{[node.name for node in self.to_nodes]}" + "}"
        return repr

    def structure_hash(self):
        hash_value = self.name + self.node_type
        for child in self.from_nodes:
            hash_value += str(hash(child))
        hash_key = hash(hash_value)
        return hash_key

    def short_repr(self):
        return self.name

    def set_from_node(self, node):
        self.from_nodes.append(node)
        node.to_nodes.append(self)

    def set_to_node(self, node):
        self.to_nodes.append(node)
        node.from_nodes.append(self)

    def get_to_nodes(self, name):
        res_list = list()
        for node in self.to_nodes:
            if name == node.name:
                res_list.append(node)
        return res_list

    def delete(self):
        for node in self.from_nodes:
            node.to_nodes.remove(self)

    def save_to_node_dict(self, node_dict):
        self.node_dict_key = node_dict.set_value(self.name, self)


def parse_formula_to_tokens(formula):
    #              float
    #                |             word               sign             []
    #                ↓              ↓                  ↓              ↓  ↓
    pattern = r'(\d*\.?\d+|[\w\u4e00-\u9fa5]+|[+\-*/()^,@\.\<\>\{\}]|\[|\])'
    tokens = re.findall(pattern, formula)
    agg_tokens = list()
    while tokens:
        token = tokens.pop(0)
        if token in oprt_dict.keys():
            # ['+', '-', '*', '/']等类似符号直接加入tokens
            agg_tokens.append((token, 'oprt'))
        elif token == '@':
            # 如果以'@'开头，说明是其他操作符(@AND@, @OR@, @FILT@), 一直弹出token并拼接直到读取到下一个'@'
            while True:
                new_token = tokens.pop(0)
                token += new_token
                if new_token == '@':
                    break
            agg_tokens.append((token, 'oprt'))
        elif token == '[':  # '['说明是一元运算符, 一直弹出token并拼接直到读取到下一个']'
            while True:
                new_token = tokens.pop(0)
                token += new_token
                if new_token == ']':
                    break
            agg_tokens.append((token, 'unary_oprt'))
        elif token == '<':  # '<'说明是函数参数, 一直弹出token并拼接直到读取到下一个'>'
            while True:
                new_token = tokens.pop(0)
                token += new_token
                if new_token == '>':
                    break
            agg_tokens.append((token, 'param'))
        elif token == '{':  # '<'说明是函数参数, 一直弹出token并拼接直到读取到下一个'>'
            while True:
                new_token = tokens.pop(0)
                token += new_token
                if new_token == '}':
                    break
            agg_tokens.append((token, 'param'))
        else:  # 说明是因子, 数字, 函数
            if token in func_dict.keys():
                new_token = tokens.pop(0)
                if new_token != '(':  # 如果是函数, 则弹出下一个'('
                    raise ValueError(f'"(" should appear behind func keyword {token}.')
                token = token + new_token
                token_type = 'func'
            else:
                try:
                    float(token)
                    token_type = 'num'
                except ValueError:
                    token_type = 'fac'
            if len(agg_tokens) > 0:  # 如果不是第一个token, 则有可能是带有负号的
                last_token, last_node_type = agg_tokens.pop()
                if last_token == '-' and (
                    len(agg_tokens) == 0 or (  # 如果是第一个token或者前一个token是'-'，则说明是负号
                        agg_tokens[-1][1] == 'oprt' and agg_tokens[-1][0] != ')'  # 再前一个token不是')'的运算符才是负号
                )):
                    # 如果上一个token是负号, 且上一个token不是'('和','，则说明是负号而不是减号
                    token = last_token + token
                else:
                    agg_tokens.append((last_token, last_node_type))
            agg_tokens.append((token, token_type))
    return agg_tokens


def trans_token_to_fac(token):
    if isinstance(token, str):
        if token.isdigit():
            return float(token)
        else:
            raise NotImplementedError
    else:
        return token


def load_factor_from_tokens(tokens, begin_date, end_date, symbols, fac_lib_dict):
    dict_df_factor = dict()
    for token in tokens:
        token, token_type = token
        token = token.strip('-')
        if token_type == 'fac' and token not in dict_df_factor.keys():  # token是已经加载的因子
            lib_name = fac_lib_dict[token]
            df_fac = get_lru_factor(token, lib_name, begin_date, end_date).rename_axis('symbol', axis=1)
            df_fac = df_fac[df_fac.columns.intersection(symbols)]
            dict_df_factor[token] = convert_frame_to_stack(df_fac)
    return dict_df_factor


def trans_tokens_to_expr_ast_tree(tokens, debug_bool=True):
    print(f"{pd.Timestamp.now()} {os.getpid()} token转换为expr_ast_tree {tokens}") if debug_bool else ''
    tokens = [('(', 'oprt')] + tokens + [(')', 'oprt')]
    stack_node_done = list()
    stack_node_undone = list()
    curr_expr_node = None  # 存储当前正在处理的ast_tree_node的指针

    def _insert_node_to_stack(curr_expr_node, node_done_bool=True):
        if curr_expr_node:  # 初始第一轮, 传入的curr_expr_node是None, 将这种情况排除
            if node_done_bool:
                stack_node_done.append(curr_expr_node)
            else:
                stack_node_undone.append(curr_expr_node)

    def _pop_stack_oprt():
        last_oprt = stack_node_undone.pop()
        return last_oprt

    def _pop_stack_expr():
        if stack_node_done:
            df_fac_a = stack_node_done.pop()
            df_fac_b = stack_node_done.pop()
            return df_fac_a, df_fac_b
        else:
            raise ValueError('Empty "stack_expr".')

    while tokens:
        curr_token, _ = tokens.pop(0)
        if curr_token[-1] == '(':
            node_type = 'brkt' if len(curr_token) == 1 else 'func'
            _insert_node_to_stack(curr_expr_node, False)  # 括号之前一定是没有计算完成的操作符或函数，因为括号内部元素的最终计算值是其子树
            curr_expr_node = ExprTreeNode(name=curr_token, node_type=node_type)
        elif curr_token == ')':
            _insert_node_to_stack(curr_expr_node, True)
            while True:
                last_oprt = _pop_stack_oprt()
                if last_oprt.node_type in ['brkt', 'func']:
                    fac = stack_node_done.pop()
                    if last_oprt.node_type == 'brkt':
                        curr_expr_node = fac
                    else:  # last_oprt.node_type == 'func'
                        last_oprt.set_children([fac])
                        curr_expr_node = last_oprt
                    break
                else:
                    fac_a, fac_b = _pop_stack_expr()
                    last_oprt.set_children([fac_b, fac_a])
                    _insert_node_to_stack(last_oprt, True)
        elif curr_token in oprt_dict.keys() and curr_token != '(':
            _insert_node_to_stack(curr_expr_node)  # 操作符之前(不是)
            curr_oprt = curr_token
            curr_expr_node = ExprTreeNode(name=curr_oprt, node_type='oprt')
            last_oprt = _pop_stack_oprt()
            if oprt_dict[curr_oprt] <= oprt_dict.get(last_oprt.name, 0):
                fac_a, fac_b = _pop_stack_expr()
                last_oprt.set_children([fac_b, fac_a])
                _insert_node_to_stack(last_oprt, True)
            else:
                _insert_node_to_stack(last_oprt, False)
        elif curr_token[0] == '[':
            unary_oprts = curr_token.strip('[]').split(',')
            for unary_oprt in unary_oprts:
                unary_oprt_expr_node = ExprTreeNode(unary_oprt, 'unary_oprt')
                unary_oprt_expr_node.set_children([curr_expr_node])
                curr_expr_node = unary_oprt_expr_node
        else:
            # 数字或因子之前，或者是操作符，或者是函数，而且是会以当前因子为子树的节点，因此必须定义为False
            _insert_node_to_stack(curr_expr_node, False)
            if re.match(r'([-]?\d*\.?\d+)', curr_token):
                node_type = 'num'
            elif curr_token[0] == '<':
                node_type = 'param'
            else:
                node_type = 'fac'
            curr_expr_node = ExprTreeNode(curr_token, node_type)

    return curr_expr_node


def trans_expr_tree_to_graph(fac_name, expr_tree_node, org_endpoints, res_endpoints, node_dict):
    def _judge_identical_node(node, similar_nodes):
        for similar_node in similar_nodes:
            if similar_node.structure_hash() == node.structure_hash():  # 有同样的from_nodes结构
                node.delete()
                return similar_node
        node.save_to_node_dict(node_dict)
        return node

    name, node_type = expr_tree_node.name, expr_tree_node.node_type
    if node_type in ['fac', 'num', 'param']:
        if name in org_endpoints.keys():
            node = org_endpoints[name]
        else:
            node = ExprGraphNode(name, node_type)
            node.save_to_node_dict(node_dict)
            org_endpoints[name] = node
    elif node_type in ['unary_oprt', 'oprt', 'func']:
        similar_nodes = list()
        node = ExprGraphNode(name, node_type)
        for child_expr_tree_node in expr_tree_node.children:
            child_node = trans_expr_tree_to_graph(fac_name, child_expr_tree_node, org_endpoints, res_endpoints, node_dict)
            similar_nodes += child_node.get_to_nodes(name)
            node.set_from_node(child_node)
        node = _judge_identical_node(node, similar_nodes)
    else:
        raise ValueError(f'Invalid node type: {node_type}')
    return node


def calc_formula_ast_tree(node: ExprTreeNode, dict_df_factor: Dict[str, pd.DataFrame], debug_bool=True) -> ExprTreeNode:
    # first, traverse all the children to get their value
    if len(node.children) > 0:
        for child_node in node.children:
            calc_formula_ast_tree(child_node, dict_df_factor, debug_bool)
    # then, calc the value of current node to get the value
    if node.node_type == 'num':
        node.value = float(node.name)
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值数字 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    elif node.node_type == 'fac':
        neg_fac = -1 if node.name[0] == '-' else 1
        node.value = dict_df_factor[node.name.strip('-')]
        node.value['raw_factor'] = neg_fac * node.value['raw_factor']
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值因子 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    elif node.node_type == 'param':
        node.value = node.name
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 解析参数 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    elif node.node_type == 'unary_oprt':
        node.value = calc_unary_oprt_on_node(node)
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算一元运算符 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    elif node.node_type == 'oprt':
        calc_func = oprt_calc_func_dict[node.name]
        node.value = calc_func(*_parse_node_params(*node.children))
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算二元运算符 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    elif node.node_type == 'func':
        calc_func = func_dict[node.name[:-1].strip('-')]
        neg_fac = -1 if node.name[0] == '-' else 1
        node.value = convert_frame_to_stack(calc_func(*node.children[0].value))
        node.value['raw_factor'] = neg_fac * node.value['raw_factor']
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算函数 "
              f"{node.name}[{node.node_type}]-{[child.name for child in node.children]}") if debug_bool else ''
    return node


def calc_formula_graph(node: ExprGraphNode, dict_df_factor: Dict[str, pd.DataFrame], debug_bool=True):
    if node.node_type == 'num':
        node.value = float(node.name)
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值数字 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    elif node.node_type == 'fac':
        neg_fac = -1 if node.name[0] == '-' else 1
        node.value = dict_df_factor[node.name.strip('-')]
        node.value['raw_factor'] = neg_fac * node.value['raw_factor']
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值因子 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    elif node.node_type == 'param':
        node.value = node.name
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 解析参数 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    elif node.node_type == 'unary_oprt':
        node.value = calc_unary_oprt_on_node(node)
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算一元运算符 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    elif node.node_type == 'oprt':
        calc_func = oprt_calc_func_dict[node.name]
        node.value = calc_func(*_parse_node_params(*node.from_nodes))
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算二元运算符 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    elif node.node_type == 'func':
        calc_func = func_dict[node.name[:-1].strip('-')]
        neg_fac = -1 if node.name[0] == '-' else 1
        node.value = convert_frame_to_stack(calc_func(*node.from_nodes[0].value))
        node.value['raw_factor'] = neg_fac * node.value['raw_factor']
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算函数 "
              f"{node.name}[{node.node_type}]->{node}") if debug_bool else ''
    return node


def calc_formula_graph_stepwise(node: ExprGraphNode, dict_df_factor: Dict[str, pd.DataFrame], debug_bool=True, steps=1):
    if node.node_type == 'num':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值数字 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        node.value = float(node.name)
    elif node.node_type == 'fac':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 赋值因子 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        neg_fac = -1 if node.name[0] == '-' else 1
        node.value = dict_df_factor[node.name.strip('-')]
        node.value['raw_factor'] = neg_fac * node.value['raw_factor']
    elif node.node_type == 'param':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 解析参数 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        node.value = node.name
    elif node.node_type == 'unary_oprt':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算一元运算符 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        node.value = calc_unary_oprt_stepwise_on_node(node)
    elif node.node_type == 'oprt':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算二元运算符 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        calc_func = oprt_calc_func_dict[node.name]
        if node.name == ',':
            node.value = calc_func(*_parse_node_params(*node.from_nodes))
        else:
            node.value = pd.concat([node.value, _calc_fac_stepwise(calc_func, steps, *_parse_node_params(*node.from_nodes))])
    elif node.node_type == 'func':
        print(f"{pd.Timestamp.now()} {os.getpid()} 节点 计算函数 "
              f"{node.name}[{node.node_type}]-{[to_node.name for to_node in node.to_nodes]}") if debug_bool else ''
        calc_func = func_dict[node.name[:-1].strip('-')]
        neg_fac = -1 if node.name[0] == '-' else 1
        df_fac = _calc_fac_stepwise(calc_func, steps, *node.from_nodes[0].value)
        df_fac['raw_factor'] = neg_fac * df_fac['raw_factor']
        node.value = pd.concat([node.value, df_fac])
    return node


def loop_calc_formula_graph(unupdated_nodes: List[ExprGraphNode], dict_df_data: Dict[str, pd.DataFrame], calc_func: Callable = lambda *x: None, debug_bool: bool = True, *args, **kwargs):
    """
    遍历整个图，按照图结构，依次更新每个节点的值
    算法从末梢(fac, num, param)开始向上计算，知道计算得到因子值(ExprNodeTree的根节点)
    设计思想:
    - unupdated_nodes是一个队列, 其中存储着所有nodes
    - 只有在当前节点计算完成时, 才将新的节点推入unupdated_nodes
      - 这是确保每个更新后的graph都已将所有to_nodes都加入到列表中
      - 但仍然可能存在部分节点，其被添加了2次
    - 但是判断新的节点能否计算，还是要判断其上游节点是否已计算完成

    Parameters
    ----------
    nodes
    dict_df_data
    calc_func

    Returns
    -------

    """
    # step = 0
    updated_nodes = list()
    curr_flag = unupdated_nodes[0].act_flag
    while unupdated_nodes:
        node = unupdated_nodes.pop(0)  # 队列先进后出
        if node.act_flag - 1 == curr_flag:
            continue
        else:  # 当前点已经更新, 已经更新的点肯定已经被添加进入updated_nodes和计算过, 直接pass
            bool_ready_calc = True
            for from_node in node.from_nodes:
                if from_node.act_flag == curr_flag:  # 只要有一个上游节点没有更新，则当前点无法计算
                    bool_ready_calc = False
                    break
            if bool_ready_calc:
                calc_func(node, dict_df_data, debug_bool=debug_bool, *args, **kwargs)
                node.act_flag += 1
                updated_nodes.append(node)
                for to_node in node.to_nodes:
                    unupdated_nodes.append(to_node)
            else:  # 当前点无法更新
                unupdated_nodes.append(node)
        # graph_draw_debug(node, org_nodes, updated_nodes, step)  # debug用, 每计算一个点，绘制一个图
        # step += 1


def loop_calc_formula_graph_paralell(unupdated_nodes: List[ExprGraphNode], dict_df_data: Dict[str, pd.DataFrame], calc_func: Callable = lambda *x: None, debug_bool: bool = True, node_dict: NodeDict = NodeDict(), *args, **kwargs):
    """
    遍历整个图，按照图结构，依次更新每个节点的值
    算法从末梢(fac, num, param)开始向上计算，知道计算得到因子值(ExprNodeTree的根节点)
    设计思想:
    - unupdated_nodes是一个队列, 其中存储着所有nodes
    - 只有在当前节点计算完成时, 才将新的节点推入unupdated_nodes
      - 这是确保每个更新后的graph都已将所有to_nodes都加入到列表中
      - 但仍然可能存在部分节点，其被添加了2次
    - 但是判断新的节点能否计算，还是要判断其上游节点是否已计算完成
    多进程改进，多进程情况下，需要将原有函数分为2个子函数：
    - 生产者函数-任务调度：判断当前未计算的node是否完成计算，获取锁，修改计算图，获取下一步需要计算的node
    - 消费者函数-计算与更新：调用进程池计算因子，获取锁，修改计算图

    Parameters
    ----------
    nodes
    dict_df_data
    calc_func

    Returns
    -------

    """

    manager = Manager()
    lock = manager.lock()
    unupdated_nodes = manager.list(unupdated_nodes)
    updating_nodes = manager.dict()

    def _worker(node, dict_df_data, debug_bool, *args, **kwargs):
        node_value = calc_func(node, dict_df_data, debug_bool=debug_bool, *args, **kwargs)
        with lock:
            node.node_value = node_value
            node.act_flag += 1
            for to_node in node.to_nodes:
                unupdated_nodes.append(to_node)

    curr_flag = unupdated_nodes[0].act_flag
    while True:  # unupdated_nodes or updating_nodes:  # 这种实现方式，可能造成数据变化，在访问这两个同步变量，必须获取锁来访问
        lock.acquire()
        if unupdated_nodes:
            node = unupdated_nodes.pop(0)  # 队列先进后出
            lock.release()
            if node.act_flag - 1 == curr_flag:  # 当前点已经更新, 已经更新的点肯定已经被添加进入updated_nodes和计算过, 直接pass
                continue
            else:
                bool_ready_calc = True
                with lock:
                    for from_node in node.from_nodes:
                        if from_node.act_flag == curr_flag:  # 只要有一个上游节点没有更新，则当前点无法计算
                            bool_ready_calc = False
                            break
                if bool_ready_calc:
                    with lock:
                        updating_nodes[node.node_dict_key] = node
                else:  # 当前点无法更新
                    unupdated_nodes.append(node)
        else:
            lock.release()
        with lock:
            if not (unupdated_nodes or updating_nodes):
                # 所有结点均处理完成
                break


def scrutinize_tree_endpoint(expr_tree_node, endpoint_list):
    if expr_tree_node.children:
        for child in expr_tree_node.children:
            scrutinize_tree_endpoint(child, endpoint_list)
    else:
        endpoint_list.append(expr_tree_node)


def scrutinize_tree_middlepoint(expr_tree_node, middlepoint_list):
    if expr_tree_node.children:
        middlepoint_list.append(expr_tree_node)
        for child in expr_tree_node.children:
            scrutinize_tree_middlepoint(child, middlepoint_list)


def init_drawing_graph(node, graph):
    """ 根据根节点便利整个计算图 """
    graph.add_node(node.node_dict_key)
    for to_node in node.to_nodes:
        init_drawing_graph(to_node, graph)
        graph.add_edge(node.node_dict_key, to_node.node_dict_key)


def graph_draw_debug(node, org_nodes, updated_nodes, step):
    """天蓝未处理, 粉色已处理, 红色处理中"""
    graph = nx.DiGraph()
    fig = plt.figure(figsize=(50, 15))
    for org_node in org_nodes.values():
        init_drawing_graph(org_node, graph)
    pos = graphviz_layout(graph, 'dot')  # 为图形设置布局
    graph_nodes = dict(zip(list(graph.nodes), list(graph.nodes)))
    graph_updated_nodes = list()
    for updated_node in updated_nodes:
        graph_nodes.pop(updated_node.node_dict_key)
        graph_updated_nodes.append(updated_node.node_dict_key)
    labels = dict(zip(list(graph.nodes), list(graph.nodes)))
    nx.draw_networkx_nodes(graph, pos, nodelist=graph_nodes, node_color='skyblue', node_size=2000)
    nx.draw_networkx_nodes(graph, pos, nodelist=graph_updated_nodes, node_color='#ffc0cb', node_size=2000)
    nx.draw_networkx_nodes(graph, pos, nodelist=[node.node_dict_key], node_color='red', node_size=2000)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True, arrowstyle='wedge')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.savefig(f'fac_step{step}.png')
    plt.close(fig)


def calc_unary_oprt_on_node(node):
    attr_name = 'children' if isinstance(node, ExprTreeNode) else 'from_nodes'
    df_fac, unary_oprt = getattr(node, attr_name)[0].value, '[' + node.name + ']'
    df_fac = calc_unary_oprt(df_fac, unary_oprt)
    return df_fac


def calc_unary_oprt(df_fac, unary_oprt):
    df_fac = convert_stacked_frame_to_frame(df_fac)
    unary_oprts = unary_oprt[1:-1].split(',')
    for unary_oprt in unary_oprts:
        unary_oprt, unary_oprt_params = unary_oprt.split('_')[0].lower(), unary_oprt.split('_')[1:]
        if unary_oprt in ['rbd', 'rbw', 'rbp', 'rbr']:
            if unary_oprt == 'rbd':
                df_fac = df_fac.rank(axis=1)
            elif unary_oprt == 'rbw':
                df_fac = df_fac.rolling(int(unary_oprt_params[0])).rank()
            elif unary_oprt == 'rbp':
                df_fac = df_fac.rolling(int(unary_oprt_params[0]))
            else:  # 'rbr'
                raise NotImplementedError(f'"Rank" unary operator "{unary_oprt}" is not implemented for now.')
        elif unary_oprt in ['ffill', 'fillna']:
            if unary_oprt == 'ffill':
                df_fac = df_fac.ffill()
            elif unary_oprt == 'fillna':
                df_fac = df_fac.fillna(float(unary_oprt_params[0]))
            else:  #
                raise NotImplementedError(f'"Fill" unary operator "{unary_oprt}" is not implemented for now.')
        elif unary_oprt in ['abs', 'exp', 'log']:
            if unary_oprt == 'abs':
                df_fac = df_fac.abs()
            elif unary_oprt == 'exp':
                df_fac = df_fac.apply(lambda x: np.exp(x))
            elif unary_oprt == 'log':
                df_fac = df_fac.apply(np.log)
        elif unary_oprt in ['lt', 'gt', 'eq', 'ne', 'in', 'out', 'among', 'isna', 'notna']:
            if unary_oprt == 'gt':  # 大于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(df_fac > threshold, 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'lt':  # 小于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(df_fac < threshold, 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'eq':  # 等于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(df_fac == threshold, 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'ne':  # 等于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(df_fac != threshold, 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'among':
                threshold = [float(unary_oprt_params[0]), float(unary_oprt_params[1])]
                df_fac = pd.DataFrame(np.where(df_fac.isin(threshold), 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'in':  # 介于
                left, right = float(unary_oprt_params[0]), float(unary_oprt_params[1])
                df_fac = pd.DataFrame(np.where((df_fac >= left) & (df_fac <= right), 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'out':  # 在区域之外
                left, right = float(unary_oprt_params[0]), float(unary_oprt_params[1])
                df_fac = pd.DataFrame(np.where((df_fac <= left) | (df_fac >= right), 1, np.nan), index=df_fac.index,
                                      columns=df_fac.columns)
            elif unary_oprt == 'isna':
                df_fac = pd.DataFrame(np.where(df_fac.isna(), 1, np.nan), index=df_fac.index, columns=df_fac.columns)
            elif unary_oprt == 'notna':
                df_fac = pd.DataFrame(np.where(~df_fac.isna(), 1, np.nan), index=df_fac.index, columns=df_fac.columns)
        elif unary_oprt in ['sum', 'ma', 'mean', 'std', 'median', 'min', 'max', 'zscore', 'skew', 'kurt', 'quantile', 'ewm']:
            min_periods = 1
            rolling_window = int(unary_oprt_params[0])
            if unary_oprt == 'sum':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).sum()
            elif unary_oprt == 'ma' or unary_oprt == 'mean':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).mean()
            elif unary_oprt == 'std':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).std()
            elif unary_oprt == 'median':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).median()
            elif unary_oprt == 'min':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).min()
            elif unary_oprt == 'max':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).max()
            elif unary_oprt == 'zscore':
                df_fac = (df_fac - df_fac.rolling(rolling_window, min_periods=min_periods).mean()) / df_fac.rolling(
                    rolling_window, min_periods=min_periods).std()
            elif unary_oprt == 'skew':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).skew()
            elif unary_oprt == 'kurt':
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).kurt()
            elif unary_oprt == 'quantile':
                quantile = float(unary_oprt_params[1])  # 第二个参数是quantile
                df_fac = df_fac.rolling(rolling_window, min_periods=min_periods).quantile(quantile)
            elif unary_oprt == 'ewm':
                df_fac = df_fac.ewm(span=rolling_window).mean()
        elif unary_oprt in ['cumsum', 'cummean', 'cumstd', 'cummedian', 'cummin', 'cummax', 'cumzscore', 'cumskew', 'cumquantile']:
            if unary_oprt == 'cumsum':
                df_fac = df_fac.groupby(df_fac.index.date).cumsum()
            elif unary_oprt == 'cummean':
                df_fac = df_fac.groupby(df_fac.index.date).expanding().mean().droplevel(0)
            elif unary_oprt == 'cumstd':
                df_fac = df_fac.groupby(df_fac.index.date).expanding().std().droplevel(0)
            elif unary_oprt == 'cummedian':
                df_fac = df_fac.groupby(df_fac.index.date).expanding().median().droplevel(0)
            elif unary_oprt == 'cummin':
                df_fac = df_fac.groupby(df_fac.index.date).cummin()
            elif unary_oprt == 'cummax':
                df_fac = df_fac.groupby(df_fac.index.date).cummax()
            elif unary_oprt == 'cumzscore':
                df_fac_mean = df_fac.groupby(df_fac.index.date).expanding().mean().droplevel(0)
                df_fac_std = df_fac.groupby(df_fac.index.date).expanding().std().droplevel(0)
                df_fac = (df_fac - df_fac_mean) / df_fac_std
            elif unary_oprt == 'cumskew':
                df_fac = df_fac.groupby(df_fac.index.date).expanding().skew().droplevel(0)
            elif unary_oprt == 'cumquantile':
                quantile = float(unary_oprt_params[1])
                df_fac = df_fac.groupby(df_fac.index.date).expanding().quantile(quantile).droplevel(0)
            elif unary_oprt == 'cumfill':
                df_fac = df_fac.groupby(df_fac.index.date).ffill()
        elif unary_oprt in ['shift', 'diff', 'pctchg']:  # 移动处理
            window = int(unary_oprt_params[0])
            if unary_oprt == 'shift':
                df_fac = df_fac.shift(window)
            elif unary_oprt == 'diff':
                df_fac = df_fac.diff(window)
            elif unary_oprt == 'pctchg':
                df_fac = df_fac.pct_change(window)
        elif unary_oprt in ['betweentime','filltime']:  # 日内分钟截取
            if unary_oprt == 'betweentime':
                start_time_str, end_time_str = unary_oprt_params
                start_time_str, end_time_str = convert_time_str(start_time_str), convert_time_str(end_time_str)
                df_fac = df_fac.between_time(start_time_str, end_time_str)
        elif unary_oprt in ['todaily', 'tominute', 'tofreq']:
            if unary_oprt == 'todaily':
                time_str = unary_oprt_params[0]
                time_str = convert_time_str(time_str)
                df_fac = df_fac.at_time(time_str)  # df.between_time(time,time)
                df_fac.index = pd.to_datetime(df_fac.index.date)
        elif unary_oprt in ['agg']:
            if unary_oprt_params[0] == 'nci':
                df_fac = _calc_agg_INDEX_NCI(df_fac)
            elif unary_oprt_params[0] == 'sum':
                df_fac = _calc_agg(df_fac, 'sum')
            elif unary_oprt_params[0] == 'mean':
                df_fac = _calc_agg(df_fac, 'mean')
            elif unary_oprt_params[0] == 'demean':
                # df_fac.to_excel(r'D:\python_projects\quant_trade_go\test\fac_test2.xlsx')
                df_fac = _calc_agg(df_fac, 'demean')
            elif unary_oprt_params[0] == 'zscore':
                df_fac = _calc_agg(df_fac, 'zscore')
        else:
            raise NotImplementedError(f'Unimplemented unary operator {unary_oprt}')
    df_fac = convert_frame_to_stack(df_fac)
    return df_fac


def calc_unary_oprt_stepwise_on_node(node, steps=1):
    attr_name = 'children' if isinstance(node, ExprTreeNode) else 'from_nodes'
    child_df_fac, unary_oprt = getattr(node, attr_name)[0].value, '[' + node.name + ']'
    df_fac = node.value
    df_fac = pd.concat([df_fac, calc_unary_oprt_stepwise(child_df_fac, df_fac, unary_oprt, steps=1)])
    return df_fac


def calc_unary_oprt_stepwise(child_df_fac, df_fac, unary_oprt, steps=1):
    df_fac = convert_stacked_frame_to_frame(df_fac)
    child_df_fac = convert_stacked_frame_to_frame(child_df_fac)

    unary_oprts = unary_oprt[1:-1].split(',')
    for unary_oprt in unary_oprts:
        unary_oprt, unary_oprt_params = unary_oprt.split('_')[0].lower(), unary_oprt.split('_')[1:]
        if unary_oprt in ['rbd', 'rbw', 'rbp', 'rbr']:
            if unary_oprt == 'rbd':
                df_fac = child_df_fac.iloc[[-1]].rank(axis=1)
            elif unary_oprt == 'rbw':
                window = int(unary_oprt_params[0])
                df_fac = child_df_fac.iloc[-(window + steps - 1):].rolling(window).rank().iloc[[-1]]
            # elif unary_oprt == 'rbp':
            #     window = int(unary_oprt_params[0])
            #     df_fac = df_fac.rolling(window)
            else:  # 'rbr'
                raise NotImplementedError(f'"Rank" unary operator "{unary_oprt}" is not implemented for now.')
        elif unary_oprt in ['ffill', 'fillna']:
            if unary_oprt == 'ffill':
                df_fac = child_df_fac.iloc[-(steps + 1):].ffill().iloc[-steps:]
            elif unary_oprt == 'fillna':
                df_fac = child_df_fac.iloc[-steps:].fillna(float(unary_oprt_params[0]))
            else:  #
                raise NotImplementedError(f'"Fill" unary operator "{unary_oprt}" is not implemented for now.')
        elif unary_oprt in ['abs', 'exp', 'log']:
            if unary_oprt == 'abs':
                df_fac = child_df_fac.iloc[-steps:].abs()
            elif unary_oprt == 'exp':
                df_fac = child_df_fac.iloc[-steps:].apply(lambda x: np.exp(x))
            elif unary_oprt == 'log':
                df_fac = child_df_fac.iloc[-steps:].apply(np.log)
        elif unary_oprt in ['lt', 'gt', 'eq', 'ne', 'in', 'out', 'among', 'isna', 'notna']:
            if unary_oprt == 'gt':  # 大于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:] > threshold, 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'lt':  # 小于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:] < threshold, 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'eq':  # 等于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:] == threshold, 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'ne':  # 等于
                threshold = float(unary_oprt_params[0])
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:] != threshold, 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'among':
                threshold = [float(unary_oprt_params[0]), float(unary_oprt_params[1])]
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:].isin(threshold), 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'in':  # 介于
                left, right = float(unary_oprt_params[0]), float(unary_oprt_params[1])
                df_fac = pd.DataFrame(np.where((child_df_fac.iloc[-steps:] >= left) &\
                                               (child_df_fac.iloc[-steps:] <= right), 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'out':  # 在区域之外
                left, right = float(unary_oprt_params[0]), float(unary_oprt_params[1])
                df_fac = pd.DataFrame(np.where((child_df_fac.iloc[-steps:] <= left) |\
                                               (child_df_fac.iloc[-steps:] >= right), 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'isna':
                df_fac = pd.DataFrame(np.where(child_df_fac.iloc[-steps:].isna(), 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
            elif unary_oprt == 'notna':
                df_fac = pd.DataFrame(np.where(~child_df_fac.iloc[-steps:].isna(), 1, np.nan),
                                      index=child_df_fac.iloc[-steps:].index, columns=child_df_fac.iloc[-steps:].columns)
        elif unary_oprt in ['sum', 'ma', 'mean', 'std', 'median', 'min', 'max', 'zscore', 'skew', 'kurt', 'quantile', 'ewm']:
            min_periods = 1
            rolling_window = int(unary_oprt_params[0])
            if unary_oprt == 'sum':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).sum().iloc[-steps:]
            elif unary_oprt == 'ma' or unary_oprt == 'mean':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).mean().iloc[-steps:]
            elif unary_oprt == 'std':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).std().iloc[-steps:]
            elif unary_oprt == 'median':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).median().iloc[-steps:]
            elif unary_oprt == 'min':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).min().iloc[-steps:]
            elif unary_oprt == 'max':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).max().iloc[-steps:]
            elif unary_oprt == 'zscore':
                df_fac = ((child_df_fac.iloc[-(rolling_window + steps -1):] - child_df_fac.iloc[-(rolling_window + steps -1):].rolling(
                    rolling_window, min_periods=min_periods).mean()) / child_df_fac.iloc[-(rolling_window + steps -1):].rolling(
                    rolling_window, min_periods=min_periods).std()).iloc[-steps:]
            elif unary_oprt == 'skew':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).skew().iloc[-steps:]
            elif unary_oprt == 'kurt':
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).kurt().iloc[-steps:]
            elif unary_oprt == 'quantile':
                quantile = float(unary_oprt_params[1])  # 第二个参数是quantile
                df_fac = child_df_fac.iloc[-(rolling_window + steps -1):].rolling(rolling_window, min_periods=min_periods).quantile(quantile).iloc[-steps:]
            elif unary_oprt == 'ewm':
                df_fac = child_df_fac.ewm(span=rolling_window).mean().iloc[-steps:]
        # elif unary_oprt in ['cumsum', 'cummean', 'cumstd', 'cummedian', 'cummin', 'cummax', 'cumzscore', 'cumskew', 'cumquantile']:
        #     if unary_oprt == 'cumsum':
        #         df_fac = df_fac.groupby(df_fac.index.date).cumsum()
        #     elif unary_oprt == 'cummean':
        #         df_fac = df_fac.groupby(df_fac.index.date).expanding().mean().droplevel(0)
        #     elif unary_oprt == 'cumstd':
        #         df_fac = df_fac.groupby(df_fac.index.date).expanding().std().droplevel(0)
        #     elif unary_oprt == 'cummedian':
        #         df_fac = df_fac.groupby(df_fac.index.date).expanding().median().droplevel(0)
        #     elif unary_oprt == 'cummin':
        #         df_fac = df_fac.groupby(df_fac.index.date).cummin()
        #     elif unary_oprt == 'cummax':
        #         df_fac = df_fac.groupby(df_fac.index.date).cummax()
        #     elif unary_oprt == 'cumzscore':
        #         df_fac_mean = df_fac.groupby(df_fac.index.date).expanding().mean().droplevel(0)
        #         df_fac_std = df_fac.groupby(df_fac.index.date).expanding().std().droplevel(0)
        #         df_fac = (df_fac - df_fac_mean) / df_fac_std
        #     elif unary_oprt == 'cumskew':
        #         df_fac = df_fac.groupby(df_fac.index.date).expanding().skew().droplevel(0)
        #     elif unary_oprt == 'cumquantile':
        #         quantile = float(unary_oprt_params[1])
        #         df_fac = df_fac.groupby(df_fac.index.date).expanding().quantile(quantile).droplevel(0)
        #     elif unary_oprt == 'cumfill':
        #         df_fac = df_fac.groupby(df_fac.index.date).ffill()
        elif unary_oprt in ['shift', 'diff', 'pctchg']:  # 移动处理
            window = int(unary_oprt_params[0])
            if unary_oprt == 'shift':
                df_fac = child_df_fac.iloc[-(window + 1):].shift(window).iloc[-steps:]
            elif unary_oprt == 'diff':
                df_fac = child_df_fac.iloc[-(window + 1):].diff(window).iloc[-steps:]
            elif unary_oprt == 'pctchg':
                df_fac = child_df_fac.iloc[-(window + 1):].pct_change(window).iloc[-steps:]
        # elif unary_oprt in ['betweentime','filltime']:  # 日内分钟截取
        #     if unary_oprt == 'betweentime':
        #         start_time_str, end_time_str = unary_oprt_params
        #         start_time_str, end_time_str = convert_time_str(start_time_str), convert_time_str(end_time_str)
        #         df_fac = df_fac.between_time(start_time_str, end_time_str)
        # elif unary_oprt in ['todaily', 'tominute', 'tofreq']:
        #     if unary_oprt == 'todaily':
        #         time_str = unary_oprt_params[0]
        #         time_str = convert_time_str(time_str)
        #         df_fac = df_fac.at_time(time_str)  # df.between_time(time,time)
        #         df_fac.index = pd.to_datetime(df_fac.index.date)
        elif unary_oprt in ['agg']:
            if unary_oprt_params[0] == 'nci':
                df_fac = _calc_agg_INDEX_NCI(child_df_fac.iloc[-steps:])
            elif unary_oprt_params[0] == 'sum':
                df_fac = _calc_agg(child_df_fac.iloc[-steps:], 'sum')
            elif unary_oprt_params[0] == 'mean':
                df_fac = _calc_agg(child_df_fac.iloc[-steps:], 'mean')
            elif unary_oprt_params[0] == 'demean':
                df_fac = _calc_agg(child_df_fac.iloc[-steps:], 'demean')
            elif unary_oprt_params[0] == 'zscore':
                df_fac = _calc_agg(child_df_fac.iloc[-steps:], 'zscore')
        else:
            raise NotImplementedError(f'Unimplemented unary operator {unary_oprt}')
    df_fac = convert_frame_to_stack(df_fac)
    return df_fac


def print_tree(node, indent=0):
    print(indent * ' ' + node.name)
    if len(node.children) > 0:
        for child in node.children:
            print_tree(child, indent + 2)
