from typing import Dict

import pandas as pd

from test_func.main import iter_flatten_df_fac_info
from test_func.util import convert_stacked_frame_to_frame
from test_func.formula.calc_formula import (
    NodeDict, trans_expr_tree_to_graph, loop_calc_formula_graph, calc_formula_graph, calc_formula_graph_stepwise)
from test_func.formula import parse_formula_to_tokens, trans_tokens_to_expr_ast_tree


class FactorHandler:
    def __init__(self, fac_file_path: str, condition: Dict) -> None:
        self.df_fac = None
        self.fac_file_path = fac_file_path
        self.condition = condition
        self.df_fac_task_list = None

    def init_factor_tasks(self):
        self.df_fac_task_list = pd.read_excel(self.fac_file_path)
        key, value = list(self.condition.items())[0]
        self.df_fac_task_list = self.df_fac_task_list[self.df_fac_task_list[key].fillna('').apply(self._examine_strategy, args=(value,))]
        self.df_fac_task_list = iter_flatten_df_fac_info(self.df_fac_task_list)

    def calc_fac(self, *args, **kwargs):
        pass

    @staticmethod
    def _examine_strategy(val, condition_val):
        val = val.split(',')
        loc_bool = len(set(val) & set(condition_val)) > 0
        return loc_bool


class FactorHandlerV1(FactorHandler):
    def __init__(self, fac_file_path: str, condition: Dict) -> None:
        super().__init__(fac_file_path, condition)
        self.org_nodes = dict()
        self.res_nodes = dict()
        self.node_dict = NodeDict()

    def calc_fac(self, dict_df_data):
        list_expr_tree_node = list()
        for i, row in self.df_fac_task_list.iterrows():
            fac_name, formula = row['name'], row['formula']
            tokens = parse_formula_to_tokens(formula)
            formula_ast_tree = trans_tokens_to_expr_ast_tree(tokens, debug_bool=False)
            list_expr_tree_node.append((fac_name, formula_ast_tree))

        for fac_name, expr_tree_node in list_expr_tree_node:
            self.res_nodes[fac_name] = trans_expr_tree_to_graph(fac_name, expr_tree_node, self.org_nodes, self.res_nodes, self.node_dict)

        loop_calc_formula_graph(list(self.org_nodes.values()), dict_df_data, calc_formula_graph, debug_bool=True)
        self.df_fac = pd.concat([
            convert_stacked_frame_to_frame(node.value).iloc[-1]
            for node in self.res_nodes.values()
        ], keys=self.res_nodes.keys(), axis=1)

    def update_fac(self, dict_df_data):
        loop_calc_formula_graph(list(self.org_nodes.values()), dict_df_data, calc_formula_graph_stepwise, steps=1)
        self.df_fac = pd.concat(self.res_nodes.values(), keys=self.res_nodes.keys(), axis=1)

    def get_df_fac(self):
        return self.df_fac


class FactorHandlerV2:
    def __init__(self, fac_file_path: str) -> None:
        self.df_fac = None
        self.fac_file_path = fac_file_path
        self.org_nodes = dict()
        self.res_nodes = dict()
        self.node_dict = NodeDict()

    def init_factor_tasks(self):
        self.df_fac_task_list = pd.read_excel(self.fac_file_path)

    @staticmethod
    def _examine_strategy(val, condition_val):
        val = val.split(',')
        loc_bool = len(set(val) & set(condition_val)) > 0
        return loc_bool

    def filter_df_fac_task_list(self, strategy_name):
        df_fac_task_list = self.df_fac_task_list[
            self.df_fac_task_list['strategies'].str.contains(strategy_name).fillna(False)]
        df_fac_task_list = iter_flatten_df_fac_info(df_fac_task_list)
        return df_fac_task_list

    def calc_fac(self, dict_df_data, df_fac_task_list):
        # 每次计算前更新
        self.org_nodes = dict()
        self.res_nodes = dict()
        self.node_dict = NodeDict()

        list_expr_tree_node = list()
        for i, row in df_fac_task_list.iterrows():
            fac_name, formula = row['name'], row['formula']
            tokens = parse_formula_to_tokens(formula)
            formula_ast_tree = trans_tokens_to_expr_ast_tree(tokens, debug_bool=False)
            list_expr_tree_node.append((fac_name, formula_ast_tree))

        for fac_name, expr_tree_node in list_expr_tree_node:
            self.res_nodes[fac_name] = trans_expr_tree_to_graph(
                fac_name, expr_tree_node, self.org_nodes, self.res_nodes, self.node_dict)

        loop_calc_formula_graph(list(self.org_nodes.values()), dict_df_data, calc_formula_graph, debug_bool=True)
        self.df_fac = pd.concat([
            convert_stacked_frame_to_frame(node.value).iloc[-1]
            for node in self.res_nodes.values()
        ], keys=self.res_nodes.keys(), axis=1)

    def update_fac(self, dict_df_data):
        loop_calc_formula_graph(list(self.org_nodes.values()), dict_df_data, calc_formula_graph_stepwise, steps=1)
        self.df_fac = pd.concat(self.res_nodes.values(), keys=self.res_nodes.keys(), axis=1)

    def get_df_fac(self):
        return self.df_fac
