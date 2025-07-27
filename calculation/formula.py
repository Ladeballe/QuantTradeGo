import re

import numpy as np
import pandas as pd

from .formula_cfuncs import _trans_sbv_hurdle_4, _conj_entryand_accelerator, _conj_entryfilt_accelerator


_CONJ = {'_AND_': 2, '_NOP_': 2, '_SOR_': 2, '_FOR_': 2, '_FILT_': 2, '_DROP_': 2, '_INV_': 2, '_ENTRYAND_': 2,
         '_ENTRYFILT_': 2, '_ENTRYNOP_': 2}
_CONJ_PATTERN = rf"(_AND_|_OR_|_NOP_|_SOR_|_FOR_|_FILT_|_DROP_|_INV_|_ENTRYAND_|_ENTRYFILT_|_ENTRYNOP_)"
_SIG_PATTERN_DICT = {
    "SBV": r"(-?\d+\.?\d*)",
    "RANK": r"(RBD|RBP|RBT|RBW)",
    "FILL": r"(FFILL|FILLNA)",
    "PUTOFF": r"(PUTOFF)"
}


def expr_trans_sig(formula, df_factor):
    list_infix_token = _split_conj(formula)
    list_postfix_token = _infix_to_postfix(list_infix_token)

    stack = list()
    for token in list_postfix_token:
        if token in _CONJ:
            sig_b = stack.pop()
            sig_a = stack.pop()
            sig = connect_sig(sig_a, sig_b, token)
            stack.append(sig)
        else:
            sig = token_trans_sig(token, df_factor)
            stack.append(sig)

    sig = stack.pop()
    return sig


def _split_conj(formula):
    tokens = re.split(_CONJ_PATTERN, formula)
    return tokens


def _infix_to_postfix(list_infix_token):
    list_postfix_token = list()
    op_stack = []

    # 按字符遍历中缀表达式
    for token in list_infix_token:
        if token == '(':
            op_stack.append(token)  # 左括号入栈
        elif token == ')':
            while op_stack[-1] != '(':  # 右括号出现时，弹出栈顶运算符直至遇到左括号
                list_postfix_token.append(op_stack.pop())
            op_stack.pop()  # 弹出左括号
        elif token in _CONJ:  # 处理运算符
            while op_stack and op_stack[-1] != '(' and \
                    _CONJ[token] <= _CONJ[op_stack[-1]]:
                list_postfix_token.append(op_stack.pop())  # 当前运算符优先级低于或等于栈顶运算符时，弹出栈顶运算符
            op_stack.append(token)  # 当前运算符入栈
        else:
            list_postfix_token.append(token)

    # 将栈中剩余的运算符全部弹出添加到结果列表
    while op_stack:
        list_postfix_token.append(op_stack.pop())

    return list_postfix_token


def connect_sig(sig_a, sig_b, conj):
    if conj == "_AND_":
        sig = (((sig_a == 1) & (sig_b == 1)).astype(int) - ((sig_a == -1) & (sig_b == -1)).astype(int))
    elif conj == "_OR_":
        sig = (((sig_a == 1) | (sig_b == 1)).astype(int) - ((sig_a == -1) | (sig_b == -1)).astype(int))
    elif conj == "_ENTRYAND_":
        # 连接符有顺序的，注意栈是先进后出
        sig = conj_entry(sig_a, sig_b, conj)
    elif conj == "_ENTRYFILT_":
        sig = conj_entry(sig_a, sig_b, conj)
    else:
        raise NotImplementedError("Unsupported conjunction: ", conj)
    return sig


def conj_entry(sig_a, sig_b, conj):
    sig_a, sig_b = sig_a.unstack(), sig_b.unstack()
    symbols = sig_a.columns
    sig = list()
    for symbol in symbols:
        sub_sig_a, sub_sig_b = sig_a[symbol].values.astype(int), sig_b[symbol].values.astype(int)
        if conj == "_ENTRYAND_":
            sub_sig = _conj_entryand_accelerator(sub_sig_a, sub_sig_b)
        elif conj == "_ENTRYFILT_":
            sub_sig = _conj_entryfilt_accelerator(sub_sig_a, sub_sig_b)
        sig.append(sub_sig)
    sig = pd.DataFrame(
        np.vstack(sig).T, index=sig_a.index, columns=symbols
    ).stack().rename('signal')
    return sig


def _split_token(token):
    fac_name, sig_args = token.strip(']').split('[')
    if fac_name[0] == '-':
        fac_side = '-'
        fac_name = fac_name[1:]
    else:
        fac_side = ''
    sig_args = sig_args.split(',')
    return fac_side, fac_name, sig_args


def token_trans_sig(token, df_factor):
    fac_side, fac_name, sig_args = _split_token(token)
    fac = df_factor[fac_name].unstack()
    for sig_arg in sig_args:
        if re.match(_SIG_PATTERN_DICT["SBV"], sig_arg) is not None:
            fac = _trans_sbv(sig_arg, fac)
        elif re.match(_SIG_PATTERN_DICT["RANK"], sig_arg) is not None:
            fac = _trans_rank(sig_arg, fac)
        elif re.match(_SIG_PATTERN_DICT["FILL"], sig_arg) is not None:
            fac = _trans_fill(sig_arg, fac)
        elif re.match(_SIG_PATTERN_DICT['PUTOFF'], sig_arg) is not None:
            fac = _trans_put_off(sig_arg, fac)
        else:
            raise NotImplementedError
    fac = -fac if fac_side == "-" else fac
    fac = fac.stack().rename('signal')
    return fac


def _trans_sbv(sig_arg, fac):
    sig_hurdles = sig_arg.split('_')
    sig_hurdles = [float(hurdle) for hurdle in sig_hurdles]
    if len(sig_hurdles) == 1:
        fac = (fac > sig_hurdles[0]).astype(int) * 2 - 1
    elif len(sig_hurdles) == 2:
        fac = (fac > sig_hurdles[1]).astype(int) - (fac < sig_hurdles[0]).astype(int)
    elif len(sig_hurdles) == 4:
        fac_idx = fac.index
        fac = fac.apply(lambda sr: pd.Series(
            _trans_sbv_hurdle_4(
                np.array(sig_hurdles, dtype=np.double),
                sr.to_list()
            ), index=fac_idx))
    else:
        raise NotImplementedError(f"The number of hurdles is not supportable: {sig_arg}")
    return fac


def _trans_sbv_hurdle_4(hurdles, fac_sr):
    sig_status = 0
    sig_arr = []

    for fac in fac_sr:
        if sig_status == 0:
            if fac <= hurdles[0]:
                sig_status = -1
            elif fac >= hurdles[3]:
                sig_status = 1
            else:
                sig_status = 0
        elif sig_status == 1:
            if fac <= hurdles[2]:
                sig_status = 0
            else:
                sig_status = 1
        elif sig_status == -1:
            if fac >= hurdles[1]:
                sig_status = 0
            else:
                sig_status = -1

        sig_arr.append(sig_status)

    return sig_arr


# def _trans_rank(sig_arg, fac):
#     match sig_arg:
#         case "RBD":
#             fac = fac.rank(axis=1, pct=True)
#         case "RBP":
#             pass
#         case "RBT":
#             pass
#         case _:
#             raise NotImplementedError(f"The rank type is not supportable: {sig_arg}")
#     return fac


def _trans_rank(sig_arg, fac):
    if sig_arg == "RBD":
        fac = fac.rank(axis=1, pct=True)
    elif sig_arg == "RBP":
        pass
    elif sig_arg == "RBT":
        pass
    elif sig_arg.startswith("RBW"):
        window = int(sig_arg.split("_")[-1])
        fac = fac.rolling(window=window).rank(pct=True)
    else:
        raise NotImplementedError(f"The rank type is not supportable: {sig_arg}")
    return fac


def _trans_fill(sig_arg, fac):
    if sig_arg.startswith("FFILL"):
        fac = fac.ffill()
    elif sig_arg.startswith("FILLNA"):
        fill_val = float(sig_arg.split("_")[-1])
        fac = fac.fillna(fill_val)
    return fac


def _trans_put_off(sig_arg, fac):
    if sig_arg.startswith("PUTOFF"):
        put_off_step = int(sig_arg.split("_")[-1])
        fac = fac.shift(put_off_step)
    else:
        raise NotImplementedError(f"The put off type is not supportable: {sig_arg}")
    return fac
