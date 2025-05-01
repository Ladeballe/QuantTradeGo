import pandas as pd
from test_func.formula import *


if __name__ == "__main__":
    # formula = "bolling[FFILL,RBW_3]"
    # # df_factor = pd.DataFrame(np.random.normal(0, 1, (1000000, 4)), columns=list("ABCD")).stack().to_frame('bolling')
    # df_factor = pd.DataFrame([
    #     [3, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [4, 2, 3, 1],
    #     [1, 2, 3, 4],
    #     [2, 1, 3, 4],
    #     [2, 1, 4, 3],
    # ], columns=list("ABCD")).stack().to_frame('bolling')
    # df_sig = expr_trans_sig(formula, df_factor)

    df_factor = pd.DataFrame([
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [-1, 0],
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 0],
    ],
        columns=['a', 'b'],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=12).tolist(), ["ETHUSDT"]], names=['date', 'symbol'])
    )
    formula = "a[FFILL]_ENTRYAND_b[FFILL]"
    df_sig = expr_trans_sig(formula, df_factor).rename("signal")
    df_analysis = pd.concat([df_factor, df_sig], axis=1)
    print('done')