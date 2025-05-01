import numpy as np
import pandas as pd
import numba as nb


@nb.jit(nopython=True)
def func(arr):
    fac = arr[:, 0, :]
    fac_wgt = arr[:, 1, :]
    fac_wgt.sort(axis=1)
    fac_quantile = np.quantile(fac_wgt, 0.25, axis=0)
    fac_wgt = fac_wgt > fac_quantile
    fac_res = (fac * fac_wgt).sum(axis=0)
    return fac_res


if __name__ == '__main__':
    arr_fac = np.array([
        [[200, 10, 3],
         [10000, 20000, 5000]],
        [[204, 10.5, 3.1],
         [16000, 30000, 15000]],
        [[205, 10.3, 3.1],
         [13000, 17000, 12000]],
        [[206, 10.2, 3.1],
         [14000, 18000, 13000]]
    ])
    func(arr_fac)

