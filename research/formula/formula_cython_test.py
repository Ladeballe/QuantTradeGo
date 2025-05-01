import numpy as np
from formula_cfuncs import _trans_sbv_hurdle_4

# 示例输入数据
hurdles = np.array([-3, 0, 0, 3], dtype=np.double)  # 将字符串列表转换为浮点数列表
fac_sr = [-3.1, -2, -1, 0, 0, 3, 2, 3, -1, 0, 2, 1]

# 调用 C 扩展函数
result = _trans_sbv_hurdle_4(hurdles, fac_sr)

# 输出结果
print("Input hurdles:", hurdles)
print("Input fac_sr:", fac_sr)
print("Result:", result)