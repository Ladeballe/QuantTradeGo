import pandas as pd
from lightweight_charts import Chart


if __name__ == '__main__':
    df_ohlcv = pd.read_csv('D:/python_projects/test_func/examples/ohlcv.csv', index_col=1)

    chart = Chart(inner_width=1, inner_height=0.7)
    chart1 = chart.create_subchart(width=1, height=0.3, sync=True)

    ma10 = df_ohlcv['open'].rolling(10).mean()
    ma40 = df_ohlcv['open'].rolling(40).mean()
    ma160 = df_ohlcv['open'].rolling(160).mean()

    chart.set(df_ohlcv)
    ma10_line = chart.create_line('ma10', color='red')
    ma40_line = chart.create_line('ma40', color='blue')
    ma160_line = chart.create_line('ma160', color='purple')

    ma10_line.set(ma10.to_frame("ma10"))
    ma40_line.set(ma40.to_frame("ma40"))
    ma160_line.set(ma160.to_frame("ma160"))

    ma10_ma40 = (ma10.agg('log') - ma40.agg('log')).to_frame("ma_diff")

    ma_diff_line_10_40 = chart1.create_line('ma_diff', color='green')
    ma_diff_line_10_40.set(ma10_ma40)

    chart.show(block=True)
