import numpy as np
import pandas as pd
import sqlalchemy

from test_func import data as tfd


def main():
    """
    universe_ma7d_AmtMinGt2500_LiquidityAmtTradeNumRankTop200:
    - 滚动7日
      - 日平均最小成交额>=2500
      - 使用成交额中位数与交易数中位数衡量的流动性指标 排名前200
    - 首次添加时间：20250215
    """
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    conn = engine.connect()
    df_amt = tfd.get_factor('amt', 'fac_5m.fac_basic', '2024-01-01', '2025-02-28')
    df_trade_num = tfd.get_factor('trade_num', 'fac_5m.fac_basic', '2024-01-01', '2025-02-28')
    df_symbols = pd.read_sql("SELECT * FROM bnc_symbols", conn)
    df_market_cap = pd.read_sql(
        f"SELECT * FROM gecko_market_cap_1d WHERE ts < {int(pd.Timestamp('2024-02-14 00:00:00').timestamp() * 1e3)}",
        conn)

    # 计算aggregate指标
    df_amt_agg = df_amt.groupby(df_amt.index.date).agg(['median', 'min'])
    df_trade_num_agg = df_trade_num.groupby(df_trade_num.index.date).agg(['mean', 'min']) / 5
    df_market_cap_agg = df_market_cap.pivot(index='ts', columns='symbol', values='market_cap')
    df_market_cap_agg.index = pd.to_datetime(df_market_cap_agg.index * 1e6)

    df_data = pd.concat([
        df_amt_agg.rolling(7).mean().stack(level=0).rename({'median': 'amt_median', 'min': 'amt_min'}, axis=1),
        df_trade_num_agg.rolling(7).mean().stack(level=0).rename({'mean': 'trade_num_mean', 'min': 'trade_num_min'}, axis=1),
        df_market_cap_agg.rolling(7).mean().stack().to_frame('market_cap')
    ], axis=1).dropna(subset=['amt_min'])
    df_data = df_data.rename_axis(['date', 'symbol'], axis=0).reset_index()
    df_data_rank = pd.concat([
        df_data.groupby('date')[['amt_median', 'amt_min', 'trade_num_mean', 'trade_num_min', 'market_cap']].rank(),
        df_data[['date', 'symbol']]
    ], axis=1).set_index(['date', 'symbol'])
    df_data = df_data[df_data['symbol'].str[-4:] == 'USDT']
    df_data = df_data.set_index(['date', 'symbol'])
    idx_universe = df_data[(df_data['amt_min'] > 2500) & \
                          ((df_data_rank['amt_median'].rank(pct=True) + \
                            df_data['trade_num_mean'].rank(pct=True)).groupby('date').rank(ascending=False) <= 200)]\
        .index
    df_universe = pd.Series(1, index=idx_universe).unstack()
    df_universe.index = pd.to_datetime(df_universe.index)
    df_universe = df_universe.stack()
    tfd.save_factor(df_universe, "universe_ma7d_AmtMinGt2500_LiquidityAmtTradeNumRankTop200", "universe.universe_1d")


if __name__ == '__main__':
    main()
