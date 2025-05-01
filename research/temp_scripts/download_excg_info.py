import numpy as np
import pandas as pd
import sqlalchemy
import binance as bnc


if __name__ == "__main__":
    client = bnc.Client()
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data?charset=utf8')

    excg_info = client.futures_exchange_info()
    df_excg_info = pd.DataFrame(excg_info['symbols'])
    df_excg_info['underlyingSubType'] = df_excg_info['underlyingSubType'].apply(
        lambda x: x[0] if len(x) > 0 else np.nan)
    df_excg_info = df_excg_info.loc[:, :"maxMoveOrderLimit"]
    df_excg_info.to_sql(name="bnc_excg_info", con=engine, if_exists="append", index=False)
