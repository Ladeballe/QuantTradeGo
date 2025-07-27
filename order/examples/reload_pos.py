import json
import pandas as pd
import sqlalchemy


def save_json():
    with open(r'D:\research\CRYPTO_cross_sec_strategies\pos.json', 'w') as f:
        json.dump(xgbrank_strategy_0600_0800.pos, f)

    with open(r'D:\research\CRYPTO_cross_sec_strategies\pos.json', 'r') as f:
        xgbrank_strategy_0600_0800.pos = json.load(f)
        xgbrank_fac_test_strategy.status_list += [1, 2]
        xgbrank_strategy_0600_0800.status = 2


if __name__ == '__main__':
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/trader_data')
    conn = engine.connect()
    ts = 1739426751503
    df_data = pd.DataFrame(conn.execute(sqlalchemy.text(f"SELECT * FROM order_trade_update where event_time >= {ts}")))
    pos = {
        row['symbol']: {
            'side': row['side'],
            'quantity': row['cumulative_quantity'],
            'price': row['order_price'],
            'order_status': row['order_status'],
            'average_price': row['average_price']
        } for i, row in df_data[df_data['order_status']=='FILLED'].iterrows()}
