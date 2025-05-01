import pandas as pd
import sqlalchemy


if __name__ == '__main__':
    engine = sqlalchemy.create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    conn = engine.connect()
    df_symbols = pd.DataFrame(conn.execute(
        sqlalchemy.text("""
        SELECT 
            *
        FROM
            bnc_symbols
        WHERE
            geckoName IS NULL
        """)
    ))
    conn.close()
    conn = engine.connect()
    for i, row in df_symbols.iterrows():
        conn.begin()
        symbol = row['symbol']
        print(row)
        gecko_name = input(f'{symbol} geckoName:')
        gecko_name = None if gecko_name == 'none' else gecko_name
        conn.execute(
            sqlalchemy.text(
                """
                    UPDATE bnc_symbols 
                    SET geckoName = :gecko_name 
                    WHERE symbol = :symbol
                """
            ),
            {'gecko_name': gecko_name, 'symbol': symbol}
        )
        conn.commit()
