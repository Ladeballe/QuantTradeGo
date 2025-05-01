import os
import zipfile

import numpy as np
import pandas as pd
import requests as rq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sqlalchemy
from sqlalchemy.dialects.mysql import insert
from sqlalchemy import create_engine, text
import pymongo


def insert_on_conflict_update(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = (
        insert(table.table)
        .values(data)
    )
    stmt = stmt.on_duplicate_key_update(ts=stmt.inserted.ts, symbol=stmt.inserted.symbol, level=stmt.inserted.level)
    result = conn.execute(stmt)
    return result.rowcount


if __name__ == '__main__':
    db = pymongo.MongoClient('mongodb://localhost:27017')['crypto_data']
    symbols = db['bnc-future-universe'].find(
        {'name': 'universe_bnc_perp_future_usdt_basic_onboard_over_1yr_or_amt_gt_200k'},
    ).sort('created_time', -1)[0]['symbols']
    mysql_conn = create_engine('mysql+pymysql://root:444666@localhost:3306/market_data')
    driver = webdriver.Chrome()
    driver.get("https://data.binance.vision/?prefix=data/futures/um/daily/bookDepth/")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//tbody//tr//td//a"))
    )
    table = driver.find_elements(By.XPATH, '//tbody//tr//td//a')[1:]
    symbol_href_list = [ele.get_attribute('href') for ele in table]
    for symbol_href in symbol_href_list:
        symbol = symbol_href.split('/')[-2]
        if symbol not in symbols:
            continue
        driver.get(symbol_href)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//tbody//tr//td//a"))
        )
        orderbook_eles = driver.find_elements(By.XPATH, '//tbody//tr//td//a')[2::2]
        orderbook_href_list = [ele.get_attribute('href') for ele in orderbook_eles]
        for orderbook_href in orderbook_href_list:
            print(pd.Timestamp.now(), orderbook_href)
            date = "-".join(orderbook_href.split('-')[-3:]).split('.')[0]
            if int(date.split('-')[0]) < 2024:
                break
            try:
                with mysql_conn.connect() as connection:
                    connection.begin()
                    sql_code0 = text(
                        "INSERT INTO bnc_depth5_30s_record (symbol, `date`) "
                        "VALUES (:symbol, :date)"
                    )
                    params = {
                        'symbol': symbol,
                        'date': date
                    }
                    connection.execute(sql_code0, params)
                    connection.commit()

                    res = rq.get(orderbook_href)
                    with open(f'temp_data.zip', 'wb') as f:
                        f.write(res.content)

                    with zipfile.ZipFile('temp_data.zip', 'r') as zip_file:
                        zip_file.extractall('temp_data')

                    df1 = pd.read_csv(f'temp_data/{os.listdir('temp_data')[0]}')
                    df1.columns = ['ts', 'level', 'v', 'q']
                    df1['symbol'] = symbol
                    df1['p'] = df1['q'] / df1['v']
                    df1 = df1[['ts', 'symbol', 'level', 'p', 'v', 'q']]
                    df1['ts'] = (pd.to_datetime(df1['ts']).values.astype(np.int64) / 1e6).astype(np.int64)
                    df1.to_sql(
                        name="bnc_depth5_30s", con=mysql_conn,
                        if_exists="append", index=False, method=insert_on_conflict_update
                    )

                    os.remove(f'temp_data/{os.listdir('temp_data')[0]}')
            except sqlalchemy.exc.IntegrityError as e:
                if e.args[0].split('(')[-1].split(',')[0]:
                    print(e)
                    continue
    print('done')
