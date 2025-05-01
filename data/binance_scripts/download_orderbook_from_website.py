import aiohttp
import asyncio
import aiomysql
import pandas as pd
import zipfile
import io

from test_func import data


async def task_generator(queue_task):
    sr_symbols = pd.Series(data.load_symbols_from_exchange_info())
    sr_symbols = sr_symbols[sr_symbols.str[-4:] == 'USDT']
    for date in pd.date_range('2024-12-01', '2025-01-09', freq='1D')[::-1]:
        for _, symbol in sr_symbols.items():
            await queue_task.put((symbol, date))
            print(f"{pd.Timestamp.now()}, task_generator, {symbol}, {date}, {queue_task.qsize()}")


async def request_worker(queue_task, queue_content, stop_event):
    session = aiohttp.ClientSession()
    while not stop_event.is_set():
        symbol, date = await queue_task.get()
        res = await session.get(f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date.strftime('%Y-%m-%d')}.zip")
        print(f"{pd.Timestamp.now()}, request_worker, {symbol}, {date}, {res.status}")
        if res.status != 200:
            continue
        content = await res.content.read()
        await queue_content.put((symbol, date, content))


async def mysql_worker(queue_content, stop_event, loop):
    mysql_conn = await aiomysql.connect(
            host='127.0.0.1', port=3306, user='root', password='444666',
            db='market_data', loop=loop, autocommit=True
    )
    while not stop_event.is_set():
        cursor0 = await mysql_conn.cursor()

        symbol, date, content = await queue_content.get()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            with zip_ref.open(zip_ref.namelist()[0]) as file_bookDepth:
                df_bookDepth = pd.read_csv(file_bookDepth)

        df_bookDepth.columns = ['a', 'p', 'q', 'f', 'l', 'ts', 'm']
        df_bookDepth['symbol'] = symbol
        df_bookDepth['t'] = pd.to_datetime(df_bookDepth['ts'] * 1e6)
        df_bookDepth = df_bookDepth.astype({'p': float, 'q': float})
        df_bookDepth['v'] = df_bookDepth['p'] * df_bookDepth['q']
        df_bookDepth['n'] = df_bookDepth['l'] - df_bookDepth['f']

        data = df_bookDepth.values
        data = [tuple(d) for d in data]

        table_name = 'bnc_aggtrades'
        columns = ', '.join(df_aggtrades.columns)
        placeholders = ', '.join(['%s'] * df_aggtrades.shape[1])
        replace_query = f"REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"
        print(f"{pd.Timestamp.now()}, mysql_worker, {symbol}, {date}, executing...")
        await cursor0.executemany(replace_query, data)
        await cursor0.close()
        print(f"{pd.Timestamp.now()}, mysql_worker, {symbol}, {date}, executed.")


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
