import json
import importlib
import logging
from multiprocessing import Pool

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s',
)


def parse_tasks():

    df_tasks = pd.read_excel(r'D:\python_projects\data_downloader\tasks.xlsx')
    df_tasks = df_tasks[df_tasks['update']==1]
    df_tasks['params'] = df_tasks['params'].apply(lambda x: json.loads(x))
    return df_tasks


if __name__ == '__main__':
    df_tasks = parse_tasks()
    for i, row in df_tasks.iterrows():
        print(i, row)
        module = importlib.import_module('data_downloader.' + row['name'])
        params = row['params']
        downloader_function = module.main
        res = downloader_function(**params)
    print('done.')
