import arctic
import pandas as pd


def convert_freq_to_min(freq: str):
    if freq.endswith('h'):
        return int(freq[:-1]) * 60
    elif freq.endswith('m'):
        return int(freq[:-1])
    elif freq.endswith('d'):
        return int(freq[:-1]) * 24 * 60
    else:
        raise ValueError(f'Invalid frequency unit "{freq[-1]}" for param freq "{freq}"')


def convert_frame_to_stack(df_factor: pd.DataFrame, is_return_date_symbols=True, fac_name='raw_factor'):
    df_factor = df_factor.rename_axis('date', axis=0).rename_axis('symbol', axis=1)
    df_index = df_factor.copy()
    df_index.index = df_index.index.astype(str)
    df_index = df_index.stack().index.to_frame()
    df_index = df_index['date'] + '_' + df_index['symbol']
    df_factor = df_factor.stack().rename(fac_name)
    if is_return_date_symbols:
        df_factor = df_factor.reset_index()
    df_factor.index = df_index
    return df_factor


def convert_stacked_frame_to_frame(df_factor: pd.DataFrame, fac_name='raw_factor'):
    df_factor = df_factor.reset_index().pivot(index='date', columns='symbol', values=fac_name)
    return df_factor


def rank_factor(df_factor, bins=50):
    df_factor['factor_rank'] = df_factor.groupby('date')['raw_factor'].rank(pct=True).copy()
    df_factor['factor_group'] = (df_factor['factor_rank'] * bins // 1).clip(0, bins - 1)
    return df_factor


def get_fac_lib_dict(lib_names_list):
    store = arctic.Arctic('localhost')
    fac_lib_dict = dict()
    for lib_name in lib_names_list:
        lib = store.get_library(lib_name)
        fac_names = lib.list_symbols()
        fac_lib_dict.update(
            dict(
                zip(fac_names, [lib_name] * len(fac_names))
            )
        )
    return fac_lib_dict


def convert_time_str(time_str: str) -> str:
    return f"{time_str[:2]}:{time_str[2:]}"


def align_stack_frames(*dfs, how='inner'):
    df_idx = dfs[0].index
    for df in dfs[1:]:
        if how == 'inner':
            df_idx = df_idx.intersection(df.index)
        elif how == 'outer':
            df_idx = df_idx.union(df.index)
        else:
            raise ValueError(f'Unknown parameter {how} for "how"!')
    df_fac = pd.concat([df_i[['date', 'symbol']] for df_i in dfs]).drop_duplicates()
    df_fac = df_fac.reindex(df_idx)
    for i, df in enumerate(dfs):
        df_fac[f'raw_factor{i}'] = df['raw_factor']
    return df_fac


def convert_frame_to_stack(df_fac, strftime_format='%Y-%m-%d %H:%M:%S', fac_name='raw_factor'):
    df_fac = df_fac.copy()
    df_date_str = pd.MultiIndex.from_frame(
        df_fac.index.to_series().rename('date_str').dt.strftime(strftime_format).reset_index())
    df_fac.index = df_date_str
    df_fac = df_fac.stack().rename(fac_name).reset_index()
    df_fac.index = df_fac['date_str'] + '_' + df_fac['symbol']
    return df_fac[['date', 'symbol', fac_name]]
