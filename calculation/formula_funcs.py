

def _calc_max(*args):
    if isinstance(args[0], pd.Series):
        df_factor = pd.concat(*args, axis=1)
        df_factor = df_factor.max(axis=1)
        return df_factor
    else:
        max_value = max(*args)
        return max_value
