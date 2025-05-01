def main():
    strategy = 'single_factor_test'  # 预先决定
    resample_freq = '15min'  # 预先决定
    calc_rtn_method = 'ByRtn'  # 预先决定
    version = '0.1.0'  # 预先决定
    qty_param = ''  # 可更改
    sector = 'all'  # 可更改
    output_path = f'D:\\research\\single_factor_test\\technical_indicator\\heikin_ashi_smoothed_sig\\{version}'  # 预先决定
    fwd_rtn_name = 'fwd_log_rtn_15min'  # 预先决定
    fwd_rtn_lib = ''  # 预先决定
    cost_ratio = 0.0003  # 预先决定
    initial_capital = 1000  # 预先决定
    begin_date = '2019-01-01'  # 预先决定
    end_date = '2022-12-31'  # 预先决定
    between_time = ['09:00:00', '22:59:00']

    skip_finished = True
    mode = 'debug'
    process_num = 8

    if mode == 'multiprocess':
        process_pool = mp.Pool(processes=process_num)

    products = [
        'A', 'AP', 'B', 'C', 'CF', 'CJ', 'CS', 'JD', 'LH', 'M', 'OI', 'P', 'PK', 'RM', 'SP', 'SR', 'Y',
        'FG', 'HC', 'I', 'J', 'JM', 'RB', 'SA', 'SF', 'SM', 'ZC',
        'AG', 'AL', 'AU', 'CU', 'NI', 'PB', 'SN', 'SS', 'ZN',
        'IC', 'IF', 'IH', 'IM',
        'BU', 'EB', 'EG', 'FU', 'L', 'LU', 'MA', 'NR', 'PF', 'PG', 'PP', 'RU', 'SC', 'TA', 'UR', 'V',
        # 'BC', 'CY', 'TC',
        # 'T', 'TF', 'TS',
    ]
    mongo_client = pymongo.MongoClient("localhost:27017")
    coll = mongo_client["FS_test"]["test_results"]

    base_formula = '{side}{fac_name0}[FFILL,{sig_trans0}]_{connector}_{fac_name1}[FFILL,{sig_trans1}]'
    fac_names = {'fac_name0': 'bolling_{bolling_window}',
                 'fac_name1': 'net_oi_ratio_{net_oi_ratio_window}_{rank_num}_{net_oi_ratio_ma_window}'}
    fac_libs = {'fac_name0': 'luyiyang.factor_UNI_15m',
                'fac_name1': 'luyiyang.factor_UNI_1d'}

    side = ['']
    sig_trans0 = ['-3_0_0_3', '-2_0_0_2']
    sig_trans1 = ['RBD,0.1_0.5_0.5_0.1', 'RBD,0.1_0.9']
    bolling_window = [100, 200, 400, 800, 1200]
    net_oi_ratio_window = [10, 20, 40, 60]
    rank_num = [5, 10, 20]
    net_oi_ma_window = [20]
    connector = ['AND', 'NOP']
    fac_name_params_dict = {
        'bolling_window': bolling_window,
        'net_oi_ratio_window': net_oi_ratio_window,
        'rank_num': rank_num,
        'net_oi_ratio_ma_window': net_oi_ma_window
    }
    formula_params_dict = {
        'connector': connector,
        'side': side,
        'sig_trans0': sig_trans0,
        'sig_trans1': sig_trans1,
    }

    for format_fac_names, fac_names_params_dict in iter_fac_names(fac_names, fac_name_params_dict):
        df_factor = factor_load(products, fac_names, fac_libs, begin_date, end_date, between_time)
        for formula, fac_names, res_params_dict in iter_formula(base_formula, format_fac_names, formula_params_dict):
            params = {
                'names': fac_names_params_dict,
                'params': res_params_dict
            }
            test_params = {
                'params': params, 'skip_finished': skip_finished, 'products': products,
                'fwd_rtn_name': fwd_rtn_name, 'fwd_rtn_lib': fwd_rtn_lib,
                'cost_ratio': cost_ratio, 'calc_rtn_method': calc_rtn_method, 'qty_param': qty_param,
                'initial_capital': initial_capital, 'resample_freq': resample_freq,
                'fac_names': fac_names, 'fac_libs': fac_libs
            }
            if mode == 'debug':
                factor_test(strategy, version, formula, sector, begin_date, end_date,
                            df_factor, output_path, coll, test_params)
            elif mode == 'multiprocess':
                process_pool.apply_async(
                    factor_test, args=(
                        strategy, version, formula, sector, begin_date, end_date,
                        df_factor, output_path, coll, test_params
                    )
                )
        if mode == 'multiprocess':
            process_pool.close()
            process_pool.join()


if __name__ == '__main__':
    main()