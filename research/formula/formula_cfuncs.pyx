def _trans_sbv_hurdle_4(double[:] hurdles not None, list fac_sr not None) -> list:
    cdef int sig_status = 0
    cdef list sig_arr = []

    for fac in fac_sr:
        if sig_status == 0:
            if fac <= hurdles[0]:
                sig_status = -1
            elif fac >= hurdles[3]:
                sig_status = 1
            else:
                sig_status = 0
        elif sig_status == 1:
            if fac <= hurdles[2]:
                sig_status = 0
            else:
                sig_status = 1
        elif sig_status == -1:
            if fac >= hurdles[1]:
                sig_status = 0
            else:
                sig_status = -1

        sig_arr.append(sig_status)

    return sig_arr


def _conj_entryand_accelerator(int[:] arr_sig_a, int[:] arr_sig_b):
    cdef int sig_status = 0
    cdef list sig_arr = []

    for i in range(len(arr_sig_a)):
        if sig_status == 0:
            if arr_sig_a[i] == 1 and arr_sig_b[i] == 1:
                sig_status = 1
            elif arr_sig_a[i] == -1 and arr_sig_b[i] == -1:
                sig_status = -1
            else:
                sig_status = 0
        elif sig_status == 1:
            if arr_sig_a[i] == 1:
                sig_status = 1
            else:
                sig_status = 0
        elif sig_status == -1:
            if arr_sig_a[i] == -1:
                sig_status = -1
            else:
                sig_status = 0
        sig_arr.append(sig_status)

    return sig_arr


def _conj_entryfilt_accelerator(int[:] arr_sig_a, int[:] arr_sig_b):
    cdef int sig_status = 0
    cdef list sig_arr = []

    for i in range(len(arr_sig_a)):
        if sig_status == 0:
            if arr_sig_a[i] == 1 and arr_sig_b[i] == 1:
                sig_status = 1
            elif arr_sig_a[i] == -1 and arr_sig_b[i] == 1:
                sig_status = -1
            else:
                sig_status = 0
        if sig_status == 1:
            if arr_sig_a[i] == 1:
                sig_status = 1
            else:
                sig_status = 0
        if sig_status == -1:
            if arr_sig_a[i] == -1:
                sig_status = -1
            else:
                sig_status = 0
        sig_arr.append(sig_status)

    return sig_arr
