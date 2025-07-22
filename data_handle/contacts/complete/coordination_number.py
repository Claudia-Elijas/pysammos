import numpy as np
from numba import njit, int64
from numba.types import Tuple  

# Calculate coordination number
@njit(Tuple((int64[:], int64[:]))(int64[:],int64[:]))
def count(particle_inContacts_dup, global_id_all):
    max_id = int(np.max(particle_inContacts_dup))
    count_array = np.zeros(max_id + 1, dtype=np.int64)
    bincounts = np.bincount(particle_inContacts_dup)
    count_array[:len(bincounts)] = bincounts

    CN = np.zeros(len(global_id_all), dtype=np.int64)
    for i in range(len(global_id_all)):
        pid = global_id_all[i]
        if pid <= max_id:
            CN[i] = count_array[pid]

    CN_no_rattlers = CN[CN > 1]
    return CN, CN_no_rattlers
