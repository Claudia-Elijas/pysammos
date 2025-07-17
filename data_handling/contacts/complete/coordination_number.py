import numpy as np
from numba import njit

# Calculate coordination number
@njit
def count(particle_inContacts_dup, global_id_all):
  
    # Step 1: Count occurrences using np.bincount
    max_id = np.max(particle_inContacts_dup)
    count_array = np.zeros(max_id + 1, dtype=np.int64)
    count_array[:len(np.bincount(particle_inContacts_dup))] = np.bincount(particle_inContacts_dup)

    # Step 2: Map counts to global_id_all
    CN = np.zeros(len(global_id_all), dtype=np.int64)
    for i in range(len(global_id_all)):
        pid = global_id_all[i]
        if pid <= max_id:
            CN[i] = count_array[pid]  # 0 if not found in contacts

    # Step 3: Filter out rattlers (particles with CN == 0)
    CN_no_rattlers = CN[CN > 1]

    return CN, CN_no_rattlers