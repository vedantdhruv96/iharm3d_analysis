#####################################################

# Calculate no. of thread, parallelize a function   #

#####################################################


import multiprocessing as mp
import psutil


# Function to calculate number of threads
# Argument must be padding
def calc_threads(pad = 0.4):
    Nthreads = int(psutil.cpu_count(logical=False) * pad)
    return Nthreads

def run_parallel(function, dumps_list, Nthreads):
    pool = mp.Pool(Nthreads)
    pool.map_async(function, dumps_list).get(720000)
    pool.close()
    pool.join()





