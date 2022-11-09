from src.utils import memory
import psutil
from sys import platform
import numpy as np

if platform != 'win32':
    import resource


@memory(percentage=1.1)
def test_memory_usage():
    print('In main: \n')
    mem = psutil.virtual_memory()
    start_gb = mem.used / (1024 ** 3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    process_gb = soft / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)

    ml = []
    for j in range(26):
        arr = np.random.rand(10000, 5000).astype(np.float64)  # 381.5 MB
        ml.append(arr)
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        remaining = mem.available * 100 / mem.total
        proc_remain = (1 - ((used_gb - start_gb) / process_gb)) * 100
        print(f'Iteration {j}: Total remaining RAM: {remaining:.1f} %  ({used_gb:.2f}/{total_gb:.2f} GB used)   ' 
              f'Process remaining RAM: {proc_remain:.1f} % ({used_gb-start_gb:.2f}/{process_gb:.2f} process GB used)')