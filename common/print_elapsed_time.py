import time
from typing import Union, Dict


_global: Dict[str, Union[float, None]] = {
    'start_time': None
}

def print_elapsed_time():
    start_time = _global['start_time']
    if start_time is None:
        start_time = time.time()
        _global['start_time'] = start_time
    elapsed = time.time() - start_time
    print(f':::::::::::::::::::: PROCESSOR ELAPSED TIME: {elapsed:.3f} s')

def start_timer():
    _global['start_time'] = time.time()