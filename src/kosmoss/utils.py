import json
import os
import os.path as osp
import shutil
import time
from typing import Callable, List, Union
import yaml

import kosmoss as km

def prime_factors(n: int) -> List[int]:
    i = 2
    factors = []
    
    while i ** 2 <= n:
        
        if n % i:
            i += 1
            
        else:
            n //= i
            factors.append(i)
            
    if n > 1:
        factors.append(n)
        
    return factors



def save_params(
    step: int, 
    params: dict, 
    type_: Union['flattened', 'features']) -> None:
    
    params_path = osp.join(km.ROOT_PATH, 'params.json')
    
    data = {}
    if osp.isfile(params_path):
        with open(params_path, 'r') as file:
            data = json.load(file)
    
    if str(step) in data:
        data[str(step)][type_] = params
        
    else:
        data.update({step: {type_: params}})
    
    with open(params_path, 'w') as file:
        json.dump(data, file)
        
        

def timing(fn: Callable) -> Callable:
    
    def wrap(*args, **kwargs):
        
        s = time.perf_counter()
        result = fn(*args, **kwargs)
        e = time.perf_counter()
        
        print("%.2f ms" % ((e - s) * 1000))
        return result
    
    return wrap



def purgedirs(paths: Union[str, list]) -> Union[str, list]:
    
    if isinstance(paths, str):
        paths = [paths]
    
    for p in paths: 
        if osp.isdir(p):
            shutil.rmtree(p)

        os.makedirs(p, exist_ok=True)
    
    if len(paths) == 1:
        return paths[0]
    
    return paths



def makedirs(paths: Union[str, list]) -> Union[str, list]:
    
    if isinstance(paths, str):
        paths = [paths]

    for p in paths:
        os.makedirs(p, exist_ok=True)
    
    if len(paths) == 1:
        return paths[0]
    
    return paths



def load_attr(path: str, loader_type: Union['json', 'yaml']) -> dict:
    
    if loader_type == 'json':
        loader = json.load
        
    elif loader_type == 'yaml':
        loader = yaml.safe_load
        
    else:
        raise ValueError("expected 'json' or 'yaml'")
        
    with open(path, "r") as stream:
        attr = loader(stream)
    
    return attr