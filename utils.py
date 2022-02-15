# MIT License
# 
# Copyright (c) 2022 alxyok
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import os.path as osp
import shutil
import time
from typing import Callable, List, Union

import config

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
    
    params_path = osp.join(config.root_path, f'params.json')
    
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
            print(p)
            shutil.rmtree(p)

        os.makedirs(p, exist_ok=True)
    
    if len(paths) == 1:
        return paths[0]
    
    return paths