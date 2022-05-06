# io & tools
import os, psutil, tempfile
import itertools, functools, collections
import random
import re

# parallization
import multiprocessing, subprocess, time

# data processing
import pandas as pd
import numpy as np
import pickle

import sklearn.preprocessing
from sklearn.model_selection import KFold, LeaveOneOut

#import saver
#import losses 




class IsMetacentrum:
    '''
    Parse environment variables in order to get
        a) Num of cpus
        b) Num of gpus
        c) Size of memory
        d) JOBID - metacentrum
  
        enter - nice level + pool enter (N-of cpus)
    '''
    def __init__(self, set_nice_level=False):
        self.set_nice_level = set_nice_level
        
        e = {k:v for k, v in os.environ.items() if k.startswith(('PBS', 'TORQUE'))}
        for k,v in e.items():
            if re.match('^\d+$', v):
                v = int(v)
            setattr(self, k, v)
    
    @property
    def ismetacentrum(self):
        return hasattr(self, 'PBS_JOBID')

    @property
    def cpus(self):
        if self.ismetacentrum:
            return self.PBS_NCPUS
        else:
            return multiprocessing.cpu_count()     
    @property
    def gpus(self):
        if self.ismetacentrum:
            return self.PBS_NGPUS
        raise NotImplementedError()
        
    @property
    def memory(self, divide=2**30):
        if self.ismetacentrum:
            if hasattr(self, 'PBS_RESC_TOTAL_MEM'):
                return self.PBS_RESC_TOTAL_MEM / divide
            elif hasattr(self, 'TORQUE_RESC_TOTAL_MEM'):
                return self.TORQUE_RESC_TOTAL_MEM / divide
            print('Warning memory is predicted ... no PBS_RESC_TOTAL_MEM or TORQUE_RESC_TOTAL_MEM found')
        return psutil.virtual_memory().total / divide

    def __enter__(self):
        def initializer():
            import os
            if self.set_nice_level:
                os.nice(100)
        
        self.pool = multiprocessing.Pool(self.cpus, initializer=initializer)
        return self.pool.__enter__()
 
    def __exit__(self, type, value, traceback):
        return self.pool.__exit__(type, value, traceback)
           
    
    
class ArrayIsMetacentrum(IsMetacentrum):
    '''
    Extends IsMetacentrum in order ot add 
        a) isarray() -> bool
        b) array_index
        
        generator - 
    '''
    def __init__(self, *args, slice_type='interleave', seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_type = slice_type
        self.seed = seed
    
    @property
    def isarray(self):
        return hasattr(self, 'PBS_ARRAY_INDEX')
    
    @property
    def array_index(self):
        if self.isarray:
            return self._get_all_array_indexes().index(self.PBS_ARRAY_INDEX)
        return 0
    
    @functools.lru_cache()
    def _get_all_array_indexes(self):
        if self.isarray:
            v = subprocess.run(['qstat', '-J', '-t', self.PBS_ARRAY_ID], capture_output=True).stdout.decode('utf-8')
            v = re.findall(r'^\d+\[(\d+)\]', v, re.MULTILINE)
            return list(sorted(map(int, v)))
        return [1]
    
            
    def generator(self, func, iterator):
        alen = len(self._get_all_array_indexes())
        index = self.array_index

        if self.slice_type in ('offset', 'random'):
            if self.slice_type == 'random':
                cache_state = random.getstate()
                random.seed(self.seed, version=2)
                random.shuffle(iterator)
                random.setstate(cache_state)
            splits = list(reversed(np.array_split(np.array(iterator, dtype=object) , alen)))
            iterator = list(splits[index])
        elif self.slice_type in ('interleave', ):
            iterator = itertools.islice(iterator, index, None, alen)
        else:
            raise NotImplementedError('ArrayIsMetacentrum: slice_type is not implemented')

        with self as pool:
            yield from pool.imap_unordered(func, iterator)