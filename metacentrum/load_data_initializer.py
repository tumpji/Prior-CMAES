#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import tempfile
import re
import numpy as np
import collections
import pickle
import sklearn.preprocessing

import os
import re
import multiprocessing
import psutil
import time

import saver
import losses 
import itertools

import os
import re
import multiprocessing
import psutil
import time
import itertools
import functools
import random
import numpy as np
import subprocess

from sklearn.model_selection import KFold, LeaveOneOut

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# In[2]:


class IsMetacentrum:
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
    


# In[3]:


def load_auto_mpg():
    csv = pd.read_csv('data/auto_mpg/dataset.csv')
    
    labels = np.array(np.array(csv['mpg']))
    features = np.array(csv.drop(columns=['mpg']))
    return features, labels

def load_boston_housing():
    csv = pd.read_csv('data/boston_housing/dataset.csv')

    labels = csv['CRIM']
    features = csv.drop(columns=['CRIM'])
    return np.array(features), np.array(labels)


def iterate_gov(dry_run=False):
    def load_gov(name):
        data = pd.read_csv('data/nist.gov/' + name, sep=' ', header=None)
        labels = data[0]
        features = data.drop(columns=[0])
        if len(features.columns) == 1:
            return np.array(features).reshape(-1,1), np.array(labels)
        else:
            return np.array(features), np.array(labels)
    
    for name in os.listdir('data/nist.gov'):
        if dry_run:
            yield name, (None, None)
        else:
            yield name, load_gov(name)

def load_california_housing():
    with open('data/california_housing.pickle', 'rb') as f:
        ds = pickle.load(f)
    return ds['features'], ds['labels']
    

def load_proben_building_WBE():
    csv = pd.read_csv('data/proben/dataset.csv')

    labels = csv['WBE']
    features = csv.drop(columns=['WBE', 'WBCW', 'WBHW'])
    return np.array(features), np.array(labels)
        
def load_proben_building_WBCW():
    csv = pd.read_csv('data/proben/dataset.csv')

    labels = csv['WBCW']
    features = csv.drop(columns=['WBE', 'WBCW', 'WBHW'])
    return np.array(features), np.array(labels)

def load_proben_building_WBHW():
    csv = pd.read_csv('data/proben/dataset.csv')

    labels = csv['WBHW']
    features = csv.drop(columns=['WBE', 'WBCW', 'WBHW'])
    return np.array(features), np.array(labels)

def load_concrete():
    pass
    csv = pd.read_csv('data/concrete_data.csv')
    labels = csv['concrete_compressive_strength']
    features = csv.drop(columns=['concrete_compressive_strength'])
    
    return np.array(features), np.array(labels)


# In[4]:


def _all_dataset_iterator(dry_run=False):
    yield from iterate_gov(dry_run=dry_run)

    if dry_run:
        yield ('boston_housing', (None, None))
        yield ('california_housing', (None, None))
        yield ('auto_mpg', (None, None))
        yield ('proben_building_WBHW', (None, None))
        yield ('proben_building_WBCW', (None, None))
        yield ('proben_building_WBE', (None, None))
    else:
        yield ('boston_housing', load_boston_housing())
        yield ('california_housing', load_california_housing())
        yield ('auto_mpg', load_auto_mpg())
        yield ('proben_building_WBHW', load_proben_building_WBHW())
        yield ('proben_building_WBCW', load_proben_building_WBCW())
        yield ('proben_building_WBE', load_proben_building_WBE())
        
def all_dataset_iterator(dry_run=False):
    yield from map(lambda x: (x[0], *x[1]), 
        zip(itertools.count(start=1), _all_dataset_iterator(dry_run)))
    
def all_normalized_dataset_iterator(dry_run=False):
    if dry_run:
        yield from all_dataset_iterator(dry_run=dry_run)
    else:
        for (dsid, name, (features, labels)) in all_dataset_iterator():
            scaler_features = sklearn.preprocessing.StandardScaler()
            scaler_labels = sklearn.preprocessing.StandardScaler()

            features = scaler_features.fit_transform(features)
            labels = scaler_labels.fit_transform(labels.reshape(-1,1))

            yield dsid, name, (features, labels)


# In[5]:


def all_cv_normalized_dataset_iterator(dry_run=False):
    THREASHOLD_ONE_LEAVE_OUT = 10
    N_SPLITS = 10
    
    if dry_run:
        for (dsid, name, (features, labels)) in all_normalized_dataset_iterator():
            for fold in range(1, max(THREASHOLD_ONE_LEAVE_OUT, N_SPLITS)+1):
                yield (dsid, name, fold, (train_features, train_labels, test_features, test_labels))
    else:
        for (dsid, name, (features, labels)) in all_normalized_dataset_iterator():
            if len(labels) < THREASHOLD_ONE_LEAVE_OUT:
                kf = LeaveOneOut()
            else:
                kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

            for fold, (train_index, test_index) in enumerate(kf.split(features), start=1):
                train_features = features[train_index, ...]
                test_features = features[test_index, ...]
                train_labels = labels[train_index]
                test_labels = labels[test_index]
                yield (dsid, name, fold, (train_features, train_labels, test_features, test_labels))
        


# In[6]:


# prepare experiment


# In[ ]:





# In[7]:


class TrimedAbsoluteLoss(keras.losses.Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        num_of_res = tf.cast(tf.shape(y_true)[0], tf.float32)
        num_of_res = num_of_res * (9. / 10.)
        num_of_res = tf.cast(num_of_res, tf.int32)
        
        residuals = tf.abs(y_true - y_pred)
        residuals = tf.sort(residuals)
        residuals = residuals[:num_of_res]
        
        loss = tf.reduce_mean(residuals, axis=-1)
        if sample_weight is not None:
            raise NotImplementedError('sample weight is not supported')
        return loss
    
class TrimedSquaredLoss(keras.losses.Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        num_of_res = tf.cast(tf.shape(y_true)[0], tf.float32)
        num_of_res = num_of_res * (9. / 10.)
        num_of_res = tf.cast(num_of_res, tf.int32)
        
        residuals = tf.square(y_true - y_pred)
        residuals = tf.sort(residuals)
        residuals = residuals[:num_of_res]
        
        loss = tf.reduce_mean(residuals, axis=-1)
        if sample_weight is not None:
            raise NotImplementedError('sample weight is not supported')
        return loss


# In[8]:


class RootModel(tf.keras.models.Sequential):
    def fit(self, x, y, **kwargs):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=1e-10)
        ]
        super().fit(x, y, batch_size=100000, epochs=10000, callbacks=callbacks, verbose=False)

class BasicModel(RootModel):
    def __init__(self, model_size, loss):
        super().__init__([
            keras.layers.Dense(model_size, activation='selu'),
            keras.layers.Dense(model_size//2, activation='selu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())
        
class DropoutModel(RootModel):
    def __init__(self, model_size, loss):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(model_size//2, activation='selu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='linear')
        ])

        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss()) 
        
class AlphaDropoutModel(RootModel):
    def __init__(self, model_size, loss):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu'),
                keras.layers.AlphaDropout(0.5),
                keras.layers.Dense(model_size//2, activation='selu'),
                keras.layers.AlphaDropout(0.5),
                keras.layers.Dense(1, activation='linear')
        ])

        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())


# In[9]:

class AdvancedDropoutModel(RootModel):
    def __init__(self, model_size, loss, alpha):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu'),
                keras.layers.Dropout(alpha),
                keras.layers.Dense(model_size//2, activation='selu'),
                keras.layers.Dropout(alpha),
                keras.layers.Dense(1, activation='linear')
        ])
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())

class AdvancedNormalModel(RootModel):
    def __init__(self, model_size, loss, alpha):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu'),
                keras.layers.Dense(model_size//2, activation='selu'),
                keras.layers.Dense(1, activation='linear')
        ])
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())

class AdvancedAlphaDropoutModel(RootModel):
    def __init__(self, model_size, loss, alpha):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu'),
                keras.layers.AlphaDropout(alpha),
                keras.layers.Dense(model_size//2, activation='selu'),
                keras.layers.AlphaDropout(alpha),
                keras.layers.Dense(1, activation='linear')
        ])
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())

        
class L2NormalizedModel(RootModel):
    def __init__(self, model_size, loss, alpha):
        super().__init__([
                keras.layers.Dense(model_size, activation='selu', kernel_regularizer=tf.keras.regularizers.L2(alpha)),
                keras.layers.Dense(model_size//2, activation='selu', kernel_regularizer=tf.keras.regularizers.L2(alpha)),
                keras.layers.Dense(1, activation='linear')
        ])
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())
        
class L2ActivityNormalizedModel(RootModel):
    def __init__(self, model_size, loss, alpha):
        kernel_regularizer = tf.keras.regularizers.L2(alpha)
        super().__init__([
                keras.layers.Dense(model_size, activation='selu', activity_regularizer=tf.keras.regularizers.L2(alpha)),
                keras.layers.Dense(model_size//2, activation='selu', activity_regularizer=tf.keras.regularizers.L2(alpha)),
                keras.layers.Dense(1, activation='linear')
        ])
        self.compile(optimizer=keras.optimizers.Nadam(), loss=loss())


# In[10]:


# ----------
# 2        4-2-1
# 3        8-4-1
# 4       16-8-1
# 5       32
# 6       64
# 7      128
# 8      256
# 9      512
# 10    1024
# 11    2048
# 12    4096
# 13    8192
# 14   16384
# 15   32768
# 16   65536

MAXSIZE = 12
'''  old stuff
selected_losses = {
    'TrimedAbsoluteLoss': TrimedAbsoluteLoss,
    'TrimedSquaredLoss': TrimedSquaredLoss,
    'MeanSquaredError': keras.losses.MeanSquaredError,
    'MeanAbsoluteError': keras.losses.MeanAbsoluteError,
    'MeanSquaredLogarithmicError': keras.losses.MeanSquaredLogarithmicError,
    'Huber': keras.losses.Huber,
}

model_sizes = list(map(lambda x: 2**x, range(2, MAXSIZE+1)))

models = {
    'BasicModel': BasicModel,
    #'DropoutModel' : DropoutModel,
    #'AlphaDropoutModel': AlphaDropoutModel
}
'''

selected_losses = {
    #'TrimedAbsoluteLoss': TrimedAbsoluteLoss,
    #'TrimedSquaredLoss': TrimedSquaredLoss,
    'MeanSquaredError': keras.losses.MeanSquaredError,
    'MeanAbsoluteError': keras.losses.MeanAbsoluteError,
    'MeanSquaredLogarithmicError': keras.losses.MeanSquaredLogarithmicError,
    'Huber': keras.losses.Huber,
}

model_sizes = list(map(lambda x: 2**x, range(2, MAXSIZE+1)))

models = {
    'AdvancedAlphaDropoutModel': AdvancedAlphaDropoutModel,
    'AdvancedDropoutModel': AdvancedDropoutModel,
    'AdvancedNormalModel' : AdvancedNormalModel,
    'L2NormalizedModel' : L2NormalizedModel,
    'L2ActivityNormalizedModel': L2ActivityNormalizedModel
}


# In[11]:


class ModelInContext:
    def __init__(self, dsid, fold, model_name, model_cls, model_size, loss_name, loss_cls, 
                 train_features, train_labels, test_features, test_labels):
        m = locals()
        del m['self']
        self.__dict__.update(m)
    
    def init(self):
        saver.Initializer(
            model=self.model_name,
            hyperparameters = {'size': self.model_size, 'loss':  self.loss_name},
            function_id = self.dsid,
            run = self.fold,
            losses = [losses.LossL1(), losses.LossL2(), losses.LossL1Drop10P(), losses.LossL2Drop10P()]
        )
        
    def compute(self):
        s = saver.Saver(
                model=self.model_name,
                hyperparameters = {'size': self.model_size, 'loss': self.loss_name},
                function_id = self.dsid,
                run = self.fold,
                losses = [losses.LossL1(), losses.LossL2(), losses.LossL1Drop10P(), losses.LossL2Drop10P()]
            )
            
        if s.completed:
            return
        else:
            model = self.model_cls(self.model_size, self.loss_cls)
            model.fit(self.train_features, self.train_labels)
            test_predict = np.array(model(self.test_features))

            s.write_results_to_dataset(
                prediction = test_predict.reshape(-1,), 
                target = self.test_labels.reshape(-1,))
            s.finalize()
            
class AdvancedModelInContext:
    def __init__(self, hyperparameter, dsid, fold, model_name, model_cls, model_size, loss_name, loss_cls, 
                 train_features, train_labels, test_features, test_labels):
        m = locals()
        del m['self']
        self.__dict__.update(m)
        
    def init(self):
        saver.Initializer(
            model=self.model_name,
            hyperparameters = {'size': self.model_size, 'loss':  self.loss_name, 'hyperparameter':self.hyperparameter},
            function_id = self.dsid,
            run = self.fold,
            losses = [losses.LossL1(), losses.LossL2(), losses.LossL1Drop10P(), losses.LossL2Drop10P()]
        )
    
    def compute(self):
        s = saver.Saver(
                model=self.model_name,
                hyperparameters = {'size': self.model_size, 'loss': self.loss_name, 'hyperparameter':self.hyperparameter},
                function_id = self.dsid,
                run = self.fold,
                losses = [losses.LossL1(), losses.LossL2(), losses.LossL1Drop10P(), losses.LossL2Drop10P()]
            )
            
        if s.completed:
            return
        else:
            model = self.model_cls(self.model_size, self.loss_cls, self.hyperparameter)
            model.fit(self.train_features, self.train_labels)
            test_predict = np.array(model(self.test_features))

            s.write_results_to_dataset(
                prediction = test_predict.reshape(-1,), 
                target = self.test_labels.reshape(-1,))
            s.finalize()
''' old
task_generator = [
    ModelInContext(dsid, fold, model_name, model_cls, model_size, loss_name, loss_cls, 
        train_features, train_labels, test_features, test_labels)
    
    for (dsid, name, fold, (train_features, train_labels, test_features, test_labels)) \
        in all_cv_normalized_dataset_iterator() \
    for (model_name, model_cls) in models.items() \
    for model_size in model_sizes \
    for (loss_name, loss_cls) in selected_losses.items() 
]
'''

def decision_about_hyperparameter(modelname):
    if modelname == 'AdvancedAlphaDropoutModel':
        return [0.025, 0.05, 0.1, 0.2, 0.4]
    elif modelname == 'AdvancedDropoutModel':
        return [0.025, 0.05, 0.1, 0.2, 0.4]
    elif modelname == 'L2NormalizedModel':
        return [0.0, 0.001, 0.01, 0.1, 1]
    elif modelname == 'L2ActivityNormalizedModel':
        return [0.0, 0.001, 0.01, 0.1, 1]
    elif modelname == 'AdvancedNormalModel':
        return [0.0]
    else:
        raise ''


task_generator = [
    AdvancedModelInContext(hyperparameter, dsid, fold, model_name, model_cls, model_size, loss_name, loss_cls, 
        train_features, train_labels, test_features, test_labels)
    
    for (dsid, name, fold, (train_features, train_labels, test_features, test_labels)) \
        in all_cv_normalized_dataset_iterator() \
    for (model_name, model_cls) in models.items() \
    for hyperparameter in decision_about_hyperparameter(model_name) \
    for model_size in model_sizes \
    for (loss_name, loss_cls) in selected_losses.items() 
]

print(len(task_generator))


# In[12]:


print(model_sizes)


# In[13]:


progress = True
INIT = True
clasic = False

import warnings
warnings.simplefilter('error')

if INIT:
    if progress:
        import tqdm
        for task in tqdm.tqdm(task_generator, total=len(task_generator)):
            task.init()
    else:
        for task in task_generator:
            task.init()
elif clasic:
    if progress:
        import tqdm
        for _ in tqdm.tqdm(map(ModelInContext.compute, task_generator), total=len(task_generator)):
            pass
    else:
        for task in task_generator:
            task.compute()
else:
    if progress:
        import tqdm
        r = list(tqdm.tqdm(
            ArrayIsMetacentrum(slice_type='random').generator(AdvancedModelInContext.compute, task_generator),
            total = len(task_generator)
        ))
    #with IsMetacentrum() as pool:
    #    r = list(tqdm.tqdm(pool.imap_unordered(ModelInContext.compute, task_generator), total=len(task_generator)))
    else:    
        for res in ArrayIsMetacentrum(slice_type='random').generator(AdvancedModelInContext.compute, task_generator):
            pass
    

