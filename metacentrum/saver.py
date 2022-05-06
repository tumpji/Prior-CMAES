#!/usr/bin/env python
# coding: utf-8

import os
import sys

import functools
import itertools
import copy
import datetime
import collections
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

import csv
import json
import tempfile
import h5py

from losses import Loss

def for_each_h5py(root, ignore=None):
    if ignore is None:
        ignore = set()

    for value in root.values():
        if value.name in ignore:
            continue
        elif isinstance(value, h5py.Dataset):
            ignore.add(value.name)
            #yield (value.name[1:], np.array(value))
            yield (value.name[1:], value)
        elif isinstance(value, h5py.Group):
            yield from for_each_h5py(value, ignore=ignore)
        else:
            logging.warning(f'Dataset is not cleaned property becouse unknown type returned: {v.name}')
    


# ok
class _HDF5_Model_base(ABC):
    def __init__(self, *, model: str, root_directory='results'):
        self.model = model
        self.root_directory = root_directory
        
    @property
    def _model_path(self):
        return os.path.join(self.root_directory, self.model)

    @property
    def _dispatcher_path(self):
        return os.path.join(self._model_path, 'dispatcher.csv')
        
    def _create_dispatcher(self):
        with open(self._dispatcher_path, mode='w', newline='') as dis_file:
            dis_writer = csv.writer(dis_file)
            dis_writer.writerow(['dir_name', 'hyperparameters'])

# ok
class HyperparameterInspector(_HDF5_Model_base):
    def __iter__(self):
        v = []
        with open(self._dispatcher_path, mode='r', newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for (directory, hp) in itertools.islice(dis_reader, 1, None):
                v.append(json.loads(hp))
        return iter(v)
    
    def __getitem__(self, item):
        assert isinstance(item, int)
        with open(self._dispatcher_path, mode='r', newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for (directory, hp) in itertools.islice(dis_reader, 1 + item, None):
                return json.loads(hp)
    
    def __repr__(self):
        result = ''
        with open(self._dispatcher_path, mode='r', newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for i, (directory, hp) in enumerate(itertools.islice(dis_reader, 1, None)):
                result += f'[{i}]: < {str(json.loads(hp))} >\n'
        return result


# ok
class _HDF5_Hyperparameters_base(_HDF5_Model_base):
    def __init__(self, *, model: str, hyperparameters: dict, root_directory = 'results'):
        super().__init__(model = model, root_directory=root_directory)
        self.hyperparameters = hyperparameters
        self._model_directory_cached = None
            
    def _create_model_directory(self):
        d_path = self._dispatcher_path
        serialization = json.dumps(self.hyperparameters)
        directory_prefix =  datetime.datetime.now().strftime('%Y-%m-%d_%H:%M_')
        directory = tempfile.mkdtemp(prefix=directory_prefix, dir=os.path.dirname(d_path) )
        directory = os.path.relpath(directory, start=os.path.dirname(d_path))
        
        with open(d_path, mode='a', newline='') as dis_file:
            dis_writer = csv.writer(dis_file)
            dis_writer.writerow([directory, serialization])
        return directory
        
    def _compare_hyperparameters(self, loaded, searched):
        return loaded == searched
    
    @property
    def _model_directory(self): # == use dispatcher
        if self._model_directory_cached is not None:
            return self._model_directory_cached
        
        with open(self._dispatcher_path, newline='') as dis_file:
            dis_reader = csv.reader(dis_file)
            for (directory, hp) in itertools.islice(dis_reader, 1, None):
                if self._compare_hyperparameters(json.loads(hp), self.hyperparameters):
                    self._model_directory_cached = os.path.join(self._model_path,directory)
                    return self._model_directory_cached
            return None

class _HDF5_Concrete_base(_HDF5_Hyperparameters_base):
    def __init__(self, *
        , model: str , hyperparameters: dict , root_directory = 'resluts'

        , function_id: int
        , run: int
        , losses = None
        , additional_datasets = None

        ):
        super().__init__(model=model, hyperparameters=hyperparameters, root_directory=root_directory)
        
        self.run = run
        self.function_id = function_id
        # losses
        if losses is None:
            losses = []
        assert all(isinstance(x, Loss) for x in losses)
        self.losses = losses
        if additional_datasets is None:
            additional_datasets = []
        assert all(isinstance(x, str) for x in additional_datasets)
        self.additional_datasets = additional_datasets
        self.init()
    
    @property
    def hdf5_final_path(self):
        return os.path.join(self._model_directory, 
            str(self.function_id) + '_fid',
            f'{self.run}.hdf5')
    @property
    def hdf5_tmp_path(self):
        return os.path.join(self._model_directory, 
            str(self.function_id) + '_fid',
            f'{self.run}.hdf5_tmp')

    @abstractmethod
    def init(self):
        pass

class Initializer(_HDF5_Concrete_base):
    def __init__(self, *args, 
            remove_existing_files = False, 
            **kwargs):
        self.remove_existing_files = remove_existing_files
        super().__init__(*args, **kwargs)

    def init(self):
        os.makedirs(os.path.dirname(self._dispatcher_path), exist_ok=True)
        
        if not os.path.exists(self._dispatcher_path):
            self._create_dispatcher()
            
        model_directory = self._model_directory
        if model_directory is None:
            model_directory = self._create_model_directory()

        if self.remove_existing_files:
            try:
                os.remove(self.hdf5_final_path)
            except OSError:
                pass

            try:
                os.remove(self.hdf5_tmp_path)
            except OSError:
                pass
        
        if not os.path.exists(self.hdf5_tmp_path) and not os.path.exists(self.hdf5_final_path):
            self._construct_hdf5_dataset(self.hdf5_tmp_path)

    @staticmethod
    def _add_database(hdf5_file, name):
        hdf5_file.create_dataset(name
                    , shape=(0,)
                    , chunks=(32,)
                    , maxshape=(None,)
                    , dtype=np.float32
                    , fillvalue=-np.inf)
    
    def _construct_hdf5_dataset(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # use only 'w'
        with h5py.File(path, 'w-') as h5f:
            h5f.create_dataset('prediction'
                        , shape=(0,0)
                        , chunks=(128,32)
                        , maxshape=(None, None)
                        , dtype=np.float32
                        , fillvalue=-np.inf)
            h5f.create_dataset('target'
                        , shape=(0,0)
                        , chunks=(128,32)
                        , maxshape=(None, None)
                        , dtype=np.float32
                        , fillvalue=-np.inf)

            for loss in self.losses:
                self._add_database(h5f, loss.name)

            for adsn in self.additional_datasets:
                self._add_database(h5f, adsn)


class Saver(_HDF5_Concrete_base):
    def __init__(self, *args, disable=False, **kwargs):
        self.disable = disable
        super().__init__(*args, **kwargs)

    def init(self):
        self._skip = 0
        
        self.completed = False
        if os.path.exists(self.hdf5_final_path):
            self.completed = True

        if  os.path.exists(self.hdf5_tmp_path):
            self._skip = self._check_hdf5_dataset_state(self.hdf5_tmp_path)
        elif not self.completed:
            e = 'The saver is not property initialized'
            logging.error(e)
            raise RuntimeError(e)
        
    def finalize(self):
        os.rename(self.hdf5_tmp_path, self.hdf5_final_path)
        self.completed = True

    def clean(self, dset):
        if not self.disable:
            with h5py.File(dset, mode='a') as f:
                for (name, value) in for_each_h5py(f):
                    value.resize((0,*value.shape[1:]))

    def _check_hdf5_dataset_state(self, path):
        with h5py.File(path, 'r') as h5f:
            prediction = h5f['prediction']
            target = h5f['target']
            
            if prediction.shape != target.shape:
                logging.warning(f'Shape mismatch prediction:{prediction.shape} target:{target.shape} in file {path}')
            return min(prediction.shape[0], target.shape[0])
            
    def write_results_to_dataset(self, *, prediction, target, **kwargs):
        assert isinstance(prediction, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert len(prediction.shape) == 1
        assert target.shape == prediction.shape

        # Add losses
        others = {}
        for loss_obj in self.losses:
            loss_value = loss_obj(prediction, target)
            others[loss_obj.name] = loss_value
        # Add other
        others.update(kwargs)
        
        if self.disable:
            return others
        
        with h5py.File(self.hdf5_tmp_path, 'a') as h5f:
            pred_d = h5f['prediction']
            pred_d.resize( (self._skip+1, max(pred_d.shape[1], len(prediction))) )
            pred_d[-1, :len(prediction)] = prediction

            targ_d = h5f['target']
            targ_d.resize( (self._skip+1, max(targ_d.shape[1], len(prediction))) )
            targ_d[-1, :len(target)] = target 

            for key, val in others.items():
                try:
                    d = h5f[key]
                except KeyError:
                    logging.warning(f"Key: {key} cannot be saved: inner database not found: creating database")
                    Initializer._add_database(h5f, key)
                d = h5f[key]
                d.resize( (self._skip+1, *d.shape[1:]) )
                d[-1, ...] = val

            self._skip += 1
        return others
            
    @property
    def already_computed(self):
        return self._skip

class Loader(_HDF5_Concrete_base):
    def init(self):
        self.completed = False
        
        if os.path.exists(self.hdf5_final_path):
            self.completed = True
            self._data_path = self.hdf5_final_path
        else:
            logging.warning('Final result not found: using tmp')
            self._data_path = self.hdf5_tmp_path
            
    @property
    def data(self):
        with h5py.File(self._data_path, 'r') as h5f:
            prediction = np.array(h5f['prediction'])
            target = np.array(h5f['target'])
            others = {}

            for (name, value) in for_each_h5py(h5f, ignore=set(['/target', '/prediction'])):
                others[name] = np.array(value)

        return prediction, target, others
            
class LoaderIterator(_HDF5_Hyperparameters_base):
    def __iter__(self):
        m = copy.deepcopy(self)
        m.files = []
        for root, _, files in os.walk(self._model_directory):
            for fil in files:
                m.files.append(os.path.normpath(os.path.join(root, fil)))
        m.type_output = collections.namedtuple(
            'RunDescription', 
            ['function_id', 'run']
        )
        return m
    
    def __next__(self):
        try:
            filepath = self.files.pop() 
        except IndexError:
            raise StopIteration()

        func_id, basefilename = filepath.split(os.sep)[-2:]
        func_id = func_id[:-4]
        run = basefilename.split('.')[0]

        conf = self.type_output(int(func_id), int(run))
        l = Loader(
                model=self.model, 
                hyperparameters=self.hyperparameters, 
                function_id=conf.function_id,
                run=conf.run,
                root_directory=self.root_directory
            )
        return l, conf


if __name__ == '__main__':
    import unittest
    import shutil
    import random
    import collections

    import losses

if __name__ == '__main__':
    class TestInitializer(unittest.TestCase):
        testFileDirName = 's1412_test'
        
        '''
        @classmethod
        def setUpClass(cls):
            shutil.rmtree(cls.testFileDirName, ignore_errors=True)
        '''
            
        @classmethod
        def tearDownClass(cls):
            shutil.rmtree(cls.testFileDirName)
            
        def setUp(self):
            shutil.rmtree(self.testFileDirName, ignore_errors=True)
            
        def test_loader(self):
            # CREATE FIRST MODEL
            Initializer( model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=3, dimension=2, kernel_id=1, run=0, root_directory = self.testFileDirName,
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv') ))
            m1_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel1')))
            m1_dir.remove('dispatcher.csv')
            m1_dir = m1_dir.pop()
            
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', m1_dir, '3_fid', '2_dim', '1_0.hdf5_tmp') ))
            with open(os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 2)
            self.assertEqual(len(os.listdir(self.testFileDirName)), 1)
            
            # CREATE SECOND MODEL
            Initializer( model='testModel2', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=2, dimension=4, kernel_id=1, run=2, root_directory = self.testFileDirName
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel2', 'dispatcher.csv') ))
            m2_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel2')))
            m2_dir.remove('dispatcher.csv')
            m2_dir = m2_dir.pop()
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel2', m2_dir, '2_fid', '4_dim', '1_2.hdf5_tmp') ))
            with open(os.path.join(self.testFileDirName, 'testModel2', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 2)
            self.assertEqual(len(os.listdir(self.testFileDirName)), 2)
                
                
            # CREATE THIRD...
            Initializer( model='testModel1', hyperparameters={'a': 12.3, 'b': 16} # <--- minor change of hyp.
                    , function_id=1, dimension=2, kernel_id=1, run=0 , root_directory = self.testFileDirName
                  )
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv') ))
            with open(os.path.join(self.testFileDirName, 'testModel1', 'dispatcher.csv'), 'r') as f:
                self.assertEqual(sum(1 for line in f), 3)
            m3_dir = set(os.listdir(os.path.join(self.testFileDirName, 'testModel1')))
            m3_dir.remove('dispatcher.csv')
            m3_dir.remove(m1_dir)
            m3_dir = m3_dir.pop()
            
            self.assertTrue(os.path.exists(
                os.path.join(self.testFileDirName, 'testModel1', m3_dir, '1_fid', '2_dim',  '1_0.hdf5_tmp') ))
            self.assertEqual(len(os.listdir(self.testFileDirName)), 2)
            
            f = list(itertools.chain.from_iterable((files for subdir, dirs, files in os.walk(self.testFileDirName))))
            self.assertEqual(len([files for files in f if '.hdf5' in files]), 3)
            self.assertEqual( len([files for files in f if files == 'dispatcher.csv']), 2)

        def test_loss(self):
            Initializer( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=2, run=0, root_directory = self.testFileDirName
                    , losses = [losses.LossL2(), losses.LossL1()]
                  )
            s = Saver( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=2, run=0, root_directory = self.testFileDirName
                    , losses = [losses.LossL2(), losses.LossL1()]
                  )
            s.write_results_to_dataset(prediction=np.array([1,2,4]), target=np.array([2,2,2]))
            s.write_results_to_dataset(prediction=np.array([1,2,5]), target=np.array([2,2,2]))

            l = Loader( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=2, run=0 , root_directory = self.testFileDirName
                  )

            prediction, target, other_obj = l.data

            self.assertIn('L1', other_obj)
            self.assertIn('L2', other_obj)
            self.assertTrue(np.all(
                np.abs(other_obj['L1'] - np.array([3/3,4/3])) < 1e-5
                    ))
            self.assertTrue(np.all(
                np.abs(other_obj['L2'] - np.array([5/3,10/3])) < 1e-5
                ))

        def test_additional_datasets(self):
            Initializer( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=3, run=0, root_directory = self.testFileDirName
                    , losses = [losses.LossL2(), losses.LossL1()]
                    , additional_datasets = ['a']
                  )
            s = Saver( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=3, run=0 , root_directory = self.testFileDirName
                    , losses = [losses.LossL2(), losses.LossL1()]
                  )
            s.write_results_to_dataset(prediction=np.array([1,2,4]), target=np.array([2,2,2]), a=2)
            s.write_results_to_dataset(prediction=np.array([1,2,5]), target=np.array([2,2,2]), a=3)

            l = Loader( model='testModel1_loss', hyperparameters={'a': 12.3, 'b': 16}
                    , function_id=1, dimension=2, kernel_id=3, run=0, root_directory = self.testFileDirName
                  )

            prediction, target, other_obj = l.data
            self.assertIn('a', other_obj)
            self.assertTrue( np.all(other_obj['a'] == np.array([2,3])) )
        

if __name__ == '__main__':
    class TestSaverLoader(unittest.TestCase):
        testFileDirName = 's1412_test'
        
        '''
        @classmethod
        def setUpClass(cls):
            shutil.rmtree(cls.testFileDirName, ignore_errors=True)
        '''
            
        @classmethod
        def tearDownClass(cls):
            shutil.rmtree(cls.testFileDirName)
            
        def setUp(self):
            shutil.rmtree(self.testFileDirName, ignore_errors=True)
            
        def test_loader(self):
            # CREATE FIRST MODEL
            Initializer(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=4,  run=0 , root_directory = self.testFileDirName
                  )
            
            s = Saver(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=4, run=0, root_directory = self.testFileDirName
                  )
            
            self.assertEqual(s.already_computed, 0)
            
            sizes = [10, 72, 111, 12, 312]
            
            for i,size in enumerate(sizes):
                # saver
                s.write_results_to_dataset(
                    prediction = np.arange(size),
                    target = np.arange(size) + 1,
                    training_samples = size
                )
                
                # loader
                l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                    , function_id=3, dimension=2, kernel_id=4, run=0 , root_directory = self.testFileDirName
                    )
                
                pred, tar, si = l.data
                self.assertEqual(pred.shape[0], i+1)
                self.assertEqual(tar.shape[0], i+1)
                self.assertEqual(si['training_samples'].shape[0], i+1)
                
                for y in range(i+1):
                    self.assertTrue(np.all(pred[y, :sizes[y]] == np.arange(sizes[y])))
                    self.assertTrue(np.all(pred[y, sizes[y]:] == -np.inf))
                    
                    self.assertTrue(np.all(tar[y, :sizes[y]] == 1 + np.arange(sizes[y])))
                    self.assertTrue(np.all(tar[y, sizes[y]:] == -np.inf))
                    
                    self.assertEqual(si['training_samples'][y], sizes[y])
                    
            ####  NEW SAVER
            s = Saver(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2, kernel_id=4, run=0 , root_directory = self.testFileDirName )
            self.assertEqual(len(sizes), s.already_computed)
            
            s.write_results_to_dataset(
                prediction = np.arange(10),
                target = np.arange(10) + 1,
                training_samples = 10)
            
            l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=4, run=0 , root_directory = self.testFileDirName)
            
            pred, tar, si = l.data
            self.assertEqual(pred.shape[0], len(sizes) + 1)
            self.assertEqual(tar.shape[0], len(sizes) + 1)
            self.assertEqual(si['training_samples'].shape[0], len(sizes) + 1)
            
            self.assertTrue(np.all(pred[-1, :10] == np.arange(10)))
            self.assertTrue(np.all(pred[-1, 10:] == -np.inf))

            self.assertTrue(np.all(tar[-1, :10] == 1 + np.arange(10)))
            self.assertTrue(np.all(tar[-1, 10:] == -np.inf))

            self.assertEqual(si['training_samples'][y], sizes[y])
            
            s.finalize()
            
            #### FINALIZE + LOADER AGAIN
            
            l = Loader(model='testModel1', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=4, run=0 , root_directory = self.testFileDirName)
            pred, tar, si = l.data
            
            sizes = sizes + [10]
            
            for y in range(len(sizes)):
                self.assertTrue(np.all(pred[y, :sizes[y]] == np.arange(sizes[y])))
                self.assertTrue(np.all(pred[y, sizes[y]:] == -np.inf))

                self.assertTrue(np.all(tar[y, :sizes[y]] == 1 + np.arange(sizes[y])))
                self.assertTrue(np.all(tar[y, sizes[y]:] == -np.inf))

                self.assertEqual(si['training_samples'][y], sizes[y])

        def test_clean_and_emergency_generation_of_dataset(self):
            Initializer(model='testModel1_21', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=5, run=0, root_directory = self.testFileDirName,
                losses = [losses.LossL1()]
                  )
            
            s = Saver(model='testModel1_21', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3 , dimension=2, kernel_id=5, run=0 , root_directory = self.testFileDirName,
                losses = [losses.LossL1()]
                  )

            sizes = [1,10,111,32]
            for i,size in enumerate(sizes):
                # saver
                s.write_results_to_dataset(
                    prediction = np.arange(size),
                    target = np.arange(size) + 1,
                    training_samples = size
                )

            s.clean(s.hdf5_tmp_path)
        
            l = Loader(model='testModel1_21', hyperparameters={'a': 12.3, 'b': 15}
                , function_id=3, dimension=2, kernel_id=5, run=0, root_directory = self.testFileDirName)

            pred, tar, si = l.data
            self.assertEqual(pred.shape[0], 0)
            self.assertEqual(tar.shape[0], 0)
            for v in si.values():
                self.assertEqual(v.shape[0], 0)



if __name__ == '__main__':
    class TestIterator(unittest.TestCase):
            testFileDirName = 's1412_test'

            '''
            @classmethod
            def setUpClass(cls):
                shutil.rmtree(cls.testFileDirName, ignore_errors=True)
            '''

            @classmethod
            def tearDownClass(cls):
                shutil.rmtree(cls.testFileDirName)

            def setUp(self):
                shutil.rmtree(self.testFileDirName, ignore_errors=True)

            def create_data(self, model, hyperparameters, fid, dim, run, offset=0):
                Initializer(model=model, 
                        hyperparameters=hyperparameters, 
                    function_id=fid, dimension=dim, kernel_id=6, run=run, 
                    root_directory = self.testFileDirName, additional_datasets=['training_samples'])

                s = Saver(model=model, hyperparameters=hyperparameters,
                        function_id=fid, dimension=dim, kernel_id=6, 
                        run=run, root_directory = self.testFileDirName)

                for size in [2, 5, 3, 17, 13]:
                    s.write_results_to_dataset(
                        prediction = np.arange(size),
                        target = np.arange(size) + fid + dim*100 + run*10000 + offset,
                        training_samples = size)

                s.finalize()


            def check_data(self, data, fid, dim, run, offset = 0):
                pred, targ, other_obj = data

                for i, size in enumerate([2, 5, 3, 17, 13]):
                    self.assertTrue(np.all(np.arange(size) == pred[i,:size]))
                    self.assertTrue(np.all(-np.inf == pred[i,size:]))
                    self.assertTrue(np.all(np.arange(size) + fid + dim*100 + run*10000 + offset == targ[i,:size]))
                    self.assertTrue(np.all(-np.inf == targ[i,size:]))

                    self.assertEqual(other_obj['training_samples'][i], size)


            def test_loader(self):
                c = collections.defaultdict(set)
                for (model, of1) in [('testModel111', 10), ('testModel222', 11)]:
                    for hyp in [{'hyp': 2}, {'hyp': 3}, {'hyp': 4}]:
                        for fid in [11,22,33]:
                            for dim in [10,20,30]:
                                for run in [1,2,3]:
                                    c[(model, hyp['hyp'])].add((fid,dim,run))
                                    self.create_data(model, hyp, fid, dim, run, offset=of1 + hyp['hyp'])

                for (model, of1) in [('testModel111', 10), ('testModel222', 11)]:
                    for hyp in [{'hyp': 2}, {'hyp': 3}, {'hyp': 4}]:
                        it = LoaderIterator(model=model, hyperparameters=hyp, 
                                            root_directory=self.testFileDirName )
                        for (loader, ids) in it:
                            c[(model, hyp['hyp'])].remove( (ids.function_id, ids.dimension, ids.run))
                            self.check_data(loader.data, ids.function_id, ids.dimension, ids.run, offset=of1 + hyp['hyp'])
                
                for t in c.values():
                    self.assertEqual(0, len(t))
                    
                hi = HyperparameterInspector(model='testModel111', root_directory=self.testFileDirName)
                print(hi)
                print(hi[2])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

