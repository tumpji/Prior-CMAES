#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from abc import ABC, abstractmethod

from scipy.stats import kendalltau

class Loss(ABC):
    name = 'error'
    
    @abstractmethod
    def __call__(self, predict, target):
        assert isinstance(predict, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert predict.shape == target.shape
        assert len(predict.shape) == 1
        return None

class LossL1(Loss):
    name = 'L1'
    
    def __call__(self, predict, target):
        #super().__call__(predict, target)
        return np.mean(np.abs(predict - target))

class LossL2(Loss):
    name = 'L2'
    
    def __call__(self, predict, target):
        #super().__call__(predict, target)
        return np.mean(np.square(predict - target))

class LossL1Drop10P(Loss):
    name = 'LossL1Drop10P'
    def __call__(self, predict, target):
         super().__call__(predict, target)
         num_of_res = int(round(float(target.shape[0])*(9./10.)))
         residuals = np.abs(predict - target)
         residuals = np.sort(residuals, axis=0)[:num_of_res]
         return np.mean(residuals)

class LossL2Drop10P(Loss):
    name = 'LossL2Drop10P'

    def __call__(self, predict, target):
         super().__call__(predict, target)
         num_of_res = int(round(float(target.shape[0])*(9./10.)))
         residuals = np.square(predict - target)
         residuals = np.sort(residuals, axis=0)[:num_of_res]
         return np.mean(residuals)


class LossKendall(Loss):
    name = 'Kendall'
    
    def __call__(self, predict, target):
        super().__call__(predict, target)
        c, _ = kendalltau(predict, target)
        return c

class LossRDE(Loss):
    name = 'RDE'
    cache = {}
    
    def __init__(self, mu):
        self._mu = mu

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        assert isinstance(mu, int)
        self._mu = mu
        
    def _compute_normalization_coefficient(self, lam, mu):
        assert mu <= lam
        
        prvni_sloupec = np.arange(1, -mu, step=-1)[:, np.newaxis]
        assert len(prvni_sloupec) == mu + 1
        
        radek = np.arange(1,mu+1)[np.newaxis, :]
        radek_obraceny = np.arange(lam, lam-mu, step=-1)[np.newaxis, :]
        assert radek.shape[1] == mu
        assert radek_obraceny.shape[1] == mu
        
        tabulka = prvni_sloupec + (radek - 1)
        tabulka = np.where(tabulka > 0, tabulka, radek_obraceny)
        vysledek = np.amax(np.sum(np.abs(tabulka - radek), axis=1))
        return vysledek
    
    def __call__(self, predict, target):
        assert self._mu is not None
        #super().__call__(predict, target)
        lam = len(predict)
        try:
            err_max = self.cache[(lam, self._mu)]
        except KeyError:
            err_max = self._compute_normalization_coefficient(lam, self._mu)
            self.cache[(lam, self._mu)] = err_max
            
        si_predict = np.argsort(predict)
        si_target  = np.argsort(target)[:self._mu]
        
        inRank = np.zeros(lam)
        inRank[si_predict] = np.arange(lam)
        
        r1 = inRank[si_target[:self._mu]]
        r2 = np.arange(self._mu)
        return np.sum(np.abs(r1 - r2))/err_max

class LossRDE_auto(LossRDE):
    def __init__(self):
        pass
    def __call__(self, predict, target):
        lam = len(target)
        self._mu = int(math.floor(lam / 2))
        return super().__call__(predict, target)

if __name__ == '__main__':
    import unittest
    
    class TestLosses(unittest.TestCase):
        def test_RDE(self):
            vec = np.array([0.8147, 0.9058, 0.1270, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
            tar = np.array([0.1576, 0.9706, 0.9572, 0.4854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
            
            vysledek = [0, 0.2500, 0.1905, 0.3333, 0.4286, 0.4062, 0.4444, 0.5500, 0.5111, 0.5200]
            
            for mu in range(1, 10+1):
                loss = LossRDE(mu)
                res = loss(vec, tar)
                self.assertAlmostEqual(res, vysledek[mu-1], places=4)
                
        def test_RDE_2(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760, 
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            vysledek = [0.1500, 0.2105, 0.4630, 0.6176, 0.5875, 0.5222, 0.5510, 0.5524, 0.5893, 0.5750,
                0.5859, 0.5809, 0.5694, 0.6209, 0.5864, 0.5965, 0.6500, 0.6526, 0.7000, 0.6714, 0.6545]
            
            for mu in range(1, 21+1):
                loss = LossRDE(mu)
                res = loss(vec, tar)
                self.assertAlmostEqual(res, vysledek[mu-1], places=4)
                
        def test_l1(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760, 
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = LossL1()
            loss(vec, tar)
            
        def test_l2(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760, 
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = LossL2()
            a = loss(vec, tar)
            
            
        def test_kendall(self):
            vec = np.array([0.6557, 0.0357, 0.8491, 0.9340, 0.6787, 0.7577, 0.7431, 0.3922, 0.6555, 0.1712,
                0.7060, 0.0318, 0.2769, 0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, 0.0344, 0.4387])
            tar = np.array([0.3816, 0.7655, 0.7952, 0.1869, 0.4898, 0.4456, 0.6463, 0.7094, 0.7547, 0.2760, 
                  0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 0.9597, 0.3404, 0.5853, 0.2238, 0.7513, 0.2551, ])
            loss = LossKendall()
            loss(vec, tar)
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


# In[ ]:




