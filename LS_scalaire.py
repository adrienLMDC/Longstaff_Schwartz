import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random 
import math
import numpy as np


class Market:
    def __init__(self, spot: float, r: float, vol: float, div_rate: float):
        self.r = r
        self.vol = vol
        self.spot = spot
        self.q = div_rate


class Option:
    def __init__(self, T, K, type: str, spec: 'americaine'):
        self.T = T
        self.K = K
        self.price = None

        if spec.lower() not in ['europeene', 'americaine']:
            raise ValueError("Veuillez rentrer le type de l'option sous le format : call ou put")
        else:
            self.spec = spec.lower()

        if type.lower() not in ['call', 'put']:
            raise ValueError("Veuillez rentrer le type de l'option sous le format : call ou put")
        else:
            self.type = type.lower()

    def payoff(self, underlying_price):
        if self.type == 'call':
            return max(underlying_price - self.K, 0)
        if self.type == 'put':
            return max(self.K - underlying_price, 0)
        else:
            raise ValueError('Option Type not defined')  
    


class BrownianMotion:
    def __init__(self, option: Option, nbobs):
        self.T = option.T
        self.nbobs = nbobs
        self.dt = self.T / self.nbobs
        self.bm_path = None

    def generate_brownian(self, seed=42):
        np.random.seed(seed)
        increment = []
        
        for i in range(self.nbobs):
            nb = max(random.uniform(0,1.0), 0.0)
            increment.append(norm.ppf(nb) * math.sqrt(0.1))

        cumsum = []
        cumsum.append(increment[0])   
        previous = increment[0]    
        for i in range(1,len(increment)):
            actual = previous + increment[i]
            cumsum.append(actual)
            previous = cumsum[i]

        self.bm_path = cumsum

class Stock:
    def __init__(self, mkt: Market, bm: BrownianMotion):
        self.mkt = mkt
        self.bm = bm
        self.path = None

    def set_path(self):
        path = []
        path.append(self.mkt.spot)

        for i in range(1,len(self.bm.bm_path)):            
            path.append(self.mkt.spot * math.exp((self.mkt.r - self.mkt.q - 0.5 * self.mkt.vol**2) * self.bm.T + self.mkt.vol * self.bm.bm_path[i]))
        

        self.path = path

    def plot_stock_path(self):
        if self.path is None:
            print("Erreur : la trajectoire du prix n'est pas encore générée. Exécute set_path() d'abord.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.path.T, alpha=0.1, color='blue')
        plt.xlabel("Temps")
        plt.ylabel("Prix de l'actif")
        plt.title(f"Évolution de l'actif")
        plt.grid(True)
        plt.show()


class LongstaffSchwartz:
    @staticmethod
    def basis_function(k, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("The type of the vector X should be np.ndarray")

        if k > 4:
            raise ValueError("The value of k should be less than or equal to 4")

        basis_funcs = [
            [np.exp(-X / 2)],
            [np.exp(-X / 2), np.exp(-X / 2) * (1 - X)],
            [np.exp(-X / 2), np.exp(-X / 2) * (1 - X), np.exp(-X / 2) * (1 - 2 * X - X**2 / 2)],
            [np.exp(-X / 2), np.exp(-X / 2) * (1 - X), np.exp(-X / 2) * (1 - 2 * X - X**2 / 2),
             np.exp(-X / 2) * (1 - 3 * X + 3 * (X**2) / 2 - (X**3) / 6)]
        ]
        return tuple(basis_funcs[k - 1])


