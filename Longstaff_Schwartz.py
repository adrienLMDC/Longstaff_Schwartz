import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.api as sm


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
            return np.maximum(underlying_price - self.K, 0)
        if self.type == 'put':
            return np.maximum(self.K - underlying_price, 0)
        else:
            raise ValueError('Option Type not defined')

    def price_option(self, mkt: Market, stocks, k=3):
        stock_paths = stocks.path
        option_prices = self.payoff(stock_paths)

        if self.spec == 'europeene':

            option_price = (option_prices[:,-1] * np.exp(-mkt.r * self.T)).mean()
        else:
            cash_flows = option_prices[:, -1]

            for t in range(stocks.bm.nbobs - 1, 0, -1):
                in_the_money = option_prices[:, t] > 0
                if not np.any(in_the_money):
                    cash_flows *= np.exp(-mkt.r * stocks.bm.dt)
                    continue

                X = stock_paths[in_the_money, t]
                Y = cash_flows[in_the_money] * np.exp(-mkt.r * stocks.bm.dt)

                basis = np.column_stack(LongstaffSchwartz.basis_function(k, X / mkt.spot))
                model = sm.OLS(Y, basis).fit()
                continuation_value = model.predict(basis)

                intrinsic_value = option_prices[in_the_money, t]
                exercise = intrinsic_value > continuation_value

                cash_flows[in_the_money] = np.where(
                    exercise,
                    intrinsic_value,
                    cash_flows[in_the_money] * np.exp(-mkt.r * stocks.bm.dt)
                )

            option_price = np.mean(cash_flows) * np.exp(-mkt.r * stocks.bm.dt)
        return option_price
    
    


class BrownianMotion:
    def __init__(self, option: Option, nbobs, nb_path):
        self.T = option.T
        self.nbobs = nbobs
        self.dt = self.T / self.nbobs
        self.nb_path = nb_path
        self.bm_paths = None

    def generate_brownian(self, seed=42):
        np.random.seed(seed)
        unif_draws = np.random.rand(self.nb_path, self.nbobs)
        unif_draws = np.maximum(unif_draws, 1e-20)  # Evite les zéros pour éviter les erreurs de la fonction ppf
        normal_draws = norm.ppf(unif_draws)
        increment = normal_draws * np.sqrt(self.dt)
        self.bm_paths = np.cumsum(increment, axis=1)


class Stock:
    def __init__(self, mkt: Market, bm: BrownianMotion):
        self.mkt = mkt
        self.bm = bm
        self.path = None

    def set_path(self):
        self.path = self.mkt.spot * np.exp((self.mkt.r - self.mkt.q - 0.5 * self.mkt.vol**2) * self.bm.T +
                                           self.mkt.vol * self.bm.bm_paths)

    def plot_stock_path(self):
        if self.path is None:
            print("Erreur : les trajectoires de prix ne sont pas encore générées. Exécute set_path() d'abord.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.path.T, alpha=0.1, color='blue')
        plt.xlabel("Temps")
        plt.ylabel("Prix de l'actif")
        plt.title(f"Évolution de l'actif ({self.bm.nb_path} trajectoires)")
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


# Initialisation et test
market = Market(100, 0.03, 0.3, 0)
optionAM = Option(2, 100, 'call', 'Americaine')
optionEU = Option(2, 100, 'call', 'Europeene')
bm = BrownianMotion(option=optionAM, nbobs=10, nb_path=1000000)
bm.generate_brownian()
asset = Stock(market, bm)

asset.set_path()
# asset.plot_stock_path()  # Désactivé pour ne pas ralentir les tests
option_price = optionAM.price_option(market, asset, k=3)
option_price_EU = optionEU.price_option(market, asset, k=3)

print(f"American Option Price: {option_price} \nEuropean Option Price: {option_price_EU}")
'''
all_option_prices = []
for i in range(1, 1000000, 10000):
    bm = BrownianMotion(option=optionAM, nbobs=10, nb_path=i)
    bm.generate_brownian()
    asset = Stock(market, bm)
    asset.set_path()
    
    all_option_prices.append(optionAM.price_option(market, asset, k=3))
    print(i)

print('finish')


steps = [i for i in range(1, 1000000, 5000)]
plt.figure(figsize=(10, 5))
plt.plot(all_option_prices, steps, alpha=0.1, color='blue')
plt.xlabel("Temps")
plt.ylabel("Prix de l'actif")
plt.grid(True)
plt.show()
'''
