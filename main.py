import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Tuple
from scipy.stats import norm
from abc import ABC, abstractmethod
from numpy.random import default_rng

# ---------------------------------------------------------------------
#                           Market & Option
# ---------------------------------------------------------------------

@dataclass
class Market:
    stock_price: float
    int_rate: float
    vol: float
    dividend_yield: float = 0.0

@dataclass
class Option:
    S0: float
    K: float
    T: float
    sigma: float
    option_type: str = 'put'

    def __post_init__(self) -> None:
        self.option_type = self.option_type.strip().lower()
        if self.option_type not in ['put', 'call']:
            raise ValueError("option_type must be 'put' or 'call'")

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == 'put':
            return np.maximum(self.K - S, 0.0)
        else:
            return np.maximum(S - self.K, 0.0)

    @property
    def is_put(self) -> bool:
        return self.option_type == 'put'

# ---------------------------------------------------------------------
#                            Brownian Paths
# ---------------------------------------------------------------------

class Brownian:
    def __init__(
            self,
            market: Market,
            option: Option,
            n_steps: int,
            n_paths: int,
            seed: Optional[int] = None,
            antithetic: bool = False
    ) -> None:
        self.market = market
        self.option = option
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.T = option.T
        self.dt = self.T / n_steps
        self.antithetic = antithetic
        self.rng = default_rng(seed)

    def simulate_paths(self) -> np.ndarray:
        S0 = self.option.S0
        r = self.market.int_rate
        q = self.market.dividend_yield
        sigma = self.market.vol

        drift = (r - q - 0.5 * sigma ** 2) * self.dt
        diffusion = sigma * np.sqrt(self.dt)

        if self.antithetic:
            half_paths = (self.n_paths + 1) // 2
            Z_half = self.rng.standard_normal((half_paths, self.n_steps))
            Z = np.concatenate([Z_half, -Z_half], axis=0)
            Z = Z[:self.n_paths]
        else:
            Z = self.rng.standard_normal((self.n_paths, self.n_steps))

        increments = drift + diffusion * Z

        log_paths = np.zeros((self.n_paths, self.n_steps + 1))
        log_paths[:, 0] = np.log(S0)
        np.cumsum(increments, axis=1, out=log_paths[:, 1:])
        log_paths[:, 1:] += log_paths[:, [0]]

        return np.exp(log_paths)

# ---------------------------------------------------------------------
#                             Regression
# ---------------------------------------------------------------------

class Regression:
    def __init__(
            self,
            basis_functions: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None
    ) -> None:
        self.basis_functions = basis_functions or [
            lambda x: np.ones_like(x),
            lambda x: np.exp(-x / 2),
            lambda x: np.exp(-x / 2) * (1 - x),
            lambda x: np.exp(-x / 2) * (1 - 2 * x + 0.5 * x ** 2)
        ]

    def regress(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        A = np.column_stack([f(x) for f in self.basis_functions])
        beta, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = self.predict(x, beta)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return beta, r_squared

    def predict(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        A = np.column_stack([f(x) for f in self.basis_functions])
        return A @ beta

# ---------------------------------------------------------------------
#                       Abstract Pricing Model
# ---------------------------------------------------------------------

class PricingModel(ABC):
    def __init__(self, market: Market, option: Option) -> None:
        self.market = market
        self.option = option
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.market.vol <= 0:
            raise ValueError("Volatility (vol) must be positive.")
        if self.option.T <= 0:
            raise ValueError("Time to maturity (T) must be positive.")
        if self.option.K <= 0:
            raise ValueError("Strike price (K) must be positive.")
        if self.option.S0 <= 0:
            raise ValueError("Stock price (S0) must be positive.")

    @abstractmethod
    def price(self) -> float:
        pass

    @abstractmethod
    def calc_greeks(self, shift: float = 0.01) -> Dict[str, float]:
        pass

# ---------------------------------------------------------------------
#                            Monte Carlo
# ---------------------------------------------------------------------

class MCModel(PricingModel):
    def __init__(
            self,
            market: Market,
            option: Option,
            n_steps: int,
            n_paths: int,
            normalize: bool = True,
            debug: bool = False,
            debug_steps: Optional[List[int]] = None,
            seed: Optional[int] = None,
            antithetic: bool = False,
            greek_samples: int = 5
    ):
        super().__init__(market, option)
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.normalize = normalize
        self.debug = debug
        self.debug_steps = debug_steps or []
        self.seed = seed
        self.antithetic = antithetic
        self.greek_samples = greek_samples

        self.dt = option.T / n_steps
        self.brownian = Brownian(market, option, n_steps, n_paths, seed, antithetic)
        self.regression = Regression()

        self._cached_price: Optional[float] = None
        self._last_signature: Optional[Tuple] = None
        self.stock_paths: Optional[np.ndarray] = None
        self.exercise_decisions: Optional[np.ndarray] = None

    def _make_signature(self) -> Tuple:
        return (
            self.market.int_rate,
            self.market.vol,
            self.market.dividend_yield,
            self.option.S0,
            self.option.K,
            self.option.T,
            self.option.option_type,
            self.n_steps,
            self.n_paths,
            self.normalize,
            self.seed,
            self.antithetic
        )

    def price(self) -> float:
        current_signature = self._make_signature()
        if current_signature != self._last_signature:
            self._cached_price = None
            self._last_signature = current_signature

        if self._cached_price is not None:
            return self._cached_price

        if self.stock_paths is None:
            self.stock_paths = self.brownian.simulate_paths()

        N = self.n_steps
        S = self.stock_paths
        r = self.market.int_rate
        dt = self.dt
        option = self.option

        exercise_time = np.full(self.n_paths, N, dtype=int)
        cash_flow = option.payoff(S[:, N])
        exercise_decisions = np.zeros_like(S, dtype=bool)
        exercise_decisions[:, N] = cash_flow > 0.0

        discount_factor = np.exp(-r * dt)

        # Corrected loop to include t=0
        for t in range(N - 1, -1, -1):
            alive = exercise_time > t
            if not np.any(alive):
                continue

            current_prices = S[alive, t]
            immediate_payoff = option.payoff(current_prices)

            if option.is_put:
                itm_mask = current_prices < option.K
            else:
                itm_mask = current_prices > option.K

            if not np.any(itm_mask):
                cash_flow[alive] *= discount_factor
                continue

            future_discount = np.exp(-r * (exercise_time[alive] - t) * dt)
            continuation_values = cash_flow[alive] * future_discount

            scale = option.K if self.normalize else 1.0
            x_reg = current_prices[itm_mask] / scale
            y_reg = continuation_values[itm_mask] / scale

            beta, r_squared = self.regression.regress(x_reg, y_reg)
            cont_est = self.regression.predict(x_reg, beta) * scale

            exercise_now = immediate_payoff[itm_mask] > cont_est

            if self.debug and t in self.debug_steps:
                self._plot_regression_diagnostic(x_reg, y_reg, beta, t, r_squared)

            alive_indices = np.where(alive)[0]
            chosen_to_exercise = alive_indices[itm_mask][exercise_now]

            exercise_decisions[chosen_to_exercise, t] = True
            exercise_time[chosen_to_exercise] = t
            cash_flow[chosen_to_exercise] = immediate_payoff[itm_mask][exercise_now]

            not_exercised = np.logical_and(alive, ~np.isin(alive_indices, chosen_to_exercise))
            cash_flow[not_exercised] *= discount_factor

        self._cached_price = np.mean(cash_flow)
        self.exercise_decisions = exercise_decisions
        return self._cached_price

    def _reprice_with_reset(self):
        # Reset cached values to ensure a fresh simulation
        self._cached_price = None
        self._last_signature = None
        self.stock_paths = None

        # Reinitialize Brownian with current parameters
        self.brownian = Brownian(
            self.market, self.option,
            self.n_steps, self.n_paths,
            self.seed, self.antithetic
        )
        return self.price()

    def calc_greeks(self, shift: float = 0.01) -> Dict[str, float]:
        """
        Implements the calc_greeks method for the MCModel,
        using finite differences with multiple simulations
        to reduce Monte Carlo noise.
        """
        def robust_reprice():
            # Average across multiple simulations for noise reduction
            prices = [self._reprice_with_reset() for _ in range(self.greek_samples)]
            return np.mean(prices)

        # Save original parameters
        orig_params = {
            'S0': self.option.S0,
            'r': self.market.int_rate,
            'vol': self.market.vol,
            'T': self.option.T
        }

        # Baseline price
        price_0 = robust_reprice()

        # ---- Delta & Gamma ----
        h_S0 = max(shift * orig_params['S0'], 0.01)
        delta, gamma = self._calculate_delta_gamma(orig_params, price_0, h_S0, robust_reprice)

        # ---- Vega ----
        h_vol = max(shift * orig_params['vol'], 0.001)
        self.market.vol = orig_params['vol'] + h_vol
        price_vol_up = robust_reprice()
        self.market.vol = orig_params['vol'] - h_vol
        price_vol_down = robust_reprice()
        self.market.vol = orig_params['vol']
        vega = (price_vol_up - price_vol_down) / (2 * h_vol)

        # ---- Rho ----
        h_r = max(shift * orig_params['r'], 0.0001)
        self.market.int_rate = orig_params['r'] + h_r
        price_r_up = robust_reprice()
        self.market.int_rate = orig_params['r'] - h_r
        price_r_down = robust_reprice()
        self.market.int_rate = orig_params['r']
        rho = (price_r_up - price_r_down) / (2 * h_r)

        # ---- Theta ----
        h_T = max(shift * orig_params['T'], 0.01)
        if orig_params['T'] > 2 * h_T:
            self.option.T = orig_params['T'] + h_T
            price_T_up = robust_reprice()
            self.option.T = orig_params['T'] - h_T
            price_T_down = robust_reprice()
            theta = (price_T_up - price_T_down) / (2 * h_T)
        else:
            # If T is too small for both + and - shift,
            # use a one-sided difference
            self.option.T = orig_params['T'] + h_T
            price_T_up = robust_reprice()
            theta = (price_T_up - price_0) / h_T

        # Restore original parameters
        self.option.T = orig_params['T']

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Rho": rho,
            "Theta": theta
        }

    def _calculate_delta_gamma(self, orig_params, price_0, h_S0, repricer):
        # Shift up
        self.option.S0 = orig_params['S0'] + h_S0
        price_up = repricer()

        # Shift down
        self.option.S0 = orig_params['S0'] - h_S0
        price_down = repricer()

        # Restore
        self.option.S0 = orig_params['S0']

        delta = (price_up - price_down) / (2 * h_S0)
        try:
            gamma = (price_up - 2 * price_0 + price_down) / (h_S0 ** 2)
        except ZeroDivisionError:
            gamma = np.nan
        return delta, gamma

    def _plot_regression_diagnostic(self, x_reg, y_reg, beta, t, r_squared):
        """
        Optional debug plot to visualize the regression fit at a given time step.
        (Not required for normal usage.)
        """
        fig, ax = plt.subplots()
        ax.scatter(x_reg, y_reg, alpha=0.5, label='Data Points')
        x_line = np.linspace(np.min(x_reg), np.max(x_reg), 100)
        y_line = self.regression.predict(x_line, beta)
        ax.plot(x_line, y_line, label='Regression Fit')
        ax.set_title(f"Time Step {t}, R²={r_squared:.3f}")
        ax.legend()
        plt.show()

# ---------------------------------------------------------------------
#                            Trinomial
# ---------------------------------------------------------------------

class TrinomialModel(PricingModel):
    def __init__(self, market: Market, option: Option, n_steps: int) -> None:
        super().__init__(market, option)
        self.n_steps = n_steps
        self.dt = option.T / n_steps
        self._cached_price: Optional[float] = None
        self._last_signature: Optional[Tuple] = None
        self._validate_steps()

    def _validate_steps(self) -> None:
        if self.n_steps < 1:
            raise ValueError("Number of steps must be at least 1.")

    def _make_signature(self) -> Tuple:
        return (
            self.market.int_rate,
            self.market.vol,
            self.market.dividend_yield,
            self.option.S0,
            self.option.K,
            self.option.T,
            self.option.option_type,
            self.n_steps
        )

    def price(self) -> float:
        current_signature = self._make_signature()
        if current_signature != self._last_signature:
            self._cached_price = None
            self._last_signature = current_signature

        if self._cached_price is not None:
            return self._cached_price

        S0, K = self.option.S0, self.option.K
        r, sigma = self.market.int_rate, self.market.vol
        q = self.market.dividend_yield
        T, N = self.option.T, self.n_steps
        dt = T / N

        dx = sigma * np.sqrt(3 * dt)
        u = np.exp(dx)
        d = 1 / u
        nu = r - q - 0.5 * sigma ** 2
        term = nu * np.sqrt(dt / (12 * sigma ** 2))
        pu = 1/6 + term
        pd = 1/6 - term
        pm = 2/3

        # Validate probabilities
        if pu < 0 or pm < 0 or pd < 0:
            raise ValueError("Negative probabilities detected")

        # Initialize stock price tree for ALL time steps
        max_nodes = 2 * N + 1
        S = np.zeros((max_nodes, N + 1))
        V = np.zeros((max_nodes, N + 1))

        for t in range(N + 1):
            for j in range(-t, t + 1):
                idx = j + N
                S[idx, t] = S0 * np.exp(j * dx)
                if t == N:
                    V[idx, t] = self.option.payoff(S[idx, t])

        # Backward induction
        for t in range(N - 1, -1, -1):
            for j in range(-t, t + 1):
                idx = j + N
                up_idx = (j + 1) + N
                mid_idx = j + N
                down_idx = (j - 1) + N

                # Handle boundary conditions for indices
                cont_value = 0.0
                if up_idx < max_nodes:
                    cont_value += pu * V[up_idx, t + 1]
                cont_value += pm * V[mid_idx, t + 1]
                if down_idx >= 0:
                    cont_value += pd * V[down_idx, t + 1]
                cont_value *= np.exp(-r * dt)

                immediate = self.option.payoff(S[idx, t])
                V[idx, t] = max(cont_value, immediate)

        self._cached_price = V[N, 0]
        return self._cached_price

    def calc_greeks(self, shift: float = 0.01) -> Dict[str, float]:
        orig_price = self.price()
        orig_params = {
            'S0': self.option.S0,
            'r': self.market.int_rate,
            'vol': self.market.vol,
            'T': self.option.T
        }

        # Delta & Gamma
        h_S0 = max(shift * orig_params['S0'], 0.01)
        self.option.S0 = orig_params['S0'] + h_S0
        price_up = self.price()
        self.option.S0 = orig_params['S0'] - h_S0
        price_down = self.price()
        self.option.S0 = orig_params['S0']
        delta = (price_up - price_down) / (2 * h_S0)
        gamma = (price_up - 2 * orig_price + price_down) / (h_S0 ** 2)

        # Vega
        h_vol = max(shift * orig_params['vol'], 0.001)
        self.market.vol = orig_params['vol'] + h_vol
        price_vol_up = self.price()
        self.market.vol = orig_params['vol'] - h_vol
        price_vol_down = self.price()
        self.market.vol = orig_params['vol']
        vega = (price_vol_up - price_vol_down) / (2 * h_vol)

        # Rho
        h_r = max(shift * orig_params['r'], 0.0001)
        self.market.int_rate = orig_params['r'] + h_r
        price_r_up = self.price()
        self.market.int_rate = orig_params['r'] - h_r
        price_r_down = self.price()
        self.market.int_rate = orig_params['r']
        rho = (price_r_up - price_r_down) / (2 * h_r)

        # Theta
        h_T = max(shift * orig_params['T'], 0.01)
        self.option.T = orig_params['T'] + h_T
        price_T_up = self.price()
        self.option.T = orig_params['T']
        theta = (price_T_up - orig_price) / h_T

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Rho": rho,
            "Theta": theta
        }
# ---------------------------------------------------------------------
#                           Plotting Utilities
# ---------------------------------------------------------------------

def plot_exercise_regions(
        stock_paths: np.ndarray,
        exercise_decisions: np.ndarray,
        time_grid: np.ndarray,
        option: Option,
        max_paths_to_plot: int = 300
) -> plt.Figure:
    """Plot multiple simulated paths in gray, strike line,
    and red X at the times an early exercise occurs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_paths, _ = stock_paths.shape

    if n_paths > max_paths_to_plot:
        indices_to_plot = np.random.choice(n_paths, max_paths_to_plot, replace=False)
    else:
        indices_to_plot = np.arange(n_paths)

    # Plot the paths
    for i in indices_to_plot:
        ax.plot(time_grid, stock_paths[i, :], color='gray', alpha=0.4, linewidth=0.5)

    # Mark exercise points
    exercise_points = []
    exercise_prices = []
    for t, time_val in enumerate(time_grid):
        exercised = exercise_decisions[indices_to_plot, t]
        if np.any(exercised):
            idx_ex = indices_to_plot[exercised]
            exercise_points.extend([time_val] * len(idx_ex))
            exercise_prices.extend(stock_paths[idx_ex, t])

    ax.scatter(
        exercise_points,
        exercise_prices,
        color='red',
        marker='x',
        s=30,
        linewidths=1.5,
        label='Exercise Points',
        zorder=3
    )

    ax.axhline(y=option.K, color='blue', linestyle='--', alpha=0.5, label='Strike Price')
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title(
        f"Exercise Regions for {option.option_type.capitalize()} Option\n"
        f"Strike={option.K:.2f}, Maturity={option.T:.1f}y"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig

def black_scholes_price(market: Market, option: Option) -> float:
    """European Black-Scholes formula, for reference."""
    S0, K = option.S0, option.K
    r, q = market.int_rate, market.dividend_yield
    sigma = market.vol
    T = option.T

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option.is_put:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ---------------------------------------------------------------------
#                         Convergence Utilities
# ---------------------------------------------------------------------

def linear_basis():
    return [
        lambda x: np.ones_like(x),
        lambda x: x
    ]

def quadratic_basis():
    return [
        lambda x: np.ones_like(x),
        lambda x: x,
        lambda x: x**2
    ]

BASIS_SETS = {
    'American Linear': linear_basis(),
    'American Quadratic': quadratic_basis()
}

def run_convergence_experiment(market: Market,
                               option: Option,
                               steps_list: List[int]) -> None:
    """
    Example: Evaluate how American Put prices (with different polynomial bases)
    converge as we increase n_steps. Compare with European BS price.
    """
    # We'll store results in a dictionary
    results = {
        "Steps": [],
        "European": [],
    }
    # Also store each American basis set's results
    for basis_name in BASIS_SETS:
        results[basis_name] = []

    # For reference, the European price from closed-form
    european_bs_price = black_scholes_price(market, option)

    for n_steps in steps_list:
        results["Steps"].append(n_steps)
        # We can store the same BS price for reference,
        # or you can also do a binomial/trinomial for a purely European payoff
        results["European"].append(european_bs_price)

        # For each polynomial basis, run a new MC:
        for basis_name, basis_funcs in BASIS_SETS.items():
            mc_model = MCModel(
                market=market,
                option=option,
                n_steps=n_steps,
                n_paths=20000,
                normalize=False,
                seed=123,
                antithetic=True
            )
            # Overwrite the default basis functions:
            mc_model.regression.basis_functions = basis_funcs

            am_price = mc_model.price()
            results[basis_name].append(am_price)

    # Plot the results
    _plot_convergence(results, market, option)

def _plot_convergence(results: Dict[str, List[float]],
                      market: Market,
                      option: Option) -> None:
    plt.figure(figsize=(9, 6))
    steps = results["Steps"]

    # Plot the European
    plt.scatter(steps, results["European"], color='black', label='European (BS)', alpha=0.7)
    plt.plot(steps, results["European"], color='black', alpha=0.5)

    # Plot each American variant
    colors = ['red', 'blue', 'green', 'orange']
    for i, basis_name in enumerate(BASIS_SETS):
        plt.scatter(steps, results[basis_name], color=colors[i], label=basis_name, alpha=0.7)
        plt.plot(steps, results[basis_name], color=colors[i], alpha=0.5)

    plt.xscale('log')  # often steps are shown on a log scale
    plt.xlabel('Number of Steps')
    plt.ylabel('Option Price')
    plt.title(
        f"American Put @ S0={option.S0:.2f} Price vs. #Steps\n"
        f"(IR={market.int_rate*100:.1f}%, Vol={market.vol*100:.1f}%, "
        f"Div={market.dividend_yield:.1f}, T={option.T:.2f})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("american_put_convergence.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------------
#                              Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Demonstrates:
      1) MC and Trinomial pricing on a sample put,
      2) Plotting the exercise region,
      3) Running a convergence study with different n_steps.
    """
    try:
        # Example: a small Market/Option for demonstration
        market = Market(stock_price=1.0, int_rate=0.05, vol=0.2, dividend_yield=0.02)
        option = Option(S0=1.0, K=1.0, T=2.0, sigma=0.2, option_type='put')

        # ----- 1) MC & Trinomial Pricing -----
        methods = {
            'Monte Carlo': MCModel(
                market,
                option,
                n_steps=50,
                n_paths=30000,
                normalize=True,
                seed=42,
                antithetic=True,
                greek_samples=5
            ),
            'Trinomial': TrinomialModel(
                market,
                option,
                n_steps=200
            )
        }

        for name, model in methods.items():
            model_price = model.price()
            print(f"{name} Price: {model_price:.6f}")

            greeks = model.calc_greeks(shift=0.01)
            print(f"{name} Greeks:")
            for greek_name, value in greeks.items():
                print(f"  {greek_name}: {value:.6f}")
            print("")

        # European price (Black-Scholes) for reference
        european_price = black_scholes_price(market, option)
        print(f"European Price (BS): {european_price:.6f}\n")

        # ----- 2) Plot the Exercise Regions (Monte Carlo only) -----
        mc_model = methods['Monte Carlo']
        time_grid = np.linspace(0, option.T, mc_model.n_steps + 1)

        fig = plot_exercise_regions(
            stock_paths=mc_model.stock_paths,
            exercise_decisions=mc_model.exercise_decisions,
            time_grid=time_grid,
            option=option
        )
        plt.savefig("exercise_regions.png", dpi=200, bbox_inches="tight")
        plt.show()

        # ----- 3) Convergence study -----
        steps_list = [1, 2, 4, 8, 16, 32, 64]
        run_convergence_experiment(market, option, steps_list)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
