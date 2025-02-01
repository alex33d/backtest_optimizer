import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import logging
from typing import Iterable, Tuple, List, Dict, Any, Union
from itertools import combinations
from scipy.special import logit
from scipy.stats import norm, skew, kurtosis
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF
from metrics import *

try:
    matplotlib.use("TkAgg")
except:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("performance_analysis.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def pbo(M: np.ndarray, S: int) -> None:
    """
    Calculate and plot the probability of backtest overfitting (PBO).

    Args:
        M (np.ndarray): Matrix, where columns are experiments and rows are returns.
        S (int): Number of submatrices to form.

    Returns:
        None
    """
    Ms = np.array_split(M, S)
    cs_combinations = list(combinations(range(S), S // 2))

    lambdas = []
    is_sharpe = []
    oos_sharpe = []

    for c in cs_combinations:
        J = np.vstack([Ms[i] for i in c])
        J_bar = np.vstack([Ms[i] for i in range(S) if i not in c])

        R = np.apply_along_axis(annual_sharpe, 0, J)
        n_star = np.argmax(R)

        R_bar = np.apply_along_axis(annual_sharpe, 0, J_bar)
        omega_c = (R_bar <= R_bar[n_star]).mean()

        is_sharpe.append(R[n_star])
        oos_sharpe.append(R_bar[n_star])

        lambda_c = logit(omega_c)
        lambdas.append(lambda_c)

    lambdas = np.array(lambdas)
    lambdas = lambdas[np.isfinite(lambdas)]

    ecdf = ECDF(lambdas)
    PBO = ecdf(0)
    logging.info(f"Estimated PBO using statsmodels ECDF: {PBO}")

    sharpe_regression(is_sharpe, oos_sharpe)
    logging.info(
        "IS R* negative ratio: %f",
        len([x for x in is_sharpe if x < 0]) / len(is_sharpe),
    )
    logging.info(
        "OOS R* negative ratio: %f",
        len([x for x in oos_sharpe if x < 0]) / len(oos_sharpe),
    )


def monte_carlo(df: pd.DataFrame, n_simulations: int) -> pd.DataFrame:
    """
    Perform Monte Carlo simulations on a dataframe of prices.

    Args:
        df (pd.DataFrame): DataFrame containing the price data.
        n_simulations (int): Number of simulations to run.

    Returns:
        pd.DataFrame: DataFrame containing the simulated paths.
    """
    df["log_diff"] = np.log(df["close"]).diff().dropna()

    period = 100
    df["mu"] = df["log_diff"].rolling(period).mean()
    df["std"] = df["log_diff"].rolling(period).std()
    df["var"] = df["std"] ** 2
    df["drift"] = df["mu"] - 0.5 * df["var"]
    s_0 = df["close"].iloc[period - 1]
    df.dropna(inplace=True)

    def simulate(df: pd.DataFrame, i: int) -> pd.Series:
        epsilon = np.random.normal(size=len(df))
        returns = df["drift"] + df["std"] * epsilon

        S = np.zeros_like(returns)
        S[0] = s_0
        for t in range(1, len(S)):
            S[t] = S[t - 1] * np.exp(returns.iloc[t])
        return pd.Series(S, name=f"sim_{i}", index=df.index)

    simulation_arrays = Parallel(n_jobs=4)(
        delayed(simulate)(df, i) for i in range(n_simulations)
    )
    simulations = pd.concat(simulation_arrays, axis=1)

    return pd.concat([df["close"], simulations], axis=1)


def get_simulations_stats(
    simulation_type: str,
    init_df: pd.DataFrame,
    params_dict: Dict[str, Any],
    pl_function: callable,
    n_simulations: int,
    timeframe: str,
) -> None:
    """
    Generate and plot statistics from simulations.

    Args:
        simulation_type (str): Type of simulation ('monte_carlo' or other).
        init_df (pd.DataFrame): Initial dataframe with price data.
        params_dict (Dict[str, Any]): Parameters for the pl_function.
        pl_function (callable): Function to calculate performance metrics.
        n_simulations (int): Number of simulations to run.
        timeframe (str): Resampling timeframe.

    Returns:
        None
    """
    returns_list = []
    sharpe_list = []

    if simulation_type.lower() == "monte_carlo":
        df = monte_carlo(init_df, n_simulations)
        for i in range(n_simulations):
            path = df[f"sim_{i}"]
            sample_df = path.resample(timeframe).ohlc()
            sample_df["size_mult"] = 1
            sample_df["traded"] = True
            sample_df["ticker"] = "BTCUSDT"
            returns = pl_function(sample_df, params_dict)
            returns = returns.sum(axis=1)
            returns_list.append(returns)
    else:
        meboot_df = pd.read_csv(
            "BTCUSDT_spot_1m_meboot.csv",
            index_col=0,
            parse_dates=[0],
            on_bad_lines="skip",
        )
        init_price = init_df["close"].iloc[0]
        for col in meboot_df.columns:
            returns = meboot_df[col]
            cum_returns = returns.cumsum()
            prices = [init_price]
            for r in returns:
                new_price = prices[-1] * (1 + r)
                prices.append(new_price)
            path = pd.Series(prices[1:], index=returns.index)
            path.index = pd.to_datetime(path.index, errors="coerce")

            sample_df = path.resample(timeframe).ohlc()
            sample_df["size_mult"] = 1
            sample_df["traded"] = True
            sample_df["ticker"] = "BTCUSDT"
            returns = pl_function(sample_df, params_dict)
            returns = returns.sum(axis=1)
            returns_list.append(returns)

    fig, ax = plt.subplots(figsize=(10, 6))
    for returns in returns_list:
        ax.plot(returns.cumsum())
        sharpe_list.append(annual_sharpe(returns))
    ax.set_ylabel("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_title(f"{simulation_type} simulation results")
    ax.legend()
    plt.show()

    plt.hist(sharpe_list, alpha=0.5, color="blue", edgecolor="black")
    mean = np.mean(sharpe_list)
    percentile_5 = np.percentile(sharpe_list, 5)
    plt.axvline(
        mean, color="red", linestyle="dashed", linewidth=1, label=f"Mean: {mean:.2f}"
    )
    plt.axvline(
        percentile_5,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"5% Percentile: {percentile_5:.2f}",
    )
    plt.xlabel("Data Values")
    plt.ylabel("Frequency")
    plt.title("Sharpe distribution")
    plt.legend()
    plt.show()


def plot_scatter_sharpe(df: pd.DataFrame, strategy_name: str = "Strategy name") -> None:
    """
    Plot a scatter plot of Sharpe ratios for hyperparameters.

    Args:
        df (pd.DataFrame): DataFrame containing hyperparameters and Sharpe ratios.
        strategy_name (str): Name of the strategy.

    Returns:
        None
    """
    from matplotlib.colors import ListedColormap

    cols = [col for col in df.columns if col != "sharpe"]
    sharpe_ratios = df["sharpe"]

    colors = ["red", "green"]
    cmap = ListedColormap(colors)

    fig = plt.figure()
    plt.title(strategy_name)

    if len(cols) == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(
            df[cols[0]], df[cols[1]], c=sharpe_ratios, cmap=cmap, marker="o"
        )
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])

    elif len(cols) == 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            df[cols[0]],
            df[cols[1]],
            df[cols[2]],
            c=sharpe_ratios,
            cmap=cmap,
            marker="o",
        )
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_zlabel(df.columns[2])
        ax.view_init(elev=30, azim=45)
    else:
        logging.error("Unsupported number of dimensions.")
        return

    fig.colorbar(sc, ax=ax, label="Sharpe Ratio")
    plt.show()


def plot_performance_distributions(
    M: np.ndarray, actual_sharpe: float, actual_cagr: float
) -> None:
    """
    Plot distributions of Sharpe Ratios and CAGR.

    Args:
        M (np.ndarray): Matrix of returns.
        actual_sharpe (float): Actual Sharpe ratio.
        actual_cagr (float): Actual CAGR.

    Returns:
        None
    """
    sharpe_ratios = np.apply_along_axis(annual_sharpe, 0, M)
    cagr = np.apply_along_axis(annual_returns, 0, M)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(sharpe_ratios, bins=20, color="blue", alpha=0.7, label="Sharpe Ratios")
    axs[0].axvline(
        actual_sharpe,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Actual Sharpe: {actual_sharpe}",
    )
    axs[0].set_title("Distribution of Sharpe Ratios")
    axs[0].set_xlabel("Sharpe Ratio")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    axs[1].hist(cagr, bins=20, color="green", alpha=0.7, label="CAGR")
    axs[1].axvline(
        actual_cagr,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Actual CAGR: {actual_cagr}",
    )
    axs[1].set_title("Distribution of Annual Returns")
    axs[1].set_xlabel("R")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    plt.show()


def calculate_mcc(TP: int, TN: int, FP: int, FN: int) -> float:
    """
    Calculate the Matthews correlation coefficient (MCC).

    Args:
        TP (int): True Positives.
        TN (int): True Negatives.
        FP (int): False Positives.
        FN (int): False Negatives.

    Returns:
        float: The MCC value.
    """
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        return 0

    return numerator / denominator


def calculate_psr(returns: pd.DataFrame, benchmark_sr: float) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    Args:
        returns (np.ndarray): Array of returns.
        benchmark_sr (float): Benchmark Sharpe ratio.

    Returns:
        float: The PSR value.
    """
    T = len(returns) - 1
    sr = estimated_sharpe_ratio(returns)
    skewness = skew(returns)
    kurt = kurtosis(returns)
    numerator = sr - benchmark_sr
    denominator = np.sqrt((1 - skewness * sr + (kurt - 1) / 4.0 * sr**2) / T)
    z = numerator / denominator
    return norm.cdf(z)


def calculate_dsr(all_returns: pd.DataFrame, selected_returns: pd.DataFrame) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).

    Args:
        all_returns (pd.DataFrame): All returns.
        selected_returns (np.ndarray): Selected returns.

    Returns:
        float: The DSR value.
    """
    emc = 0.5772156649
    e = 2.718281828459045
    sharpe_std = np.std(estimated_sharpe_ratio(all_returns))
    N = num_independent_trials(all_returns)

    sharpe_star = sharpe_std * (
        (1 - emc) * norm.ppf(1 - 1 / N) + emc * norm.ppf(1 - e ** (-1) / N)
    )
    logging.info("Maximum expected daily sharpe: %f", sharpe_star)

    return probabilistic_sharpe_ratio(selected_returns, sharpe_star)


def calculate_smart_sharpe(returns: np.ndarray) -> float:
    """
    Calculate the Smart Sharpe Ratio.

    Args:
        returns (np.ndarray): Array of returns.

    Returns:
        float: The Smart Sharpe Ratio.
    """
    sr = annual_sharpe(returns)
    rho = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    return sr * (1 - rho)


def merton_brownian_motion_jump_diffusion(prices: np.ndarray) -> Dict[str, float]:
    """
    Fit Merton's Brownian Motion with Jump Diffusion model to the prices.

    Args:
        prices (np.ndarray): Array of prices.

    Returns:
        Dict[str, float]: Estimated parameters.
    """
    from scipy.optimize import minimize

    prices = np.array(prices)
    log_returns = np.log(prices[1:] / prices[:-1])

    def neg_log_likelihood(params, returns):
        mu, sigma, lambda_, mu_jump, sigma_jump = params
        n = len(returns)

        merton_part = returns - lambda_ * (np.exp(mu_jump + 0.5 * sigma_jump**2) - 1)
        ll_merton = np.sum(norm.logpdf(merton_part, loc=mu, scale=sigma))
        ll_jump = np.sum(norm.logpdf(returns, loc=mu_jump, scale=sigma_jump))

        return -(ll_merton + ll_jump)

    initial_params = np.array([0.0001, 0.01, 0.01, 0.0001, 0.01])
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(log_returns,),
        bounds=[(None, None), (0.001, 10), (0, 10), (None, None), (0.001, 10)],
    )
    estimated_params = result.x
    estimated_params = {
        "mu": estimated_params[0],
        "sigma": estimated_params[1],
        "lambda": estimated_params[2],
        "mu_J": estimated_params[3],
        "sigma_J": estimated_params[4],
    }
    logging.info("Estimated Parameters: %s", estimated_params)

    n_params = len(initial_params)
    n_obs = len(log_returns)
    log_likelihood = -result.fun

    AIC = 2 * n_params - 2 * log_likelihood
    BIC = n_params * np.log(n_obs) - 2 * log_likelihood

    logging.info("AIC: %f", AIC)
    logging.info("BIC: %f", BIC)
    return estimated_params


def generate_stochastic_process(
    mu: float,
    sigma: float,
    lambda_: float,
    mu_jump: float,
    sigma_jump: float,
    init_price: float,
    T: int = 3,
    with_chart: bool = False,
    n_paths: int = 100,
) -> pd.DataFrame:
    """
    Generate stochastic process using the GBMJD model.

    Args:
        mu (float): Drift.
        sigma (float): Volatility.
        lambda_ (float): Jump intensity.
        mu_jump (float): Jump mean.
        sigma_jump (float): Jump volatility.
        init_price (float): Initial price.
        T (int): Time horizon in years.
        with_chart (bool): Whether to plot the paths.
        n_paths (int): Number of paths.

    Returns:
        pd.DataFrame: DataFrame containing the simulated paths.
    """
    dt = 1
    N = int(T * 365 * 24 * 60)
    S = np.zeros((N, n_paths))
    S[0] = init_price

    np.random.seed(0)
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1, n_paths)
        dN = np.random.poisson(lambda_ * dt, n_paths)
        dJ = np.random.normal(mu_jump, sigma_jump, n_paths) * dN
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + dJ)

    if with_chart:
        plt.figure(figsize=(10, 6))
        for i in range(min(10, n_paths)):
            plt.plot(np.linspace(0, T, N), S[:, i])
        plt.xlabel("Time (Years)")
        plt.ylabel("Price")
        plt.title("Random Paths using GBMJD")
        plt.show()
    return pd.DataFrame(S)


def sharpe_regression(IS_sharpe: List, OOS_sharpe: List) -> None:
    """
    Perform linear regression between IS and OOS Sharpe ratios and plot the results.

    Args:
        IS_sharpe (List[float]): In-sample Sharpe ratios.
        OOS_sharpe (List[float]): Out-of-sample Sharpe ratios.

    Returns:
        None
    """
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame(data=[IS_sharpe, OOS_sharpe]).T
    df.dropna(inplace=True)

    IS_sharpe = df[0].values
    OOS_sharpe = df[1].values

    X = IS_sharpe.reshape(-1, 1)
    Y = OOS_sharpe
    reg = LinearRegression().fit(X, Y)

    plt.figure(figsize=(10, 6))
    plt.scatter(IS_sharpe, OOS_sharpe, label="Data Points")
    plt.plot(
        IS_sharpe,
        reg.predict(X),
        color="red",
        label=f"Linear Fit: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
    )
    plt.xlabel("In-Sample Sharpe Ratio")
    plt.ylabel("Out-of-Sample Sharpe Ratio")
    plt.title("Regression of OOS Sharpe on IS Sharpe")
    plt.legend()
    plt.show()

    logging.info("Coefficient: %f", reg.coef_[0])
    logging.info("Intercept: %f", reg.intercept_)


def estimated_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> float:
    """
    Calculate the estimated Sharpe ratio (risk-free rate = 0).

    Args:
        returns (Union[np.ndarray, pd.Series, pd.DataFrame]): Returns data.

    Returns:
        float: The estimated Sharpe ratio.
    """
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
    periods: int = 261,
    sr: Union[float, np.ndarray, pd.Series, pd.DataFrame] = None,
) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Calculate the annualized estimated Sharpe ratio (risk-free rate = 0).

    Args:
        returns (Union[np.ndarray, pd.Series, pd.DataFrame], optional): Returns data.
        periods (int): Number of periods in a year.
        sr (Union[float, np.ndarray, pd.Series, pd.DataFrame], optional): Sharpe ratio to be annualized.

    Returns:
        Union[float, np.ndarray, pd.Series, pd.DataFrame]: The annualized Sharpe ratio.
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    return sr * np.sqrt(periods)


def estimated_sharpe_ratio_stdev(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
    n: int = None,
    skewness: float = None,
    kurt: float = None,
    sr: float = None,
) -> Union[float, pd.Series]:
    """
    Calculate the standard deviation of the estimated Sharpe ratio.

    Args:
        returns (Union[np.ndarray, pd.Series, pd.DataFrame], optional): Returns data.
        n (int, optional): Number of return samples.
        skewness (float, optional): Skewness of returns.
        kurt (float, optional): Kurtosis of returns.
        sr (float, optional): Sharpe ratio.

    Returns:
        Union[float, pd.Series]: The standard deviation of the Sharpe ratio.
    """
    if type(returns) != pd.DataFrame:
        returns = pd.DataFrame(returns)

    if n is None:
        n = len(returns)
    if skewness is None:
        skewness = pd.Series(skew(returns), index=returns.columns)
    if kurt is None:
        kurt = pd.Series(kurtosis(returns, fisher=False), index=returns.columns)
    if sr is None:
        sr = ann_estimated_sharpe_ratio(returns, 365)

    sr_std = np.sqrt(
        (1 + (0.5 * sr**2) - (skewness * sr) + (((kurt - 3) / 4) * sr**2)) / (n - 1)
    )

    if type(returns) == pd.DataFrame:
        sr_std = pd.Series(sr_std, index=returns.columns)
    elif type(sr_std) not in (float, np.float64, pd.DataFrame):
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
    sr_benchmark: float = 0.0,
    sr: Union[float, np.ndarray, pd.Series, pd.DataFrame] = None,
    sr_std: Union[float, np.ndarray, pd.Series, pd.DataFrame] = None,
) -> Union[float, pd.Series]:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    Args:
        returns (Union[np.ndarray, pd.Series, pd.DataFrame], optional): Returns data.
        sr_benchmark (float): Benchmark Sharpe ratio.
        sr (Union[float, np.ndarray, pd.Series, pd.DataFrame], optional): Sharpe ratio.
        sr_std (Union[float, np.ndarray, pd.Series, pd.DataFrame], optional): Standard deviation of the Sharpe ratio.

    Returns:
        Union[float, pd.Series]: The PSR value.
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    psr = norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(
    returns: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
    sr_benchmark: float = 0.0,
    prob: float = 0.95,
    n: int = None,
    sr: float = None,
    sr_std: float = None,
) -> Union[float, pd.Series]:
    """
    Calculate the Minimum Track Record Length (minTRL).

    Args:
        returns (Union[np.ndarray, pd.Series, pd.DataFrame], optional): Returns data.
        sr_benchmark (float): Benchmark annual Sharpe ratio (which will be converted to daily unit)
        prob (float): Confidence level.
        n (int, optional): Number of return samples.
        sr (float, optional): Sharpe ratio.
        sr_std (float, optional): Standard deviation of the Sharpe ratio.

    Returns:
        Union[float, pd.Series]: The minTRL value.
    """

    sr_benchmark = sr_benchmark / np.sqrt(365)

    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns, sr=sr)

    min_trl = 1 + (sr_std**2 * (n - 1)) * (norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def num_independent_trials(
    trials_returns: pd.DataFrame = None, m: int = None, p: float = None
) -> int:
    """
    Calculate the number of independent trials.

    Args:
        trials_returns (pd.DataFrame, optional): All trial returns.
        m (int, optional): Number of total trials.
        p (float, optional): Average correlation between trials.

    Returns:
        int: Number of independent trials.
    """
    if m is None:
        m = trials_returns.shape[1]

    if p is None:
        corr_matrix = trials_returns.corr()
        p = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    n = p + (1 - p) * m
    return int(n) + 1


def expected_maximum_sr(
    trials_returns: pd.DataFrame = None,
    expected_mean_sr: float = 0.0,
    independent_trials: int = None,
    trials_sr_std: float = None,
) -> float:
    """
    Calculate the expected maximum Sharpe ratio.

    Args:
        trials_returns (pd.DataFrame, optional): All trial returns.
        expected_mean_sr (float): Expected mean Sharpe ratio.
        independent_trials (int, optional): Number of independent trials.
        trials_sr_std (float, optional): Standard deviation of the Sharpe ratios.

    Returns:
        float: The expected maximum Sharpe ratio.
    """
    emc = 0.5772156649

    if independent_trials is None:
        independent_trials = num_independent_trials(trials_returns)

    if trials_sr_std is None:
        trials_sr_std = estimated_sharpe_ratio(trials_returns).std()

    maxZ = (1 - emc) * norm.ppf(1 - 1.0 / independent_trials) + emc * norm.ppf(
        1 - 1.0 / (independent_trials * np.e)
    )
    return expected_mean_sr + (trials_sr_std * maxZ)


def deflated_sharpe_ratio(
    trials_returns: pd.DataFrame = None,
    returns_selected: pd.DataFrame = None,
    expected_mean_sr: float = 0.0,
    expected_max_sr: float = None,
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).

    Args:
        trials_returns (pd.DataFrame, optional): All trial returns.
        returns_selected (pd.Series): Selected returns.
        expected_mean_sr (float): Expected mean Sharpe ratio.
        expected_max_sr (float, optional): Expected maximum Sharpe ratio.

    Returns:
        float: The DSR value.
    """
    if expected_max_sr is None:
        expected_max_sr = expected_maximum_sr(trials_returns, expected_mean_sr)
        logging.info("Maximum expected daily sharpe: %f", expected_max_sr)
    return probabilistic_sharpe_ratio(
        returns=returns_selected, sr_benchmark=expected_max_sr
    )


def select_best_sr_strategies(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Select the top N strategies based on Sharpe ratio.

    Args:
        df (pd.DataFrame): DataFrame containing the strategies.
        N (int): Number of top strategies to select.

    Returns:
        pd.DataFrame: DataFrame containing the top N strategies.
    """
    sharpe_ratios = df.apply(annual_sharpe)
    sorted_strategies = sharpe_ratios.sort_values(ascending=False)
    top_strategies = sorted_strategies.index[:N]
    return df[top_strategies]


def comb_k_fold_cv_grouped(
    n_splits: int, n_test_splits: int, data_length: int
) -> Iterable[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Generate grouped indices for combinatorial K-Fold Cross Validation.

    Args:
        n_splits (int): Number of total splits.
        n_test_splits (int): Number of test splits.
        data_length (int): Total number of data points.

    Yields:
        Tuple[List[np.ndarray], List[np.ndarray]]: Grouped train and test indices.
    """
    indices = np.arange(data_length)
    fold_size = data_length // n_splits
    folds = [indices[i * fold_size : (i + 1) * fold_size] for i in range(n_splits)]

    for test_fold_indices in combinations(range(n_splits), n_test_splits):
        test_indices = np.hstack([folds[i] for i in test_fold_indices])
        train_indices = np.setdiff1d(indices, test_indices)

        grouped_train_indices = np.split(
            train_indices, np.where(np.diff(train_indices) != 1)[0] + 1
        )
        grouped_test_indices = [folds[i] for i in test_fold_indices]

        yield grouped_train_indices, grouped_test_indices


def run_stress_tests(returns: pd.DataFrame, params: Dict[str, Any] = None) -> None:
    """
    Run stress tests on the returns.

    Args:
        returns (pd.DataFrame): DataFrame containing returns.
        params (Dict[str, Any], optional): Parameters for the stress tests.

    Returns:
        None
    """
    if params is None:
        params = {}

    pbo_splits = params.get("pbo_splits", 10)
    annual_benchmark_sharpe = params.get("annual_benchmark_sharpe", 1)
    top_n_strategies = params.get("top_n_strategies", 5)

    returns = returns.T.drop_duplicates().T
    returns.columns = range(len(returns.columns))
    returns = returns.drop(
        [col for col in returns.columns if returns[col].nunique() == 1], axis=1
    )

    logging.info("---")
    pbo(np.array(returns), pbo_splits)
    logging.info("---")

    daily_benchmark_sharpe = annual_benchmark_sharpe / np.sqrt(365)
    logging.info("Daily benchmark Sharpe: %f", daily_benchmark_sharpe)

    top_strategies = select_best_sr_strategies(returns, top_n_strategies)
    for col in top_strategies.columns:
        logging.info(
            "Strategy %s annual sharpe: %f",
            col,
            annual_sharpe(top_strategies.loc[:, col]),
        )
        logging.info(
            "Strategy %s daily sharpe: %f",
            col,
            estimated_sharpe_ratio(top_strategies.loc[:, col]),
        )
    logging.info("---")

    logging.info(
        "Probabilistic sharpe - chances that top %d will show sharpe higher than %f",
        top_n_strategies,
        annual_benchmark_sharpe,
    )
    logging.info("Probabilistic sharpe v1")
    logging.info(calculate_psr(top_strategies, daily_benchmark_sharpe))
    logging.info("Probabilistic sharpe v2")
    logging.info(
        probabilistic_sharpe_ratio(top_strategies, sr_benchmark=daily_benchmark_sharpe)
    )
    logging.info("---")

    logging.info("Deflated sharpe")
    logging.info(calculate_dsr(returns, top_strategies))
    logging.info("Deflated sharpe v2")
    logging.info(
        deflated_sharpe_ratio(
            trials_returns=returns,
            returns_selected=top_strategies,
            expected_mean_sr=daily_benchmark_sharpe,
        )
    )


def rolling_sharpe_threshold(returns, window=126, threshold_sr=0.0):
    """
    Rolling test if Sharpe < threshold_sr. We do:
      H0: Sharpe == threshold_sr vs H1: Sharpe < threshold_sr
    or equivalently we can rewrite as testing mean - threshold_sr*(std dev) < 0.

    For simplicity, let's do a direct approach:
      - Compute sample mean, sample stdev in the window
      - Estimate NW standard error of the mean (we must decide how to incorporate the 'threshold_sr' factor).
      - Then form a test statistic:
          T = (Sharpe - threshold_sr) / SE_of_Sharpe
      - We'll produce the rolling Sharpe and also the "Sharpe threshold" needed for significance if you want to see it.

    This function returns a DataFrame with:
      'rolling_sharpe' : the rolling Sharpe
      'test_stat'      : t-like statistic for H1: SR < threshold_sr
      'pval'           : one-sided p-value for SR < threshold_sr
    """
    s_values = []
    t_values = []
    p_values = []
    p_values_alt = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            s_values.append(np.nan)
            t_values.append(np.nan)
            p_values.append(np.nan)
            p_values_alt.append(np.nan)
        else:
            window_data = returns.iloc[i - window : i]
            r = window_data.values

            # sample mean, stdev
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            # naive Sharpe
            sample_sr = mu / sigma

            # Alternative approach
            pval = probabilistic_sharpe_ratio(window_data, sr_benchmark=threshold_sr)

            s_values.append(sample_sr)
            p_values.append(pval)

    print(len(p_values), len(p_values_alt))
    df_out = pd.DataFrame(
        {
            "rolling_sharpe": s_values,
            "pval": p_values,
        },
        index=idx,
    )
    return df_out


def cusum_breaks(returns, h=0.01):
    """
    A simplistic CUSUM approach to detect structural breaks in the mean.
    h: threshold for detection (depends on scale of data).

    Returns a list of break indices.
    """
    s = 0
    indices = []
    for i in range(1, len(returns)):
        s = s + (returns.iloc[i] - returns.iloc[i - 1])
        if abs(s) > h:
            indices.append(returns.index[i])
            s = 0  # reset
    return indices


###############################################################################
# 2) Demo / Putting It All Together
###############################################################################


def plot_rolling_sharpe(returns, sharpe_threshold=1, window=180, alpha=0.1):
    daily_sharpe_threshold = round(1 / np.sqrt(365), 3)

    if isinstance(returns, pd.Series):
        returns_ser = returns
    elif isinstance(returns, pd.DataFrame):
        returns_ser = returns.iloc[:, 0]
    else:
        raise Exception("Supported returns format are Series or DataFrame")

    # 2) Rolling Sharpe test if SR < threshold (e.g., 0 or 1)
    # Here let's test if SR < 0. We'll produce a DataFrame
    df_sharpe_test = rolling_sharpe_threshold(
        returns_ser, window=window, threshold_sr=daily_sharpe_threshold
    )

    # Prepare data for plotting
    cumret = returns_ser.cumsum()

    ############################################################################
    # Plot the requested subplots
    ############################################################################
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # (1) Cumulative returns with breakpoints
    axes[0].plot(cumret, label="Cumulative Returns")
    axes[0].legend()

    # (2) Alpha statistics p-values
    # axes[1].plot(rolling_pvals, label='Rolling p-value (alpha>0)')
    # axes[1].axhline(0.05, color='r', linestyle='--', label='5% cutoff')
    # axes[1].set_ylim([0, 1])
    # axes[1].set_title("Rolling Alpha p-values (Newey-West, window=60)")
    # axes[1].legend()

    # (3) Rolling Sharpe with threshold
    axes[1].plot(
        df_sharpe_test["rolling_sharpe"] * np.sqrt(365), label="Rolling Sharpe"
    )
    # We can plot lines for "Sharpe = 0" if we want
    axes[1].axhline(
        sharpe_threshold,
        color="r",
        linestyle="--",
        label=f"SR={sharpe_threshold} threshold",
    )
    axes[1].set_title(
        f"Rolling Sharpe, Testing SR < {sharpe_threshold} with NW adjustment"
    )
    axes[1].legend()

    # (4) Sharpe test p-values
    axes[2].plot(df_sharpe_test["pval"], label=f"p-value (SR < {sharpe_threshold})")
    axes[2].axhline(alpha, color="r", linestyle="--", label=f"{alpha} cutoff")
    axes[2].set_ylim([0, 1])
    axes[2].set_title("Rolling Sharpe Test p-values (one-sided)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
