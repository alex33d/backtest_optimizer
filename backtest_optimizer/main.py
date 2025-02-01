import sys
import os
from datetime import timedelta

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the PYTHONPATH
sys.path.append(current_dir)

from pandas import DataFrame
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import math
from tqdm import tqdm
import optuna
from optuna.pruners import BasePruner
from multiprocessing import Pool
import itertools
from typing import Any, List, Dict, Union, Tuple, Optional, Set
import matplotlib.pyplot as plt
from collections.abc import Iterable
from functools import partial
import logging
import itertools as itt
import gc

from backtest_stress_tests import run_stress_tests
from metrics import *


class RepeatPruner(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """
        Prune trials that have duplicate parameters.

        Args:
            study (optuna.Study): The study object.
            trial (optuna.Trial): The trial object.

        Returns:
            bool: True if the trial should be pruned, False otherwise.
        """
        print("RepeatPruner is called.")
        trials = study.get_trials(deepcopy=False)
        params_list = [t.params for t in trials]
        if params_list.count(trial.params) > 1:
            print(
                f"Trial {trial.number} pruned due to duplicate parameters: {trial.params}"
            )
            return True
        return False


def calc_returns(args):
    calc_pl_func, params, train_data = args
    try:
        returns = calc_pl_func(train_data, params)
    except Exception as e:
        logging.info(
            f"PL calculation for stress tests failed for {params}, with error {e}"
        )
        returns = pd.Series()
    if not returns.empty:
        returns = returns.resample("D").sum()
    return returns


def create_objective(group_indices, params_dict, calc_pl_func, data_dict):
    """Create a standalone objective function that doesn't rely on instance variables."""

    def objective(trial):
        try:
            trial_params = {}
            for k, v in params_dict.items():
                if not isinstance(v, Iterable) or isinstance(v, (str, bytes)):
                    trial_params[k] = v
                elif all(isinstance(item, int) for item in v):
                    trial_params[k] = trial.suggest_categorical(k, v)
                elif any(isinstance(item, float) for item in v):
                    trial_params[k] = trial.suggest_categorical(k, v)
                else:
                    trial_params[k] = trial.suggest_categorical(k, v)

            current_params = trial.params
            existing_trials = trial.study.get_trials(deepcopy=False)
            completed_trials = [
                t
                for t in existing_trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            existing_params = [t.params for t in completed_trials]
            if current_params in existing_params:
                logging.info(
                    f"Pruning trial {trial.number} due to duplicate parameters: {trial_params}"
                )
                raise optuna.TrialPruned()

            sharpe, _ = combcv_pl(trial_params, group_indices, calc_pl_func, data_dict)

            if np.isnan(sharpe):
                logging.warning(f"Trial {trial.number} returned NaN Sharpe ratio.")
                return -1e6

            return sharpe

        except optuna.TrialPruned:
            raise  # Allow Optuna to handle pruning
        except Exception as e:
            logging.exception(f"Error in trial {trial.number}")
            return -1e6  # Assign a large negative penalty

    return objective


def combcv_pl(
    params: dict, group_indices: dict, calc_pl_func: callable, data_dict: dict
) -> tuple:
    final_returns = []
    for group_num, ticker_indices in group_indices.items():
        group_data = {}
        for ticker, indices in ticker_indices.items():
            df = data_dict.get(ticker)
            if df is None or df.empty:
                logging.warning(f"Ticker {ticker} has no data for group {group_num}.")
                continue
            group_data[ticker] = df.iloc[indices]
        if not group_data:
            logging.warning(f"No valid tickers in group {group_num}. Skipping.")
            continue
        returns = calc_pl_func(group_data, params)
        if returns is None or returns.empty:
            logging.warning(
                f"calc_pl_func returned empty returns for group {group_num}."
            )
            continue
        final_returns.append(returns)

    if not final_returns:
        logging.error("All groups returned empty or invalid returns.")
        return np.nan, final_returns  # Return NaN to indicate failure

    sharpe_ratios = []
    for r in final_returns:
        sharpe = annual_sharpe(r)
        if np.isnan(sharpe):
            logging.warning("Annual Sharpe ratio calculated as NaN.")
            continue
        sharpe_ratios.append(sharpe)

    if not sharpe_ratios:
        logging.error("All Sharpe ratios are NaN.")
        return np.nan, final_returns

    final_sharpe = np.nanmean(sharpe_ratios)
    return final_sharpe, final_returns


class ParameterOptimizer:

    def __init__(
        self, calc_pl: callable, save_path: str, save_file_prefix: str, n_jobs: int
    ):
        """
        Initialize the parameter optimizer.

        Args:
            calc_pl (callable): Function to calculate performance metrics.
            save_path: path for saving and loading results (folder)
        """
        self.calc_pl = calc_pl
        self.combcv_dict = {}
        self.params_dict = {}
        self.all_tested_params = []
        self.best_params_by_fold = {}
        self.backtest_paths = {}
        self.top_params_list = None
        self.current_group = None
        self.group_indices = None
        self.data_dict = {}
        self.save_path = save_path
        self.file_prefix = save_file_prefix
        self.n_jobs = n_jobs

    def check_datetime_index_integrity(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        Memory-optimized version of datetime index integrity checker.
        Uses sequential processing and minimal memory footprint.

        Args:
            data_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames with datetime indices.

        Returns:
            Tuple[bool, List[str]]: Tuple of (integrity_check_passed, error_messages)
        """
        error_messages = []

        if not data_dict:
            error_messages.append("The input dictionary is empty.")
            return False, error_messages

        frequencies = []
        expected_freq = None

        for ticker, df in data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                continue

            # Infer frequency from first few rows to save memory
            if len(df) > 1:
                try:
                    sample_size = min(len(df), 1000)
                    sample = df.index[:sample_size]
                    freq = pd.infer_freq(sample)
                    if freq:
                        frequencies.append(freq)
                except Exception:
                    continue

        # Determine expected frequency
        if frequencies:
            from collections import Counter

            freq_counter = Counter(frequencies)
            expected_freq = freq_counter.most_common(1)[0][0]
        else:
            # Default to 'B' if unable to infer
            expected_freq = "B"

        # Process DataFrames sequentially to conserve memory
        all_errors = []

        for ticker, df in data_dict.items():
            errors = self.check_single_dataframe(ticker, df)
            all_errors.extend(errors)

        return len(all_errors) == 0, all_errors

    def align_dataframes_to_max_index(self, data_dict: dict):
        # Find the maximum date range
        all_dates = pd.DatetimeIndex([])
        for df in data_dict.values():
            all_dates = all_dates.union(df.index)

        # Sort the dates to ensure they're in chronological order
        all_dates = all_dates.sort_values()

        for ticker, df in data_dict.items():
            # Reindex the DataFrame to the full date range
            aligned_df = df.reindex(all_dates)

            # If you want to forward fill a limited number of NaNs (e.g., 5 days), uncomment the next line
            # aligned_df = aligned_df.fillna(method='ffill', limit=5)

            data_dict[ticker] = aligned_df

        return data_dict

    def split_data(self, data_dict: dict, train_end: str):
        """
        Memory-optimized version of split_data using Dask that includes integrity checks
        and alignment while managing memory efficiently.

        Args:
            data_dict (dict): Dictionary containing the data.
            train_end (str): The end date for the training data.
        """
        logging.info(
            f"Splitting data into train and test sets with cutoff: {train_end}"
        )

        # Step 1: Check datetime index integrity and infer expected frequency
        error_messages = []
        frequencies = []

        for ticker, df in data_dict.items():
            try:
                # Ensure index is a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Perform index integrity checks
                errors = self.check_single_dataframe(ticker, df)
                if errors:
                    error_messages.extend(errors)

            except Exception as e:
                logging.warning(f"Error processing {ticker}: {str(e)}")
                continue

        # If any errors were found during index checks, log them and exit
        if error_messages:
            logging.warning(f"Data integrity check errors: {error_messages}")

        # Step 2: Align DataFrames to max index using memory-efficient approach
        try:
            data_dict = self.align_dataframes_to_max_index(data_dict)
        except Exception as e:
            logging.error(f"Failed to align DataFrames: {str(e)}")
            return

        # Convert train_end to a Timestamp once
        train_end_ts = pd.Timestamp(train_end)

        # Prepare directories to save individual ticker data
        train_dir = os.path.join(self.save_path, f"{self.file_prefix}_train_data")
        test_dir = os.path.join(self.save_path, f"{self.file_prefix}_test_data")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Process each DataFrame individually
        for ticker in tqdm(list(data_dict.keys()), desc="Processing tickers"):
            df = data_dict[ticker]
            if df.empty:
                continue
            try:
                # Ensure index is a DatetimeIndex and remove timezone
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Ensure train_end_ts is timezone-naive
                if (
                    isinstance(train_end_ts, pd.Timestamp)
                    and train_end_ts.tz is not None
                ):
                    train_end_ts = train_end_ts.tz_localize(None)
                elif not isinstance(train_end_ts, pd.Timestamp):
                    train_end_ts = pd.Timestamp(train_end_ts).tz_localize(None)

                # Sort the index to ensure chronological order
                df.sort_index(inplace=True)

                # Split the data into training and testing sets
                train_df = df.loc[df.index < train_end_ts]
                test_df = df.loc[df.index >= train_end_ts]

                # Downcast numerical columns to save memory
                numerical_cols = train_df.select_dtypes(
                    include=["float64", "int64"]
                ).columns
                for col in numerical_cols:
                    if train_df[col].dtype == "float64":
                        train_df[col] = train_df[col].astype("float32", copy=False)
                        test_df[col] = test_df[col].astype("float32", copy=False)
                    elif train_df[col].dtype == "int64":
                        train_df[col] = train_df[col].astype("int32", copy=False)
                        test_df[col] = test_df[col].astype("int32", copy=False)

                # Save the training and testing sets to disk in Parquet format
                train_file_path = os.path.join(train_dir, f"{ticker}.parquet")
                test_file_path = os.path.join(test_dir, f"{ticker}.parquet")

                train_df.to_parquet(train_file_path, index=True)
                test_df.to_parquet(test_file_path, index=True)

                # Clean up to free memory
                del df, train_df, test_df
                del data_dict[ticker]
                gc.collect()

            except Exception as e:
                logging.warning(f"Error processing {ticker}: {str(e)}")
                continue

        logging.info(
            f"Successfully split data into training and testing sets and saved to {self.save_path}"
        )

    def cpcv_generator(
        self, t_span: int, n: int, k: int, verbose: bool = True
    ) -> tuple:
        """
        Generate combinatorial purged cross-validation (CPCV) splits.

        Args:
            t_span (int): Total time span.
            n (int): Number of groups.
            k (int): Number of test groups.
            verbose (bool): Whether to print information about the splits.

        Returns:
            tuple: (is_test, paths, path_folds) arrays for CPCV.
        """
        group_num = np.arange(t_span) // (t_span // n)
        group_num[group_num == n] = n - 1

        test_groups = np.array(list(itt.combinations(np.arange(n), k))).reshape(-1, k)
        C_nk = len(test_groups)
        n_paths = C_nk * k // n

        if verbose:
            print("n_sim:", C_nk)
            print("n_paths:", n_paths)

        is_test_group = np.full((n, C_nk), fill_value=False)
        is_test = np.full((t_span, C_nk), fill_value=False)

        if k > 1:
            for k, pair in enumerate(test_groups):
                for i in pair:
                    is_test_group[i, k] = True
                    mask = group_num == i
                    is_test[mask, k] = True
        else:
            for k, i in enumerate(test_groups.flatten()):
                is_test_group[i, k] = True
                mask = group_num == i
                is_test[mask, k] = True

        path_folds = np.full((n, n_paths), fill_value=np.nan)

        for i in range(n_paths):
            for j in range(n):
                s_idx = is_test_group[j, :].argmax().astype(int)
                path_folds[j, i] = s_idx
                is_test_group[j, s_idx] = False

        paths = np.full((t_span, n_paths), fill_value=np.nan)

        for p in range(n_paths):
            for i in range(n):
                mask = group_num == i
                paths[mask, p] = int(path_folds[i, p])

        return (is_test, paths, path_folds)

    def load_data_from_parquet(self, data_type: str, tickers: list = None):
        """
        Load data from individual Parquet files into ParameterOptimizer.shared_data.

        Args:
            data_type (str): Specify 'train' or 'test' to load the corresponding data.
            tickers (list): List of tickers to load. If None, load all tickers.
        """
        if data_type not in ("train", "test"):
            raise ValueError("data_type must be 'train' or 'test'")

        dir_path = os.path.join(self.save_path, f"{self.file_prefix}_{data_type}_data")

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_path} does not exist.")

        logging.info(f"Loading {data_type} data from {dir_path}")

        if tickers is None:
            tickers = [f[:-8] for f in os.listdir(dir_path) if f.endswith(".parquet")]

        data_dict = {}
        for ticker in tqdm(tickers, desc="Loading tickers"):
            file_path = os.path.join(dir_path, f"{ticker}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                data_dict[ticker] = df
            else:
                logging.warning(f"File {file_path} does not exist.")

        logging.info(f"{data_type.capitalize()} data loaded successfully")

        return data_dict

    def generate_group_indices(self, data_dict: dict, is_train: bool):
        """
        Generate group indices for training or testing.

        Args:
            data_dict (dict): dictionary with data
            is_train (bool): Whether to generate indices for training or testing.

        Returns:
            dict: A dictionary mapping group numbers to indices.
        """
        group_indices = {}
        for ticker, df in data_dict.items():
            if ticker not in self.current_group:
                continue
            select_idx = self.current_group[ticker]["train"]
            if self.current_group[ticker]["test"] and not is_train:
                select_idx = self.current_group[ticker]["test"]
            for i, idx in enumerate(select_idx):
                if i not in group_indices:
                    group_indices[i] = {}
                group_indices[i][ticker] = idx  # Store indices instead of DataFrames
        return group_indices

    def plot_returns(self, data_dict, params: dict, save_returns=False):
        """
        Calculate out-of-sample Sharpe ratio and plot cumulative returns.

        Args:
            data_dict: dictionary with data in format {'ticker':DF,...}
            params (dict): Parameters for the performance calculation.
        """

        def plot_results(returns, save_file_name):
            metrics = calculate_metrics(returns)
            text = ""
            for metric, v in metrics.items():
                text += f"{metric}: {round(v, 2)}\n"

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(returns.cumsum())
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel("Cumulative Returns", fontsize=14)
            ax.set_title(f"{save_file_name}", fontsize=16)
            plt.savefig(self.save_path + save_file_name)
            plt.show()

        logging.info("Plotting returns")

        returns = self.calc_pl(data_dict, params)

        if isinstance(returns, pd.Series):
            if save_returns:
                returns.to_csv(self.save_path + self.file_prefix + "_returns.csv")
                logging.info(
                    f"Returns saved as {self.save_path + self.file_prefix + '_returns.csv'}"
                )
            plot_results(returns, self.file_prefix + "total_returns.png")
        elif isinstance(returns, dict):
            for returns_type, returns_series in returns.items():
                if save_returns:
                    returns_series.to_csv(
                        self.save_path
                        + self.file_prefix
                        + f"_{returns_type}"
                        + "_returns.csv",
                    )
                    logging.info(
                        f"Returns saved as {self.save_path + self.file_prefix + returns_type + '_returns.csv'}"
                    )
                plot_results(
                    returns_series, self.file_prefix + f"{returns_type}_returns.png"
                )
        else:
            raise Exception(
                "Wrong data type for plotting returns, accepted types are pd.Series and dict"
            )

    def create_combcv_dict(self, data_dict, n_splits: int, n_test_splits: int):
        """
        Create a dictionary for combinatorial cross-validation.

        Args:
            n_splits (int): Number of total splits.
            n_test_splits (int): Number of test splits.
        """

        def split_consecutive(arr):
            # Ensure the input is a NumPy array
            arr = np.asarray(arr)

            # Calculate the differences between adjacent elements
            diff = np.diff(arr)

            # Find where the difference is not 1 (i.e., where sequences break)
            split_points = np.where(diff != 1)[0] + 1

            # Use these split points to create chunks
            chunks = np.split(arr, split_points)

            return chunks

        total_comb = math.comb(n_splits, n_test_splits)
        if n_test_splits == 0 or n_splits == 0:
            logging.info(
                "Using the entire dataset as the training set with no validation groups."
            )
            self.combcv_dict[0] = {}
            for ticker, df in data_dict.items():
                self.combcv_dict[0][ticker] = {
                    "train": [np.arange(len(df))],
                    "test": None,
                }
        else:
            logging.info(
                f"Creating combinatorial train-val split, total_split: {n_splits}, out of which val groups: {n_test_splits}"
            )
            for ticker, df in data_dict.items():

                if len(df) > total_comb * 50:
                    data_length = len(df)
                    is_test, paths, path_folds = self.cpcv_generator(
                        data_length, n_splits, n_test_splits, verbose=False
                    )
                    self.backtest_paths[ticker] = paths
                    for combination_num in range(is_test.shape[1]):
                        if combination_num not in self.combcv_dict:
                            self.combcv_dict[combination_num] = {}

                        train_indices = np.where(~is_test[:, combination_num])[0]
                        test_indices = np.where(is_test[:, combination_num])[0]

                        train_indices = split_consecutive(train_indices)
                        test_indices = split_consecutive(test_indices)

                        self.combcv_dict[combination_num][ticker] = {
                            "train": train_indices,
                            "test": test_indices,
                        }

    def optimize(
        self,
        data_dict: {},
        n_splits: int,
        n_test_splits: int,
        params: dict,
        n_runs: int,
        best_trials_pct: float,
    ):
        """
        Optimize parameters using Optuna.

        Args:
            n_splits (int): number of total groups.
            n_test_splits (int): number of test groups.
            params (dict): Initial parameters for the optimization.
            n_runs (int): Number of optimization runs.
            best_trials_pct (float): Percentage of best trials to consider.
        """
        result = []
        self.params_dict = params.copy()
        all_tested_params = []
        if not data_dict:
            data_dict = self.load_data_from_parquet("train")

        self.create_combcv_dict(
            data_dict, n_splits=n_splits, n_test_splits=n_test_splits
        )

        # Create the objective function closure
        for fold_num, train_test_splits in self.combcv_dict.items():
            logging.info(f"Starting optimization for group: {fold_num}")
            self.current_group = train_test_splits

            # Generate indices once before optimization
            group_indices = self.generate_group_indices(data_dict, is_train=True)

            # Pass calc_pl and data_dict explicitly
            objective_func = create_objective(
                group_indices, self.params_dict, self.calc_pl, data_dict
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(multivariate=True),
            )

            study.optimize(objective_func, n_trials=n_runs, n_jobs=self.n_jobs)

            all_trials = sorted(
                [
                    trial
                    for trial in study.trials
                    if trial.value is not None
                    and trial.state == optuna.trial.TrialState.COMPLETE
                ],
                key=lambda trial: trial.value,
                reverse=True,
            )
            all_trials = [trial.params for trial in all_trials]
            for trial_params in all_trials:
                for key, value in self.params_dict.items():
                    trial_params.setdefault(key, value)
            all_tested_params.extend(all_trials)
            top_params = all_trials[: max(1, int(len(all_trials) * best_trials_pct))]
            logging.info(f"Top {best_trials_pct} param combinations are: {top_params}")

            group_indices = self.generate_group_indices(data_dict, is_train=False)
            for i, trial_params in enumerate(top_params):
                sharpe, returns_list = combcv_pl(
                    params=trial_params,
                    group_indices=group_indices,
                    calc_pl_func=self.calc_pl,
                    data_dict=data_dict,
                )
                if i == 0:
                    trial_params["fold_num"] = fold_num
                else:
                    trial_params["fold_num"] = np.nan
                trial_params["sharpe"] = sharpe
                result.append(trial_params)
                logging.info(f"Val performance: {trial_params}")

            logging.info(f"Best params: {result}")

            self.top_params_list = result
            self.all_tested_params = list(
                {frozenset(d.items()): d for d in all_tested_params}.values()
            )
            self.best_params_by_fold[fold_num] = top_params[0]

            if self.save_path is not None:
                param_df = pd.DataFrame(self.top_params_list).sort_values(
                    "sharpe", ascending=False
                )
                param_df.to_csv(
                    self.save_path + self.file_prefix + "top_params.csv", index=False
                )

                all_tested_params_df = pd.DataFrame(self.all_tested_params)
                all_tested_params_df.to_csv(
                    self.save_path + self.file_prefix + "all_tested_params.csv",
                    index=False,
                )
                logging.info(f"Interim optimization results saved to {self.save_path}")

    def load_best_params(self, file_name: str = None, params: dict = None):
        """
        Load the best parameters from a file or dictionary.

        Args:
            file_name (str, optional): File name to load parameters from.
            params (dict, optional): Dictionary of parameters.
        """
        if params is not None:
            self.top_params_list = params
        elif file_name is not None:
            self.top_params_list = pd.read_csv(file_name)

    def reconstruct_equity_curves(self):
        """
        Reconstruct equity curves based on the best parameters.
        """

        data_dict = self.load_data_from_parquet(data_type="train")

        logging.info("Reconstructing val equity curves")

        arrays = list(self.backtest_paths.values())
        num_columns = arrays[0].shape[1]
        if not all(arr.shape[1] == num_columns for arr in arrays):
            raise Exception("Tickers have different number of backtest paths")

        for col in range(num_columns):
            unique_values_set = set(np.unique(arrays[0][:, col]))
            for arr in arrays[1:]:
                if unique_values_set != set(np.unique(arr[:, col])):
                    raise Exception(
                        "Tickers have different parameter folds within same backtest path number"
                    )

        n_paths = num_columns
        tmp_dict = {}
        final_metrics = []
        final_returns = []
        for path_num in tqdm(range(n_paths), desc="Path num"):
            logging.info(f"Starting for path {path_num}")
            path_returns = []
            unique_folds = np.unique(arrays[0][:, path_num])
            for fold in unique_folds:
                logging.info(f"Starting for fold {fold}")
                fold = int(fold)
                params = self.best_params_by_fold[fold]
                for ticker, path_array in self.backtest_paths.items():
                    test_indices = np.where(path_array[:, path_num] == fold)[0]
                    tmp_dict[ticker] = data_dict[ticker].iloc[test_indices]
                returns = self.calc_pl(tmp_dict, params)
                path_returns.append(returns)

            path_returns = pd.concat(path_returns)
            final_returns.append(path_returns)
            metrics = calculate_metrics(path_returns)
            final_metrics.append(metrics)

        final_metrics = pd.DataFrame(final_metrics).mean().to_dict()

        text = ""
        for metric, v in final_metrics.items():
            text += f"Mean {metric}: {round(v, 2)}\n"

        fig, ax = plt.subplots(figsize=(12, 8))
        for returns in final_returns:
            ax.plot(returns.resample("D").sum().cumsum())

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Cumulative Returns", fontsize=14)
        ax.set_title("Cumulative Returns Over Time", fontsize=16)
        plt.savefig(self.save_path + f"{self.file_prefix}_CombCV_equity_curves.png")
        plt.show()

    def run_stress_tests(self, data_dict, num_workers=5):
        """
        Run stress tests on the best parameter sets.
        """
        logging.info(f"Running stress tests, num_workers: {num_workers}")

        # Create local references to necessary data

        all_tested_params = self.all_tested_params.copy()
        calc_pl = self.calc_pl  # Local reference to avoid pickling self

        with Pool(processes=num_workers) as pool:
            # Create arguments for the calc_returns function
            args = [(calc_pl, params, data_dict) for params in all_tested_params]
            results = list(
                tqdm(
                    pool.imap(calc_returns, args),
                    total=len(all_tested_params),
                    desc="Calculating individual returns",
                )
            )

        # Filter out empty DataFrames
        results = [r for r in results if not r.empty]

        if results:
            result_df = pd.concat(results, axis=1).dropna()
            run_stress_tests(result_df)

    def calculate_wcss(self, data: np.ndarray, max_clusters: int) -> list:
        """
        Calculate the within-cluster sum of squares (WCSS) for clustering.

        Args:
            data (np.ndarray): The data to be clustered.
            max_clusters (int): The maximum number of clusters.

        Returns:
            list: The WCSS for each number of clusters.
        """
        wcss = []
        for n_clusters in range(1, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(data)
            centroids = [
                data[cluster_labels == i].mean(axis=0) for i in range(n_clusters)
            ]
            wcss.append(
                sum(
                    np.linalg.norm(data[cluster_labels == i] - centroids[i]) ** 2
                    for i in range(n_clusters)
                )
            )
        return wcss

    def find_optimal_clusters(
        self, param_matrix_scaled: np.ndarray, max_clusters: int
    ) -> int:
        """
        Find the optimal number of clusters using the elbow method.

        Args:
            param_matrix_scaled (np.ndarray): The scaled parameter matrix.
            max_clusters (int): The maximum number of clusters.

        Returns:
            int: The optimal number of clusters.
        """
        wcss = self.calculate_wcss(param_matrix_scaled, max_clusters)

        first_derivative = np.diff(wcss)
        second_derivative = np.diff(first_derivative)
        elbow_point = np.argmin(second_derivative) + 2  # +2 to correct the index offset

        plt.figure(figsize=(10, 7))
        plt.plot(range(1, max_clusters + 1), wcss, marker="o")
        plt.axvline(x=elbow_point, color="r", linestyle="--")
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.show()

        return elbow_point

    def cluster_and_aggregate(self, min_sharpe=None) -> dict:
        """
        Cluster parameter sets and aggregate the best parameters.

        Returns:
            dict: The aggregated best parameter set.
        """

        logging.info("Starting clustering")

        if isinstance(self.top_params_list, list):
            param_df = pd.DataFrame(self.top_params_list)
        elif isinstance(self.top_params_list, pd.DataFrame):
            param_df = self.top_params_list
            self.top_params_list = self.top_params_list.to_dict("records")
        else:
            raise Exception(
                "Wrong data format for top params, accepted formats are list/DataFrame"
            )

        if min_sharpe:
            param_df = param_df[param_df["sharpe"] > min_sharpe]
        param_df = param_df.drop(columns=["sharpe"]).dropna(axis=1)

        max_clusters = min(max(3, len(param_df) // 3), len(param_df))
        if max_clusters > 2:
            logging.info(
                f"Starting clustering with max clusters: {max_clusters}, len of param set {len(param_df)}"
            )

            column_transformer = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        StandardScaler(),
                        make_column_selector(dtype_include=np.number),
                    ),
                    (
                        "cat",
                        OneHotEncoder(),
                        make_column_selector(dtype_exclude=np.number),
                    ),
                ],
                remainder="passthrough",
            )

            param_matrix_scaled = column_transformer.fit_transform(param_df)
            best_n_clusters = (
                self.find_optimal_clusters(param_matrix_scaled, max_clusters)
                if max_clusters < len(param_df)
                else max_clusters
            )
            logging.info(f"Optimal number of clusters: {best_n_clusters}")
            clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
            cluster_labels = clustering.fit_predict(param_matrix_scaled)

            clustered_params = {i: [] for i in range(best_n_clusters)}
            for param, cluster in zip(self.top_params_list, cluster_labels):
                clustered_params[cluster].append(param)

            best_cluster = max(
                clustered_params.keys(),
                key=lambda c: np.mean([p["sharpe"] for p in clustered_params[c]]),
            )
            best_cluster_params = clustered_params[best_cluster]
            logging.info(f"Best cluster: {best_cluster_params}")
            best_param_set = self.aggregate_params(best_cluster_params)
        else:
            logging.info(
                f"Len of params set less than 3, choosing best params out of 3"
            )
            best_param_set = pd.DataFrame(self.top_params_list).iloc[0].to_dict()
        logging.info(f"Best params: {best_param_set}")
        return best_param_set

    def param_to_vector(self, param_set: dict) -> list:
        """
        Convert a parameter set to a vector.

        Args:
            param_set (dict): The parameter set.

        Returns:
            list: The parameter vector.
        """
        vector = []
        for key in sorted(param_set.keys()):
            value = param_set[key]
            if isinstance(value, list):
                vector.extend(value)
            else:
                vector.append(value)
        return vector

    def aggregate_params(self, params_list: list) -> dict:
        """
        Aggregate parameters by computing the mean of numerical values and the most frequent value of categorical values.

        Args:
            params_list (list): List of parameter sets.

        Returns:
            dict: The aggregated parameter set.
        """
        aggregated = {}
        for key in params_list[0].keys():
            values = [param[key] for param in params_list]
            if isinstance(values[0], list):
                aggregated[key] = list(np.mean(values, axis=0))
            elif isinstance(values[0], bool):
                aggregated[key] = max(set(values), key=values.count)
            elif isinstance(values[0], (int, float, np.number)):
                aggregated[key] = np.mean(values)
            else:
                aggregated[key] = max(set(values), key=values.count)
        return aggregated

    def read_saved_params(self):

        logging.info("Loading saved params")

        top_params_df = pd.read_csv(
            self.save_path + self.file_prefix + "top_params.csv"
        )
        self.top_params_list = top_params_df.drop(columns=["fold_num"])
        self.all_tested_params = pd.read_csv(
            self.save_path + self.file_prefix + "all_tested_params.csv"
        ).to_dict("records")

        top_params_list = top_params_df.dropna(subset="fold_num").to_dict("records")
        for tp in top_params_list:
            self.best_params_by_fold[tp["fold_num"]] = tp

        logging.info("Params loaded")

    def _plot_and_save(self, returns, metrics, param_set, index):
        """Plot and save results for a single parameter combination."""
        fig, (ax, text_ax) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot the cumulative returns
        ax.plot(returns.cumsum())
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Cumulative Returns", fontsize=14)
        ax.set_title(
            f"Cumulative Returns Over Time (Combination {index + 1})", fontsize=16
        )

        # Prepare text for metrics and parameters
        text = "Metrics:\n"
        for metric, v in metrics.items():
            text += f"{metric}: {round(v, 2)}, "
        text += "\nParameters:\n"
        for param, value in param_set.items():
            text += f"{param}: {value}, "

        # Display text below the chart
        text_ax.axis("off")
        text_ax.text(
            0.5,
            1.0,
            text,
            ha="center",
            va="top",
            fontsize=10,
            wrap=True,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{self.save_path}Equity_curve_combination_{index + 1}.png")
        plt.close(fig)

        logging.info(f"Saved plot for combination {index + 1}")

    def plot_multiple_param_combinations(
        self,
        data_dict: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ):
        """
        Calculate out-of-sample Sharpe ratio, plot cumulative returns for all combinations on a single chart,
        and create a table with metrics and parameter sets.

        Args:
            data_dict (Dict[str, pd.DataFrame]): Data for processing.
            params (Dict[str, Any]): Parameters for the performance calculation. May contain lists of values.
        """
        logging.info("Processing parameter combinations")

        # Generate all combinations of parameters
        param_names = list(params.keys())
        param_combinations = list(
            itertools.product(
                *[
                    params[name] if isinstance(params[name], list) else [params[name]]
                    for name in param_names
                ]
            )
        )
        total_combinations = len(param_combinations)
        logging.info(f"Total parameter combinations to process: {total_combinations}")

        # Prepare the partial function for processing
        partial_process = partial(
            process_combination,
            data_dict=data_dict,
            param_names=param_names,
            calc_pl=self.calc_pl,
            calculate_metrics=calculate_metrics,
        )

        # Choose between multiprocessing and single-process execution
        if self.n_jobs > 1:
            # Use multiprocessing for parallel execution
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(partial_process, param_combinations),
                        total=total_combinations,
                    )
                )
        else:
            # Single-process execution
            results = list(
                tqdm(
                    (partial_process(combo) for combo in param_combinations),
                    total=total_combinations,
                )
            )

        # Filter out None results (from errors)
        valid_results = [r for r in results if r is not None]
        successful_combinations = len(valid_results)
        logging.info(
            f"Processed {successful_combinations} out of {total_combinations} combinations successfully"
        )

        if not valid_results:
            logging.warning("No valid results to process. Exiting.")
            return

        # Create DataFrame for metrics and parameters
        df = pd.DataFrame(
            [
                {**{"Combination": i + 1}, **result["metrics"], **result["params"]}
                for i, result in enumerate(valid_results)
            ]
        )

        # Sort by Sharpe ratio if available
        if "sharpe" in df.columns:
            df_sorted = df.sort_values(by="sharpe", ascending=False)
        else:
            df_sorted = df

        # Save metrics and parameters as CSV
        metrics_csv_path = os.path.join(
            self.save_path, f"{self.file_prefix}Metrics_and_Parameters.csv"
        )
        df_sorted.to_csv(metrics_csv_path, index=False)
        logging.info(f"Metrics and parameters saved to {metrics_csv_path}")

        # Plot cumulative returns
        self._plot_cumulative_returns(valid_results)

    def _plot_cumulative_returns(self, results: List[Dict[str, Any]]):
        """
        Plot cumulative returns for all combinations.

        Args:
            results (List[Dict[str, Any]]): List of processed results.
        """
        plt.figure(figsize=(12, 8))
        for i, result in enumerate(results, 1):
            returns = result["returns"]
            plt.plot(returns.index, returns.cumsum(), label=f"Combination {i}")

        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cumulative Returns", fontsize=14)
        plt.title("Cumulative Returns Over Time (All Combinations)", fontsize=16)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        # Save the cumulative returns plot
        plot_path = os.path.join(
            self.save_path, f"{self.file_prefix}All_Equity_Curves.png"
        )
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Cumulative returns plot saved to {plot_path}")

    def check_single_dataframe(self, ticker: str, df: pd.DataFrame) -> List[str]:
        """
        Checks the integrity of a single DataFrame's datetime index.

        Args:
            ticker (str): The identifier for the DataFrame.
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            List[str]: List of error messages for this DataFrame.
        """
        errors = []

        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append(f"DataFrame for {ticker} does not have a DatetimeIndex.")
            return errors

        if df.index.has_duplicates:
            errors.append(f"DataFrame for {ticker} contains duplicate timestamps.")

        if not df.index.is_monotonic_increasing:
            errors.append(f"Index for {ticker} is not monotonically increasing.")

        if len(df) > 1:
            try:
                # Calculate the most common time difference
                diffs = df.index.to_series().diff().dropna()
                expected_timedelta = diffs.mode()[0]

                if expected_timedelta.total_seconds() == 0:
                    errors.append(
                        f"Invalid zero time difference detected for {ticker}."
                    )

                # Check for gaps in the index based on the most common difference
                gaps = diffs > expected_timedelta
                if gaps.any():
                    gap_count = gaps.sum()
                    max_gap = diffs[gaps].max()
                    errors.append(
                        f"DataFrame for {ticker} has {gap_count} gaps in its index. "
                        f"Largest gap is {max_gap}. Expected interval is {expected_timedelta}."
                    )

            except Exception as e:
                errors.append(
                    f"Could not analyze time differences for {ticker}: {str(e)}"
                )

        return errors


def process_combination(
    combination: Tuple[Any, ...],
    data_dict: Dict[str, pd.DataFrame],
    param_names: List[str],
    calc_pl: Any,
    calculate_metrics: Any,
) -> Optional[Dict[str, Any]]:
    """
    Process a single parameter combination and return the results.

    Args:
        combination (Tuple[Any, ...]): A tuple of parameter values.
        data_dict (Dict[str, pd.DataFrame]): Data for processing.
        param_names (List[str]): Names of the parameters.
        calc_pl (Callable): Function to calculate returns.
        calculate_metrics (Callable): Function to calculate metrics.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing parameters, returns, and metrics, or None if failed.
    """
    param_set = dict(zip(param_names, combination))

    try:
        # Calculate returns and metrics
        returns = calc_pl(data_dict, param_set)
        if isinstance(returns, dict):
            returns = returns["Total"]
        metrics = calculate_metrics(returns)

        return {"params": param_set, "returns": returns, "metrics": metrics}
    except Exception as e:
        logging.error(f"Error processing combination {param_set}: {str(e)}")
        return None


class ImmutableDataFrame:
    """
    A thread-safe immutable wrapper for pandas DataFrame.
    Prevents any modifications to the underlying data while allowing read operations.
    """

    def __init__(self, data: Union[DataFrame, Dict, List]):
        if isinstance(data, DataFrame):
            self._df = data  # Do not copy the DataFrame
        else:
            self._df = DataFrame(data)

        # Freeze the DataFrame by making it immutable
        self._df.flags.writeable = False

    def __getattr__(self, name: str) -> Any:
        """
        Delegate read-only operations to the underlying DataFrame.
        Block any operations that could modify the DataFrame.
        """
        forbidden_methods = {
            "iloc",
            "loc",
            "at",
            "iat",  # Direct access modifiers
            "drop",
            "drop_duplicates",
            "dropna",  # Data removal
            "fillna",
            "ffill",
            "bfill",
            "replace",
            "update",  # Data modification
            "set_index",
            "reset_index",  # Index modification
            "assign",
            "insert",
            "append",  # Data addition
        }

        if name in forbidden_methods:
            raise AttributeError(
                f"'{name}' operation is not allowed on ImmutableDataFrame"
            )

        attr = getattr(self._df, name)

        if callable(attr):

            def wrapped_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, DataFrame):
                    return ImmutableDataFrame(result)
                return result

            return wrapped_method
        return attr

    def __repr__(self) -> str:
        return f"ImmutableDataFrame(\n{self._df.__repr__()})"

    def __str__(self) -> str:
        return self._df.__str__()

    # Safe read-only operations
    def head(self, n: int = 5) -> "ImmutableDataFrame":
        return ImmutableDataFrame(self._df.head(n))

    def tail(self, n: int = 5) -> "ImmutableDataFrame":
        return ImmutableDataFrame(self._df.tail(n))

    def copy(self) -> "ImmutableDataFrame":
        return ImmutableDataFrame(self._df.copy())

    @property
    def values(self):
        return self._df.values

    @property
    def columns(self):
        return self._df.columns

    @property
    def index(self):
        return self._df.index

    def to_dict(self) -> Dict:
        return self._df.to_dict()

    def to_pandas(self) -> pd.DataFrame:
        """Return a shallow copy of the DataFrame for safe modifications."""
        df_copy = self._df.copy(deep=False)  # Create a shallow copy
        df_copy.values.flags.writeable = True  # Allow modifications
        return df_copy
