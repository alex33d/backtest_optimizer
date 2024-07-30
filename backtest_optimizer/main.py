import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the PYTHONPATH
sys.path.append(current_dir)

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import math

import optuna
from optuna.pruners import BasePruner

import matplotlib.pyplot as plt
from itertools import combinations
from collections.abc import Iterable
import logging
import matplotlib
import itertools as itt

from metrics import *
from backtest_stress_tests import run_stress_tests

try:
    matplotlib.use('TkAgg')
except:
    pass

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log", mode='w'),
        logging.StreamHandler()
    ]
)


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
            print(f"Trial {trial.number} pruned due to duplicate parameters: {trial.params}")
            return True
        return False


class ParameterOptimizer:
    def __init__(self, calc_pl: callable):
        """
        Initialize the parameter optimizer.

        Args:
            calc_pl (callable): Function to calculate performance metrics.
        """
        self.calc_pl = calc_pl
        self.combcv_dict = {}
        self.params_dict = {}
        self.all_tested_params = []
        self.best_params_by_fold = {}
        self.backtest_paths = {}
        self.top_params_list = None
        self.current_group = None
        self.train_data = {}
        self.test_data = {}

    def split_data(self, data_dict: dict, train_end: str):
        """
        Split data into training and testing sets based on the specified date.

        Args:
            data_dict (dict): Dictionary containing the data.
            train_end (str): The end date for the training data.
        """
        logging.info(f'Splitting data to train-test, cutoff: {train_end}')
        self.train_data = {}
        self.test_data = {}

        for ticker, df in data_dict.items():
            train_df = df.loc[:train_end].copy()
            if not train_df.empty:
                self.train_data[ticker] = train_df

            test_df = df.loc[train_end:].copy()
            if not train_df.empty:
                self.test_data[ticker] = test_df

    def align_dfs(self, dfs_dict: dict) -> dict:
        """
        Align DataFrames by reindexing them to a common date range.

        Args:
            dfs_dict (dict): Dictionary of DataFrames.

        Returns:
            dict: Dictionary of aligned DataFrames.
        """
        # Find the minimum and maximum datetime index across all DataFrames
        min_index = min(df.index.min() for df in dfs_dict.values())
        max_index = max(df.index.max() for df in dfs_dict.values())

        # Determine the frequency from the DataFrames
        inferred_freqs = [pd.infer_freq(df.index) for df in dfs_dict.values() if pd.infer_freq(df.index) is not None]
        if not inferred_freqs:
            raise ValueError("No valid frequency could be inferred from the DataFrames' indices.")

        # Use the most common frequency
        freq = max(set(inferred_freqs), key=inferred_freqs.count)

        # Create a new date range from the minimum to the maximum datetime index with the inferred frequency
        new_index = pd.date_range(start=min_index, end=max_index, freq=freq)

        # Reindex each DataFrame to the new date range, filling missing rows with NaN
        aligned_dfs_dict = {ticker: df.reindex(new_index) for ticker, df in dfs_dict.items()}

        return aligned_dfs_dict

    def cpcv_generator(self, t_span: int, n: int, k: int, verbose: bool = True) -> tuple:
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
            print('n_sim:', C_nk)
            print('n_paths:', n_paths)

        is_test_group = np.full((n, C_nk), fill_value=False)
        is_test = np.full((t_span, C_nk), fill_value=False)

        if k > 1:
            for k, pair in enumerate(test_groups):
                for i in pair:
                    is_test_group[i, k] = True
                    mask = (group_num == i)
                    is_test[mask, k] = True
        else:
            for k, i in enumerate(test_groups.flatten()):
                is_test_group[i, k] = True
                mask = (group_num == i)
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
                mask = (group_num == i)
                paths[mask, p] = int(path_folds[i, p])

        return (is_test, paths, path_folds)

    def combcv_pl(self, params: dict, is_train: bool) -> tuple:
        """
        Calculate performance metrics using combinatorial cross-validation.

        Args:
            params (dict): Parameters for the performance calculation.
            is_train (bool): Whether the data is for training or testing.

        Returns:
            tuple: (final_sharpe, final_returns) calculated metrics.
        """
        final_returns = []
        group_dict = {}
        for ticker, df in self.train_data.items():
            if ticker not in self.current_group:
                continue
            df = df.copy()
            df['ticker'] = ticker
            select_idx = self.current_group[ticker]['train']
            if self.current_group[ticker]['test'] and not is_train:
                select_idx = self.current_group[ticker]['test']
            for i, idx in enumerate(select_idx):
                if i not in group_dict:
                    group_dict[i] = {}
                new_df = df.iloc[idx]
                if not new_df.empty:
                    group_dict[i][ticker] = new_df

        for group_num, group_data in group_dict.items():
            returns = self.calc_pl(group_data, params)
            final_returns.append(returns)

        sharpe_ratios = [annual_sharpe(returns) for returns in final_returns if not returns.empty]
        final_sharpe = np.nanmean(sharpe_ratios)
        return final_sharpe, final_returns

    def calc_oos_sharpe(self, params: dict):
        """
        Calculate out-of-sample Sharpe ratio and plot cumulative returns.

        Args:
            params (dict): Parameters for the performance calculation.
        """
        returns = self.calc_pl(self.test_data, params)
        metrics = calculate_metrics(returns)
        text = ''
        for metric, v in metrics.items():
            text += f'{metric}: {round(v, 2)}\n'

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(returns.cumsum())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Returns', fontsize=14)
        ax.set_title('Cumulative Returns Over Time', fontsize=16)
        plt.show()

    def create_combcv_dict(self, n_splits: int, n_test_splits: int):
        """
        Create a dictionary for combinatorial cross-validation.

        Args:
            n_splits (int): Number of total splits.
            n_test_splits (int): Number of test splits.
        """
        total_comb = math.comb(n_splits, n_test_splits)
        if n_test_splits == 0 or n_splits == 0:
            logging.info('Using the entire dataset as the training set with no validation groups.')
            self.combcv_dict[0] = {}
            for ticker, df in self.train_data.items():
                self.combcv_dict[0][ticker] = {"train": [np.arange(len(df))], "test": None}
        else:
            logging.info(
                f'Creating combinatorial train-val split, total_split: {n_splits}, out of which val groups: {n_test_splits}')
            for ticker, df in self.train_data.items():

                if not df.empty and len(df) > total_comb * 50:
                    data_length = len(df)
                    is_test, paths, path_folds = self.cpcv_generator(data_length, n_splits, n_test_splits, verbose=False)
                    self.backtest_paths[ticker] = paths
                    for combination_num in range(is_test.shape[1]):
                        if combination_num not in self.combcv_dict:
                            self.combcv_dict[combination_num] = {}

                        train_indices = np.where(~is_test[:, combination_num])[0]
                        test_indices = np.where(is_test[:, combination_num])[0]

                        self.combcv_dict[combination_num][ticker] = {
                            "train": [train_indices],
                            "test": [test_indices]
                        }

    def optimize(self, params: dict, n_jobs: int, n_runs: int, best_trials_pct: float, save_file_name: str = None):
        """
        Optimize parameters using Optuna.

        Args:
            params (dict): Initial parameters for the optimization.
            n_jobs (int): Number of parallel jobs.
            n_runs (int): Number of optimization runs.
            best_trials_pct (float): Percentage of best trials to consider.
            save_file_name (str, optional): File name to save the results.
        """
        result = []
        self.params_dict = params.copy()

        for fold_num, train_test_splits in self.combcv_dict.items():
            logging.info(f'Starting optimization for group: {fold_num}')
            self.current_group = train_test_splits
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(multivariate=True))
            study.optimize(self.objective, n_trials=n_runs, n_jobs=n_jobs)
            top_trials = sorted([trial for trial in study.trials if
                                 trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE],
                                key=lambda trial: trial.value, reverse=True)
            top_trials = [trial.params for trial in top_trials]
            for params in top_trials:
                for key, value in self.params_dict.items():
                    params.setdefault(key, value)
            self.all_tested_params.extend(top_trials)
            top_params = top_trials[:max(1, int(len(top_trials) * best_trials_pct))]
            logging.info(f'Top {best_trials_pct} param combinations are: {top_params}')
            self.best_params_by_fold[fold_num] = top_params[0]

            for i, params in enumerate(top_params):
                sharpe, returns_list = self.combcv_pl(params, is_train=False)
                params['sharpe'] = sharpe
                result.append(params)
                logging.info(f'Val perfomance: {params}')

        logging.info(f'Best params: {result}')
        self.top_params_list = result
        self.all_tested_params = list({frozenset(d.items()): d for d in self.all_tested_params}.values())

        if save_file_name is not None:
            param_df = pd.DataFrame(self.top_params_list).sort_values('sharpe', ascending=False)
            param_df.to_csv(save_file_name, index=False)

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
        arrays = list(self.backtest_paths.values())
        num_columns = arrays[0].shape[1]
        if not all(arr.shape[1] == num_columns for arr in arrays):
            raise Exception('Tickers have different number of backtest paths')

        for col in range(num_columns):
            unique_values_set = set(np.unique(arrays[0][:, col]))
            for arr in arrays[1:]:
                if unique_values_set != set(np.unique(arr[:, col])):
                    raise Exception('Tickers have different parameter folds within same backtest path number')

        n_paths = num_columns
        tmp_dict = {}
        final_metrics = []
        final_returns = []
        for path_num in range(n_paths):
            path_returns = []
            unique_folds = np.unique(arrays[0][:, path_num])
            for fold in unique_folds:
                fold = int(fold)
                params = self.best_params_by_fold[fold]
                for ticker, path_array in self.backtest_paths.items():
                    test_indices = np.where(path_array[:, path_num] == fold)[0]
                    tmp_dict[ticker] = self.train_data[ticker].iloc[test_indices]
                returns = self.calc_pl(tmp_dict, params)
                path_returns.append(returns)

            path_returns = pd.concat(path_returns)
            final_returns.append(path_returns)
            metrics = calculate_metrics(path_returns)
            final_metrics.append(metrics)

        final_metrics = pd.DataFrame(final_metrics).mean().to_dict()

        text = ''
        for metric, v in final_metrics.items():
            text += f'Mean {metric}: {round(v, 2)}\n'

        fig, ax = plt.subplots(figsize=(12, 8))
        for returns in final_returns:
            ax.plot(returns.resample('D').sum().cumsum())

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Returns', fontsize=14)
        ax.set_title('Cumulative Returns Over Time', fontsize=16)
        plt.show()

    def run_stress_tests(self):
        """
        Run stress tests on the best parameter sets.
        """
        result = []
        for params in self.all_tested_params:
            returns = self.calc_pl(self.train_data, params)
            if not returns.empty:
                returns = returns.resample('D').sum()
            result.append(returns)
        result = pd.concat(result, axis=1).dropna()
        run_stress_tests(result)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for parameter optimization.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            float: The Sharpe ratio.
        """
        params = {}
        for k, v in self.params_dict.items():
            if not isinstance(v, Iterable) or isinstance(v, (str, bytes)):
                params[k] = v
            elif all(isinstance(item, int) for item in v):
                params[k] = int(trial.suggest_categorical(k, v))
            elif any(isinstance(item, float) for item in v):
                params[k] = float(trial.suggest_categorical(k, v))
            elif any(isinstance(item, str) for item in v) or any(isinstance(item, list) for item in v):
                params[k] = (trial.suggest_categorical(k, v))

        current_params = trial.params
        existing_trials = trial.study.get_trials(deepcopy=False)
        completed_trials = [t for t in existing_trials if t.state == optuna.trial.TrialState.COMPLETE]
        existing_params = [t.params for t in completed_trials]
        if current_params in existing_params:
            print(f"Pruning trial {trial.number} due to duplicate parameters: {params}")
            raise optuna.TrialPruned()
        sharpe, _ = self.combcv_pl(params, is_train=True)
        return sharpe

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
            centroids = [data[cluster_labels == i].mean(axis=0) for i in range(n_clusters)]
            wcss.append(sum(np.linalg.norm(data[cluster_labels == i] - centroids[i]) ** 2 for i in range(n_clusters)))
        return wcss

    def find_optimal_clusters(self, param_matrix_scaled: np.ndarray, max_clusters: int) -> int:
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
        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.axvline(x=elbow_point, color='r', linestyle='--')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

        return elbow_point

    def cluster_and_aggregate(self) -> dict:
        """
        Cluster parameter sets and aggregate the best parameters.

        Returns:
            dict: The aggregated best parameter set.
        """
        if isinstance(self.top_params_list, list):
            param_df = pd.DataFrame(self.top_params_list).drop(columns=['sharpe'])
        elif isinstance(self.top_params_list, pd.DataFrame):
            param_df = self.top_params_list.drop(columns=['sharpe'])
        else:
            raise Exception('Wrong data format for top params, accepted formats are list/DataFrame')

        max_clusters = min(max(3, len(param_df) // 3), len(param_df))
        if max_clusters > 2:
            logging.info(f'Starting clustering with max clusters: {max_clusters}, len of param set {len(param_df)}')

            column_transformer = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
                    ('cat', OneHotEncoder(), make_column_selector(dtype_exclude=np.number))
                ],
                remainder='passthrough'
            )

            param_matrix_scaled = column_transformer.fit_transform(param_df)
            best_n_clusters = self.find_optimal_clusters(param_matrix_scaled, max_clusters) if max_clusters < len(param_df) else max_clusters
            logging.info(f'Optimal number of clusters: {best_n_clusters}')
            clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
            cluster_labels = clustering.fit_predict(param_matrix_scaled)

            clustered_params = {i: [] for i in range(best_n_clusters)}
            for param, cluster in zip(self.top_params_list, cluster_labels):
                clustered_params[cluster].append(param)

            best_cluster = max(clustered_params.keys(), key=lambda c: np.mean([p['sharpe'] for p in clustered_params[c]]))
            best_cluster_params = clustered_params[best_cluster]
            logging.info(f'Best cluster: {best_cluster_params}')
            best_param_set = self.aggregate_params(best_cluster_params)
        else:
            logging.info(f'Len of params set less than 3, choosing best params out of 3')
            best_param_set = pd.DataFrame(self.top_params_list).iloc[0].to_dict()
        logging.info(f'Best params: {best_param_set}')
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


