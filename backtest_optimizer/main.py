import sys
import os
from datetime import timedelta
import time
import traceback

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
import tempfile
import shutil
import pyarrow.parquet as pq

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
        trials = study.get_trials(deepcopy=False)
        completed_trials = [
            t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        params_list = [t.params for t in completed_trials]

        if trial.params in params_list:
            logging.info(
                f"Trial {trial.number} pruned due to duplicate parameters: {trial.params}"
            )
            return True
        return False


def calc_returns(args):
    optimizer, params, train_data = args
    try:
        returns = optimizer.calc_pl(train_data, params)
    except Exception as e:
        logging.info(
            f"PL calculation for stress tests failed for {params}, with error {e}"
        )
        returns = pd.Series()
    if not returns.empty:
        returns = returns.resample("D").sum()
    return returns


class ParameterOptimizer:
    def __init__(
        self,
        calc_pl: callable,
        save_path: str,
        save_file_prefix: str,
        scale_pos_func: callable = None,
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
        self.data_info = {}  # Store metadata about data files (lengths, etc.)
        self.average_row_size_bytes = None
        self.use_batch_processing = None
        self.index_batch_size = None
        self.warmup_period = None
        self.scale_pos_func = scale_pos_func

    def combcv_pl(
        self,
        params: dict,
        group_indices: dict,
        data_source: dict,
    ) -> tuple:
        """
        For each CV group in group_indices, aggregate returns.

        Data_source can be either:
        - A dictionary mapping tickers to DataFrames (legacy mode)
        - A string path to data directory

        Returns:
            (final_sharpe, final_returns) where final_returns is a list (one per group) of aggregated returns.
        """
        final_returns = []

        logging.info(f"Processing {len(group_indices)} groups with params: {params}")

        for group_num, ticker_indices in group_indices.items():
            ticker_count = len(ticker_indices)
            logging.info(f"Processing group {group_num} with {ticker_count} tickers")

            if not self.use_batch_processing:
                # Non-batch mode: Process entire group data at once
                group_data = {}
                if isinstance(data_source, str):
                    # Load data from file for all tickers and indices
                    load_args = [
                        (ticker, data_source, indices)
                        for ticker, indices in ticker_indices.items()
                    ]
                    n_workers = min(self.n_jobs, len(load_args), 32)
                    with Pool(processes=n_workers) as pool:
                        results = list(
                            tqdm(
                                pool.imap(load_single_ticker, load_args),
                                total=len(load_args),
                                desc=f"Loading data for group {group_num}",
                            )
                        )
                    group_data = {
                        ticker: df
                        for ticker, df in results
                        if df is not None and not df.empty
                    }
                else:
                    # Legacy mode: Data already in memory
                    for ticker, indices in ticker_indices.items():
                        df = data_source.get(ticker)
                        if df is None or df.empty:
                            logging.warning(
                                f"Group {group_num}: Ticker {ticker} has no data"
                            )
                            continue
                        if isinstance(indices, pd.DatetimeIndex):
                            mask = df.index.isin(indices)
                            group_data[ticker] = df.loc[mask]
                        else:
                            group_data[ticker] = df.iloc[indices]

                if not group_data:
                    logging.warning(f"Group {group_num}: No valid tickers. Skipping.")
                    continue

                # Calculate returns for the entire group
                logging.info(
                    f"Group {group_num}: Calculating returns for {len(group_data)} tickers"
                )
                returns = self.calc_pl(group_data, params)
                returns.to_csv("rets_no_batch.csv")
                if returns is not None and not returns.empty:
                    final_returns.append(returns)
                else:
                    logging.warning(
                        f"Group {group_num}: No returns generated. Skipping."
                    )

            else:
                # Batch mode: Process timestamp-based batches
                reference_ticker = params["reference_ticker"]
                common_index = ticker_indices[reference_ticker]  # pd.DatetimeIndex
                if not isinstance(common_index, pd.DatetimeIndex):
                    logging.error(
                        f"Group {group_num}: common_index is not a DatetimeIndex"
                    )
                    continue

                if len(common_index) == 0:
                    logging.warning(
                        f"Group {group_num}: No indices available. Skipping."
                    )
                    continue

                # Batch parameters
                index_batch_size = self.index_batch_size  # e.g., '10D' for 10 days
                warmup_period = self.warmup_period  # e.g., '1H' for 1 hour

                # Convert time strings to Timedelta for calculations
                batch_timedelta = pd.Timedelta(index_batch_size)
                warmup_timedelta = pd.Timedelta(warmup_period)

                # Calculate batch start and end dates
                start_date = common_index[0]
                end_date = common_index[-1]
                batch_starts = pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq=index_batch_size,
                    inclusive="left",
                )
                num_batches = len(batch_starts)
                if end_date not in batch_starts:
                    batch_starts = batch_starts.append(pd.Index([end_date]))
                    num_batches += 1

                logging.info(f"Group {group_num}: Splitting into {num_batches} batches")

                group_returns_list = []

                for k in range(num_batches):
                    # Define main period for this batch
                    main_start = batch_starts[k]
                    if k + 1 < num_batches:
                        main_end = batch_starts[k + 1]
                    else:
                        main_end = end_date + pd.Timedelta(
                            minutes=1
                        )  # Ensure inclusion of end_date
                    if k == 0:
                        main_start = start_date
                    main_period = common_index[
                        (common_index >= main_start) & (common_index < main_end)
                    ]

                    # Define warmup period
                    if k == 0:
                        warmup_start = start_date
                    else:
                        warmup_start = main_start - warmup_timedelta

                    # Indices to load (warmup + main period)
                    data_period = common_index[
                        (common_index >= warmup_start) & (common_index <= main_end)
                    ]
                    logging.info(
                        f"Group {group_num}, Batch {k}/{num_batches}: Loading {len(data_period)} indices "
                        f"from {data_period[0]} to {data_period[-1]}"
                    )

                    # Prepare batch indices for all tickers
                    batch_indices = {
                        ticker: data_period for ticker in ticker_indices.keys()
                    }

                    # Load data only for this batch
                    batch_data = {}
                    if isinstance(data_source, str):
                        load_args = [
                            (ticker, data_source, indices)
                            for ticker, indices in batch_indices.items()
                        ]
                        n_workers = min(self.n_jobs, len(load_args), 32)
                        with Pool(processes=n_workers) as pool:
                            results = list(
                                tqdm(
                                    pool.imap(load_single_ticker, load_args),
                                    total=len(load_args),
                                    desc=f"Loading data for batch {k}/{num_batches} in group {group_num}",
                                )
                            )
                        batch_data = {
                            ticker: df
                            for ticker, df in results
                            if df is not None and not df.empty
                        }
                    else:
                        # Legacy mode: Data in memory, filter by batch indices
                        for ticker, indices in batch_indices.items():
                            df = data_source.get(ticker)
                            if df is None or df.empty:
                                logging.warning(
                                    f"Group {group_num}, Batch {k}/{num_batches}: Ticker {ticker} has no data"
                                )
                                continue
                            mask = df.index.isin(indices)
                            batch_data[ticker] = df.loc[mask]

                    if not batch_data:
                        logging.warning(
                            f"Group {group_num}, Batch {k}/{num_batches}: No valid data. Skipping."
                        )
                        continue

                    # Log batch data summary
                    logging.info(
                        f"Group {group_num}, Batch {k}/{num_batches}: Loaded data for {len(batch_data)} tickers"
                    )
                    for ticker, df in batch_data.items():
                        logging.info(
                            f"  Ticker {ticker}: {len(df)} rows, from {df.index[0]} to {df.index[-1]}"
                        )

                    # Calculate returns for this batch
                    logging.info(
                        f"Group {group_num}, Batch {k}/{num_batches}: Calculating returns"
                    )
                    batch_returns = self.calc_pl(batch_data, params)

                    if batch_returns is None or batch_returns.empty:
                        logging.warning(
                            f"Group {group_num}, Batch {k}/{num_batches}: No returns generated. Skipping."
                        )
                        continue

                    # Keep only main period returns
                    batch_returns_main = batch_returns.loc[
                        batch_returns.index.intersection(main_period)
                    ]
                    if batch_returns_main.empty:
                        logging.warning(
                            f"Group {group_num}, Batch {k}/{num_batches}: No returns in main period. Skipping."
                        )
                        continue

                    group_returns_list.append(batch_returns_main)

                # Aggregate batch returns for the group
                if group_returns_list:
                    group_returns = pd.concat(group_returns_list, axis=0)
                    # Ensure it's a Series, sort by index, and keep first occurrence of duplicates
                    if not isinstance(group_returns, pd.Series):
                        logging.warning(
                            f"Group {group_num}: Concatenated returns is not a Series"
                        )
                        continue
                    group_returns = group_returns.sort_index()
                    group_returns = group_returns[
                        ~group_returns.index.duplicated(keep="first")
                    ]
                    group_returns.to_csv("rets_batch.csv")
                    final_returns.append(group_returns)
                    logging.info(
                        f"Group {group_num}: Aggregated {len(group_returns)} unique returns"
                    )
                else:
                    logging.warning(
                        f"Group {group_num}: No returns generated for any batch."
                    )

        # Compute Sharpe ratios from each group's aggregated returns
        if not final_returns:
            logging.error("No valid returns from any group")
            return np.nan, final_returns

        logging.info(f"Computing Sharpe ratios for {len(final_returns)} groups")
        sharpe_ratios = []

        for i, r in enumerate(final_returns):
            sharpe = annual_sharpe(r)
            if np.isnan(sharpe):
                logging.warning(f"Group {i}: Annual Sharpe ratio calculated as NaN")
                continue
            logging.info(
                f"Group {i}: Annual Sharpe = {sharpe:.4f}, from {len(r)} returns"
            )
            sharpe_ratios.append(sharpe)

        if not sharpe_ratios:
            logging.error("All Sharpe ratios are NaN")
            return np.nan, final_returns

        final_sharpe = np.nanmean(sharpe_ratios)
        logging.info(
            f"Final mean Sharpe ratio: {final_sharpe:.4f} (across {len(sharpe_ratios)} groups)"
        )
        return final_sharpe, final_returns

    def create_objective(self, group_indices, params_dict, data_source):
        # 1) Partition params into fixed vs. to-be-optimized
        fixed_params = {
            k: v
            for k, v in params_dict.items()
            if not (isinstance(v, Iterable) and
                    not isinstance(v, (str, bytes, pd.DataFrame, pd.Series)))
        }
        search_space = {
            k: v
            for k, v in params_dict.items()
            if isinstance(v, Iterable) and
            not isinstance(v, (str, bytes, pd.DataFrame, pd.Series))
        }

        def objective(trial):
            # 2) Start with your fixed params…
            trial_params = fixed_params.copy()

            # 3) …and only suggest over the actual lists
            for k, choices in search_space.items():
                trial_params[k] = trial.suggest_categorical(k, choices)

            # 4) (Optional) if you want to prune duplicates, compare only the tuned keys:
            existing = [
            {kk: t.params[kk] for kk in search_space}
            for t in trial.study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
            current = {kk: trial_params[kk] for kk in search_space}
            if current in existing:
                raise optuna.TrialPruned()

            # 5) Finally call your combcv_pl exactly as before
            sharpe, _ = self.combcv_pl(trial_params, group_indices, data_source)
            return np.nan if np.isnan(sharpe) else sharpe

        return objective


    def collect_data_info(self, data_dir: str, file_pattern: str = None):
        """
        Collect metadata about data files without loading them fully into memory.

        Args:
            data_dir (str): Directory containing the data files
            file_pattern (str): Pattern to match data files. If None, all supported formats are used.
        """
        import glob

        logging.info(f"Collecting data info from {data_dir}")
        self.data_info = {}

        # Get list of all data files with supported extensions
        supported_patterns = (
            ["*.parquet", "*.csv.gz", "*.csv"]
            if file_pattern is None
            else [file_pattern]
        )

        file_paths = []
        for pattern in supported_patterns:
            file_paths.extend(glob.glob(os.path.join(data_dir, pattern)))

        for file_path in tqdm(file_paths, desc="Processing data files"):
            try:
                filename = os.path.basename(file_path)
                ticker = filename.split(".")[0]  # Extract ticker from filename

                # Read only metadata depending on file type
                if file_path.endswith(".parquet"):
                    df_sample = pd.read_parquet(file_path, columns=[])
                    # Get date range from parquet metadata
                    df_sample_start_date = (
                        df_sample.index.min()
                        if len(df_sample) > 0 and hasattr(df_sample.index, "min")
                        else None
                    )
                    df_sample_end_date = (
                        df_sample.index.max()
                        if len(df_sample) > 0 and hasattr(df_sample.index, "max")
                        else None
                    )
                    df_sample_freq = "D"  # Default
                    if len(df_sample) > 1 and hasattr(df_sample.index, "inferred_freq"):
                        inferred = pd.infer_freq(df_sample.index)
                        if inferred is not None:
                            df_sample_freq = inferred
                elif file_path.endswith(".csv.gz") or file_path.endswith(".csv"):
                    # For CSV files, use efficient date range function
                    # Use the first column as the date column
                    compression = "gzip" if file_path.endswith(".csv.gz") else None

                    # Read just the first row to get first column name
                    try:
                        first_row = pd.read_csv(
                            file_path, nrows=1, compression=compression
                        )
                        date_col = first_row.columns[0]  # Use first column as date

                        # Get date range and frequency
                        (
                            df_sample_start_date,
                            df_sample_end_date,
                            df_sample_freq,
                        ) = self._get_csv_date_range(
                            file_path, date_col=date_col, compression=compression
                        )
                    except Exception as e:
                        logging.warning(
                            f"Error processing first column as date in {file_path}: {str(e)}"
                        )
                        continue
                else:
                    logging.warning(f"Unsupported file format for {file_path}")
                    continue

                self.data_info[ticker] = {
                    "path": file_path,
                    "format": file_path.split(".")[-1]
                    if not file_path.endswith(".csv.gz")
                    else "csv.gz",
                    "start_date": df_sample_start_date,
                    "end_date": df_sample_end_date,
                    "freq": df_sample_freq,
                }
            except Exception as e:
                logging.warning(f"Error processing {file_path}: {str(e)}")

        logging.info(f"Collected info for {len(self.data_info)} tickers")

    def _calculate_frequency(self, dates):
        """
        Calculate the frequency of a time series based on the most common time difference.
        Uses pandas built-in frequency detection with fallback to mode-based detection.

        Args:
            dates (pd.DatetimeIndex or Series): Datetime values

        Returns:
            str: Frequency string in pandas format
        """
        from pandas.tseries.frequencies import to_offset

        # Ensure we have a DatetimeIndex
        if isinstance(dates, pd.Series):
            dates = pd.DatetimeIndex(dates)

        # First try pandas' built-in frequency inference
        freq = dates.inferred_freq or pd.infer_freq(dates)
        if freq:
            logging.debug(f"Frequency inferred by pandas: {freq}")
            return freq

        # Fallback: calculate the most common time difference
        if len(dates) < 2:
            logging.debug(
                "Not enough data points to determine frequency, using default 'D'"
            )
            return "D"

        try:
            # Get the mode (most common) of time differences
            td = dates.to_series().diff().dropna().mode().iloc[0]
            freq_str = to_offset(td).freqstr

            if freq_str:
                logging.debug(
                    f"Frequency determined from mode of time differences: {freq_str}"
                )
                return freq_str

            # If to_offset can't determine a frequency string, convert to seconds
            total_seconds = td.total_seconds()
            logging.debug(f"Using total seconds ({total_seconds}) as frequency")

            # Format based on the time scale
            if total_seconds < 60:
                return f"{int(total_seconds)}S"
            elif total_seconds < 3600:
                minutes = int(total_seconds / 60)
                return f"{minutes}min"
            elif total_seconds < 86400:
                hours = int(total_seconds / 3600)
                return f"{hours}h"
            else:
                days = int(total_seconds / 86400)
                return f"{days}D"

        except Exception as e:
            logging.warning(f"Error determining frequency: {str(e)}. Using default 'D'")
            return "D"

    def create_combcv_dict_old(self, data_dict, n_splits: int, n_test_splits: int):
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

    def _get_csv_date_range(
        self, filepath, date_col="date", chunksize=100_000, compression=None
    ):
        """
        Efficiently get the date range and frequency of a CSV file by reading only the date column in chunks.

        Args:
            filepath (str): Path to the CSV file
            date_col (str): Name of the date column (not used if first column is assumed to be date)
            chunksize (int): Size of chunks to read
            compression (str): Compression type (None, 'gzip', etc.)

        Returns:
            tuple: (start_date, end_date, frequency) as (Timestamp, Timestamp, str)
        """
        try:
            # Read first row to get column names
            first_row = pd.read_csv(filepath, nrows=1, compression=compression)
            # Use the first column as the date column
            date_col = first_row.columns[0]
            logging.debug(f"Using column '{date_col}' as date column for {filepath}")

            # First read the entire date column to get accurate start and end dates
            date_series = pd.read_csv(
                filepath,
                usecols=[date_col],
                parse_dates=[date_col],
                compression=compression,
            )[date_col]

            # Get min and max dates
            start = pd.to_datetime(date_series.min())
            end = pd.to_datetime(date_series.max())
            logging.debug(f"Date range for {filepath}: {start} to {end}")

            # Sort the dates and calculate frequency using custom function
            sorted_dates = date_series.sort_values()

            freq = self._calculate_frequency(sorted_dates)
            logging.debug(f"Calculated frequency for {filepath}: {freq}")

            return start, end, freq
        except Exception as e:
            logging.warning(f"Error getting date range for {filepath}: {str(e)}")
            return None, None, "D"  # Default to daily frequency on error

    def load_ticker_data(self, ticker: str, date_range=None, data_dir=None):
        # If we're loading from a directory that's not in data_info
        if data_dir is not None:
            file_path = None
            file_format = None
            extensions = [".parquet", ".csv", ".csv.gz", ".h5"]
            for ext in extensions:
                possible_path = os.path.join(data_dir, f"{ticker}{ext}")
                if os.path.exists(possible_path):
                    file_path = possible_path
                    file_format = ext.lstrip(".")
                    break
            if file_path is None:
                logging.debug(f"No data file found for {ticker} in {data_dir}")
                return pd.DataFrame()
        else:
            if ticker not in self.data_info:
                logging.debug(f"No info for ticker {ticker}")
                return pd.DataFrame()
            file_path = self.data_info[ticker]["path"]
            file_format = self.data_info[ticker]["format"]

        date_range_str = "all data"
        if date_range is not None:
            if isinstance(date_range, pd.DatetimeIndex):
                if len(date_range) > 0:
                    date_range_str = f"{date_range[0]} to {date_range[-1]}"
                    logging.debug(
                        f"Loading {ticker} with DatetimeIndex range: {date_range_str}"
                    )
                else:
                    logging.debug(f"Empty DatetimeIndex provided for {ticker}")
                    return pd.DataFrame()
            elif isinstance(date_range, list):
                all_dates = pd.concat([pd.Series(idx) for idx in date_range]).unique()
                date_range_str = f"{min(all_dates)} to {max(all_dates)}"
                logging.debug(
                    f"Loading {ticker} with list of indices: {date_range_str}"
                )
            else:
                date_range_str = f"{date_range[0]} to {date_range[1]}"
                logging.debug(f"Loading {ticker} with date range: {date_range_str}")
        else:
            logging.debug(f"Loading all data for {ticker}")

        try:
            logging.debug(f"Loading {ticker} from {file_path}")

            # Helper function to convert date_range into a list of (start, end) tuples
            def parse_date_ranges(date_range):
                if date_range is None:
                    return None
                elif isinstance(date_range, tuple) and len(date_range) == 2:
                    return [
                        (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
                    ]
                elif isinstance(date_range, pd.DatetimeIndex):
                    # Group consecutive dates into ranges
                    dates = sorted(date_range)
                    ranges = []
                    start = dates[0]
                    prev = start
                    for curr in dates[1:] + [None]:
                        if curr is None or (curr - prev).days > 1:
                            ranges.append((start, prev))
                            start = curr
                        prev = curr if curr is not None else prev
                    return ranges
                elif isinstance(date_range, list):
                    # Flatten list of DatetimeIndex or tuples into ranges
                    all_ranges = []
                    for item in date_range:
                        if isinstance(item, pd.DatetimeIndex):
                            all_ranges.extend(parse_date_ranges(item))
                        elif isinstance(item, tuple) and len(item) == 2:
                            all_ranges.append(
                                (pd.to_datetime(item[0]), pd.to_datetime(item[1]))
                            )
                    return all_ranges
                else:
                    raise ValueError(f"Unsupported date_range type: {type(date_range)}")

            def find_datetime_column(file_path):
                possible_names = [
                    "index",
                    "date",
                    "time",
                    "datetime",
                    "timestamp",
                    "Index",
                    "Date",
                    "Time",
                    "Datetime",
                    "Timestamp",
                ]
                try:
                    # Read Parquet metadata using pyarrow
                    parquet_file = pq.ParquetFile(file_path)
                    schema = parquet_file.schema
                    for i in range(len(schema)):
                        col_name = schema[i].name
                        col_type = str(schema[i].physical_type).lower()
                        # Check if the column is a timestamp type or has a matching name
                        if (
                            col_type.startswith("timestamp")
                            or col_name in possible_names
                        ):
                            return col_name
                    return None
                except Exception as e:
                    logging.warning(
                        f"Error reading Parquet schema for {file_path}: {e}"
                    )
                    return None

            # Parse the date ranges to load
            ranges = parse_date_ranges(date_range) if date_range is not None else None

            # Load data based on file format
            if file_format == "parquet":
                datetime_column = find_datetime_column(file_path)
                if datetime_column is None:
                    logging.warning(
                        f"No suitable datetime column found for {ticker}. Loading full file."
                    )
                    df = pd.read_parquet(file_path)
                else:
                    if ranges:
                        dfs = []
                        for start_date, end_date in ranges:
                            try:
                                df_part = pd.read_parquet(
                                    file_path,
                                    filters=[
                                        (datetime_column, ">=", start_date),
                                        (datetime_column, "<=", end_date),
                                    ],
                                )
                                dfs.append(df_part)
                            except Exception as e:
                                logging.warning(
                                    f"Failed to load Parquet part for {ticker} ({start_date} to {end_date}): {e}"
                                )
                                continue
                        df = pd.concat(dfs) if dfs else pd.DataFrame()
                        if datetime_column in df.columns:
                            df.set_index(datetime_column, inplace=True)
                    else:
                        df = pd.read_parquet(file_path)
                        if datetime_column in df.columns:
                            df.set_index(datetime_column, inplace=True)
            elif file_format == "h5":
                key = "data"
                if ranges:
                    dfs = []
                    for start_date, end_date in ranges:
                        try:
                            df_part = pd.read_hdf(
                                file_path,
                                key=key,
                                where=f"index >= '{start_date}' and index <= '{end_date}'",
                            )
                            dfs.append(df_part)
                        except Exception as e:
                            logging.warning(
                                f"Failed to load HDF5 part for {ticker} ({start_date} to {end_date}): {e}"
                            )
                            continue
                    df = pd.concat(dfs) if dfs else pd.DataFrame()
                else:
                    df = pd.read_hdf(file_path, key=key)
            elif file_format in ["csv", "csv.gz"]:
                if file_format == "csv.gz":
                    df = pd.read_csv(file_path, compression="gzip")
                else:
                    df = pd.read_csv(file_path)
                first_col = df.columns[0]
                df.set_index(first_col, inplace=True)
                df.index = pd.to_datetime(df.index)
                if ranges:
                    masks = [
                        (df.index >= start) & (df.index <= end) for start, end in ranges
                    ]
                    combined_mask = masks[0]
                    for mask in masks[1:]:
                        combined_mask |= mask
                    df = df.loc[combined_mask]
            else:
                logging.warning(f"Unsupported file format: {file_format}")
                return pd.DataFrame()

            # Ensure index is DatetimeIndex and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if len(df) > 0:
                logging.debug(
                    f"Loaded {ticker}: {len(df)} rows, range: {df.index[0]} to {df.index[-1]}"
                )
            else:
                logging.warning(f"Loaded empty DataFrame for {ticker}")
            return df
        except Exception as e:
            logging.warning(f"Error loading data for {ticker}: {str(e)}")
            return pd.DataFrame()

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

    def split_consecutive(self, arr):
        """
        Split an array into consecutive chunks.

        Args:
            arr: Array to split

        Returns:
            list: List of consecutive chunks
        """
        # Ensure the input is a NumPy array
        arr = np.asarray(arr)

        # Calculate the differences between adjacent elements
        diff = np.diff(arr)

        # Find where the difference is not 1 (i.e., where sequences break)
        split_points = np.where(diff != 1)[0] + 1

        # Use these split points to create chunks
        chunks = np.split(arr, split_points)

        return chunks

    def create_combcv_dict(self, data_dir, n_splits: int, n_test_splits: int):
        """
        Create a dictionary for combinatorial cross-validation based on data info.

        Args:
            data_dir (str): Directory containing data files
            n_splits (int): Number of total splits.
            n_test_splits (int): Number of test splits.
        """
        if not self.data_info:
            self.collect_data_info(data_dir)

        # Find common date range across all tickers
        min_dates = [
            info["start_date"]
            for info in self.data_info.values()
            if info["start_date"] is not None
        ]
        max_dates = [
            info["end_date"]
            for info in self.data_info.values()
            if info["end_date"] is not None
        ]

        if not min_dates or not max_dates:
            raise ValueError("No valid date ranges found in data info.")

        global_start_date = min(min_dates)  # Earliest of all start dates
        global_end_date = max(max_dates)  # Latest of all end dates

        logging.info(
            f"Using common date range: {global_start_date} to {global_end_date}"
        )

        # Determine most common frequency
        frequencies = [
            info["freq"] for info in self.data_info.values() if "freq" in info
        ]
        if not frequencies:
            common_freq = "D"  # Default to daily if no frequencies found
        else:
            from collections import Counter

            freq_counter = Counter(frequencies)
            common_freq = freq_counter.most_common(1)[0][0]

        logging.info(f"Using common frequency: {common_freq}")

        # Calculate a common date range length for all tickers using inferred frequency
        common_date_range = pd.date_range(
            start=global_start_date, end=global_end_date, freq=common_freq
        )
        common_length = len(common_date_range)

        logging.info(f"Common date range length: {common_length} time periods")

        total_comb = math.comb(n_splits, n_test_splits)
        if n_test_splits == 0 or n_splits == 0:
            logging.info(
                "Using the entire dataset as the training set with no validation groups."
            )
            self.combcv_dict[0] = {}
            for ticker, info in self.data_info.items():
                self.combcv_dict[0][ticker] = {
                    "train": [np.arange(common_length)],
                    "test": None,
                }
        else:
            logging.info(
                f"Creating combinatorial train-val split, total_split: {n_splits}, out of which val groups: {n_test_splits}"
            )

            # Generate CPCV indices once based on common length
            is_test, paths, path_folds = self.cpcv_generator(
                common_length, n_splits, n_test_splits, verbose=False
            )

            for ticker, info in self.data_info.items():
                # Map the common indices to ticker-specific data
                self.backtest_paths[ticker] = paths

                for combination_num in range(is_test.shape[1]):
                    if combination_num not in self.combcv_dict:
                        self.combcv_dict[combination_num] = {}

                    train_indices = np.where(~is_test[:, combination_num])[0]
                    test_indices = np.where(is_test[:, combination_num])[0]

                    train_indices = self.split_consecutive(train_indices)
                    test_indices = self.split_consecutive(test_indices)

                    self.combcv_dict[combination_num][ticker] = {
                        "train": train_indices,
                        "test": test_indices,
                    }

    def generate_group_indices(self, is_train: bool):
        """
        Generate group indices for training or testing without loading all data.

        Args:
            is_train (bool): Whether to generate indices for training or testing.

        Returns:
            dict: A dictionary mapping group numbers to indices.
        """
        group_indices = {}
        ticker_count = len(self.current_group)
        mode = "training" if is_train else "testing"
        logging.info(f"Generating {mode} indices for {ticker_count} tickers")

        total_indices = 0
        min_date = None
        max_date = None

        for ticker, group_info in self.current_group.items():
            select_idx = group_info["train"]
            if group_info["test"] and not is_train:
                select_idx = group_info["test"]

            if not select_idx:
                logging.debug(f"No {mode} indices for ticker {ticker}")
                continue

            for i, idx in enumerate(select_idx):
                if i not in group_indices:
                    group_indices[i] = {}

                group_indices[i][ticker] = idx  # Store indices instead of DataFrames

                # Update statistics for logging
                total_indices += len(idx)

                # Track date range if possible
                if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                    curr_min = idx.min()
                    curr_max = idx.max()

                    if min_date is None or curr_min < min_date:
                        min_date = curr_min

                    if max_date is None or curr_max > max_date:
                        max_date = curr_max

        # Log summary of generated indices
        group_count = len(group_indices)
        avg_tickers_per_group = sum(
            len(indices) for indices in group_indices.values()
        ) / max(1, group_count)

        if min_date is not None and max_date is not None:
            date_range = f"{min_date} to {max_date}"
            logging.info(
                f"Generated {group_count} {mode} groups with {total_indices} total indices, date range: {date_range}"
            )
        else:
            logging.info(
                f"Generated {group_count} {mode} groups with {total_indices} total indices"
            )

        logging.info(f"Average tickers per group: {avg_tickers_per_group:.1f}")

        for group_num, group_data in group_indices.items():
            ticker_count = len(group_data)
            index_counts = {
                ticker: len(indices) for ticker, indices in group_data.items()
            }
            avg_indices = sum(index_counts.values()) / max(1, ticker_count)
            logging.debug(
                f"Group {group_num}: {ticker_count} tickers, avg {avg_indices:.1f} indices per ticker"
            )

        return group_indices

    def load_data_for_indices(
        self,
        group_indices: dict,
        data_dir: str,
        max_memory_gb: float,
        fold_num: int = None,
    ):
        """
        Load data for the given group indices.

        Args:
            group_indices (dict): Mapping of group numbers to ticker indices.
            data_dir (str): Directory containing data files.
            max_memory_gb (float): Max allowed amount of RAM
            fold_num (int): number of fold for train

        Returns:
            data_source (str or dict): Data source with desired indices
        """

        if self.use_batch_processing:
            return data_dir

        fold_or_test = f"Fold {fold_num}" if fold_num is not None else "Test"

        if self.average_row_size_bytes is None:
            sample_ticker = list(self.data_info.keys())[0]
            sample_start = self.data_info[sample_ticker]["start_date"]
            sample_end = sample_start + pd.Timedelta(days=10)
            sample_date_range = (sample_start, sample_end)
            self.average_row_size_bytes = self.calculate_average_row_size(
                data_dir, sample_ticker, sample_date_range
            )

        total_rows = 0
        for group_num, ticker_indices in group_indices.items():
            for ticker, indices in ticker_indices.items():
                if isinstance(indices, list):
                    total_rows += sum(len(chunk) for chunk in indices)
                else:
                    total_rows += len(indices)

        # Estimate memory with a 20% buffer
        total_memory_bytes = total_rows * self.average_row_size_bytes * 1.2
        estimated_memory = total_memory_bytes / (1024 * 1024 * 1024)  # Convert to GB
        logging.info(f"{fold_or_test}: Estimated memory: {estimated_memory:.2f} GB")

        ticker_index_pairs = {}
        for group_num, ticker_indices in group_indices.items():
            for ticker, indices in ticker_indices.items():
                if ticker not in ticker_index_pairs:
                    ticker_index_pairs[ticker] = []
                if isinstance(indices, list):
                    ticker_index_pairs[ticker].extend(indices)
                else:
                    ticker_index_pairs[ticker].append(indices)

        if estimated_memory <= max_memory_gb:
            logging.info(f"Preloading data for {fold_or_test}")

            n_workers = min(self.n_jobs, len(ticker_index_pairs), 32)

            if n_workers <= 1:
                data = {}
                for ticker, indices in tqdm(
                    ticker_index_pairs.items(), desc="Loading ticker data"
                ):
                    df = self.load_ticker_data(
                        ticker, date_range=indices, data_dir=data_dir
                    )
                    if df is not None and not df.empty:
                        data[ticker] = df
            else:
                logging.info(
                    f"Loading {len(ticker_index_pairs)} tickers using {n_workers} parallel workers"
                )

                load_args = [
                    (self, ticker, indices, data_dir)
                    for ticker, indices in ticker_index_pairs.items()
                ]
                with Pool(processes=n_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap(_load_single_ticker, load_args),
                            total=len(load_args),
                            desc="Loading ticker data",
                        )
                    )

                data = {}
                for ticker, df in results:
                    if df is not None and not df.empty:
                        data[ticker] = df

            return data
        else:
            if all(
                info["format"] in ["csv", "csv.gz"] for info in self.data_info.values()
            ):
                logging.info(f"Converting CSV to HDF5 for {fold_or_test}")
                # Create a TemporaryDirectory that will be returned and managed by the caller
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_obj.name

                try:
                    conversion_args = [
                        (self, ticker, indices, data_dir, temp_dir)
                        for ticker, indices in ticker_index_pairs.items()
                    ]
                    with Pool(processes=self.n_jobs) as pool:
                        results = list(
                            tqdm(
                                pool.imap(convert_to_hdf5, conversion_args),
                                total=len(conversion_args),
                                desc="Converting to HDF5",
                            )
                        )
                    if all(results):
                        # Return both the directory name and the context manager object
                        return {"path": temp_dir, "context": temp_dir_obj}
                    else:
                        logging.error(
                            "Some conversions failed. Falling back to original data directory."
                        )
                        # Clean up the temporary directory by allowing the context manager to go out of scope
                        temp_dir_obj.cleanup()
                        return data_dir
                except Exception as e:
                    logging.error(f"Failed to convert CSV to HDF5: {e}")
                    # Clean up the temporary directory
                    temp_dir_obj.cleanup()
                    return data_dir
            else:
                return data_dir

    def calculate_average_row_size(
        self, data_dir: str, sample_ticker: str, sample_date_range: tuple
    ) -> float:
        """
        Calculate the average memory size per row based on a sample ticker.

        Args:
            data_dir (str): Directory containing data files.
            sample_ticker (str): Ticker to use for sampling.
            sample_date_range (tuple): Date range for the sample (start, end).

        Returns:
            float: Average memory size per row in bytes.
        """

        df = self.load_ticker_data(
            sample_ticker, date_range=sample_date_range, data_dir=data_dir
        )
        if df.empty:
            logging.warning(f"Sample data for {sample_ticker} is empty.")
            return 0

        total_memory = df.memory_usage(deep=True).sum()
        num_rows = len(df)
        average_row_size = total_memory / num_rows if num_rows > 0 else 0
        logging.debug(
            f"Average row size for {sample_ticker}: {average_row_size:.2f} bytes"
        )
        return average_row_size

    def optimize(
        self,
        data_dir: str,
        params: dict,
        optimizer_params: dict,
    ):
        """
        Optimize parameters using combinatorial cross-validation.

        Args:
            data_dir (str): Directory containing data files
            params (dict): Initial parameters for the optimization.
            optimizer_params (dict): Internal parameters for optimization.
        """
        # Start timing for overall process
        optimization_start_time = time.time()

        self.use_batch_processing = optimizer_params["use_batch_processing"]
        self.warmup_period = optimizer_params["warmup_period"]
        self.index_batch_size = optimizer_params["index_batch_size"]
        self.n_jobs = optimizer_params["n_jobs"]

        # Log optimization parameters
        param_keys_to_optimize = [k for k, v in params.items() if isinstance(v, list)]
        logging.info(
            f"Starting optimization with {self.n_jobs} jobs, {optimizer_params.get('n_runs')} trials per fold and {len(param_keys_to_optimize)} parameters to optimize"
        )
        if param_keys_to_optimize:
            logging.info(f"Parameters to optimize: {param_keys_to_optimize}")

        # Initialize result tracking
        result = []
        self.params_dict = params.copy()
        all_tested_params = []
        max_memory_gb = optimizer_params.get(
            "max_memory_gb", 1000
        )  # Default to 1000 GB (1 TB)

        # Dictionary to keep references to temporary directory context managers
        # Using a list to store context managers for each fold
        temp_dir_contexts = {}

        # Collect data info instead of loading all data
        if not self.data_info:
            data_info_start = time.time()
            self.collect_data_info(data_dir)
            data_info_duration = time.time() - data_info_start
            logging.info(
                f"Collected info for {len(self.data_info)} tickers in {data_info_duration:.2f}s"
            )

        # If train_test_date is provided, limit max date in data_info
        if optimizer_params.get("train_test_date") is not None:
            train_test_datetime = pd.Timestamp(optimizer_params["train_test_date"])
            logging.info(
                f"Limiting data to before {optimizer_params['train_test_date']}"
            )

            # Update data_info to limit end_date to train_test_date
            for ticker in self.data_info:
                if self.data_info[ticker]["end_date"] > train_test_datetime:
                    self.data_info[ticker]["end_date"] = train_test_datetime

        if optimizer_params.get("start_date") is not None:
            start_time = pd.Timestamp(optimizer_params["start_date"])
            logging.info(f"Limiting data to after {start_time}")

            # Update data_info to limit start_date to start_time
            for ticker in self.data_info:
                if self.data_info[ticker]["start_date"] < start_time:
                    self.data_info[ticker]["start_date"] = start_time

        # Store data directory path for subsequent operations
        self.params_dict["data_path"] = data_dir

        # Check if n_splits and n_test_splits are provided
        if (
            optimizer_params.get("n_splits") is None
            or optimizer_params.get("n_test_splits") is None
        ):
            raise ValueError("n_splits and n_test_splits must be provided")

        # Create combinatorial splits based on data info
        splits_start = time.time()
        self.create_date_combcv_dict(
            data_dir,
            n_splits=optimizer_params.get("n_splits"),
            n_test_splits=optimizer_params.get("n_test_splits"),
        )
        splits_duration = time.time() - splits_start
        logging.info(
            f"Created {len(self.combcv_dict)} combinatorial splits in {splits_duration:.2f}s"
        )

        try:
            # Process each combinatorial split
            total_folds = len(self.combcv_dict)
            for fold_idx, (fold_num, date_ranges) in enumerate(
                self.combcv_dict.items()
            ):
                fold_start_time = time.time()
                logging.info(f"Starting fold {fold_num} ({fold_idx+1}/{total_folds})")

                # Set current group for this fold
                self.current_group = date_ranges

                # Generate indices before optimization
                group_indices = self.generate_group_indices(is_train=True)
                data_source = self.load_data_for_indices(
                    group_indices, data_dir, max_memory_gb, fold_num
                )
                # Handle the case where data_source is a dictionary containing path and context
                actual_data_source = data_source
                if (
                    isinstance(data_source, dict)
                    and "path" in data_source
                    and "context" in data_source
                ):
                    # Store the context manager for cleanup later
                    temp_dir_contexts[fold_num] = data_source["context"]
                    # Use the path for data processing
                    actual_data_source = data_source["path"]

                # Define the objective function using combcv_pl
                objective_func = self.create_objective(
                    group_indices,
                    self.params_dict,
                    actual_data_source,  # Use the actual data source (path or data dict)
                )

                # Create and configure the Optuna study
                pruner = RepeatPruner()
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(multivariate=True),
                    pruner=pruner,
                )

                # Run the optimization
                logging.info(
                    f"Running {optimizer_params.get('n_runs')} trials for fold {fold_num}"
                )
                study.optimize(
                    objective_func,
                    n_trials=optimizer_params.get("n_runs"),
                    n_jobs=optimizer_params.get("n_jobs_optuna", 1),
                )
                optimization_duration = time.time() - fold_start_time

                if "train_data" in locals():
                    del train_data

                # Process and sort trials
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

                completed_trials = len(all_trials)
                logging.info(
                    f"Fold {fold_num}: {completed_trials}/{optimizer_params.get('n_runs')} trials completed in {optimization_duration:.2f}s"
                )

                if completed_trials == 0:
                    logging.warning(
                        f"Fold {fold_num}: No successfully completed trials"
                    )
                    continue

                # Extract parameters from trials
                all_trials = [trial.params for trial in all_trials]

                # Fill in default parameters
                for trial_params in all_trials:
                    for key, value in self.params_dict.items():
                        if key not in trial_params:
                            trial_params[key] = value

                # Add to the list of all tested parameters
                all_tested_params.extend(all_trials)

                # Select top parameters based on percentage
                top_count = max(
                    1, int(len(all_trials) * optimizer_params.get("best_trials_pct"))
                )
                top_params = all_trials[:top_count]
                logging.info(
                    f"Fold {fold_num}: Selected top {top_count} parameter sets (best Sharpe: {study.best_value:.4f})"
                )

                # Generate test indices
                test_indices = self.generate_group_indices(is_train=False)
                test_data = self.load_data_for_indices(
                    test_indices, data_dir, max_memory_gb
                )

                # Handle test data if it's a dictionary with path and context
                if (
                    isinstance(test_data, dict)
                    and "path" in test_data
                    and "context" in test_data
                ):
                    # Store the context manager for cleanup later (use a different key)
                    temp_dir_contexts[f"test_{fold_num}"] = test_data["context"]
                    # Use the path for data processing
                    test_data = test_data["path"]

                # Evaluate top parameters on validation data
                evaluation_start = time.time()
                for i, trial_params in enumerate(top_params):
                    sharpe, _ = self.combcv_pl(trial_params, test_indices, test_data)

                    # Record results
                    if i == 0:
                        trial_params["fold_num"] = fold_num
                    else:
                        trial_params["fold_num"] = np.nan

                    trial_params["sharpe"] = sharpe
                    result.append(trial_params)
                    logging.info(
                        f"Fold {fold_num}: Parameter set {i+1}/{len(top_params)} - Test Sharpe: {sharpe:.4f}"
                    )

                evaluation_duration = time.time() - evaluation_start
                logging.info(
                    f"Fold {fold_num}: Validation completed in {evaluation_duration:.2f}s"
                )

                # Save the best parameters for this fold
                self.best_params_by_fold[fold_num] = top_params[0]

                # Save interim results
                if self.save_path is not None:
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    interim_df = pd.DataFrame(result).sort_values(
                        "sharpe", ascending=False
                    )
                    if "data_path" in interim_df.columns:
                        del interim_df["data_path"]

                    interim_file = (
                        self.save_path
                        + self.file_prefix
                        + f"interim_fold_{fold_num}.csv"
                    )
                    interim_df.to_csv(interim_file, index=False)
                    logging.info(f"Interim results saved to {interim_file}")

                # Cleanup temporary directories for this fold
                if fold_num in temp_dir_contexts:
                    temp_dir_contexts[fold_num].cleanup()
                    del temp_dir_contexts[fold_num]
                    logging.info(f"Cleaned up temporary directory for fold {fold_num}")

                if f"test_{fold_num}" in temp_dir_contexts:
                    temp_dir_contexts[f"test_{fold_num}"].cleanup()
                    del temp_dir_contexts[f"test_{fold_num}"]
                    logging.info(
                        f"Cleaned up temporary directory for test fold {fold_num}"
                    )

                fold_duration = time.time() - fold_start_time
                logging.info(f"Completed fold {fold_num} in {fold_duration:.2f}s")

        finally:
            # Ensure all temporary directories are cleaned up
            for key, context in list(temp_dir_contexts.items()):
                try:
                    context.cleanup()
                    logging.info(f"Cleaned up temporary directory for {key}")
                except Exception as e:
                    logging.warning(
                        f"Error cleaning up temporary directory for {key}: {e}"
                    )
                del temp_dir_contexts[key]

        # Save final results
        self.top_params_list = result
        self.all_tested_params = list(
            {frozenset(d.items()): d for d in all_tested_params}.values()
        )

        if self.save_path is not None:
            param_df = pd.DataFrame(self.top_params_list).sort_values(
                "sharpe", ascending=False
            )

            if "data_path" in param_df.columns:
                del param_df["data_path"]

            top_file = self.save_path + self.file_prefix + "top_params.csv"
            param_df.to_csv(top_file, index=False)

            all_params_file = (
                self.save_path + self.file_prefix + "all_tested_params.csv"
            )
            all_tested_params_df = pd.DataFrame(self.all_tested_params)
            all_tested_params_df.to_csv(all_params_file, index=False)

            logging.info(f"Final results saved to {self.save_path}")

        # Log summary statistics
        total_duration = time.time() - optimization_start_time
        hrs, rem = divmod(total_duration, 3600)
        mins, secs = divmod(rem, 60)
        logging.info(f"Total optimization time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

        # Print best parameters
        if result:
            best_params = param_df.iloc[0].to_dict()
            best_sharpe = best_params.get("sharpe", np.nan)
            logging.info(f"Best parameters found: Sharpe={best_sharpe:.4f}")
            for k, v in best_params.items():
                if k != "sharpe" and k != "fold_num":
                    logging.info(f"  {k}: {v}")
        else:
            logging.warning("No valid parameter sets found")

    def plot_returns(self, data_dir, params: dict, save_returns=False):
        """
        Calculate out-of-sample Sharpe ratio and plot cumulative returns.

        Args:
            data_dir (str): Directory containing data files
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

        # For plotting, we need to load all data
        if not self.data_info:
            self.collect_data_info(data_dir)

        # Get list of tickers to load
        tickers = list(self.data_info.keys())

        # Use parallel loading for better performance
        all_data = self.load_multiple_tickers(tickers, data_dir)

        logging.info(f"Calculating returns for {len(all_data)} tickers")
        returns = self.calc_pl(all_data, params)

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

    def read_saved_params(self):
        """
        Read saved parameters from disk.
        """
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

    def run_stress_tests(self, data_dir, num_workers=5):
        """
        Run stress tests on the best parameter sets using the combcv_pl function.

        Args:
            data_dir (str): Directory containing data files
            num_workers (int): Number of workers for parallel processing
        """
        logging.info(f"Running stress tests, num_workers: {num_workers}")

        # Collect data info if not already done
        if not self.data_info:
            self.collect_data_info(data_dir)

        if not self.all_tested_params:
            logging.warning("No tested parameters available for stress tests.")
            return

        # Create a single set of indices covering the entire dataset
        all_indices = {}
        all_tickers = list(self.data_info.keys())

        # Load all ticker data in parallel for better performance
        logging.info(f"Loading data for {len(all_tickers)} tickers")
        all_data = self.load_multiple_tickers(all_tickers, data_dir)

        if not all_data:
            logging.error("Failed to load any valid data. Cannot run stress tests.")
            return

        logging.info(f"Successfully loaded data for {len(all_data)} tickers")

        # Function to process a single parameter set
        def process_param_set(params):
            try:
                # Use calc_pl directly on the loaded data
                returns = self.calc_pl(all_data, params)

                if returns is not None and not (
                    isinstance(returns, pd.Series) and returns.empty
                ):
                    if isinstance(returns, dict):
                        # If returns is a dictionary, use the "Total" key or first key available
                        if "Total" in returns:
                            return returns["Total"]
                        else:
                            return list(returns.values())[0]
                    else:
                        return returns
                return pd.Series()
            except Exception as e:
                logging.warning(f"Error processing parameters {params}: {str(e)}")
                return pd.Series()

        # Process all parameter sets
        stress_test_workers = min(num_workers, len(self.all_tested_params), self.n_jobs)
        logging.info(
            f"Processing {len(self.all_tested_params)} parameter sets with {stress_test_workers} workers"
        )

        if stress_test_workers > 1:
            # Use multiprocessing for parallel execution
            with multiprocessing.Pool(processes=stress_test_workers) as pool:
                results = list(
                    tqdm(
                        pool.map(process_param_set, self.all_tested_params),
                        total=len(self.all_tested_params),
                        desc="Processing parameter sets",
                    )
                )
        else:
            # Single-process execution
            results = []
            for params in tqdm(
                self.all_tested_params, desc="Processing parameter sets"
            ):
                results.append(process_param_set(params))

        # Filter out empty DataFrames and run stress tests
        results = [r for r in results if not r.empty]
        if results:
            result_df = pd.concat(results, axis=1).dropna()
            run_stress_tests(result_df)
        else:
            logging.warning("No valid results for stress tests")

    def reconstruct_equity_curves(self, data_dir):
        """
        Reconstruct equity curves based on the best parameters using combcv_pl for consistency.

        Args:
            data_dir (str): Directory containing data files
        """
        logging.info("Reconstructing validation equity curves")

        # Collect data info if not already done
        if not self.data_info:
            self.collect_data_info(data_dir)

        arrays = list(self.backtest_paths.values())
        if not arrays:
            logging.error(
                "No backtest paths available. Cannot reconstruct equity curves."
            )
            return

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
        final_metrics = []
        final_returns = []

        for path_num in tqdm(range(n_paths), desc="Processing path"):
            logging.info(f"Starting for path {path_num}")
            path_returns = []
            unique_folds = np.unique(arrays[0][:, path_num])

            for fold in unique_folds:
                logging.info(f"Starting for fold {fold}")
                fold = int(fold)
                params = self.best_params_by_fold[fold]

                # Create indices for this fold and path
                test_group = {}
                for ticker, path_array in self.backtest_paths.items():
                    # Find where this path has this fold
                    indices = np.where(path_array[:, path_num] == fold)[0]
                    if len(indices) > 0:
                        if 0 not in test_group:
                            test_group[0] = {}
                        # If we have date mapping, use it, otherwise use integer indices
                        if (
                            hasattr(self, "date_mapping")
                            and path_num in self.date_mapping
                        ):
                            test_group[0][ticker] = self.date_mapping[path_num][fold]
                        else:
                            test_group[0][ticker] = indices

                # Use combcv_pl to get returns
                if test_group:
                    _, returns_list = self.combcv_pl(params, test_group, data_dir)

                    if returns_list:
                        # Combine returns from all groups (usually just one for test data)
                        fold_returns = pd.concat(returns_list)
                        path_returns.append(fold_returns)

            if path_returns:
                # Combine returns from all folds in this path
                path_returns = pd.concat(path_returns)
                path_returns = path_returns.sort_index()
                final_returns.append(path_returns)

                # Calculate metrics for this path
                metrics = calculate_metrics(path_returns)
                final_metrics.append(metrics)

        if final_metrics:
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
        else:
            logging.warning("No valid return data to plot equity curves")

    def cluster_and_aggregate(self, min_sharpe=None, plot_elbow=False) -> dict:
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
            if param_df.empty:
                logging.warning(
                    f"None of parameters generated sharpe above {min_sharpe}"
                )
                return {}

        param_df = param_df.drop(columns=["sharpe"]).dropna(axis=1)

        # For simplicity, we'll just return the best parameters without clustering
        if not param_df.empty:
            best_param_set = (
                pd.DataFrame(self.top_params_list)
                .sort_values(by="sharpe", ascending=False)
                .iloc[0]
                .to_dict()
            )
            logging.info(f"Best params: {best_param_set}")
            return best_param_set
        else:
            logging.warning("No valid parameters found")
            return {}

    def plot_multiple_param_combinations(
        self,
        data_dir: str,
        params: Dict[str, Any],
    ):
        """
        Calculate out-of-sample Sharpe ratio, plot cumulative returns for all combinations on a single chart,
        and create a table with metrics and parameter sets using combcv_pl.

        Args:
            data_dir (str): Directory containing data files
            params (Dict[str, Any]): Parameters for the performance calculation. May contain lists of values.
        """
        logging.info("Processing parameter combinations")

        # Collect data info if not already done
        if not self.data_info:
            self.collect_data_info(data_dir)

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
            data_dir=data_dir,
            param_names=param_names,
            optimizer=self,
            calculate_metrics=calculate_metrics,
            n_jobs=self.n_jobs,  # Pass n_jobs from the class
        )

        # Choose between multiprocessing and single-process execution
        if self.n_jobs > 1:
            # Determine how many jobs to allocate to data loading vs parameter combinations
            # If we have many cores, allocate some for data loading and some for param combinations
            n_param_jobs = max(1, min(self.n_jobs // 2, total_combinations))
            n_data_jobs = max(
                1, min(self.n_jobs // 2, 8)
            )  # Limit data loading parallelism

            # Update the partial function to use the adjusted number of data loading workers
            partial_process = partial(
                process_combination,
                data_dir=data_dir,
                param_names=param_names,
                optimizer=self,
                calculate_metrics=calculate_metrics,
                n_jobs=n_data_jobs,  # Use a portion of available cores for data loading
            )

            # Use multiprocessing for parallel execution of parameter combinations
            with multiprocessing.Pool(n_param_jobs) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(partial_process, param_combinations),
                        total=total_combinations,
                        desc="Processing parameter combinations",
                    )
                )
        else:
            # Single-process execution
            results = list(
                tqdm(
                    (partial_process(combo) for combo in param_combinations),
                    total=total_combinations,
                    desc="Processing parameter combinations",
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
                {**{"Combination": i + 1}, **result["metrics"], **result["parameters"]}
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

    def split_data(self, data_dir: str, train_end_date: str):
        """
        Split data into training and validation sets based on a date.

        Args:
            data_dir (str): Directory containing data files
            train_end_date (str): End date for training data in format YYYY-MM-DD

        Returns:
            dict: Dictionary with training and validation date ranges
        """
        logging.info(f"Splitting data with train end date: {train_end_date}")

        # Collect data info if not already done
        if not self.data_info:
            self.collect_data_info(data_dir)

        # Convert train_end_date to datetime
        train_end = pd.Timestamp(train_end_date)

        # Find the earliest and latest dates in the data
        min_dates = [
            info["start_date"]
            for info in self.data_info.values()
            if info["start_date"] is not None
        ]
        max_dates = [
            info["end_date"]
            for info in self.data_info.values()
            if info["end_date"] is not None
        ]

        if not min_dates or not max_dates:
            raise ValueError("No valid date ranges found in data info.")

        data_start = min(min_dates)
        data_end = max(max_dates)

        # Ensure train_end is within the data range
        if train_end < data_start or train_end > data_end:
            logging.warning(
                f"Train end date {train_end} is outside data range ({data_start} to {data_end}). Adjusting."
            )
            train_end = min(max(train_end, data_start), data_end)

        # Create date ranges for training and validation
        train_range = (data_start, train_end)
        val_range = (train_end, data_end)

        logging.info(
            f"Data split: Training from {train_range[0]} to {train_range[1]}, Validation from {val_range[0]} to {val_range[1]}"
        )

        return {"train": train_range, "val": val_range}

    def create_date_combcv_dict(self, data_dir: str, n_splits: int, n_test_splits: int):
        """
        Create a dictionary for combinatorial cross-validation using date ranges.
        Based on the same algorithm as create_combcv_dict but storing datetime indices.

        Args:
            data_dir (str): Directory containing data files
            n_splits (int): Number of total splits.
            n_test_splits (int): Number of test splits.

        Returns:
            dict: Dictionary mapping combination numbers to date ranges
        """
        logging.info(
            f"Creating date-based combinatorial splits: {n_splits} total, {n_test_splits} test"
        )

        # Collect data info if not already done
        if not self.data_info:
            self.collect_data_info(data_dir)

        # Find the common date range across all tickers
        min_dates = [
            info["start_date"]
            for info in self.data_info.values()
            if info["start_date"] is not None
        ]
        max_dates = [
            info["end_date"]
            for info in self.data_info.values()
            if info["end_date"] is not None
        ]

        if not min_dates or not max_dates:
            raise ValueError("No valid date ranges found in data info.")

        # Use the earliest start date and latest end date to ensure all tickers have the same date range
        global_start_date = min(min_dates)
        global_end_date = max(max_dates)

        logging.info(
            f"Using common date range for all tickers: {global_start_date} to {global_end_date}"
        )

        # Determine most common frequency
        frequencies = [
            info["freq"] for info in self.data_info.values() if "freq" in info
        ]
        if not frequencies:
            common_freq = "min"  # Default to daily if no frequencies found
        else:
            from collections import Counter

            freq_counter = Counter(frequencies)
            common_freq = freq_counter.most_common(1)[0][0]

        logging.info(f"Using common frequency: {common_freq}")

        # Create a common date range for all tickers
        common_date_range = pd.date_range(
            start=global_start_date, end=global_end_date, freq=common_freq
        )
        total_periods = len(common_date_range)

        # Clear existing data
        self.combcv_dict = {}
        self.backtest_paths = {}

        total_comb = math.comb(n_splits, n_test_splits)

        if n_test_splits == 0 or n_splits == 0:
            logging.info(
                "Using the entire dataset as the training set with no validation groups."
            )
            self.combcv_dict[0] = {}
            for ticker in self.data_info.keys():
                self.combcv_dict[0][ticker] = {
                    "train": [common_date_range],
                    "test": None,
                }
            return self.combcv_dict

        logging.info(
            f"Creating combinatorial train-val split, total_split: {n_splits}, out of which val groups: {n_test_splits}"
        )

        # Generate CPCV indices once based on common length
        is_test, paths, path_folds = self.cpcv_generator(
            total_periods, n_splits, n_test_splits, verbose=False
        )

        # Precompute train and test datetime indices for each combination
        combination_train_test = {}
        for combination_num in tqdm(
            range(is_test.shape[1]), desc="Precomputing train/test indices"
        ):
            train_mask = ~is_test[:, combination_num]
            test_mask = is_test[:, combination_num]
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            train_chunks = self.split_consecutive(train_indices)
            test_chunks = self.split_consecutive(test_indices)
            train_datetimes = [
                common_date_range[chunk] for chunk in train_chunks if len(chunk) > 0
            ]
            test_datetimes = [
                common_date_range[chunk] for chunk in test_chunks if len(chunk) > 0
            ]
            combination_train_test[combination_num] = {
                "train": train_datetimes,
                "test": test_datetimes,
            }

        # Assign precomputed indices to each ticker
        for ticker in self.data_info.keys():
            self.backtest_paths[ticker] = paths
            for combination_num in combination_train_test:
                if combination_num not in self.combcv_dict:
                    self.combcv_dict[combination_num] = {}
                self.combcv_dict[combination_num][ticker] = combination_train_test[
                    combination_num
                ]

        # Store the mapping between indices and dates for equity curve reconstruction
        self.date_mapping = {}
        for p in range(paths.shape[1]):
            self.date_mapping[p] = {}
            for fold in np.unique(paths[:, p]):
                indices = np.where(paths[:, p] == fold)[0]
                self.date_mapping[p][fold] = common_date_range[indices]

        logging.info(f"Created {len(self.combcv_dict)} combinatorial splits")
        return self.combcv_dict

    def load_multiple_tickers(self, tickers, data_dir=None, date_range=None):
        """
        Load data for multiple tickers in parallel using multiprocessing.

        Args:
            tickers (list): List of ticker symbols to load
            data_dir (str, optional): Directory to load data from
            date_range (tuple or DatetimeIndex, optional): Date range to filter data

        Returns:
            dict: Dictionary mapping tickers to DataFrames
        """
        if not tickers:
            return {}

        # Determine number of worker processes to use
        n_workers = min(self.n_jobs, len(tickers), 32)  # Cap at 32 workers max

        if n_workers <= 1:
            # Single process loading
            result = {}
            for ticker in tqdm(tickers, desc="Loading ticker data"):
                df = self.load_ticker_data(
                    ticker, date_range=date_range, data_dir=data_dir
                )
                if df is not None and not df.empty:
                    result[ticker] = df
            return result

        # Prepare arguments for parallel processing
        load_args = [(ticker, data_dir or "", date_range) for ticker in tickers]

        # Use process pool for parallel loading
        logging.info(
            f"Loading {len(tickers)} tickers using {n_workers} parallel workers"
        )
        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(load_single_ticker, load_args),
                    total=len(load_args),
                    desc="Loading ticker data",
                )
            )

        # Process results into a dictionary
        data_dict = {}
        loaded_count = 0
        total_rows = 0

        for ticker, df in results:
            if df is not None and not df.empty:
                data_dict[ticker] = df
                loaded_count += 1
                total_rows += len(df)

        logging.info(
            f"Successfully loaded {loaded_count}/{len(tickers)} tickers with {total_rows} total rows"
        )
        return data_dict


def convert_to_hdf5(args):
    optimizer, ticker, indices, data_dir, temp_dir = args
    try:
        df = optimizer.load_ticker_data(ticker, date_range=indices, data_dir=data_dir)
        if not df.empty:
            h5_path = os.path.join(temp_dir, f"{ticker}.h5")
            df.to_hdf(
                h5_path,
                key="data",
                mode="w",
                format="table",
                data_columns=True,
            )
            return True
        return False
    except Exception as e:
        logging.error(f"Error converting {ticker} to HDF5: {str(e)}")
        return False


def process_combination(
    combination, data_dir, param_names, optimizer=None, calculate_metrics=None, n_jobs=1
):
    """
    Process a parameter combination for the backtest.

    Args:
        combination (tuple): Parameter combination to test.
        data_dir (str): Directory with data files.
        param_names (list): List of parameter names.
        optimizer (ParameterOptimizer, optional): Optimizer instance with calc_pl method.
        calculate_metrics (callable, optional): Function to calculate additional metrics.
        n_jobs (int): Number of parallel workers for data loading.

    Returns:
        dict: Dictionary with parameters, returns and metrics if successful, None otherwise.
    """
    start_time = time.time()
    params = dict(zip(param_names, combination))
    logging.info(f"Processing parameter combination: {params}")

    try:
        # Use the optimizer instance passed in, or create a temporary one if needed
        temp_optimizer = None
        if optimizer is None:
            # Create temporary ParameterOptimizer for this combination
            raise ValueError("optimizer parameter must be provided")

        # Get ticker files
        ticker_files = get_ticker_filenames(data_dir)
        if not ticker_files:
            logging.warning(f"No ticker files found in {data_dir}")
            return None

        logging.info(
            f"Found {len(ticker_files)} ticker files, loading with {n_jobs} workers"
        )

        # Prepare data structure
        all_data = {}

        # Use multiprocessing for data loading if n_jobs > 1
        if n_jobs > 1:
            # Prepare arguments for parallel processing
            load_args = [(ticker, data_dir, None) for ticker in ticker_files]

            # Use process pool for parallel loading
            with Pool(processes=n_jobs) as pool:
                results = list(
                    tqdm(
                        pool.imap(load_single_ticker, load_args),
                        total=len(load_args),
                        desc="Loading ticker data",
                    )
                )

            # Process results
            loaded_data_count = 0
            total_rows = 0

            for ticker, df in results:
                if df is not None and not df.empty:
                    all_data[ticker] = df
                    loaded_data_count += 1
                    total_rows += len(df)
        else:
            # Single-process loading for n_jobs = 1
            loaded_data_count = 0
            total_rows = 0

            for ticker in tqdm(ticker_files, desc="Loading ticker data"):
                df = optimizer.load_ticker_data(ticker, data_dir=data_dir)
                if df is not None and not df.empty:
                    all_data[ticker] = df
                    loaded_data_count += 1
                    total_rows += len(df)

        if loaded_data_count == 0:
            logging.error("Failed to load any valid data")
            return None

        logging.info(
            f"Loaded {loaded_data_count} ticker datasets with {total_rows} total rows"
        )

        # Calculate returns for the combination
        try:
            returns = optimizer.calc_pl(all_data, params)

            # If returned None or NaN, this combination isn't viable
            if returns is None or (isinstance(returns, float) and math.isnan(returns)):
                logging.warning(f"Invalid returns for combination {params}: {returns}")
                return None

            # Calculate additional metrics if needed
            metrics = {}
            if calculate_metrics:
                metrics = calculate_metrics(returns)

            duration = time.time() - start_time
            logging.info(
                f"Finished processing combination in {duration:.2f}s with valid returns"
            )

            # Return the results
            result = {"parameters": params, "returns": returns, "metrics": metrics}
            return result

        except Exception as e:
            logging.error(
                f"Error calculating returns for combination {params}: {str(e)}"
            )
            logging.debug(f"Exception details: {traceback.format_exc()}")
            return None

    except Exception as e:
        logging.error(f"Error processing combination {params}: {str(e)}")
        logging.debug(f"Exception details: {traceback.format_exc()}")
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


def _load_single_ticker(args):
    optimizer, ticker, indices, data_dir = args
    df = optimizer.load_ticker_data(ticker, date_range=indices, data_dir=data_dir)
    return ticker, df


def get_ticker_filenames(data_dir):
    """
    Get a list of ticker filenames from a directory.

    Args:
        data_dir (str): Directory containing data files

    Returns:
        list: List of ticker names (without file extensions)
    """
    import glob
    import os

    # Supported file types
    extensions = ["*.parquet", "*.csv", "*.csv.gz"]
    ticker_files = []

    for ext in extensions:
        pattern = os.path.join(data_dir, ext)
        files = glob.glob(pattern)
        for file_path in files:
            # Extract ticker name from filename (without extension)
            filename = os.path.basename(file_path)
            ticker = filename.split(".")[0]
            ticker_files.append(ticker)

    return list(set(ticker_files))  # Remove duplicates


def load_single_ticker(args):
    """
    Helper function to load a single ticker's data file, suitable for multiprocessing.

    Args:
        args (tuple): Tuple containing (ticker, data_dir, date_range)
            ticker (str): Ticker symbol
            data_dir (str): Directory with data files
            date_range (optional): Date range to filter data

    Returns:
        tuple: (ticker, DataFrame or None)
    """
    ticker, data_dir, date_range = args

    try:
        # Check for different file formats
        extensions = [".parquet", ".csv", ".csv.gz"]
        file_path = None
        file_format = None

        for ext in extensions:
            possible_path = os.path.join(data_dir, f"{ticker}{ext}")
            if os.path.exists(possible_path):
                file_path = possible_path
                file_format = ext.lstrip(".")
                break

        if file_path is None:
            logging.debug(f"No data file found for {ticker} in {data_dir}")
            return ticker, None

        # Load based on file format
        if file_format == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format in ["csv", "csv.gz"]:
            if file_format == "csv.gz":
                df = pd.read_csv(file_path, compression="gzip")
            else:
                df = pd.read_csv(file_path)

            # Ensure the DataFrame has a DatetimeIndex - use first column as date
            first_col = df.columns[0]
            df.set_index(first_col, inplace=True)
            df.index = pd.to_datetime(df.index)
        else:
            logging.warning(f"Unsupported file format: {file_format}")
            return ticker, None

        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort the index to ensure chronological order
        df = df.sort_index()

        if len(df) == 0:
            logging.warning(f"Loaded empty DataFrame for {ticker}")
            return ticker, None

        # Apply date range filtering if needed
        if date_range is not None:
            if isinstance(date_range, pd.DatetimeIndex):
                filtered_df = df.loc[df.index.isin(date_range)]
                return ticker, filtered_df
            else:
                start_date, end_date = date_range
                filtered_df = df.loc[start_date:end_date]
                return ticker, filtered_df

        return ticker, df

    except Exception as e:
        logging.warning(f"Error loading data for {ticker}: {str(e)}")
        return ticker, None
