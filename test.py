#!/usr/bin/env python3
import os
import logging
import pandas as pd
from backtest_optimizer.main import ParameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Test parameters
DATA_DIR = "/Users/alexanderdemachev/PycharmProjects/strategy/data/futures/1min/"
SAVE_PATH = "./test_results/"

# Create save directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

def test_collect_data_info():
    """
    Test the collect_data_info method and verify the data_info dictionary.
    """
    logging.info("=== Testing collect_data_info ===")
    
    # Initialize optimizer with dummy calc_pl function
    def dummy_calc_pl(data, params):
        return pd.Series()
    
    optimizer = ParameterOptimizer(
        calc_pl=dummy_calc_pl,
        save_path=SAVE_PATH,
        save_file_prefix="test_",
        n_jobs=1
    )
    
    # Collect data info
    optimizer.collect_data_info(DATA_DIR)
    
    # Log the results
    logging.info(f"Found {len(optimizer.data_info)} tickers")
    
    # Check all fields for each ticker
    for ticker, info in optimizer.data_info.items():
        logging.info(f"\nTicker: {ticker}")
        logging.info(f"  Path: {info['path']}")
        logging.info(f"  Format: {info['format']}")
        logging.info(f"  Start Date: {info['start_date']}")
        logging.info(f"  End Date: {info['end_date']}")
        logging.info(f"  Frequency: {info['freq']}")
        
        # Verify that all required fields exist
        assert 'path' in info, f"Missing 'path' for {ticker}"
        assert 'format' in info, f"Missing 'format' for {ticker}"
        assert 'start_date' in info, f"Missing 'start_date' for {ticker}"
        assert 'end_date' in info, f"Missing 'end_date' for {ticker}"
        assert 'freq' in info, f"Missing 'freq' for {ticker}"
        
        # Verify that dates are valid
        assert info['start_date'] is not None, f"Invalid start_date for {ticker}"
        assert info['end_date'] is not None, f"Invalid end_date for {ticker}"
        assert info['start_date'] <= info['end_date'], f"Start date is after end date for {ticker}"
    
    logging.info("All fields verified successfully")
    
    return optimizer

def test_create_combcv_dict(optimizer, n_splits_list, n_test_splits_list):
    """
    Test create_combcv_dict with various n_splits and n_test_splits combinations.
    
    Args:
        optimizer: Initialized ParameterOptimizer instance
        n_splits_list: List of n_splits values to test
        n_test_splits_list: List of n_test_splits values to test
    """
    logging.info("\n=== Testing create_combcv_dict ===")
    
    for n_splits in n_splits_list:
        for n_test_splits in n_test_splits_list:
            # Skip invalid combinations
            if n_test_splits > n_splits:
                continue
                
            logging.info(f"\nTesting with n_splits={n_splits}, n_test_splits={n_test_splits}")
            
            # Clear existing combcv_dict
            optimizer.combcv_dict = {}
            optimizer.backtest_paths = {}
            
            try:
                # Create combinatorial CV dictionary
                optimizer.create_combcv_dict(DATA_DIR, n_splits=n_splits, n_test_splits=n_test_splits)
                
                # Log results
                logging.info(f"Created {len(optimizer.combcv_dict)} combinations")
                
                # Verify the first combination
                if optimizer.combcv_dict:
                    first_combo = next(iter(optimizer.combcv_dict.values()))
                    first_ticker = next(iter(first_combo.keys()))
                    train_indices = first_combo[first_ticker]["train"]
                    test_indices = first_combo[first_ticker]["test"]
                    
                    logging.info(f"First ticker: {first_ticker}")
                    logging.info(f"Number of training segments: {len(train_indices)}")
                    if test_indices:
                        logging.info(f"Number of testing segments: {len(test_indices)}")
                    else:
                        logging.info("No test indices (using all data for training)")
                    
                    # Verify paths match the number of combinations
                    if first_ticker in optimizer.backtest_paths:
                        paths_shape = optimizer.backtest_paths[first_ticker].shape
                        logging.info(f"Backtest paths shape: {paths_shape}")
                
                # Test access to data in the combination
                if len(optimizer.combcv_dict) > 0:
                    # Set current group for testing
                    optimizer.current_group = optimizer.combcv_dict[0]
                    
                    # Generate indices for training
                    train_indices = optimizer.generate_group_indices(is_train=True)
                    logging.info(f"Generated {len(train_indices) - 1} training groups")  # -1 for parameter_optimizer
                    
                    # Generate indices for testing (if applicable)
                    if n_test_splits > 0:
                        test_indices = optimizer.generate_group_indices(is_train=False)
                        logging.info(f"Generated {len(test_indices) - 1} testing groups")  # -1 for parameter_optimizer
                
            except Exception as e:
                logging.error(f"Error testing combination (n_splits={n_splits}, n_test_splits={n_test_splits}): {e}")

def main():
    """Main test function."""
    # Test data info collection
    optimizer = test_collect_data_info()
    
    # Test combcv dictionary creation with various parameters
    n_splits_list = [2, 5, 10]
    n_test_splits_list = [0, 1, 2, 3]
    test_create_combcv_dict(optimizer, n_splits_list, n_test_splits_list)
    
    logging.info("\nAll tests completed")

if __name__ == "__main__":
    main()
