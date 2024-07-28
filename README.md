# backtest_optimizer
Optimization of trading strategy hyperparameters with combinatorial cross validation and stress tesing

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To get a local copy up and running, follow these simple steps:

### Prerequisites

Ensure you have the following software installed on your system:

- **Python**: You can download it from [python.org](https://www.python.org/downloads/).
- **Git**: You can download it from [git-scm.com](https://git-scm.com/downloads).

### Install the Repository

pip install git+https://github.com/alex33d/backtest_optimizer.git

## Usage

## Usage

To start using the Optimizer, you need to import the `ParameterOptimizer` class from `backtest_optimizer/main.py`. The main functions are:

1. **Initialize the Optimizer**:

    ```python
    from backtest_optimizer.main import ParameterOptimizer

    optimizer = ParameterOptimizer(calc_pl)
    ```

    - `calc_pl`: A function for calculating strategy profit and loss. It must accept a dictionary with data in the form `{ticker: DataFrame}`, and return the parameter you want to maximize (e.g., Sharpe ratio, returns).

2. **Train/Test Split**:

    ```python
    optimizer.split_data(data_dict, train_end)
    ```

    - `data_dict`: Dictionary containing data in the form `{ticker: DataFrame}`.
    - `train_end`: The train end date in the form of a pandas DateTime.

3. **Create Combinatorial Cross Validation Groups**:

    ```python
    optimizer.create_combcv_dict(n_splits, n_test_splits)
    ```

    - `n_splits`: Total number of groups.
    - `n_test_splits`: Number of test groups.

4. **Run the Optimization Algorithm**:

    ```python
    optimizer.optimize(params=params_dict, n_jobs=3, n_runs=10, best_trials_pct=0.25)
    ```

    - `params`: The parameter dictionary with hyperparameter space for Optuna optimization.
    - `n_jobs`: Number of parallel workers.
    - `n_runs`: Number of Optuna trials.
    - `best_trials_pct`: Percentage of trials that go to the validation set.

5. **Reconstruct Equity Curves**:

    ```python
    optimizer.reconstruct_equity_curves()
    ```

    - Reconstructs the equity curves based on validation sets.

6. **Run Stress Tests**:

    ```python
    optimizer.run_stress_tests()
    ```

    - Runs backtest stress tests.

7. **Select Best Parameters**:

    ```python
    best_params = optimizer.cluster_and_aggregate()
    ```

    - Selects the best parameters via clustering.

8. **Calculate Out-of-Sample Sharpe Ratio**:

    ```python
    optimizer.calc_oos_sharpe(best_params)
    ```

    - Tests the best parameters on the test set.

### Example Usage

Here’s a complete example of how you might use the Optimizer class:

```python
from backtest_optimizer.main import ParameterOptimizer

# Define your profit and loss calculation function
def calc_pl(data):
    # Calculate P&L logic here
    pass

# Initialize the optimizer
optimizer = ParameterOptimizer(calc_pl)

# Split data into train and test sets
data_dict = {'AAPL': data_frame_aapl, 'GOOG': data_frame_goog}
train_end = pd.to_datetime('2023-01-01')
optimizer.split_data(data_dict, train_end)

# Create combinatorial cross-validation groups
optimizer.create_combcv_dict(n_splits=5, n_test_splits=1)

# Define parameter dictionary for optimization
params_dict = {
    'param1': [0, 1, 2],
    'param2': [0, 10, 20]
}

# Run optimization
optimizer.optimize(params=params_dict, n_jobs=3, n_runs=10, best_trials_pct=0.25)

# Reconstruct equity curves
optimizer.reconstruct_equity_curves()

# Run stress tests
optimizer.run_stress_tests()

# Select best parameters
best_params = optimizer.cluster_and_aggregate()

# Calculate out-of-sample Sharpe ratio
oos_sharpe = optimizer.calc_oos_sharpe(best_params)
