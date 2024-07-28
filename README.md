# Backtest Optimizer
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
```


## Contributing

I welcome contributions to the project! Here’s how you can help:

### Reporting Issues

If you find a bug or have a feature request, please create an issue on GitHub.

1. Go to the [Issues](https://github.com/yourusername/my-python-project/issues) page.
2. Click on the "New issue" button.
3. Provide a clear and descriptive title.
4. Describe the issue or feature request in detail, including steps to reproduce the issue if applicable.

### Submitting Pull Requests

If you want to contribute code, follow these steps to submit a pull request:

1. **Fork the Repository**

    Click the "Fork" button at the top right of the repository page to create a copy of the repository on your GitHub account.

2. **Clone Your Fork**

    Clone your forked repository to your local machine:

    ```bash
    git clone https://github.com/alex33d/backtest_optimizer.git
    cd backtest_optimizer
    ```

3. **Create a Branch**

    Create a new branch for your feature or bug fix:

    ```bash
    git checkout -b my-feature-branch
    ```

4. **Make Your Changes**

    Make your changes to the codebase. Ensure your code follows the project's coding standards.

5. **Commit Your Changes**

    Commit your changes with a descriptive commit message:

    ```bash
    git add .
    git commit -m "Description of your changes"
    ```

6. **Push to Your Fork**

    Push your changes to your forked repository:

    ```bash
    git push origin my-feature-branch
    ```

7. **Create a Pull Request**

    Go to the original repository on GitHub and click the "New pull request" button.

    - Select your branch from the "compare" dropdown.
    - Provide a clear and descriptive title for your pull request.
    - Describe your changes in detail in the description field.
    - Submit the pull request.

### Coding Guidelines

To maintain consistency in the codebase, please adhere to the following guidelines:

- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- Write descriptive commit messages.
- Include docstrings for all functions and classes.
- Write unit tests for new features and bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, suggestions, or feedback, feel free to contact me:

- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/alexander-demachev-b067759b/) 


