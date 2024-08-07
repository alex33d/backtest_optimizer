from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Backtest-Optimizer",  
    version="0.1.7",
    author="Alexnader Demachev",
    author_email="alexdemachev@gmail.com",
    description="Hyperparameter search with Combinatorial Cross Validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alex33d/backtest_optimizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
      'pandas',
      'scikit-learn',
      'optuna',
      'matplotlib',
      'scipy',
      'joblib',
      'statsmodels'
    ],
)
