# Portfolio
Python implementation of Modern Portfolio Theory (MPT) to find the efficient frontier and optimized Sharpe Ratio portfolios.

# Mean-Variance Portfolio Optimization (MVO)

## Project Overview
This repository contains a professional-grade Quantitative Finance tool that implements **Modern Portfolio Theory (MPT)**. The engine determines the optimal asset allocation for a given set of equities by analyzing historical risk-adjusted returns.

## Mathematical Methodology
The core of this project is the **Markowitz Optimization** framework. We solve for the weights $w$ that minimize portfolio variance for a target level of expected return:

$$\min_{w} \sigma_p^2 = w^T \Sigma w$$

**Key Features:**
* **Efficient Frontier Mapping:** Visualizing the risk-return trade-off.
* **Sharpe Ratio Maximization:** Finding the "Tangency Portfolio" using:
  $$S_p = \frac{E[R_p] - R_f}{\sigma_p}$$
* **Monte Carlo Simulation:** Generating 10,000+ random portfolios to visualize the feasible set.

## Tech Stack
* **Language:** Python 3.x
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Optimization:** `SciPy.optimize`
* **Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
