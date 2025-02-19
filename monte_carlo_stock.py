import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def monte_carlo_simulation(stock_ticker, start_date, end_date, num_simulations=1000, time_horizon=252):
    # Fetch historical data
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    # Debugging: Print the first few rows and available columns
    print("Retrieved Data:\n", stock_data.head())
    print("Available columns:", stock_data.columns)

    # Ensure valid columns exist
    if 'Adj Close' in stock_data.columns:
        stock_data = stock_data['Adj Close']
    elif 'Close' in stock_data.columns:  # Use 'Close' if 'Adj Close' is unavailable
        stock_data = stock_data['Close']
    else:
        raise ValueError(f"No valid price data found for {stock_ticker}. Check available columns: {stock_data.columns}")

    # Check if data is empty
    if stock_data.empty:
        raise ValueError(f"No data found for {stock_ticker} in the given date range {start_date} to {end_date}.")

    # Calculate daily returns
    log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
    mu = log_returns.mean()  # Average return
    sigma = log_returns.std()  # Volatility

    # Get last stock price
    last_price = stock_data.iloc[-1]

    # Monte Carlo simulation
    simulations = np.zeros((time_horizon, num_simulations))
    for i in range(num_simulations):
        price_series = [last_price]
        for _ in range(time_horizon - 1):
            next_price = price_series[-1] * np.exp(np.random.normal(mu, sigma))
            price_series.append(next_price)
        simulations[:, i] = price_series

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(simulations, alpha=0.2, color='blue')
    plt.title(f"Monte Carlo Simulation of {stock_ticker} Stock Price")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.show()

# Example Usage
monte_carlo_simulation('GOOG', '2023-01-01', '2024-01-01')

