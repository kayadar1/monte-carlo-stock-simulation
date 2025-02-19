import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

def monte_carlo_simulation(S0, mu, sigma, days, simulations):
    dt = 1  # 1-day time step
    prices = np.zeros((days, simulations))
    prices[0] = S0  # Initial stock price
    
    for t in range(1, days):
        epsilon = np.random.normal(0, 1, simulations)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * epsilon * np.sqrt(dt))
    
    return prices

def plot_simulation(prices, ticker, actual_prices):
    plt.figure(figsize=(10, 5))
    avg_prices = np.mean(prices, axis=1)  # Average of future simulations
    days_past = np.arange(-len(actual_prices), 0)  # Past trading days
    days_future = np.arange(0, len(avg_prices))  # Future trading days
    
    # Plot actual stock prices
    plt.plot(days_past, actual_prices, color='red', linestyle='solid', linewidth=2, label='Actual Price (Last Year)')
    
    # Plot future Monte Carlo projection
    plt.plot(days_future, avg_prices, color='blue', linewidth=2, label='Simulated Price (Next Year)')
    
    plt.axvline(0, color='black', linestyle='dashed', label='Today')  # Mark today
    plt.xlabel("Days (Past to Future Trading Days)")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Stock Price Simulation - {ticker}")
    plt.legend()
    plt.show()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  # Fetch last 1 year
    
    print("Available Columns in Downloaded Data:", hist.columns)
    
    stock_data = hist['Close'].values
    log_returns = np.log(stock_data[1:] / stock_data[:-1])
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    return stock_data[-1], mu, sigma, stock_data[-252:]  # Ensure last 252 trading days

def main():
    tickers = ["AAPL", "MSFT", "GOOGL"]  # List of stocks to simulate
    days = 252  # 1 trading year
    simulations = 100  # Number of simulations
    
    for ticker in tickers:
        print(f"Running Monte Carlo Simulation for {ticker}...")
        S0, mu, sigma, actual_prices = get_stock_data(ticker)
        future_simulation = monte_carlo_simulation(S0, mu, sigma, days, simulations)
        
        # Save each stock’s data separately
        df = pd.DataFrame(future_simulation)
        df.to_csv(f"monte_carlo_simulation_{ticker}.csv", index=False)
        
        # Plot each stock’s simulation
        plot_simulation(future_simulation, ticker, actual_prices)
    
    print("Simulations completed for all stocks!")

if __name__ == "__main__":
    main()












