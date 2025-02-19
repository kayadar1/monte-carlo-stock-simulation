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
    avg_prices = np.mean(prices, axis=1)  # Calculate average across all simulations
    days = np.arange(len(avg_prices))  # Create an array for the days
    
    plt.plot(days, avg_prices, color='blue', linewidth=2, label='Simulated Average Price')
    plt.plot(range(len(actual_prices)), actual_prices, color='red', linestyle='dashed', linewidth=2, label='Actual Price')
    
    plt.xlabel("Days (Future Trading Days)")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Stock Price Simulation vs Actual - {ticker}")
    plt.legend()
    plt.show()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    # Debugging: Print columns to ensure "Close" exists
    print("Available Columns in Downloaded Data:", hist.columns)
    
    stock_data = hist['Close']  # Use 'Close' instead of 'Adj Close'
    log_returns = np.log(stock_data / stock_data.shift(1))
    mu = log_returns.mean()
    sigma = log_returns.std()
    return stock_data.iloc[-1], mu, sigma, stock_data.values  # Also return actual historical prices

def main():
    # Fetch real stock data
    ticker = "AAPL"  # Example: Apple stock
    S0, mu, sigma, actual_prices = get_stock_data(ticker)
    days = 252  # 1 trading year
    simulations = 100  # Number of simulations
    
    prices = monte_carlo_simulation(S0, mu, sigma, days, simulations)
    plot_simulation(prices, ticker, actual_prices)
    
    # Save to CSV
    df = pd.DataFrame(prices)
    df.to_csv("monte_carlo_simulation.csv", index=False)
    print(f"Simulation data for {ticker} saved to monte_carlo_simulation.csv")

if __name__ == "__main__":
    main()



