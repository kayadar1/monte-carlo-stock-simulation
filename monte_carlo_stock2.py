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

def plot_simulation(prices, ticker, actual_prices, past_simulation, past_actual_prices):
    plt.figure(figsize=(10, 5))
    avg_prices = np.mean(prices, axis=1)  # Calculate average across all simulations
    avg_past_simulation = np.mean(past_simulation, axis=1)  # Average of past simulation
    
    days_past = np.arange(-len(actual_prices), 0)  # Days for actual past data
    days_future = np.arange(0, len(avg_prices))  # Days for simulated future data
    
    plt.plot(days_past, actual_prices, color='red', linestyle='dashed', linewidth=2, label='Actual Price (Past)')
    plt.plot(days_future, avg_prices, color='blue', linewidth=2, label='Simulated Average Price (Future)')
    
    # Past Projection
    days_past_simulation = np.arange(-len(past_actual_prices), -len(past_actual_prices) + len(avg_past_simulation))
    plt.plot(days_past_simulation, avg_past_simulation, color='green', linestyle='dashed', linewidth=2, label='Simulated Projection (Last Year)')
    plt.plot(days_past_simulation, past_actual_prices, color='purple', linestyle='dotted', linewidth=2, label='Actual Price (Last Year)')
    
    plt.axvline(0, color='black', linestyle='dotted', label='Today')  # Mark separation between past and future
    plt.xlabel("Days (Past to Future Trading Days)")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Stock Price Simulation vs Actual - {ticker}")
    plt.legend()
    plt.show()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")  # Fetch last 2 years to compare past projections
    
    # Debugging: Print columns to ensure "Close" exists
    print("Available Columns in Downloaded Data:", hist.columns)
    
    stock_data = hist['Close'].values
    log_returns = np.log(stock_data[1:] / stock_data[:-1])
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    return stock_data[-1], mu, sigma, stock_data[-252:], stock_data[-504:-252]  # Return current data and past year prices

def main():
    # Fetch real stock data
    ticker = "AAPL"  # Example: Apple stock
    S0, mu, sigma, actual_prices, past_actual_prices = get_stock_data(ticker)
    days = 252  # 1 trading year
    simulations = 100  # Number of simulations
    
    prices = monte_carlo_simulation(S0, mu, sigma, days, simulations)
    past_simulation = monte_carlo_simulation(past_actual_prices[0], mu, sigma, days, simulations)  # Generate past projection
    
    plot_simulation(prices, ticker, actual_prices, past_simulation, past_actual_prices)
    
    # Save to CSV
    df = pd.DataFrame(prices)
    df.to_csv("monte_carlo_simulation.csv", index=False)
    print(f"Simulation data for {ticker} saved to monte_carlo_simulation.csv")

if __name__ == "__main__":
    main()



