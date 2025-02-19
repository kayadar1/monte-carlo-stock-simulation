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

def plot_simulation(prices, ticker, past_actual_prices, past_simulation):
    plt.figure(figsize=(10, 5))
    
    avg_prices = np.mean(prices, axis=1)  # Average of future simulations
    avg_past_simulation = np.mean(past_simulation, axis=1)  # Average of past simulation
    
    days_past = np.arange(-len(past_actual_prices), 0)  # Days for past data
    days_future = np.arange(0, len(avg_prices))  # Days for simulated future data
    days_simulation = np.concatenate([days_past, days_future])  # Full timeline

    combined_simulation = np.concatenate([avg_past_simulation, avg_prices])  # Combine past & future
    combined_actual = past_actual_prices  # Only past actual prices

    plt.plot(days_simulation, combined_simulation, color='blue', linewidth=2, label='Simulated Price')
    plt.plot(days_past, combined_actual, color='red', linestyle='solid', linewidth=2, label='Actual Price')

    plt.axvline(0, color='black', linestyle='dashed', label='Today')  # Mark today
    plt.xlabel("Days (Past to Future Trading Days)")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Stock Price Simulation vs Actual - {ticker}")
    plt.legend()
    plt.show()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y")  # Fetch last 2 years
    
    print("Available Columns in Downloaded Data:", hist.columns)

    stock_data = hist['Close'].values
    log_returns = np.log(stock_data[1:] / stock_data[:-1])
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    # Ensure past_actual_prices has exactly 252 days
    past_actual_prices = stock_data[-504:-252]
    if len(past_actual_prices) != 252:
        past_actual_prices = past_actual_prices[-252:]  # Trim excess

    return stock_data[-1], mu, sigma, stock_data[-252:], past_actual_prices

def main():
    ticker = "AAPL"  # Example: Apple stock
    S0, mu, sigma, actual_prices, past_actual_prices = get_stock_data(ticker)
    days = 252  # 1 trading year
    simulations = 100  # Number of simulations

    past_simulation = monte_carlo_simulation(past_actual_prices[0], mu, sigma, days, simulations)  # Generate past projection
    future_simulation = monte_carlo_simulation(S0, mu, sigma, days, simulations)

    plot_simulation(future_simulation, ticker, past_actual_prices, past_simulation)

    df = pd.DataFrame(future_simulation)
    df.to_csv("monte_carlo_simulation.csv", index=False)
    print(f"Simulation data for {ticker} saved to monte_carlo_simulation.csv")

if __name__ == "__main__":
    main()





