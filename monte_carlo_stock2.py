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
    
    # Ensure past_simulation and past_actual_prices have exactly 252 days
    avg_past_simulation = avg_past_simulation[-252:]
    past_actual_prices = past_actual_prices[-252:]
    
    # Ensure continuity: set the last past simulated value to match today's price
    # Adjust past simulation to smoothly connect to today's actual price
    scaling_factor = past_actual_prices[-1] / avg_past_simulation[-1]
    avg_past_simulation *= scaling_factor  # Scale past simulation to match the actual last price
    
    # Days arrays
    days_past = np.arange(-252, 0)  # Last year's actual trading days
    days_future = np.arange(0, 252)  # Next year's predicted trading days
    
    combined_simulation = np.concatenate([avg_past_simulation, avg_prices])  # Full simulated data
    days_simulation = np.concatenate([days_past, days_future])  # Full timeline
    
    plt.plot(days_simulation, combined_simulation, color='blue', linewidth=2, label='Simulated Price')
    plt.plot(days_past, past_actual_prices, color='red', linestyle='solid', linewidth=2, label='Actual Price')
    
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
    if len(past_actual_prices) > 252:
        past_actual_prices = past_actual_prices[-252:]  # Trim excess
    elif len(past_actual_prices) < 252:
        past_actual_prices = np.pad(past_actual_prices, (252 - len(past_actual_prices), 0), mode='edge')  # Pad if missing days
    
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







