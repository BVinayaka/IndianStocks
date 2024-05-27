import yfinance as yf
import pandas as pd

def save_stock_data_to_csv(ticker, start_date, end_date, filename):
    # Fetching stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Saving data to CSV
    stock_data.to_csv(filename)

# Example usage:
ticker = 'AAPL'  # Apple Inc. stock
start_date = '2023-01-01'
end_date = '2024-01-01'
filename = 'stock_data.csv'

save_stock_data_to_csv(ticker, start_date, end_date, filename)
