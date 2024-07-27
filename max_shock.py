import yfinance as yf

# Download ETH-USD historical data since inception
ticker = "ETH-USD"
data = yf.download(ticker, start="2015-08-07")  # Ethereum started trading around August 7, 2015
data = data['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate the largest return shock (absolute value)
max_return_shock = returns.max()
min_return_shock = returns.min()

print("Largest positive return shock:", max_return_shock)
print("Largest negative return shock:", min_return_shock)
