import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop

# Specifying the sample
ticker = '^GSPC'
start = '2015-12-31'
end = '2021-06-25'

# Downloading data
prices = yf.download(ticker, start, end)['Close']

# Calculating returns
returns = np.array(prices)[1:] / np.array(prices)[:-1] - 1

# Starting parameter values - sample mean and variance
mean = np.average(returns)
var = np.std(returns) ** 2

def garch_mle(params):
    # Specifying model parameters
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    # Calculating long-run volatility
    long_run = (omega / (1 - alpha - beta)) ** (1 / 2)
    # Calculating realised and conditional volatility
    resid = returns - mu
    realised = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] = long_run
    for t in range(1, len(returns)):
        conditional[t] = (omega + alpha * resid[t-1] ** 2 + beta * conditional[t-1] ** 2) ** (1 / 2)
    # Calculating log-likelihood
    likelihood = 1 / ((2 * np.pi) ** (1 / 2) * conditional) * np.exp(-realised ** 2 / (2 * conditional ** 2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood

# Maximising log-likelihood
res = spop.minimize(garch_mle, [mean, var, 0, 0], method='Nelder-Mead')

# Retrieving optimal parameters
params = res.x
mu = res.x[0]
omega = res.x[1]
alpha = res.x[2]
beta = res.x[3]
log_likelihood = -float(res.fun)

# Calculating realised and conditional volatility for optimal parameters
long_run = (omega / (1 - alpha - beta)) ** (1 / 2)
resid = returns - mu
realised = abs(resid)
conditional = np.zeros(len(returns))
conditional[0] = long_run
for t in range(1, len(returns)):
    conditional[t] = (omega + alpha * resid[t-1] ** 2 + beta * conditional[t-1] ** 2) ** (1 / 2)

# Plotting the first chart with realized and conditional volatilities from 2016 to 2021
plt.figure(figsize=(12, 6))
plt.plot(prices.index[1:], realised, label='Realized Volatility')
plt.plot(prices.index[1:], conditional, label='Conditional Volatility')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Realized and Conditional Volatility (2016 - 2021)')
plt.show()

# New data for the next days
new_data = {
    '2021-06-25': 4280.700195,
    '2021-06-28': 4290.609863,
    '2021-06-29': 4291.799805,
    '2021-06-30': 4297.500000,
    '2021-07-01': 4319.939941,
    '2021-07-02': 4352.339844
}

# Converting to pandas series
new_prices = pd.Series(new_data)

# Calculating returns for new data
new_returns = np.array(new_prices)[1:] / np.array(new_prices)[:-1] - 1

# Initialize the conditional volatility with the last value from the previous period
last_conditional_volatility = conditional[-1]

# Initialize a new array for the new conditional volatility
new_conditional = np.zeros(len(new_returns))
new_conditional[0] = (omega + alpha * (new_returns[0] - mu) ** 2 + beta * last_conditional_volatility ** 2) ** (1 / 2)

for t in range(1, len(new_returns)):
    new_conditional[t] = (omega + alpha * (new_returns[t-1] - mu) ** 2 + beta * new_conditional[t-1] ** 2) ** (1 / 2)

# Calculating the new realized volatilities
new_realised = abs(new_returns - mu)

# Plotting the second chart with realized and conditional volatilities for new data
plt.figure(figsize=(12, 6))
plt.plot(new_prices.index[1:], new_realised, label='Realized Volatility', color='blue')
plt.plot(new_prices.index[1:], new_conditional, label='Conditional Volatility', color='orange')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Realized and Conditional Volatility (June 25, 2021 - July 2, 2021)')
plt.show()
