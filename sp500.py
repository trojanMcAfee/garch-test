import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Download S&P 500 historical data for the past 5 years
ticker = "^GSPC"
end_date = pd.to_datetime("today")
start_date = end_date - pd.DateOffset(years=5)
data = yf.download(ticker, start=start_date, end=end_date)
data = data['Adj Close']

# Calculate daily returns
returns = 100 * np.log(data / data.shift(1)).dropna()

# Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
model_fitted = model.fit(disp='off')

# Extract model parameters
alpha_0 = model_fitted.params['omega']
alpha_1 = model_fitted.params['alpha[1]']
beta_1 = model_fitted.params['beta[1]']

# Generate conditional volatility series
garch_volatility = model_fitted.conditional_volatility

# Calculate realized volatility (rolling standard deviation)
realized_volatility = returns.rolling(window=22).std() * np.sqrt(252)  # 22 trading days in a month, annualize

# Plot realized volatility and GARCH volatility
plt.figure(figsize=(12, 6))
plt.plot(realized_volatility, label='Realized Volatility')
plt.plot(garch_volatility, label='GARCH Volatility', linestyle='--')
plt.title('S&P 500 Volatility: Realized vs GARCH(1,1) Model')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.show()

# Print the final values of the important variables
important_variables = {
    'alpha_0 (omega)': alpha_0,
    'alpha_1 (alpha[1])': alpha_1,
    'beta_1 (beta[1])': beta_1,
    'mu (constant)': model_fitted.params['mu']
}

print("Important Variables:")
for key, value in important_variables.items():
    print(f"{key}: {value}")
