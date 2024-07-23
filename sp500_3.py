#importing packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop

#specifying the sample
ticker = '^GSPC'
start = '2015-12-31'
end = '2021-06-25'

#downloading data
prices = yf.download(ticker, start, end)['Close']

#calculating returns
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1

#starting parameter values - sample mean and variance
mean = np.average(returns)
var = np.std(returns)**2
def garch_mle(params):
    #specifying model parameters
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    #calculating long-run volatility
    long_run = (omega/(1 - alpha - beta))**(1/2)
    #calculating realised and conditional volatility
    resid = returns - mu
    realised = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] =  long_run
    for t in range(1,len(returns)):
        conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
    #calculating log-likelihood
    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood

#maximising log-likelihood
res = spop.minimize(garch_mle, [mean, var, 0, 0], method='Nelder-Mead')

#retrieving optimal parameters
params = res.x
mu = res.x[0]
omega = res.x[1]
alpha = res.x[2]
beta = res.x[3]
log_likelihood = -float(res.fun)

#calculating realised and conditional volatility for optimal parameters
long_run = (omega/(1 - alpha - beta))**(1/2)
resid = returns - mu
realised = abs(resid)
conditional = np.zeros(len(returns))
conditional[0] =  long_run
for t in range(1,len(returns)):
    conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
    
#printing optimal parameters
print('GARCH model parameters')
print('')
print('mu '+str(round(mu, 6)))
print('omega '+str(round(omega, 6)))
print('alpha '+str(round(alpha, 4)))
print('beta '+str(round(beta, 4)))
print('long-run volatility '+str(round(long_run, 4)))
print('log-likelihood '+str(round(log_likelihood, 4)))

# visualising the results
plt.figure(1)
plt.rc('xtick', labelsize = 10)
plt.plot(prices.index[1:],realised, label='Realised Volatility')
plt.plot(prices.index[1:],conditional, label='Conditional Volatility')
plt.legend()
plt.show()

# Projections with random shocks
days_ahead = 15
num_simulations = 100  # Number of simulations to average out

projection_matrix = np.zeros((num_simulations, days_ahead))

for sim in range(num_simulations):
    # Simulate future returns and volatility
    projections = np.zeros(days_ahead)
    projections[0] = conditional[-1]  # starting from the last conditional volatility value
    shocks = np.random.normal(0, 1, days_ahead)  # Generate random shocks

    for t in range(1, days_ahead):
        projections[t] = (omega + alpha * (shocks[t-1]**2) + beta * projections[t-1]**2)**(1/2)
    
    projection_matrix[sim] = projections

# Average projections
mean_projections = projection_matrix.mean(axis=0)

# Plotting the average projections with randomness
plt.figure(2)
plt.plot(range(1, days_ahead + 1), mean_projections, label='Mean Projections with Random Shocks')
plt.xlabel('Days')
plt.ylabel('Volatility')
plt.title('GARCH Volatility Projections with Random Shocks')
plt.legend()
plt.show()
