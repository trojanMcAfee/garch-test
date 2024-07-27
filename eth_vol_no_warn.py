#importing packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop

#specifying the sample
ticker = 'ETH-USD'
start = '2020-01-01'
end = '2024-07-22'

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
    
    # Ensure non-negative parameters for stability
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    
    #calculating long-run volatility
    long_run = (omega/(1 - alpha - beta))**(1/2)
    
    # Ensure long_run is valid
    if np.isnan(long_run) or long_run <= 0:
        return np.inf
    
    #calculating realised and conditional volatility
    resid = returns - mu
    realised = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] =  long_run
    for t in range(1,len(returns)):
        conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
        if np.isnan(conditional[t]) or conditional[t] <= 0:
            return np.inf
    
    #calculating log-likelihood
    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood

#maximising log-likelihood
res = spop.minimize(garch_mle, [mean, var, 0.1, 0.8], method='Nelder-Mead')

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

print(conditional[-1])

# #printing optimal parameters
# print('GARCH model parameters')
# print('')
# print('mu '+str(round(mu, 6)))
# print('omega '+str(round(omega, 6)))
# print('alpha '+str(round(alpha, 4)))
# print('beta '+str(round(beta, 4)))
# print('long-run volatility '+str(round(long_run, 4)))
# print('log-likelihood '+str(round(log_likelihood, 4)))

# #visualising the results
# plt.figure(1)
# plt.rc('xtick', labelsize = 10)
# plt.plot(prices.index[1:],realised)
# plt.plot(prices.index[1:],conditional)
# plt.show()
