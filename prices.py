#importing packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop


#specifying the sample
ticker = '^GSPC'
start = '2021-06-20'
end = '2021-06-25'

#downloading data
prices = yf.download(ticker, start, end)['Close']

print(prices)