#importing packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop


#specifying the sample
ticker = '^GSPC'
start = '2021-06-19'
end = '2021-06-25'

#downloading data
prices = yf.download(ticker, start, end)['Close']

print(prices)


# 2021-06-21    4224.790039
# 2021-06-22    4246.439941
# 2021-06-23    4241.839844
# 2021-06-24    4266.490234