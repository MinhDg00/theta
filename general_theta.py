# Implement standard Theta method based on Fiorucci et al. (2016) 
# Reference: https://github.com/cran/forecTheta

import sys
import numpy as np 
import pandas as pd 
import statsmodels as sm 
import warnings 
from scipy.stats import norm 
from statsmodels.tsa.stattools import acf 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression


def sThetaF(y, s_period = 1, h = 10, s = None):
	"""
	@param y : array-like time series data
	@param s_period : the no. of observations before seasonal pattern repeats
	@param h : number of period for forcasting
	@s : additive or multiplicative 
	"""
	fcast = {} # store result
	n = y.index.size
	x = y.copy()
	m = s_period
	time_y = np.array(np.arange(n))/m + 1
	time_fc = time_y[n-1] + np.array(np.arange(1,h+1))/m

	s_type = 'multiplicative'
	if s is not None:
		if s == 'additive':
			s = True
			s_type = 'additive'

	# Seasonality Test & Decomposition
	if s is not None and m >= 4:
		r = (acf(x, nlags = m+1))[1:]
		clim = 1.64/sqrt(n) * np.sqrt(np.cumsum([1, 2 * np.square(r)]))
		s = abs(r[m-1]) > clim[m-1]
	else:
		if not s:
			s = False


	if s: 
		decomp = seasonal_decompose(x, model = s_type)
		if s_type == 'additive' or (s_type -- 'multiplicative' and any(decomp < 0.01)): 
			s_type = 'additive'
			decomp = seasonal_decompose(x, model = 'additive').seasonal
			x = x - decomp
		else:
			x = x/decomp


	## Find Theta Line
	model = LinearRegression().fit(time_y.reshape(-1,1), x)
	fcast['mean'] = model.intercept_ + model.coef_ * time_fc 
	
	return fcast
	




