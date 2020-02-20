# Implement standard Theta method of Assimakopoulos and Nikolopoulos (2000)
# This is Python implementation of thetaf() function in R by Prof. Hyndman 
# https://github.com/robjhyndman/forecast/blob/master/R/theta.R

# The standard theta method of Assimakopoulos and Nikolopoulos (2000) is equivalent to 
# simple exponential smoothing with drift. This is demonstrated in Hyndman and Billah (2003)

# Step 1: Check for seasonality
# Step 2: Decompose Seasonality if it is deemed seasonal
# Step 3: Applying Theta Method
# Step 4: Reseasonalize the resulting forecast


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

def sesThetaF(y, s_period , h = 10, level = np.array([90,95,99])):
	"""
	@param y : array-like time series data
	@param s_period : the no. of observations before seasonal pattern repeats
	@param h : number of period for forcasting
	@param level: confidence levels for prediction intervals
	"""
	if not s_period:
		print('ERROR: s_period variable only accepts positve integer.')
		sys.exit()


	fcast = {} # store result
	# Check seasonality
	x = y.copy()
	n = y.index.size
	m = s_period 

	if m > 1 and n > 2 * m:
		r = (acf(x, nlags = m))[1:]
		temp = np.delete(r, m-1)
		stat = np.sqrt((1+ 2 * np.sum(np.square(temp))) / n)
		seasonal = (abs(r[m-1])/stat) > norm.cdf(0.95)
	else:
		seasonal = False

	# Seasonal Decomposition
	origx = x.copy()
	if seasonal:
		decomp = seasonal_decompose(x, model = 'multiplicative')
		if decomp.seasonal < 1e-10 :
			warnings.warn('Seasonal indexes equal to zero. Using non-seasonal Theta method')
		else:
			x = decomp.observed/decomp.seasonal

	# Find theta lines
	model = SimpleExpSmoothing(x).fit()
	fcast['mean'] = model.forecast(h)
	num = np.array(range(0,n))
	temp = LinearRegression().fit(num.reshape(-1,1),x).coef_
	temp = temp/2
	alpha = np.maximum(1e-10, model.params['smoothing_level'])
	fcast['mean'] = fcast['mean'] + temp * (np.array(range(0,h)) + (1 - (1 - alpha)**n)/alpha)

	# Reseasonalize
	if seasonal:
		fcast['mean'] = fcast['mean'] *  np.repeat(decomp.seasonal[-m:], (1 + h//m))[:h]
		fcast['fitted'] = model.predict(x.index[0], x.index[n-1]) * decomp.seasonal
	else:
		fcast['fitted'] = model.predict(x.index[0], x.index[n-1])

	fcast['residuals'] = origx - fcast['fitted']

	return fcast
	# Prediction Intervals



    














	

