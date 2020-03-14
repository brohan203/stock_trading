#!/usr/bin/env python
# Input: a text file with a ticker on each line.
# Dependends on NUMPY and PANDAS

import csv, math, os, shutil, datetime, urllib, matplotlib
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices
from sys import argv
import yfinance as yf
from forex_python.converter import get_rate
from sklearn.svm import SVC

# Tickers is a text file that contains all the tickers you want to analyze. 
# Start is the year you'd like to start on
script, txt = argv
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime.now()
#matplotlib.style.use('ggplot')

# Parameters:

#start = datetime.datetime(start, 1, 1)

print "#######################################################"
print "Beginning simulation starting in " + str(start)

if os.path.isdir('csvfiles'):
	shutil.rmtree('csvfiles')
os.mkdir('csvfiles')

# Step 1: record tickers
# This simple chunk opens the "txt" file containing tickers and records all of the tickers in a list
def get_tickers(textfile):
	tickers = []
	with open(textfile) as t:
		tickers = t.readlines()
	tickers = [x.strip('\n') for x in tickers]
	print tickers
	return tickers

########################################################################################################
# Establish indicators and other important numbers:

# Moving average. Self explanatory. Data is a pandas dataframe, period is an integer
	# Apparently 9 and 13 are special numbers for this
def moving_average(data, period, type='simple'):
	"""
	Compute an n period moving average.
	type is 'simple' or 'exponential'
	"""
	try:
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)

	if type == 'simple':
		weights = np.ones(period)
	else:
		weights = np.exp(np.linspace(-1., 0., period))

	weights /= weights.sum()

	a = np.convolve(x, weights, mode='full')[:len(x)]
	a[:period] = a[period]
	return a

# Bollinger curves. Data is a pandas dataframe, period is an integer, multiplier is how many standard deviations to use. Usually people use 2 and -2
def bollinger(data, period, multiplier):
	try:
		x = data['Adj Close']
	except:
		x = data

	return x.rolling(period).mean() + multiplier*(x.rolling(period).std())

# Relative Strength Index. Data is the pandas dataframe containing , period is an integer.
def rsi(data, period):
	try:
		delta = data['Adj Close'].diff()
	except:
		delta = data.diff()
	up_days = delta.copy()
	up_days[delta<=0]=0.0
	down_days = abs(delta.copy())
	down_days[delta>0]=0.0
	RS_up = up_days.rolling(period).mean()
	RS_down = down_days.rolling(period).mean()
	rsi= 100-100/(1+RS_up/RS_down)
	return rsi

# MACD is apparently good. Here it is. No idea how to use it :D
def moving_average_convergence(data, nslow=26, nfast=12):
	"""
	Compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(data) arrays
	"""
	try:
		x = data['Adj Close'].tolist()
	except:
		x = data.tolist()

	emaslow = moving_average(x, nslow, type='exponential')
	emafast = moving_average(x, nfast, type='exponential')
	return emafast - emaslow

################################################################################################################
# Normalize data:
# Normalizes any data in a numpy array and returns it as a numpy array :D
# Formula: element - mean / (max - min)
def normalize(nparray):
	return ((nparray - nparray.min()) / (nparray.max() - nparray.min()))

# Step 2: Analyze/organize data. For any indicators you don't plan on using, feel free to comment that shiet out
def analyze_organize(ticker):
	tickerdata = yf.download(ticker, start, end)
	#tickerdata = tickerdata.iloc[::-1]
	# Establishing indicators
	#tickerdata['MACD'] = moving_average_convergence(tickerdata)
	#tickerdata['MACD_Signal'] = moving_average(tickerdata['MACD'], 9, 'exponential')
	#tickerdata['MACD_Indicator'] = tickerdata['MACD'] - tickerdata['MACD_Signal']
	tickerdata['SMA200'] = moving_average(tickerdata, 200, 'simple')
	tickerdata['SMA50'] = moving_average(tickerdata, 50, 'simple')
	tickerdata['Bol_High_20'] = bollinger(tickerdata, 20, 2)
	tickerdata['Bol_Low_20'] = bollinger(tickerdata, 20, -2)
	tickerdata['RSI14'] = rsi(tickerdata, 14)
	tickerdata = tickerdata[tickerdata['RSI14'] > -1]

	# Plot whatever data you want
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1 = tickerdata['SMA200'][-300:].plot(linestyle='dashed', label='SMA 200', legend=True)
	ax1 = tickerdata['SMA50'][-300:].plot(linestyle='dashed', label='SMA 50', legend=True)
	ax1 = tickerdata['Adj Close'][-300:].plot(label="Price", legend=True)
	ax1 = tickerdata['Bol_Low_20'][-300:].plot()
	ax1 = tickerdata['Bol_High_20'][-300:].plot()
	plt.show()

	# print tickerdata

	# Normalizing everything
	# for x in tickerdata:
	# 	if x != 'Date':
	# 		tickerdata[x] = normalize(tickerdata[x])
	return tickerdata

# This is where the actual algorithm goes. Use whatever indicators you want! :D
	# Position is either 1 (long), or 0 (short).
def backtest(data, budget):
	# This is where shit gets organised and the "position" array is developed.
	position = np.digitize( (data['RSI14']), [0,30,70,100])
	print position.shape

	#print position
	prices = data['Adj Close'].tolist()
	prices.pop(0)
	prices = np.array(prices)

	#print position
	# Here's the backtesting part. I know it's a cop-out using ZIP and turning it into a list thing
	# I'll try to figure out an alternative way of handling the data 
	money = float(budget)
	stocks = 0
	for price, pos in zip( prices, position):
		if pos == 1 and stocks == 0:
			stocks = int(money/price)
			money -= stocks*price
			print "Bought " + str(stocks) + " at " + str(price)
			print "Cash = " + str(money)
		elif pos == 3 and stocks != 0:
			print "Sold " + str(stocks) + " at " + str(price)
			money += stocks*price
			print "Cash = " + str(money)
			stocks = 0
	return money + data['Adj Close'].iloc[-1] * stocks

#Work in progress
def runSVM(tickerdata):


	# Machine learning method. Support vector machine
	adj = tickerdata['Adj Close'].tolist()
	vol = tickerdata['Volume'].tolist()
	macd = tickerdata['MACD'].tolist()
	macds = tickerdata['MACD_Signal'].tolist()
	macdi = tickerdata['MACD_Indicator'].tolist()
	sma200 = tickerdata['SMA200'].tolist()
	sma50 = tickerdata['SMA50'].tolist()
	model = SVC()
	x = []
	for a in range(0, len(macd)):
		x.append( [ vol[a], macd[a], macds[a], macdi[a], sma200[a], sma50[a]] )
	y = (tickerdata['Adj Close'].shift().diff() > 0).tolist()
	model.fit(x[0:1000], y[0:1000])
	predy = model.predict(x)
	df = pd.DataFrame( {'predicted': predy, 'actual': y} )
	print df

# This just runs all the functions. BANANAS HAVE POTASSIUM
def main(tickers):
	for t in tickers:
		print "____________________________________________________________________\n"
		print "Processing data for " + t
		data = analyze_organize(t)
		budget = 10000
		# print "Budget = $" + str(budget)
		algo = backtest(data, budget)
		gains = round(algo-budget, 2)
		long_gains = round((data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) * int(budget/(data['Adj Close'].iloc[0])), 2)

		print "Algorithm yield:    " + str(gains) + ", percent: " + str( round(100*(gains) / budget, 2) ) + "%"
		print "Holding long yield: " + str(long_gains) + ", percent: " + str( round(100*(long_gains) / budget, 2) ) + "%\n"

tickers = get_tickers(txt)
main(tickers)
#shutil.rmtree('csvfiles')


# """ TO DO """
# """ - Filling the position array between buy/sell
# 	- Multiple indicators (using AND)
# 	- Weighting different indicators (normalizing to 1?)
# 	- Scikit learn?
# 	"""