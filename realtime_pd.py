#!/usr/bin/env python
# Input: a text file with a ticker on each line.
# Dependends on NUMPY and PANDAS

import csv, math, os, shutil, datetime, urllib, matplotlib, datetime, json, time
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from googlefinance import getQuotes
from sys import argv

# Tickers is a text file that contains all the tickers you want to analyze. 
# Start is the year you'd like to start on
script, txt = argv
now = datetime.datetime.now()
#matplotlib.style.use('ggplot')

# Parameters:

interval = 20.0

print "#######################################################"
print "Beginning simulation"

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
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)
	return pd.rolling_mean(x, period) + multiplier*(pd.rolling_std(data['Adj Close'], period, min_periods=period))

# Relative Strength Index. Data is a pandas dataframe, period is an integer.
def rsi(data, period):
	try:
		delta = data['Adj Close'].diff()
	except:
		delta = data.diff()

	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = pd.rolling_mean(dUp, period)
	RolDown = pd.rolling_mean(dDown, period).abs()

	RS = RolUp / RolDown
	rsi = 100 - (100.0 / (1.0 + RS))
	return rsi

# MACD is apparently good. Here it is. No idea how to use it :D
def moving_average_convergence(data, nslow=26, nfast=12):
	"""
	Compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(data) arrays
	"""
	try:
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)
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
	tickerdata = pd.read_csv('./csvfiles/' + ticker + '.csv')
	tickerdata = tickerdata.iloc[::-1]
	# Establishing indicators
	#tickerdata['MACD'] = moving_average_convergence(tickerdata)
	#tickerdata['MACD_Signal'] = moving_average(tickerdata['MACD'], 9, 'exponential')
	#tickerdata['MACD_Indicator'] = tickerdata['MACD'] - tickerdata['MACD_Signal']
	#tickerdata['SMA20'] = moving_average(tickerdata, 20, 'simple')
	#tickerdata['SMA50'] = moving_average(tickerdata, 50, 'simple')
	#tickerdata['Bol_High_20']=bollinger(tickerdata, 20, 2)
	#tickerdata['Bol_Low_20']=bollinger(tickerdata, 20, -2)
	#tickerdata['RSI14'] = rsi(tickerdata, 14)

	# Normalizing everything
	# for x in tickerdata:
	# 	if x != 'Date':
	# 		tickerdata[x] = normalize(tickerdata[x])
	return tickerdata


# This just runs all the functions. BANANAS HAVE POTASSIUM
def main(tickers):
	print "____________________________________________________________________\n"
	budget = 1000
	ticker_dictionary = {}

	# Construct a dictionary of pandas dataframes for each ticker. These dataframes contain two columns: date/time and price
	for t in tickers:
		ticker_dictionary[t] = pd.DataFrame(columns=['Date/Time', 'Price'])
		# POSITION is a [stocks, buy price] array
		ticker_dictionary[t + "_position"] = [0,0]
	profit = 0
	# Running while market is open
	print "Waiting for market to open..."
	while time.localtime( time.time() )[3] > 13 or time.localtime( time.time() )[3] < 7:
		sleep(60.00)
	print "Beginning data collection"
	while time.localtime( time.time() )[3] < 13 and time.localtime( time.time() )[3] > 7:
		for t in tickers:
			# Add data to the pandas dataframe
			try:
				mydump = json.dumps(getQuotes(t))
				data = {'Date/Time': [str(time.localtime( time.time() )[3]) + ":" + str(time.localtime( time.time() )[4]) + ":" + str(time.localtime( time.time() )[5])], 'Price': [round(float(json.loads(mydump)[0]["LastTradePrice"]), 2)]}
				ticker_dictionary[t].append(pd.DataFrame(data))
			except:
				data = {'Date/Time': [str(time.localtime( time.time() )[3]) + ":" + str(time.localtime( time.time() )[4]) + ":" + str(time.localtime( time.time() )[5])], 'Price': ticker_dictionary[t]['Price'][-1]}
				ticker_dictionary[t].append(pd.DataFrame(data))


			# Start after a minimum of 50 data points have been recorded. This ensures most indicators can be established
			if (ticker_dictionary[t]['Price'].shape[0]) == 50:
				print "Sufficient data collected to begin simulation."
			if (ticker_dictionary[t]['Price'].shape[0]) > 50:
				# Establish indicators
				ticker_dictionary['MACD'] = moving_average_convergence(ticker_dictionary)
				ticker_dictionary['MACD_Signal'] = moving_average(ticker_dictionary['MACD'], 9, 'exponential')

				# This bit decides whether to buy/sell stocks
				# BUY CONDITIONS
				if ticker_dictionary[t]['MACD'][-1] > ticker_dictionary[t]['MACD_Signal'][-1]:
											      # APPEND Stock position (int of budget/buy price)       and         buy price
					ticker_dictionary[t + "_position"] = [int(budget/ticker_dictionary[t]['Price'][-1]), (ticker_dictionary[t]['Price'][-1])]
					print str(ticker_dictionary[t]['Date/Time'][-1]) + " - Bought at " + str(ticker_dictionary[t]['Prices'][-1])

				# SELL CONDITIONS
				elif ticker_dictionary[t]['MACD'][-1] < ticker_dictionary[t]['MACD_Signal'][-1]:
					# ADD TO PROFIT: (Sell price - buy price)*position
					profit += ticker_dictionary[t + "_position"][0] * (ticker_dictionary[t]['Prices'][-1] * ticker_dictionary[t + "_position"][1])
					ticker_dictionary[t + "_position"] = [0,0]
					print str(ticker_dictionary[t]['Date/Time'][-1]) + " - Sold at " + str(ticker_dictionary[t]['Prices'][-1])
					print "Profit so far: " + str(profit)

			print ticker_dictionary[t]
			print ticker_dictionary[t]['Price'].shape
		sleep(interval)

		# After market close report results
		# print "Algorithm yield:    " + str(gains) + ", representing change of " + str( round(100*(gains) / budget, 2) ) + "%"
		# print "Holding long yield: " + str(long_gains) + ", representing change of " + str( round(100*(long_gains) / budget, 2) ) + "%"
		# print "____________________________________________________________________\n"


main(get_tickers(txt))
shutil.rmtree('csvfiles')


# """ TO DO """
# """ - Filling the position array between buy/sell
# 	- Multiple indicators (using AND)
# 	- Weighting different indicators (normalizing to 1?)
# 	- Scikit learn?
# 	"""