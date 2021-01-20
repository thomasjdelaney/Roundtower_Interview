"""
For keeping functions that may be used in multiple scripts
"""
import os, sys
import datetime as dt
import numpy as np
import pandas as pd

def loadOpenCloseTimesCsv(csv_dir):
	"""
	For loading in the csv file that contains opening and closing times info.
	Arguments:	csv_dir, str
	Returns:	pandas DataFrame
	"""
	str_to_time_converter = lambda x:dt.datetime.strptime(x, '%H:%M:%S').time()
	open_close = pd.read_csv(os.path.join(csv_dir, 'open_close.csv'),
		converters={'FUT_TRADING_HRS':lambda x:'' if not x else x,
			'TRADING_DAY_START_TIME_EOD':str_to_time_converter,
			'TRADING_DAY_END_TIME_EOD':str_to_time_converter}, 
		dtype={'Bloomberg Ticker':str}, index_col=0)
	return open_close

def extractTickerNameFromColumns(frame_columns):
	"""
	For extracting the names of the ticker for which we have data from the column names.
	Arguments:	frame_columns, list of str, frame.columns, or single str
	Returns:	numpy array of strings, or single str
	"""
	if type(frame_columns) == str:
		tickers = frame_columns.split('_')[0]
	elif type(frame_columns) == list:
		tickers = np.unique([c.split('_')[0]for c in frame_columns])
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised type for frame_columns!')
		tickers = None
	return tickers

def getTickerBidAskMidColNames(ticker):
	"""
	For getting the bid and ask column names from a ticker.
	Arguments: 	ticker, str
	Returns:	bid_col_name, str
				ask_col_name, str
	"""
	bid_col_name = ticker + '_Bid'
	ask_col_name = ticker + '_Ask'
	mid_col_name = ticker + '_Mid'
	return bid_col_name, ask_col_name, mid_col_name

def getMatchedColNames(frame, pattern):
	"""
	Returns column names that contain the provided pattern
	Arguments:	frame, pandas DataFrame
				pattern, str
	Returns:	list of str
	"""
	return [c for c in frame.columns if -1 < c.find(pattern)]