"""
For loading in the csv data, and cleaning it all up. See README for details of cleaning required.
"""
import os, sys, glob, shutil
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

pd.set_option('max_rows',30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns-2)

# useful globals
proj_dir = os.path.join(os.environ.get('HOME'), 'Roundtower_Interview')
csv_dir = os.path.join(proj_dir, 'csv')

def loadOpenCloseTimesCsv():
	"""
	For loading in the csv file that contains opening and closing times info.
	Arguments:	None
	Returns:	pandas DataFrame
	"""
	str_to_time_converter = lambda x:dt.datetime.strptime(x, '%H:%M:%S').time()
	open_close = pd.read_csv(os.path.join(csv_dir, 'open_close.csv'),
		converters={'FUT_TRADING_HRS':lambda x:'' if not x else x,
			'TRADING_DAY_START_TIME_EOD':str_to_time_converter,
			'TRADING_DAY_END_TIME_EOD':str_to_time_converter}, 
		dtype={'Bloomberg Ticker':str}, index_col=0)
	return open_close

def getOpenCloseTimesForTicker(open_close, ticker):
	"""
	For getting the opening anc closing times for a given ticker.
	Arguments:	open_close, pandas DataFrame
				ticker, str
	Returns:	open_close_times, list of pairs of datetime.time
				num_sessions, int (1 or 2)
	"""
	str_to_time_converter = lambda x:dt.datetime.strptime(x, '%H:%M').time()
	future_trading_hrs = open_close.loc[ticker]['FUT_TRADING_HRS']
	has_future_session = future_trading_hrs != ''
	if has_future_session:
		has_two_sessions = future_trading_hrs.find('&') > -1
		if has_two_sessions:
			num_sessions = 2
			str_sessions = [session.strip().split('-') for session in future_trading_hrs.split('&')]
			open_close_times = [[str_to_time_converter(t) for t in session] for session in str_sessions]
		else:
			num_sessions = 1
			open_close_times = [[str_to_time_converter(t)for t in future_trading_hrs.split('-')]]
	else:
		num_sessions = 1
		opening_time = open_close.loc[ticker]['TRADING_DAY_START_TIME_EOD']
		closing_time = open_close.loc[ticker]['TRADING_DAY_END_TIME_EOD']
		open_close_times = [[opening_time, closing_time]]
	return open_close_times, num_sessions

def printFileInfo(csv_file):
	"""
	For printing some information about the given csv file. Assumes that the index of the file is a datetime.
	Arguments:	csv_file, str, file name with path
	Returns:	nothing
	"""
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'File name: ' + csv_file)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'File size: ' + 
		str(round(os.path.getsize(csv_file)/(1024*1024),2)) + 'MB')
	file_frame = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Number of columns: ' + str(file_frame.shape[1]))
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Number of rows: ' + str(file_frame.shape[0]))
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Start time: ' + str(file_frame.index[0]))
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'End time: ' + str(file_frame.index[-1]))
	file_column_names = file_frame.columns.to_list()
	print(', '.join(file_column_names))	

def extractContractNameFromColumns(frame_columns):
	"""
	For extracting the names of the contracts for which we have data from the column names.
	Arguments:	frame_columns, list of str, frame.columns
	Returns:	numpy array of strings
	"""
	return np.unique([c.split('_')[0]for c in frame_columns])

def getDataDatesFromFrame(frame):
	"""
	For getting the range of dates for which we expect to have data.
	Arguments:	frame, pandas DataFrame
	Returns:	
	"""
	return pd.unique(frame.index.date)

def getAllTimes():
	"""
	For getting a list of every possible time (hours, minutes) in a day
	Arguments: 	none
	Returns:	array of datetime.time 
	"""
	return np.array([dt.time(hour, minute) for hour, minute in product(range(24), range(60))])

def plotTickerTimeProfile(bid_ask_counts, bid_col_name, ask_col_name, ticker, num_weekdays):
	"""
	For plotting the number of bids/asks we have per minute of the day, per weekday
	Arguments:	bid_ask_counts, pandas dataFrame
				bid_col_name, str
				ask_col_name, str
				ticker, str
				num_weekdays, int, the total number of weekdays for which we might have data
	Returns:	nothing
	"""
	fig, axes = plt.subplots(nrows=2, ncols=1)
	bid_ax, ask_ax = axes
	bid_ask_counts[bid_col_name].plot(ax=bid_ax)
	bid_ask_counts[ask_col_name].plot(ax=ask_ax, color='orange')
	bid_ax.set_xlim((bid_ask_counts.index[0], bid_ask_counts.index[-1]))
	bid_ax.set_xlabel('')
	bid_ax.set_xticks([])
	bid_ax.set_ylabel('Num. of Bids', fontsize='large')
	ask_ax.set_xlim((bid_ask_counts.index[0], bid_ask_counts.index[-1]))
	ask_ax.set_xlabel('Time of day', fontsize='large')
	ask_ax.set_ylabel('Num. of Asks', fontsize='large')
	[tick.set_rotation(30) for tick in ask_ax.get_xticklabels()]
	[ax.spines[s].set_visible(False) for ax,s in product(axes, ['top', 'right'])]
	bid_ax.set_title(ticker + ', Total weekdays = ' + str(num_weekdays), fontsize='large')
	plt.tight_layout()

def getTickerTimeProfile(ticker, bid_ask_frame):
	"""
	For visualising the times at which we have bids and asks for a given instrument.
	Arguments:	ticker, str
				bid_ask_frame, pandas dataFrame
	Returns:	displays figures showing distributions of bid/ask times
	"""
	bid_col_name, ask_col_name = getTickerBidAskColNames(ticker)
	bid_ask_frame = bid_ask_frame[[bid_col_name, ask_col_name]].notna()
	bid_ask_frame['time_of_day'] = bid_ask_frame.index.time
	bid_ask_counts = bid_ask_frame.groupby('time_of_day').aggregate('sum')
	frame_dates = getDataDatesFromFrame(bid_ask_frame)
	num_weekdays = np.sum([date.weekday() < 5 for date in frame_dates])
	plotTickerTimeProfile(bid_ask_counts, bid_col_name, ask_col_name, ticker, num_weekdays)

def getTickerBidAskColNames(ticker):
	"""
	For getting the bid and ask column names from a ticker.
	Arguments: 	ticker, str
	Returns:	bid_col_name, str
				ask_col_name, str
	"""
	bid_col_name = ticker + '_Bid'
	ask_col_name = ticker + '_Ask'
	return bid_col_name, ask_col_name

def getOpenCloseDatetime(open_time, close_time, date):
	"""
	For getting the opening and closing date-time of a ticker, assuming the opening time 
	is on the given 'date'.
	Arguments:	open_time, datetime.time
				close_time, datetime.time
				date, datetime.date
	Returns:	open_datetime, datetime.datetime
				close_datetime, datetime.datetime
	"""
	is_single_day_session = open_time < close_time
	if is_single_day_session:
		open_datetime = dt.datetime.combine(date, open_time)
		close_datetime = dt.datetime.combine(date, close_time)
	else:
		next_date = date + dt.timedelta(days=1)
		open_datetime = dt.datetime.combine(date, open_time)
		close_datetime = dt.datetime.combine(next_date, close_time)
	return open_datetime, close_datetime

def fillForwardBidAskSeries(bid_ask_series, open_datetime, close_datetime):
	"""
	For getting the trading part of a bid/ask series and filling that forward.
	Arguments:	bid_ask_series, pandas Series
				open_datetime,
				close_datetime
	Returns:	filled_trading_bid_ask_series, pandas Series
	"""
	trading_bid_ask_series = bid_ask_series[(bid_ask_series.index >= open_datetime) & 
					(bid_ask_series.index <= close_datetime)]
	is_all_null = np.invert(trading_bid_ask_series.notna().any())
	has_nulls = trading_bid_ask_series.isna().any()
	if is_all_null:
		return pd.Series(dtype=float) # holiday or weekend, move on to the next date.
	if not has_nulls:
		return pd.Series(dtype=float) # trading series full, no update required
	filled_trading_bid_ask_series = trading_bid_ask_series.ffill()
	return filled_trading_bid_ask_series[filled_trading_bid_ask_series.notna() & trading_bid_ask_series.isna()]

def cleanTickerBidsAsks(bid_ask_series, bid_ask_dates, open_close_times, num_sessions):
	"""
	For cleaning the up the bid and ask data. We want to fill forward bids and asks during a trading session.
	But we don't to insert data on days where there is none (holidays, weekends).
	We don't want to insert bids or asks before the first bid or quote of the session either.
	Profile this code because it may be repeated often.
	Arguments:	bid_ask_series, pandas Series
				bid_ask_dates, array of datetime.date
				open_close_times, list of opening and closing times
				num_sessions, the number of trading sessions for that ticker.
	Returns:	pandas DataSeries
	"""
	# need to check that open time < close time
	filled_bid_ask_series = pd.Series(dtype=float)
	days_with_ff_data = 0
	for date in bid_ask_dates:
		for open_time, close_time in open_close_times:
			open_datetime, close_datetime = getOpenCloseDatetime(open_time, close_time, date)
			filled_trading_bid_ask_series = fillForwardBidAskSeries(bid_ask_series, 
													open_datetime, close_datetime)
			if filled_trading_bid_ask_series.size > 0:
				days_with_ff_data += 1	
				filled_bid_ask_series = pd.concat([filled_bid_ask_series, filled_trading_bid_ask_series])
	bid_ask_series.update(filled_bid_ask_series)
	print(dt.datetime.now().isoformat() + ' INFO: ' + bid_ask_series.name + 
			', num days with fills = ' + str(days_with_ff_data))
	return bid_ask_series

open_close = loadOpenCloseTimesCsv()
csv_files = glob.glob(os.path.join(csv_dir,'*'))
csv_file = csv_files[2]
bid_ask_all_frame = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
bid_ask_dates = getDataDatesFromFrame(bid_ask_all_frame)
ticker_names = extractContractNameFromColumns(bid_ask_all_frame.columns)
ticker = 'GC1 Comdty' # we need special rules for some tickers, check Dan's email regularly
open_close_times, num_sessions = getOpenCloseTimesForTicker(open_close, ticker)
bid_col_name, ask_col_name = getTickerBidAskColNames(ticker)
bid_ask_series = bid_ask_all_frame[bid_col_name]
clean_bid_ask_series = cleanTickerBidsAsks(bid_ask_series, bid_ask_dates, open_close_times, num_sessions)


#[getInstrumentTimeProfile(ticker, bid_ask_frame) for ticker in ticker_names[:3]]
#plt.show() # TODO save instead of show

# TODO: tickers: GC1 Comdty, TP1 Index
#		does the cleanTickerBidsAsks function work as required? Unit tests