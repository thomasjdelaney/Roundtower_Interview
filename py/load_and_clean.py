"""
For loading in the csv data, and cleaning it all up. See README for details of cleaning required.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

parser = argparse.ArgumentParser(description='For cleaning up the data.')
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

pd.set_option('max_rows',30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns-2)

# useful globals
proj_dir = os.path.join(os.environ.get('HOME'), 'Roundtower_Interview')
csv_dir = os.path.join(proj_dir, 'csv')
py_dir = os.path.join(proj_dir, 'py')

sys.path.append(py_dir)
from Roundtower import *

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
	bid_col_name, ask_col_name, mid_col_name = getTickerBidAskMidColNames(ticker)
	bid_ask_frame = bid_ask_frame[[bid_col_name, ask_col_name]].notna()
	bid_ask_frame['time_of_day'] = bid_ask_frame.index.time
	bid_ask_counts = bid_ask_frame.groupby('time_of_day').aggregate('sum')
	frame_dates = getDataDatesFromFrame(bid_ask_frame)
	num_weekdays = np.sum([date.weekday() < 5 for date in frame_dates])
	plotTickerTimeProfile(bid_ask_counts, bid_col_name, ask_col_name, ticker, num_weekdays)

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
	is_filled = filled_trading_bid_ask_series.notna() & trading_bid_ask_series.isna()
	num_fills = np.sum(is_filled)
	percentage_filled = 100 * num_fills / filled_trading_bid_ask_series.size
	if percentage_filled > 90:
		print(dt.datetime.now().isoformat() + ' WARN: ' + 'over 90% filled!')
	return filled_trading_bid_ask_series[is_filled]

def nullBidAskSeries(bid_ask_series, close_datetime, open_datetime):
	"""
	For nulling any prices in the data outside of the trading hours.
	Arguments:	bid_ask_series, pandas Series
				open_datetime,
				close_datetime
	Returns:	nulled_trading_bid_ask_series, pandas Series
	"""
	not_trading_bid_ask_series = bid_ask_series[(bid_ask_series.index < open_datetime) & 
					(bid_ask_series.index > close_datetime)]
	is_all_null = np.invert(not_trading_bid_ask_series.notna().any())
	if is_all_null:
		return pd.Series(dtype=float) # all nulls already, no need to update
	nulled_trading_bid_ask_series = not_trading_bid_ask_series * np.nan
	is_nulled = nulled_trading_bid_ask_series.isna() & not_trading_bid_ask_series.notna()
	num_nulls = np.sum(is_nulled)
	percentage_nulled = 100 * num_nulls / nulled_trading_bid_ask_series.size
	# if percentage_nulled > 90: # too many alerts here
	# 	print(dt.datetime.now().isoformat() + ' WARN: ' + 'over 90% nulled!')
	return nulled_trading_bid_ask_series[is_nulled]

def cleanTickerBidsAsks(bid_ask_series, bid_ask_dates, open_close_times):
	"""
	For cleaning the up the bid and ask data. We want to fill forward bids and asks during a trading session.
	But we don't to insert data on days where there is none (holidays, weekends).
	We don't want to insert bids or asks before the first bid or quote of the session either.
	Profile this code because it may be repeated often.
	Arguments:	bid_ask_series, pandas Series
				bid_ask_dates, array of datetime.date
				open_close_times, list of opening and closing times
	Returns:	pandas DataSeries
	"""
	filled_bid_ask_series = pd.Series(dtype=float)
	nulled_bid_ask_series = pd.Series(dtype=float)
	sessions_with_ff_data = 0
	non_sessions_with_data = 0
	open_close_datetimes = getOpenCloseSessionsForDates(bid_ask_dates, open_close_times)
	num_sessions = open_close_datetimes.shape[0]
	for open_datetime, close_datetime in open_close_datetimes:
		filled_trading_bid_ask_series = fillForwardBidAskSeries(bid_ask_series, open_datetime, close_datetime)
		if filled_trading_bid_ask_series.size > 0:
			sessions_with_ff_data += 1	
			filled_bid_ask_series = pd.concat([filled_bid_ask_series, filled_trading_bid_ask_series])
	bid_ask_series.update(filled_bid_ask_series)
	close_open_datetimes = open_close_datetimes.flatten()[1:-1].reshape(num_sessions-1, 2)
	for close_datetime, open_datetime in close_open_datetimes:
		nulled_not_trading_bid_ask_series = nullBidAskSeries(bid_ask_series, close_datetime, open_datetime)
		if nulled_not_trading_bid_ask_series.size > 0:
			non_sessions_with_data += 1
			nulled_bid_ask_series = pd.concat([nulled_bid_ask_series, nulled_not_trading_bid_ask_series])
	bid_ask_series.loc[nulled_bid_ask_series.index] = np.nan
	print(dt.datetime.now().isoformat() + ' INFO: ' + bid_ask_series.name + 
			', num trading sessions with fills = ' + str(sessions_with_ff_data))
	print(dt.datetime.now().isoformat() + ' INFO: ' + bid_ask_series.name + 
			', num not-trading sessions with data = ' + str(non_sessions_with_data))
	return bid_ask_series

def mergeColumns(raw_bid_ask):
	"""
	Specifically for merging 'FXY1 Index' and 'KMS1 Index' inro 'FXY1 KMS1 Index'
	Arguments:	raw_bid_ask, pandas DataFrame
	Returns:	pandas DataFrame
	"""
	old_f_bid_series, old_f_ask_series = raw_bid_ask['FXY1 Index_Bid'], raw_bid_ask['FXY1 Index_Ask']
	old_k_bid_series, old_k_ask_series = raw_bid_ask['KMS1 Index_Bid'], raw_bid_ask['KMS1 Index_Ask']
	old_f_bid_series.update(old_k_bid_series)
	old_f_ask_series.update(old_k_ask_series)
	raw_bid_ask['FXY1 KMS1 Index_Bid'] = old_f_bid_series.values
	raw_bid_ask['FXY1 KMS1 Index_Ask'] = old_f_ask_series.values
	new_columns = raw_bid_ask.columns.to_list()
	for c in ['FXY1 Index_Bid', 'FXY1 Index_Ask', 'KMS1 Index_Bid', 'KMS1 Index_Ask']:
		new_columns.remove(c)
	return raw_bid_ask[new_columns]

def readAndCleanData(csv_file_name, open_close):
	"""
	For reading in the csv file, and cleaning the data, returning the data with full 
	trading sessions (except for some ticks at the start of the session.)
	Arguments:	csv_file_name, str
				open_close, pandas DataFrame, contains info on opening and closing times
	Returns:	pandas DataFrame
	"""
	raw_bid_ask = pd.read_csv(csv_file_name, parse_dates=[0], index_col=0)
	raw_columns = raw_bid_ask.columns.to_list()
	if 'FXY1 Index' in extractTickerNameFromColumns(raw_columns):
		raw_bid_ask = mergeColumns(raw_bid_ask.copy())
	bid_ask_dates = getDataDatesFromFrame(raw_bid_ask)
	filled_bid_ask_frame = pd.DataFrame(index=raw_bid_ask.index)
	for col_name in raw_bid_ask.columns:
		raw_bid_ask_series = raw_bid_ask[col_name]
		ticker = extractTickerNameFromColumns(col_name)
		open_close_times, num_sessions = getOpenCloseTimesForTicker(open_close, ticker)
		filled_bid_ask_frame[col_name] = cleanTickerBidsAsks(raw_bid_ask_series.copy(), bid_ask_dates, open_close_times)
	return filled_bid_ask_frame

if not args.debug:
	open_close = loadOpenCloseTimesCsv(csv_dir)
	csv_files = glob.glob(os.path.join(csv_dir,'All_Fut_FX_Data*'))
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading and cleaning data...')
	clean_data_frames = [readAndCleanData(csv_file_name, open_close) for csv_file_name in csv_files]
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Joining data frames...')
	all_clean = pd.DataFrame([])
	for f in clean_data_frames:
		all_clean = all_clean.join(f, how='outer')
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving...')
	all_clean_file_name = os.path.join(csv_dir,'all_clean.csv')
	all_clean.to_csv(all_clean_file_name)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done: ' + all_clean_file_name)

# ticker_names = extractTickerNameFromColumns(bid_ask_all_frame.columns)
# ticker = 'TP1 Index' # we need special rules for some tickers, check Dan's email regularly
# open_close_times, num_sessions = getOpenCloseTimesForTicker(open_close, ticker)
# bid_col_name, ask_col_name = getTickerBidAskColNames(ticker)
# bid_ask_series = bid_ask_all_frame[bid_col_name]
# clean_bid_ask_series = cleanTickerBidsAsks(bid_ask_series.copy(), bid_ask_dates, open_close_times)


#[getInstrumentTimeProfile(ticker, bid_ask_frame) for ticker in ticker_names[:3]]
#plt.show() # TODO save instead of show

# TODO: tickers: GC1 Comdty, TP1 Index
#		does the cleanTickerBidsAsks function work as required? Unit tests