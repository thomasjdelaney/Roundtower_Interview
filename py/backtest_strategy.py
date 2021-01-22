"""
For testing a trading stratey or strategies on historical data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For making a "returns" table.')
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

def addFullMidsToAllClean(full_mids, all_clean):
	"""
	For merging the full_mids, and all_clean tables.
	Arguments: 	full_mids, pandas DataFrame, contains only mid prices 
					for all ticks (except for a few at the start)
				all_clean, pandas DataFrame, clean asks and bids
	Returns:	bid_ask_all_mids, pandas DataFrame
	"""
	return pd.merge(full_mids, all_clean, left_index=True, right_index=True, how='inner')

def getTradeDateTimes(start_trade_time, end_trade_time, take_off_time, trade_dates):
	"""
	For getting the trading sessions, and take of times as timestamps.
	Arguments:	start_trade_time, datetime time,
				end_trade_time, datetime time,
				take_off_time, datetime time,
				trade_dates, list of datetime date
	Returns:	List of 3 elements list of datetime datetime
	"""
	num_dates = len(trade_dates)
	start_end_off_datetimes = np.empty(shape=(num_dates, 3), dtype=object)
	for i,date in enumerate(trade_dates):
		start_end_off_datetimes[i,0] = dt.datetime.combine(date, start_trade_time)
		start_end_off_datetimes[i,1] = dt.datetime.combine(date, end_trade_time)
		next_date = date + dt.timedelta(days=1)
		start_end_off_datetimes[i,2] = dt.datetime.combine(next_date, take_off_time)
	return start_end_off_datetimes

def getRequiredTradingCols(ticker_to_trade, independent_mids):
	"""
	For getting the column names that we need for trading.
	Arguments:	ticker_to_trade, str
				independent_mids, list of str
	Returns:	list of strings
	"""
	independent_tickers = extractTickerNameFromColumns(independent_mids.tolist())
	required_tickers = np.hstack([ticker_to_trade, independent_tickers])
	required_cols = getTickerBidAskMidColNames(required_tickers)
	bid_cols = [c for c in required_cols if c.find('_Bid') > -1]
	ask_cols = [c for c in required_cols if c.find('_Ask') > -1]
	mid_cols = [c for c in required_cols if c.find('_Mid') > -1]
	return required_tickers, required_cols, bid_cols, ask_cols, mid_cols

def getTradingFrame(ticker_to_trade, independent_mids, start_end_off_datetimes, bid_ask_all_mids):
	"""
	For getting a data frame that contains the data required for trading nothing else.
	Arguments:	ticker_to_trade, str
				independent_mids, list of str
				start_end_off_datetimes, list of datetimes,
				bid_ask_all_mids, pandas DataFrame, the data from which we can select
	Returns:	trading_frame, pandas DataFrame
	"""
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, independent_mids)
	trading_frame = pd.DataFrame()
	for start, end, off in start_end_off_datetimes:
		is_trading_time = (bid_ask_all_mids.index >= start) & (bid_ask_all_mids.index <= end)
		session_trading_frame = bid_ask_all_mids.loc[is_trading_time,required_cols]
		take_off_record = bid_ask_all_mids.loc[off, required_cols]
		if take_off_record[bid_cols + ask_cols].isna().any(): # we need these data for take off
			continue # without these data we'll skip the day
		if session_trading_frame.isnull().all().any(): # if any one column is ALL null
			continue # we can't trade if the columns are all null
		trading_frame = pd.concat([trading_frame, session_trading_frame, take_off_record.to_frame().T])
	return trading_frame

def takeOffPosition(take_off_record, ticker, position):
	"""
	For entering the transactions to take off the positions at the end of the day. 
	Arguments:	take off record, pd.Series
				ticker, str
				position
	Returns:	the list of values for the columns of the transactions table			
	"""
	if position > 0:
		record_elements = [take_off_record.name, ticker, 's', take_off_record[ticker + '_Ask'], current_position, 0]
	elif position < 0:
		record_elements = [take_off_record.name, ticker, 'b', take_off_record[ticker + '_Bid'], current_position, 0]
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Position is 0!')
		record_elements = None
	return record_elements


def simulateDaysTrade(trading_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, ticker_to_trade, 
		independent_mids, independent_coefs, profit_required):
	"""
	Simulate a days worth of trading, using a leaning strategy.
	Arguments:	trading_frame, pandas DataFrame,
				start_trade_datetime, datetime datetime.
				end_trade_datetime, datetime datetime,
				take_off_datetime, datetime datetime,
				ticker_to_trade, str
				independent_mids, list of str
				independent_coefs, list of float
				profit_required
	Returns:	a table of trades, with times, bids and asks, position, and take off
	"""
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, independent_mids)
	mid_to_trade, bid_to_trade, ask_to_trade = mid_cols[0], bid_cols[0], ask_cols[0]
	is_today = (trading_frame.index >= start_trade_datetime) & (trading_frame.index <= end_trade_datetime)
	day_frame = trading_frame.loc[is_today, required_cols]
	transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'position_before', 'position_after'])
	current_position = 0
	independent_positions = np.zeros(5)
	for q_time, quote in day_frame.iterrows():
		fair_price = quote[mid_to_trade] # should we be getting this from some modelled source?
		max_bid = fair_price * (1 - profit_required) # maximum that we are willing to pay
		min_ask = fair_price * (1 + profit_required) # minimum at which we are willing to sell
		if (max_bid >= quote[ask_to_trade]) & (current_position < 3): 
			new_position = current_position + 1
			transactions.loc[len(transactions)] = [quote.name, ticker_to_trade, 'b', quote[bid_to_trade], current_position, new_position]
			for i,t in enumerate(required_tickers[1:]):
				independent_coef = independent_coefs[i]
				current_independent_position = independent_positions[i]
				if independent_coef > 0: # if coef is > 0, correlates with the traded ticker, therefore we take opposing position to hedge
					new_independent_position = current_independent_position - independent_coef
					transactions.loc[len(transactions)] = [quote.name, t, 's', quote[ask_cols[i+1]], 
						current_independent_position, new_independent_position]
					independent_positions[i] = new_independent_position
				else: # if coef < 0, anti-correlated with traded ticker, therefore take up same position to hedge
					new_independent_position = current_independent_position + np.abs(independent_coef)
					transactions.loc[len(transactions)] = [quote.name, t, 'b', quote[bid_cols[i+1]], 
						current_independent_position, new_independent_position]
					independent_positions[i] = new_independent_position
			current_position = new_position
		if (min_ask <= quote[bid_to_trade]) & (current_position > -3):
			new_position = current_position - 1
			transactions.loc[len(transactions)] = [quote.name, ticker_to_trade, 's', quote[ask_to_trade], current_position, new_position]
			for i,t in enumerate(required_tickers[1:]):
				independent_coef = independent_coefs[i]
				current_independent_position = independent_positions[i]
				if independent_coef > 0: # if coef is > 0, correlates with the traded ticker, therefore we take opposing position to hedge
					new_independent_position = current_independent_position + independent_coef
					transactions.loc[len(transactions)] = [quote.name, t, 'b', quote[bid_cols[i+1]], 
						current_independent_position, new_independent_position]
					independent_positions[i] = new_independent_position
				else: # if coef < 0, anti-correlated with traded ticker, therefore take up same position to hedge
					new_independent_position = current_independent_position - np.abs(independent_coef)
					transactions.loc[len(transactions)] = [quote.name, t, 's', quote[ask_cols[i+1]], 
						current_independent_position, new_independent_position]
					independent_positions[i] = new_independent_position
			current_position = new_position
	take_off_record = day_frame.loc[take_off_datetime]
	if transactions.shape[0] > 0:
		for ticker, position in zip(required_tickers, np.hstack([current_position, independent_positions])):
			if position > 0:
				transactions.loc[len(transactions)] = takeOffPosition(take_off_record, ticker, position)

			# we do a trade, 
			# and enter the info, 
			# and update the position
			# we must also trade



print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading in csvs...')
all_clean = loadCleanBidAsk(csv_dir)
full_mids = loadFullMids(csv_dir)
open_close = loadOpenCloseTimesCsv(csv_dir)
bid_ask_mid = loadBidAskMid(csv_dir)
hourly_returns = getReturnTable(bid_ask_mid)
ticker_to_trade = 'KWN+1M BGN Curncy'
thresh = 0.8
top_five_indep_mids, top_five_coefs, r_sq = getTopFiveIndependentVars(ticker_to_trade, hourly_returns, open_close, thresh)

start_trade_time = dt.time(9,0)
end_trade_time = dt.time(20,0)
take_off_time = dt.time(1,0) # next day
profit_required = 0.001 # buy at bid <= fair price * (1 - 0.001)
						# sell at ask >= fair price * (1 + 0.001)
# We have to offset each trade by taking the opposite position in the 
# independent variables. I suppose same rules apply 

bid_ask_all_mids = addFullMidsToAllClean(full_mids, all_clean)
trade_dates = getDataDatesFromFrame(bid_ask_all_mids)[:-2] # not ideal, revisit
start_end_off_datetimes = getTradeDateTimes(start_trade_time, end_trade_time, take_off_time, trade_dates)
trading_frame = getTradingFrame(ticker_to_trade, top_five_indep_mids, start_end_off_datetimes, bid_ask_all_mids)

start_trade_datetime, end_trade_datetime, take_off_datetime = start_end_off_datetimes[13]

# add the full mids to all clean
# pick a ticker, KWN+1M BGN Curncy
# get the top 5 independent vars for that ticker
# start at position zero
# trade when there the difference between the ask or bid and 
#	our modelled mid is greater than 0.001
# limit the trading so we don't get out of position
# every time we trade, take up the opposite position in the 
# top 5 independent vars
# close out at the end of the day


# check instructions again to see if I have missed anything.

# QUESTIONS: 	If, If else etc, trading in two directions at the same moment?
				# Limits in independent variables, should we have them also? 
				# There is no profit and loss recording, How to record profit and loss?
				# Model the return from one mid to the next

				