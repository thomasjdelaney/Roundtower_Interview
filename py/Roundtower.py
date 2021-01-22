"""
For keeping functions that may be used in multiple scripts
"""
import os, sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

################## LOADING FUNCTIONS #########################

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

def loadCleanBidAsk(csv_dir):
	"""
	For loading the cleaned bid/ask data.
	Arguments:	csv_dir, str
	Returns:	pandas DataFrame
	"""
	return pd.read_csv(os.path.join(csv_dir,'all_clean.csv'), parse_dates=[0], index_col=0)

def loadFullMids(csv_dir):
	"""
	For loading the csv with the full mid prices (after the first tick)
	Arguments:	csv_dir, str, the path to the directory
	Returns:	pandas, DataFrame
	"""
	return pd.read_csv(os.path.join(csv_dir, 'full_mids.csv'), parse_dates=[0], index_col=0)

def loadBidAskMid(csv_dir):
	"""
	For loading the clean bid/ask/mid table from the csv file.
	Arguments:	csv_dir, str
	Returns:	pandas DataFrame
	"""
	return pd.read_csv(os.path.join(csv_dir, 'bid_ask_mid.csv'), parse_dates=[0], index_col=0)

#### END OF LOADING FUNCTIONS

######################### TICKER & COLUMN NAMES #################################

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
	if type(ticker) == str:
		bid_col_name = ticker + '_Bid'
		ask_col_name = ticker + '_Ask'
		mid_col_name = ticker + '_Mid'
		col_names = bid_col_name, ask_col_name, mid_col_name
	elif type(ticker) == list:
		bid_col_names = [t + '_Bid' for t in ticker]
		ask_col_names = [t + '_Ask' for t in ticker]
		mid_col_names = [t + '_Mid' for t in ticker]
		col_names = bid_col_names + ask_col_names + mid_col_names
	elif type(ticker) == np.ndarray:
		col_names = getTickerBidAskMidColNames(ticker.tolist())
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised ticker type.')
		col_names = [None]
	return col_names

def getMatchedColNames(frame, pattern):
	"""
	Returns column names that contain the provided pattern
	Arguments:	frame, pandas DataFrame
				pattern, str
	Returns:	list of str
	"""
	return [c for c in frame.columns if -1 < c.find(pattern)]

###### END OF TICKER & COLUMN NAMES

######################### OPEN & CLOSE TIMES ##########################

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

def getDataDatesFromFrame(frame):
	"""
	For getting the range of dates for which we expect to have data.
	Arguments:	frame, pandas DataFrame
	Returns:	
	"""
	return pd.unique(frame.index.date)

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

def getOpenCloseSessionsForDates(bid_ask_dates, open_close_times):
	"""
	Get a two column matrix of open session times and close session times for each of the given dates.
	Arguments:	bid_ask_dates, array of datetime.date
				open_close_times, list of opening and close times
	Returns:	two column array, first column is opening times of sessions, second column is 
					closing times of same sessions
	"""
	num_sessions = len(open_close_times)
	num_dates = bid_ask_dates.size
	open_close_datetimes = np.empty(shape=(num_dates * num_sessions, 2), dtype=object)
	for d,date in enumerate(bid_ask_dates):
		previous_close_time = dt.time(0,0)
		for s,(open_time, close_time) in enumerate(open_close_times): # looping through sessions
			if open_time < previous_close_time: # the break between sessions (if there is one) included midnight (HC1) 
				date = date + dt.timedelta(days=1) # increment one day
			open_close_datetimes[d * num_sessions + s,0], open_close_datetimes[d * num_sessions + s,1] = getOpenCloseDatetime(open_time, close_time, date)
			previous_close_time = close_time
	return open_close_datetimes

##### END OF OPEN & CLOSE

######################## RETURNS ####################################################

def getReturnTable(bid_ask_mid):
	"""
	For calculating an hourly return table. Return = (mid/mid_one_hour_ago) - 1
	Arguments:	bid_ask_mid
	Returns:	pandas DataFrame
	"""
	is_hourly_index = bid_ask_mid.index.minute == 0
	mid_column_names = getMatchedColNames(bid_ask_mid, '_Mid')
	return bid_ask_mid.loc[is_hourly_index, mid_column_names].pct_change(fill_method=None)

##### END OF RETURNS

######################## MULTIPLE REGRESSION MODELLING ##############################

def getHourlyReturnsForTicker(ticker, hourly_returns, open_close):
	"""
	Get the top 5 most influential tickers for the given dependent ticker.
	Just use all the variables and L1 regularization (lasso) 
	Arguments:	ticker, str
				hourly_returns, pandas DataFrame
				open_close, pandas DataFrame
	Returns:	top_five_indep, list of str
	"""
	_,_, mid_col_name = getTickerBidAskMidColNames(ticker)
	open_close_times, num_sessions = getOpenCloseTimesForTicker(open_close, ticker)
	returns_dates = getDataDatesFromFrame(hourly_returns)
	open_close_datetimes = getOpenCloseSessionsForDates(returns_dates, open_close_times)
	ticker_returns = pd.Series(dtype=float)
	for open_datetime, close_datetime in open_close_datetimes:
		is_trading = (hourly_returns.index >= open_datetime) & (hourly_returns.index <= close_datetime)
		session_returns = hourly_returns.loc[is_trading][mid_col_name]
		is_real_session_return = session_returns.notna()
		if not is_real_session_return.any():
			continue # no data/not trading
		ticker_returns = pd.concat([ticker_returns, session_returns.loc[is_real_session_return]])
	ticker_returns.name = mid_col_name
	return ticker_returns

def getTopFiveIndependentVars(dependent_ticker, hourly_returns, open_close, thresh):
	"""
	Get the top 5 most influential tickers for the given dependent ticker.
	If 'thresh' = 0.8, independent variables must have non-null returns at 80% of the dependent tickers
	non-null return times. This controls the number of independent tickers that are considered.
	Arguments:	dependent_ticker,  pandas DataFrame
				hourly_returns, pandas DataFrame
				open_close, pandas DataFrame
				thresh, float, between 0 and 1, the threshold number of crossover returns to consider a ticker
	Returns:	top_five_indep_vars, as measured by absolute value of the coefficient 
				top_five_coefs, the coefficients themselves
				r_sq, the score of the model
	"""
	dependent_returns = getHourlyReturnsForTicker(dependent_ticker, hourly_returns, open_close)
	independent_returns = hourly_returns.loc[dependent_returns.index, hourly_returns.columns != dependent_returns.name]
	num_crossover_returns = independent_returns.notna().sum()
	crossover_cols = num_crossover_returns[num_crossover_returns > (thresh * dependent_returns.size)].index.to_list()
	crossover_cols.remove('IDO1 Index_Mid') if dependent_ticker == 'MXID Index' else None
	has_all_independent = independent_returns[crossover_cols].notna().all(axis=1)
	independent_returns_all_valid = independent_returns.loc[has_all_independent, crossover_cols]
	dependent_returns_with_crossover = dependent_returns[has_all_independent]
	# regularized linear regression modelling below
	test_size = dependent_returns_with_crossover.size//4
	train_size = dependent_returns_with_crossover.size - test_size
	train_X = independent_returns_all_valid.iloc[:train_size]
	test_X = independent_returns_all_valid.iloc[train_size:]
	train_y = dependent_returns_with_crossover.iloc[:train_size]
	test_y = dependent_returns_with_crossover.iloc[train_size:]
	regression_model = LassoCV(n_jobs=-1) # TODO: training and test
	regression_model.fit(train_X, train_y)
	r_sq = regression_model.score(test_X, test_y)
	top_five_inds = np.abs(regression_model.coef_).argsort()[-5:]
	top_five_indep_vars = np.array(crossover_cols)[top_five_inds]
	top_five_coefs = regression_model.coef_[top_five_inds]
	return top_five_indep_vars, top_five_coefs, r_sq

