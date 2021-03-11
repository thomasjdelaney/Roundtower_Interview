"""
For keeping functions that may be used in multiple scripts
"""
import os, sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

################### CONSTANTS #################################

special_mid_names = ['ES1 Index_Ret', 'MES1 Index_Ret', 'SGD BGN Curncy_Ret']
special_ticker_names = ['ES1 Index', 'MES1 Index', 'SGD BGN Curncy']

################## LOADING FUNCTIONS #########################

## See notes_on_csv_files.md for further information on csv files in the csv/ dir.

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

def addFullMidsToAllClean(full_mids, all_clean):
	"""
	For merging the full_mids, and all_clean tables.
	Arguments: 	full_mids, pandas DataFrame, contains only mid prices 
					for all ticks (except for a few at the start)
				all_clean, pandas DataFrame, clean asks and bids
	Returns:	bid_ask_all_mids, pandas DataFrame
	"""
	return pd.merge(full_mids, all_clean, left_index=True, right_index=True, how='inner')

def divideDataForBacktesting(bid_ask_mid, hist_to_backtest_ratio=6):
	"""
	For dividing the data into 'historical' and 'backtesting'. The historical data will be used for training models.
	The backtesting data is used for backtesting.
	Arguments:	bid_ask_mid,
	Returns:	historical_data, pandas Dataframe
				backtesting_data, pandas Dataframe
	"""
	unique_dates = np.unique(bid_ask_mid.index.date)
	num_dates = unique_dates.size
	num_historical_dates = int(hist_to_backtest_ratio * num_dates//(hist_to_backtest_ratio + 1))
	num_backtesting_dates = num_dates - num_historical_dates
	historical_dates = unique_dates[:num_historical_dates]
	backtesting_dates = unique_dates[num_historical_dates:]
	historical_data = bid_ask_mid[np.isin(bid_ask_mid.index.date, historical_dates)]
	backtesting_data = bid_ask_mid[np.isin(bid_ask_mid.index.date, backtesting_dates)]
	return historical_data, backtesting_data

def fillWithSpecialModelExcludeDependent(csv_dir, historical_data, ticker_to_trade):
	"""
	For loading the fully filled data except for the data for the ticker to trade.
	Arguments:	csv_dir, the csv directory
				historical_data, pandas DataFrame,
				ticker_to_trade, str
	Returns:	fill_historical_data, pandas DataFrame
	"""
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Filling data...')
	mid_col_name = ticker_to_trade + '_Mid'
	full_mids = loadFullMids(csv_dir)
	all_clean = loadCleanBidAsk(csv_dir)
	bid_ask_full_mids = addFullMidsToAllClean(full_mids, all_clean)
	historical_bid_ask_full_mids = bid_ask_full_mids.loc[historical_data.index,:].copy()
	historical_bid_ask_full_mids.loc[:,mid_col_name] = historical_data.loc[:,mid_col_name]
	return historical_bid_ask_full_mids

#### END OF LOADING FUNCTIONS

######################### TICKER & COLUMN NAMES #################################

def extractTickerNameFromColumns(frame_columns):
	"""
	For extracting the names of the tickers for which we have data from the column names.
	Note that the tickers returned may not be a unique list. This is to avoid reordering
	Arguments:	frame_columns, list of str, frame.columns, or single str
	Returns:	numpy array of strings, or single str
	"""
	if type(frame_columns) == str:
		tickers = frame_columns.split('_')[0]
	elif type(frame_columns) == list:
		tickers = [c.split('_')[0]for c in frame_columns]
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised type for frame_columns!')
		tickers = None
	return tickers

def getTickerBidAskMidRetColNames(ticker):
	"""
	For getting the bid and ask column names from a ticker.
	Arguments: 	ticker, str
	Returns:	col_names, includes bid, ask, mid, and returns columns
	"""
	if type(ticker) == str:
		bid_col_name = ticker + '_Bid'
		ask_col_name = ticker + '_Ask'
		mid_col_name = ticker + '_Mid'
		ret_col_name = ticker + '_Ret'
		col_names = bid_col_name, ask_col_name, mid_col_name, ret_col_name
	elif type(ticker) == list:
		bid_col_names = [t + '_Bid' for t in ticker]
		ask_col_names = [t + '_Ask' for t in ticker]
		mid_col_names = [t + '_Mid' for t in ticker]
		ret_col_names = [t + '_Ret' for t in ticker]
		col_names = bid_col_names + ask_col_names + mid_col_names + ret_col_names
	elif type(ticker) == np.ndarray:
		col_names = getTickerBidAskMidRetColNames(ticker.tolist())
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

def getRequiredTradingCols(ticker_to_trade, indep_rets):
	"""
	For getting the column names and tickers that we need for trading.
	Arguments:	ticker_to_trade, str, 
				indep_rets, list of str, the independent variables for modelling the ticker to trade
	Returns:	required_tickers,
				required_trading_cols, all the required cols for trading (excluding returns)
				bid_cols,
				ask_cols,
				mid_cols,
	TODO: We're returning the same thing twice here. Look at the use cases.
	"""
	indep_tickers = extractTickerNameFromColumns(indep_rets.tolist())
	required_tickers = [ticker_to_trade] + indep_tickers
	required_cols = getTickerBidAskMidRetColNames(required_tickers)
	bid_cols = [c for c in required_cols if c.find('_Bid') > -1]
	ask_cols = [c for c in required_cols if c.find('_Ask') > -1]
	mid_cols = [c for c in required_cols if c.find('_Mid') > -1]
	required_cols = bid_cols + ask_cols + mid_cols
	return required_tickers, required_cols, bid_cols, ask_cols, mid_cols

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

def getHourlyReturnTable(bid_ask_mid):
	"""
	For calculating an hourly return table. Return = (mid/mid_one_hour_ago) - 1
	Arguments:	bid_ask_mid
	Returns:	pandas DataFrame
	"""
	is_hourly_index = bid_ask_mid.index.minute == 0
	mid_column_names = getMatchedColNames(bid_ask_mid, '_Mid')
	with_returns = bid_ask_mid.loc[is_hourly_index, mid_column_names].pct_change(fill_method=None)
	with_returns.columns = [c.replace('_Mid', '_Ret') for c in with_returns.columns]
	return with_returns

def getBidMidAskHourlyReturnTable(bid_ask_mid):
	"""
	For getting hourly returns in bid, mid, and ask.
	Arguments:	bid_mid_ask, pandas DataFrame
	Returns:	pandas DataFrame
	"""
	is_hourly_index = bid_ask_mid.index.minute == 0
	with_returns = bid_ask_mid.loc[is_hourly_index, :].pct_change(fill_method=None)
	return with_returns

def AddMidReturnsCol(trading_frame, valid_start_end_off_datetimes):
	"""
	For adding a column of returns to the given trading frame. Returns are just the 
	percentage change in the mid price.
	Arguments:	trading_frame, a frame with some mid columns
				valid_start_end_off_datetimes, list of list of 3 datetimes
	Returns:	pandas DataFrame, with added returns columns
	"""
	mid_column_names = getMatchedColNames(trading_frame, '_Mid')
	return_column_names = [c.replace('_Mid', '_Ret') for c in mid_column_names]
	returns_frame = pd.DataFrame()
	for start, end, off in valid_start_end_off_datetimes:
		is_trading = (trading_frame.index >= start) & (trading_frame.index <= end)
		days_trading = trading_frame[is_trading]
		days_trading = days_trading.loc[:, mid_column_names].pct_change(fill_method=None)
		returns_frame = pd.concat([returns_frame, days_trading])
	returns_frame.columns = return_column_names
	with_returns = trading_frame.join(returns_frame, how='left')
	return with_returns, return_column_names

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
	_,_, _, ret_col_name = getTickerBidAskMidRetColNames(ticker)
	open_close_times, num_sessions = getOpenCloseTimesForTicker(open_close, ticker)
	returns_dates = getDataDatesFromFrame(hourly_returns)
	open_close_datetimes = getOpenCloseSessionsForDates(returns_dates, open_close_times)
	ticker_returns = pd.Series(dtype=float)
	for open_datetime, close_datetime in open_close_datetimes:
		is_trading = (hourly_returns.index >= open_datetime) & (hourly_returns.index <= close_datetime)
		session_returns = hourly_returns.loc[is_trading][ret_col_name]
		is_real_session_return = session_returns.notna()
		if not is_real_session_return.any():
			continue # no data/not trading
		ticker_returns = pd.concat([ticker_returns, session_returns.loc[is_real_session_return]])
	ticker_returns.name = ret_col_name
	return ticker_returns

def getTopIndependentVars(dependent_ticker, hourly_returns, open_close, thresh, num_indep_vars=5):
	"""
	Get the most influential tickers for the given dependent ticker.
	If 'thresh' = 0.8, independent variables must have non-null returns at 80% of the dependent tickers
	non-null return times. This controls the number of independent tickers that are considered.
	Arguments:	dependent_ticker,  string
				hourly_returns, pandas DataFrame
				open_close, pandas DataFrame
				thresh, float, between 0 and 1, the threshold number of crossover returns to consider a ticker
				num_indep_vars, int, number of independent variables to return
	Returns:	regression_model, the model itself, useful for predictions
				top_five_indep_vars, as measured by absolute value of the coefficient 
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
	regression_model = LassoCV(n_jobs=-1, fit_intercept=False)
	regression_model.fit(train_X, train_y)
	r_sq = regression_model.score(test_X, test_y)
	top_inds = np.abs(regression_model.coef_).argsort()[-num_indep_vars:]
	top_indep_vars = np.array(crossover_cols)[top_inds]
	top_coefs = regression_model.coef_[top_inds]
	return regression_model, top_indep_vars, top_coefs, r_sq

def getLinearModelFromDepIndep(dependent_ticker, independent_rets, hourly_returns, open_close):
	"""
	For getting the linear model of the dependent ticker given the independent tickers, and the hourly returns.
	This will mostly be used for modelling a ticker using only its top independent tickers.
	Also makes a shuffled model and measures r squared value for comparison, if desired.
	Arguments:	dependent_ticker, string
				independent_rets, list of strings, names of returns columns
				hourly_returns, pandas DataFrame
				open_close, pandas DataFrame
	Returns:	regression_model,
				r_sq,
				shuffle_r_sq
	"""
	_,_, _, dep_ret_col_name = getTickerBidAskMidRetColNames(dependent_ticker)
	all_tickers = [dep_ret_col_name] + independent_rets.tolist()
	all_returns = hourly_returns.loc[:, all_tickers]
	has_all_tickers = all_returns.notna().all(axis=1)
	all_returns = all_returns[has_all_tickers]
	test_size = all_returns.shape[0]//4
	train_size = all_returns.shape[0] - test_size
	train_X = all_returns.iloc[:train_size][independent_rets]
	test_X = all_returns.iloc[train_size:][independent_rets]
	train_y = all_returns.iloc[:train_size][dep_ret_col_name]
	test_y = all_returns.iloc[train_size:][dep_ret_col_name]
	regression_model = LassoCV(n_jobs=-1, fit_intercept=False)
	regression_model.fit(train_X, train_y)
	r_sq = regression_model.score(test_X, test_y)
	shuffle_train_X = train_X.sample(frac=1)
	shuffle_model = LassoCV(n_jobs=-1, fit_intercept=False)
	shuffle_model.fit(shuffle_train_X, train_y)
	shuffle_r_sq = shuffle_model.score(test_X, test_y)
	return regression_model, r_sq, shuffle_r_sq

#### END OF MULTIPLE REGRESSION

##################### FILLING FORWARD ###################################################

def getTickerRegressionModels(hourly_returns):
	"""
	For fitting a regression model for the returns of each ticker to ES1 Index returns and MS1 Index returns
	or SGD returns for currencies. No intercept for easier coefficient retrieval.
	Arguments:	hourly_returns, pandas DataFrame, the returns table
	Returns:	ticker_to_model, dictionary, 
	"""
	ticker_to_model = {}
	for c in hourly_returns.columns:
		if c in special_mid_names: # ES1 Index, MES1 Index, SGD BDG Curncy
			continue
		ticker = extractTickerNameFromColumns(c)
		returns_series = hourly_returns[c]
		predictor_names = getPredictorNamesForTicker(ticker)
		predictors = hourly_returns[predictor_names]
		regression_model = fitRegressionModel(returns_series, predictors)
		ticker_to_model[ticker] = regression_model
	return ticker_to_model

def getPredictorNamesForColumn(col_name):
	"""
	Get the columns names of the appropriate predictors given a ticker.
	Arguments:	col_name, str
	Returns:	list of str, column names for the hourly_returns frame
	"""
	ticker = extractTickerNameFromColumns(col_name)
	col_end = col_name[-4:]
	if ticker.endswith('Index') | ticker.endswith('Equity') | ticker.endswith('Comdty'):
		predictor_names = ['ES1 Index', 'MES1 Index']
	elif ticker.endswith('Curncy'):
		predictor_names = ['SGD BGN Curncy']
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised ticker! ' + ticker)
		predictor_names = ['']
	predictor_names = [p + col_end for p in predictor_names]
	return predictor_names

def getBidAskMidRegressionModels(bid_ask_mid_hourly_returns):
	"""
	For fitting regression models to bid ask and mid columns using the speically designated tickers.
	No intercept.
	Uses global constant 'special_ticker_names' defined at the top of this file.
	Arguments:	bid_ask_mid_hourly_returns, pandas DataFrame, the returns table
	Returns:	ticker_to_model, dictionary
	"""
	special_cols = getTickerBidAskMidRetColNames(ticker)
	special_bids = [c for c in special_cols if c.find('_Bid') > -1]
	special_asks = [c for c in special_cols if c.find('_Ask') > -1]
	special_mids = [c for c in special_cols if c.find('_Mids') > -1]
	col_to_model = {}
	for c in bid_ask_mid_hourly_returns.columns:
		if c in special_cols:
			continue # we just fill those special columns forward eventually
		returns_series = bid_ask_mid_hourly_returns[c]
		predictor_names = getPredictorNamesForColumn(c)
		predictors = bid_ask_mid_hourly_returns[predictor_names]
		regression_model = fitRegressionModel(returns_series, predictors)
		col_to_model[c] = regression_model
	return col_to_model

def fillSpecialColumns(bid_ask_mid):
	"""
	For filling the ES1 Index, MES1 Index, and SGD BGN Curncy columns forward, so that there are no blanks.
	Arguments:	bid_ask_mid, pandas DataFrame
	Returns:	special_filled, pandas DataFrame
	"""
	special_col_names = [c for c in bid_ask_mid.columns.to_list() 
							if c.startswith('ES1') | c.startswith('MES1') | c.startswith('SGD')]
	bid_ask_mid.loc[:,special_col_names] = bid_ask_mid.copy().loc[:,special_col_names].ffill()
	return bid_ask_mid

def getPredictorNamesForTicker(ticker, ret_or_mid='Ret'):
	"""
	Get the columns names of the appropriate predictors given a ticker.
	Arguments:	ticker, str
	Returns:	list of str, column names for the hourly_returns frame
	"""
	if ticker.endswith('Index') | ticker.endswith('Equity') | ticker.endswith('Comdty'):
		predictor_names = ['ES1 Index_' + ret_or_mid, 'MES1 Index_' + ret_or_mid]
	elif ticker.endswith('Curncy'):
		predictor_names = ['SGD BGN Curncy_' + ret_or_mid]
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised ticker! ' + ticker)
		predictor_names = ['']
	return predictor_names

def fitRegressionModel(returns_series, predictors):
	"""
	For fitting a regression model for returns to ES1 Index and MS1 Index, or SGD
	Arguments:	returns_series, pandas Series, the returns to be fitted, ie. 
					regressand, endogenous variable, response variable, measured variable, criterion variable, 
					or dependent variable
				predictors, pandas DataFrame
	Returns:	coefficients
	"""
	joined = returns_series.to_frame().join(predictors, how='outer')
	proper_data = joined[joined.notna().all(axis=1) & (joined != np.inf).all(axis=1)]
	if proper_data.shape[0] < 100:
		print(dt.datetime.now().isoformat() + ' WARN: ' + 'Less than 100 rows of proper data.')
	y = proper_data[returns_series.name].to_frame()
	X = proper_data[predictors.columns]
	regression_model = LassoCV(fit_intercept=False, n_jobs=-1)
	regression_model.fit(X, y)
	return regression_model

def detectBlocksOfNulls(to_be_filled_index):
	"""
	For splitting the index of nulls into blocks with the same last valid index.
	Arguments:	to_be_filled_index, list of datetimes
	Returns:	null_blocks_indices, list of list of datetimes
	"""
	index_deltas = np.diff(to_be_filled_index)
	big_difference_inds = [i+1 for i,d in enumerate(index_deltas) if d > dt.timedelta(minutes=1)]
	starts_finishes = [0] + big_difference_inds + [None]
	null_blocks_indices = []
	for i,start in enumerate(starts_finishes[:-1]):
		null_blocks_indices.append(to_be_filled_index[start:starts_finishes[i+1]])
	return null_blocks_indices

def modelNullBlock(for_filling, null_block_indices, ticker_to_model, mid_col_name, predictor_names):
	"""
	For modelling some mids for a block of nulls.
	Arguments:	for_filling, pandas DataFrame, contains last valid quotes
				null_block_indices, list of datetimes, the incides of the nulls
				ticker_to_model, dictionary, ticker => regression model
				mid_col_name, str
				predictor_names, list of str
	Returns:	modelled_series, pandas DataFrame, modelled mids
	"""
	ticker = extractTickerNameFromColumns(mid_col_name)
	last_valid_ind = for_filling.loc[:null_block_indices[0]][mid_col_name].last_valid_index()
	last_valid_record = for_filling.loc[last_valid_ind]
	predictor_returns = for_filling.loc[null_block_indices][predictor_names]/last_valid_record[predictor_names] - 1
	model_prediction = ticker_to_model.get(ticker).predict(predictor_returns).flatten()
	modelled_mid = (1 + model_prediction) * last_valid_record[mid_col_name]
	modelled_series = pd.Series(data=modelled_mid, name=mid_col_name, index=null_block_indices)
	return modelled_series

def fillRemainingBlanks(special_filled, ticker_to_model):
	"""
	For filling in the remaining mid price columns using our regression models.
	Arguments: 	special_filled, pandas DataFrame
				ticker_to_model, dictionary, ticker to linear model for returns
	Returns:	pandas DataFrame
	"""
	mid_col_names = getMatchedColNames(special_filled, '_Mid')
	for mid_col_name in mid_col_names:
		if mid_col_name in special_mid_names:
			continue
		print(dt.datetime.now().isoformat() + ' INFO: ' + 'Filling ' + mid_col_name + '...')
		ticker = extractTickerNameFromColumns(mid_col_name)
		predictor_names = getPredictorNamesForTicker(ticker, 'Mid')
		relevant_cols = special_filled[[mid_col_name] + predictor_names]
		first_valid_ind = relevant_cols.loc[relevant_cols.notna().all(axis=1)].index[0]
		for_filling = relevant_cols.loc[first_valid_ind:]
		to_be_filled = for_filling.loc[for_filling[mid_col_name].isna()].copy()
		null_blocks_indices = detectBlocksOfNulls(to_be_filled.index.to_list())
		for null_block_indices in null_blocks_indices:
			modelled_series = modelNullBlock(for_filling, null_block_indices, ticker_to_model, mid_col_name, predictor_names)
			to_be_filled.update(modelled_series)
		special_filled.update(to_be_filled[mid_col_name])
	return special_filled[mid_col_names]

############### BACKTESTING STRATEGY ####################################################

def getReferenceQuoteFromBacktestingFrame(backtesting_frame, reference_datetime):
	"""
	For getting the reference, or 'fair price' quote from the backtesting frame.
	Arguments:	backtesting_frame, pandas DataFrame, contains only bids, asks, mids for required cols
				reference_datetime, datetime datetime, the 'fair price time' for some given day
	Returns:	reference_fair_quote, pandas Series, contains bids, asks, and mids for required tickers at the required time
				is_valid, boolean, the reference_fair_quote is not valid if it contains nulls
	"""
	reference_fair_quote = backtesting_frame.loc[reference_datetime]
	is_valid = reference_fair_quote.notna().all()
	return reference_fair_quote, is_valid

def getTakeOffQuoteFromBacktestingFrame(backtesting_frame, take_off_datetime):
	"""
	For getting the 'take_off' quote from the backtesting frame.
	Arguments:	backtesting_frame, pandas DataFrame, contains only bids, asks, mids for required cols
				take_off_datetime, datetime datetime, the 'fair price time' for some given day
	Returns:	take_off_quote, pandas Series, contains bids, asks, and mids for required tickers at the required time
				is_valid, boolean, the take_off_quote is not valid if it contains nulls
	"""
	take_off_quote = backtesting_frame.loc[take_off_datetime]
	is_valid = take_off_quote.notna().all()
	return take_off_quote, is_valid

def getReferenceAndTakeOffQuote(backtesting_frame, reference_datetime, take_off_datetime):
	"""
	For getting the take off quote and reference fair price quote.
	Arguments:	backtesting_frame, pandas DataFrame contains only bids, asks, mids for required cols
				reference_datetime, datetime datetime, the 'fair price time' for some given day
				take_off_datetime, datetime datetime, the 'fair price time' for some given day
	Returns:	reference_fair_quote, pandas Series, contains bids, asks, and mids for required tickers at the required time
				take_off_quote, pandas Series, contains bids, asks, and mids for required tickers at the required time
				is_valid, if one of these quotes contains nulls, we cannot continue
	"""
	reference_fair_quote, is_valid_reference = getReferenceQuoteFromBacktestingFrame(backtesting_frame, reference_datetime)
	take_off_quote, is_valid_take_off = getTakeOffQuoteFromBacktestingFrame(backtesting_frame, take_off_datetime)
	is_valid = is_valid_reference & is_valid_take_off
	return reference_fair_quote, take_off_quote, is_valid

def getOpenTakeOffTradingFrame(bid_ask_mid, ticker_to_trade, indep_rets, reference_time, start_trade_time, end_trade_time, take_off_time):
	"""
	For getting a dataframe containing the mid prices of the ticker to trade and the independent variables during the 
	open times of the ticker to trade, and the take off times of the ticker to trade. Also returns the start, end, and
	take off datetimes for the ticker to trade on the days for which we have valid data.
	A day has valid data if it has some quotes during the open times and it has a valid take off time quote.
	Arguments:	bid_ask_mid, pandas DataFrame, this could also be the full_mids table with bids and asks.
				ticker_to_trade, string,
				indep_rets, list of strings,
				reference_time, datetime time,
				start_trade_time, datetime time,
				end_trade_time, datetime time,
				take_off_time, datetime time,
	Returns:	trading_frame, contains data for times when the ticker_to_trade is open, and take off times
				start_end_off_datetimes, list of lists of four items [reference_time, start of trading, end, take off time]
					one four item list per valid trading day
	"""
	trade_dates = np.unique(bid_ask_mid.index.date)
	start_end_off_datetimes = []
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, indep_rets)
	trading_frame = pd.DataFrame()
	for date in trade_dates:
		next_date = date + dt.timedelta(days=1)
		reference_datetime = dt.datetime.combine(date, reference_time)
		start_trade_datetime = dt.datetime.combine(date, start_trade_time)
		end_trade_datetime = dt.datetime.combine(date, end_trade_time)
		take_off_datetime = dt.datetime.combine(next_date, take_off_time)
		if not reference_datetime in bid_ask_mid.index:
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'Missing reference quote at time ' + reference_datetime.isoformat())
			continue # if we don't have a reference time quote for fair prices, we can't trade
		reference_quote = bid_ask_mid.loc[reference_datetime, required_cols]
		if reference_quote.isna().any():
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'Invalid reference quote at time ' + reference_datetime.isoformat())
			continue # if we don't have a valid quote at reference (fair price) time, we cannot trade
		if not take_off_datetime in bid_ask_mid.index:
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'Missing take off quote at time ' + reference_datetime.isoformat())
			continue # if we don't have a take off time quote, we can't trade
		take_off_quote = bid_ask_mid.loc[take_off_datetime, required_cols]
		if take_off_quote.isna().any():
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'Invalid take off quote at time ' + reference_datetime.isoformat())
			continue # If we don't have a valid quote at the take off time, we cannot trade that day
		is_trading_time = (bid_ask_mid.index >= start_trade_datetime) & (bid_ask_mid.index <= end_trade_datetime)
		if not is_trading_time.any():
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'No data for  ' + date.isoformat())
			continue # we don't have data for this date
		session_trading_frame = bid_ask_mid.loc[is_trading_time, required_cols]
		if not session_trading_frame.notna().all(axis=1).any():
			print(dt.datetime.now().isoformat() + ' WARN: ' + 'No valid quotes for  ' + date.isoformat())
			continue # if we don't have any valid quotes, we cannot trade
		trading_frame = pd.concat([trading_frame, reference_quote.to_frame().T, session_trading_frame, take_off_quote.to_frame().T])
		start_end_off_datetimes += [[reference_datetime, start_trade_datetime, end_trade_datetime, take_off_datetime]]
	trading_frame.drop_duplicates(inplace=True) # if the reference or take off time is within the open session we can get duplicates
	return trading_frame, start_end_off_datetimes

def getModelledPriceForTicker(fair_price_dep, quoted_price_indep, fair_price_indep, returns_model):
	"""
	For calculating the modelled price of a dependent variable given a model with independent variables, their quoted and 
	fair prices, and the dependent fair price. (NB: Which prices are 'fair' is decided by market knowledge, consult Dan and Stephen) 
	TODO: This function needs unit tests
	Arguments:	fair_price_dep, the reference or 'fair' price for the dependent variable
				quoted_price_indep, the current or quoted prices of the independent variables
				fair_price_indep, the reference or fair prices for the independent variables
				returns_model, the linear model that gives the return of the dependent variable given the returns of the independents
	Returns:	modelled price of dependent variable
	"""
	indep_returns = (quoted_price_indep / fair_price_indep) - 1
	modelled_return = returns_model.predict([indep_returns])[0]
	modelled_price = fair_price_dep * (1 + modelled_return)
	return modelled_price

def getMaxBidMinAskLean(modelled_price, edge_required, lean, position):
	"""
	For getting the maximum price we are willing to pay (max bid) and minimum price for which we are willing to sell (min ask)
	given the modelled/fair price, a 'lean' amount (lean as in tilt, not lean as in not fat), and our current position.
	'Leaning' is integrating the market price into our model of the fair price by adjusting our modelled price scaled by our position.
	See README.md for more details on leaning.
	Arguments:	modelled_price, float
				edge_required, float, quoted in % so 10bps should be entered as 0.001 (consider changing this)
				lean, float, also quoted in %
				position, our position in the ticker we are trading.
	Returns:	max_bid, the maximum price we are willing to pay
				min_ask, the minimum price for which we are willing to sell
	"""
	max_bid_no_lean = modelled_price - modelled_price * edge_required # maximum that we are willing to pay
	min_ask_no_lean = modelled_price + modelled_price * edge_required # minimum at which we are willing to sell
	lean_amount = modelled_price * lean
	max_bid_with_lean = max_bid_no_lean - lean_amount * position
	min_ask_with_lean = min_ask_no_lean - lean_amount * position
	return max_bid_with_lean, min_ask_with_lean
