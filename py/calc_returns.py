"""
For creating a table of hourly returns from the cleaned bid/ask data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description='For making a "returns" table.')
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

pd.set_option('max_rows',30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns-2)

# useful globals
proj_dir = os.path.join(os.environ.get('HOME'), 'Roundtower_Interview')
csv_dir = os.path.join(proj_dir, 'csv')
py_dir = os.path.join(proj_dir, 'py')

special_mid_names = ['ES1 Index_Mid', 'MES1 Index_Mid', 'SGD BGN Curncy_Mid']

sys.path.append(py_dir)
from Roundtower import *

def loadCleanBidAsk(csv_dir):
	"""
	For loading the cleaned bid/ask data.
	Arguments:	csv_dir, str
	Returns:	pandas DataFrame
	"""
	return pd.read_csv(os.path.join(csv_dir,'all_clean.csv'), parse_dates=[0], index_col=0)

def addMidColumns(all_clean):
	"""
	For adding mid columns to the cleaned data.
	Arguments:	all_clean, pandas DataFrame
	Return:		pandas DataFrame
	"""
	tickers = extractTickerNameFromColumns(all_clean.columns.to_list())
	for ticker in tickers:
		bid_col_name, ask_col_name, mid_col_name = getTickerBidAskMidColNames(ticker)
		mid_col = all_clean[[bid_col_name, ask_col_name]].mean(axis=1)
		all_clean[mid_col_name] = mid_col
	return all_clean

def getReturnTable(bid_ask_mid):
	"""
	For calculating an hourly return table. Return = (mid/mid_one_hour_ago) - 1
	Arguments:	bid_ask_mid
	Returns:	pandas DataFrame
	"""
	is_hourly_index = bid_ask_mid.index.minute == 0
	mid_column_names = getMatchedColNames(bid_ask_mid, '_Mid')
	return bid_ask_mid.loc[is_hourly_index, mid_column_names].pct_change(fill_method=None)

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
	regression_model = LinearRegression(fit_intercept=False, n_jobs=-1)
	regression_model.fit(X, y)
	return regression_model

def getPredictorNamesForTicker(ticker):
	"""
	Get the columns names of the appropriate predictors given a ticker.
	Arguments:	ticker, str
	Returns:	list of str, column names for the hourly_returns frame
	"""
	if ticker.endswith('Index') | ticker.endswith('Equity') | ticker.endswith('Comdty'):
		predictor_names = ['ES1 Index_Mid', 'MES1 Index_Mid']
	elif ticker.endswith('Curncy'):
		predictor_names = ['SGD BGN Curncy_Mid']
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Unrecognised ticker! ' + ticker)
		predictor_names = ['']
	return predictor_names

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

def fillSpecialColumns(bid_ask_mid):
	"""
	For filling the ES1 Index, MES1 Index, and SGD BGN Curncy columns forward, so that there are no blanks.
	Arguments:	bid_ask_mid, pandas DataFrame
	Returns:	special_filled, pandas DataFrame
	"""
	special_col_names = [c for c in bid_ask_mid.columns.to_list() 
							if c.startswith('ES1') | c.startswith('MES1') | c.startswith('SGD')]
	special_columns_filled = bid_ask_mid[special_col_names].ffill()
	bid_ask_mid.update(special_columns_filled)
	return bid_ask_mid

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
		predictor_names = getPredictorNamesForTicker(ticker)
		relevant_cols = special_filled[[mid_col_name] + predictor_names]
		first_valid_ind = relevant_cols.loc[relevant_cols.notna().all(axis=1)].index[0]
		for_filling = relevant_cols.loc[first_valid_ind:]
		to_be_filled = for_filling.loc[for_filling[mid_col_name].isna()].copy()
		null_blocks_indices = detectBlocksOfNulls(to_be_filled.index.to_list())
		for null_block_indices in null_blocks_indices:
			modelled_series = modelNullBlock(for_filling, null_block_indices, ticker_to_model, mid_col_name, predictor_names)
			to_be_filled.update(modelled_series)
		special_filled.update(to_be_filled[mid_col_name])
	return special_filled[mid]

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
all_clean = loadCleanBidAsk(csv_dir)
bid_ask_mid = addMidColumns(all_clean.copy())
hourly_returns = getReturnTable(bid_ask_mid)
ticker_to_model = getTickerRegressionModels(hourly_returns)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Stage 2 done.')
special_filled = fillSpecialColumns(bid_ask_mid.copy())
full_mids = fillRemainingBlanks(special_filled.copy(), ticker_to_model)
full_mids_file_name = os.path.join(csv_dir, 'full_mids.csv')
full_mids.to_csv(full_mids_file_name)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done. ' + full_mids_file_name)