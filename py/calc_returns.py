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
from sklearn.linear_model import LinearRegression, LassoCV

parser = argparse.ArgumentParser(description='For making a "returns" table.')
parser.add_argument('-f', '--fill_all', help='Flag for filling all the columns. This can take up to 15 minutes', action='store_true', default=False)
parser.add_argument('-s', '--save_bid_ask_mid', help='Flag for saving the bid_ask_mid table, or not', action='store_true', default=False)
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

def getCoefFrameForTickers(tickers_to_model, hourly_returns, open_close, thresh=0.8):
	"""
	Get the top 5 tickers for linear regression modelling of each of the given tickers.
	Returns a DataFrame containing the tickers and their coefficients.
	Arguments:	tickers_to_model, pandas DataFrame,
				hourly_returns, pandas DataFrame
				open_close, pandas DataFrame
				thresh, float, thresholding (see getTopFiveIndependentVars function)
	Returns:	coef_frame, pandas DataFrame
	"""
	coef_frame = pd.DataFrame()
	for ticker in tickers_to_model:
		regression_model, top_five_indep_vars, top_five_coefs, r_sq = getTopFiveIndependentVars(ticker, hourly_returns, open_close, thresh)
		ticker_model_frame = pd.DataFrame(columns=['independent_vars','coefs'], data=np.vstack([top_five_indep_vars, top_five_coefs]).T)
		ticker_model_frame['dependent_var'] = ticker
		ticker_model_frame['r_squared'] = r_sq
		coef_frame = pd.concat([coef_frame, ticker_model_frame])
	coef_frame['coefs'] = pd.to_numeric(coef_frame['coefs'])
	coef_frame['r_squared'] = pd.to_numeric(coef_frame['r_squared'])
	return coef_frame

if not args.debug:
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
	all_clean = loadCleanBidAsk(csv_dir)
	bid_ask_mid = addMidColumns(all_clean.copy())
	if args.save_bid_ask_mid:
		bid_ask_mid_file_name = os.path.join(csv_dir, 'bid_ask_mid.csv')
		bid_ask_mid.to_csv(bid_ask_mid_file_name)
		print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + bid_ask_mid_file_name)
	bid_ask_mid_hourly_returns = getBidMidAskHourlyReturnTable(bid_ask_mid)
	getBidAskMidRegressionModels(bid_ask_mid_hourly_returns)
	ticker_to_model = getTickerRegressionModels(hourly_returns) # we model before filling the special columns to preserve data integrity
	col_to_model...
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Stage 2 done.')
	special_filled = fillSpecialColumns(bid_ask_mid.copy())
	tickers_to_model = ['KWN+1M BGN Curncy', 'IHN+1M BGN Curncy', 'FXY1 KMS1 Index', 'MXID Index']
	open_close = loadOpenCloseTimesCsv(csv_dir)
	coef_frame = getCoefFrameForTickers(tickers_to_model, hourly_returns, open_close, thresh=0.8)
	coef_frame_file_name = os.path.join(csv_dir,'coef_frame.csv')
	coef_frame.to_csv(coef_frame_file_name)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + coef_frame_file_name)
	if args.fill_all:
		full_mids = fillRemainingBlanks(special_filled.copy(), ticker_to_model)
		full_mids_file_name = os.path.join(csv_dir, 'full_mids.csv')
		full_mids.to_csv(full_mids_file_name)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done. ' + full_mids_file_name)