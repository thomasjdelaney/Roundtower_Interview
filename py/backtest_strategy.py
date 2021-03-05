"""
For testing a trading stratey or strategies on historical data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For making a "returns" table.')
parser.add_argument('-t', '--ticker_to_trade', help='The ticker to trade', type=str, default='KWN+1M BGN Curncy')
parser.add_argument('-s', '--start_end_take_off', help='The start, end, and take off times for the ticker to trade.', nargs=3, type=str, default=['0900', '2000', '0100'])
parser.add_argument('-n', '--num_indep_vars', help='The number of independent variables to use for modelling.', type=int, default=5)
parser.add_argument('-p', '--profit_required', help='The expected profit that we require from each trade. Quoted in bps, but entered as %.', type=float, default=0.001)
parser.add_argument('-l', '--lean', help='The amount to lean after each trade. Quoted in bps, but entered as %.', type=float, default=0.001)
parser.add_argument('-v', '--size', help='The size of each trade.', type=float, default=1000000)
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

def takeOffPosition(take_off_record, ticker, position):
	"""
	For entering the transactions to take off the positions at the end of the day. 
	Arguments:	take off record, pd.Series
				ticker, str
				position
	Returns:	the list of values for the columns of the transactions table			
	"""
	if position > 0:
		record_elements = [take_off_record.name, ticker, 's', take_off_record[ticker + '_Ask'], position, 0, np.nan, np.nan, np.nan]
	elif position < 0:
		record_elements = [take_off_record.name, ticker, 'b', take_off_record[ticker + '_Bid'], position, 0, np.nan, np.nan, np.nan]
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Position is 0!')
		record_elements = None
	return record_elements

def executeTradeAndHedge(quote, required_tickers, indep_coefs, current_position, indep_positions, transaction_type):
	"""
	For getting the transaction records resulting from a trade and the hedges associated with that trade.
	Arguments:	quote, the bid/ask record for that time, name is the datetime,
				required_tickers, [ticker_to_trade  and 5 independent tickers]
				indep_coefs, array of floats, coefficients, 
				current_position, float
				indep_positions, array of floats,
				transaction_type
	Returns:	a list of lists, each element is a record for the transactions frame
	"""
	is_buy = transaction_type == 'b'
	new_position = current_position + 1 if is_buy else current_position - 1
	price = quote[ticker_to_trade + '_Ask'] if is_buy else quote[ticker_to_trade + '_Bid']
	trade_record = [quote.name, required_tickers[0], transaction_type, price, current_position, new_position]
	hedge_records = []
	for ticker, coef, position in zip(required_tickers[1:], indep_coefs, indep_positions):
		if ((coef > 0) & is_buy) | ((coef < 0) & (not is_buy)):
			new_indep_position = position - np.abs(coef)
			hedge_record = [quote.name, ticker, 's', quote[ticker + '_Bid'], position, new_indep_position]
		elif ((coef > 0) & (not is_buy)) | ((coef < 0) & is_buy):
			new_indep_position = position + np.abs(coef)
			hedge_record = [quote.name, ticker, 'b', quote[ticker + '_Ask'], position, new_indep_position]
		else:
			continue
		hedge_records.append(hedge_record)
	return [trade_record] + hedge_records

def getTransactionReturns(trans_record, take_off_record):
	"""
	For calculating the return on the given transaction after it is take off at take off time.
	Arguments:	trans_record, pandas Series
				take_off_record, pandas Series
	Returns:	trans_return, float
				trans_expected_return, float
	"""
	is_buy = trans_record.transaction_type == 'b'
	ask_col, bid_col = trans_record.ticker + '_Ask', trans_record.ticker + '_Bid'
	if is_buy:
		trans_return = (trans_record.price - take_off_record[ask_col]) / trans_record.price
		trans_expected_return = (trans_record.modelled_price - trans_record.max_bid) / trans_record.modelled_price
	else:
		trans_return = (take_off_record[bid_col] - trans_record.price) / trans_record.price
		trans_expected_return = (trans_record.min_ask - trans_record.modelled_price) / trans_record.modelled_price
	return trans_return, trans_expected_return

def addProfitAndLossColumn(transactions, ticker_to_trade, take_off_record, take_off_datetime, size):
	"""
	Add a profit and loss column to the days transactions frame. The column we add is actually a return
	rather than a profit and loss. This has the advantage of being a unitless measurement (currencyless 
	in our case).
	Arguments:	transactions, pandas DataFrame
				ticker_to_trade, str
				take_off_record, pandas Series, the bids, asks, and mids at the take off time
				take_off_datetime, datetime datetime
				size, the size of each trade in $
	Returns:	transactions, pandas DataFrame, with a profit and loss column
	"""
	transactions['return'] = 0.0
	transactions['expected_return'] = 0.0
	non_take_off_transactions = transactions[transactions.transaction_time != take_off_datetime]
	for t_ind, t in non_take_off_transactions.iterrows():
		transactions.loc[t_ind, ['return', 'expected_return']] = getTransactionReturns(t, take_off_record)
	transactions['size'] = size
	transactions['pl'] = transactions['return'] * size
	transactions['expected_pl'] = transactions['expected_return'] * size
	return transactions

def simulateDaysTrade(trading_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, fair_price_time, ticker_to_trade, indep_rets, returns_model, profit_required, lean, size):
	"""
	Simulate a days worth of trading, using a leaning strategy.
	Arguments:	trading_frame, pandas DataFrame,
				start_trade_datetime, datetime datetime.
				end_trade_datetime, datetime datetime,
				take_off_datetime, datetime datetime,
				fair_price_time, the time that we consider correct for fair prices
				ticker_to_trade, str
				indep_mids, list of str
				returns_model, a linear regression model that models a return for the ticker to trade given the returns for the independent variables
				profit_required, float, quoted in % so 10bps should be entered as 0.001 (consider changing this)
				lean, float, also quoted in %
				size, the size of the trade, set to $1000000 for now.
	Returns:	a table of trades, with times, bids and asks, position, and take off
	"""
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, indep_rets)
	mid_to_trade, bid_to_trade, ask_to_trade = mid_cols[0], bid_cols[0], ask_cols[0]
	indep_mids = mid_cols[1:] # getting useful column names
	fair_price_datetime = dt.datetime.combine(start_trade_datetime.date(), fair_price_time) # key for the reference 'fair' prices
	indep_coefs = returns_model.coef_
	is_today = (trading_frame.index >= start_trade_datetime) & (trading_frame.index <= end_trade_datetime)
	day_frame = trading_frame.loc[is_today, required_cols] # quotes for today
	reference_fair_quote = day_frame.loc[fair_price_datetime] # in the future, there may be logic required here
	transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'position_before', 'position_after', 'modelled_price', 'max_bid', 'min_ask'])
	current_position = 0 # initialising transaction table and positions
	indep_positions = np.zeros(5)
	for q_datetime, quote in day_frame.iterrows():
		if q_datetime == fair_price_datetime:
			continue # I don't think we trade at the same time as the fair price is established.
		trade_executed = False
		modelled_price = getModelledPriceForTicker(reference_fair_quote[mid_to_trade], quote[indep_mids], reference_fair_quote[indep_mids], returns_model)
		max_bid, min_ask = getMaxBidMinAskLean(modelled_price, lean, current_position, profit_required)
		if (max_bid >= quote[ask_to_trade]) & (current_position < 3):
			trade_executed = True
			transaction_type = 'b'
			new_transactions = executeTradeAndHedge(quote, required_tickers, indep_coefs, current_position, indep_positions, transaction_type,)
		if (min_ask <= quote[bid_to_trade]) * (current_position > -3):
			trade_executed = True
			transaction_type = 's'
			new_transactions = executeTradeAndHedge(quote, required_tickers, indep_coefs, current_position, indep_positions, transaction_type)
		if trade_executed:
			new_position = new_transactions[0][-1]
			new_indep_positions = np.array([transaction[-1] for transaction in new_transactions[1:]])
			current_position = new_position
			indep_positions = new_indep_positions
			for t in new_transactions:
				if t[1] == ticker_to_trade:
					t += [modelled_price, max_bid, min_ask]
				else:
					t += [np.nan, np.nan, np.nan]
				transactions.loc[len(transactions)] = t
	take_off_record = trading_frame.loc[take_off_datetime]
	if transactions.shape[0] > 0:
		for ticker, position in zip(required_tickers, np.hstack([current_position, indep_positions])):
			if position != 0:
				transactions.loc[len(transactions)] = takeOffPosition(take_off_record, ticker, position)
	transactions = addProfitAndLossColumn(transactions, ticker_to_trade, take_off_record, take_off_datetime, size)
	return transactions

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading in csvs...')
open_close = loadOpenCloseTimesCsv(csv_dir)
bid_ask_mid = loadBidAskMid(csv_dir)
historical_data, backtesting_data = divideDataForBacktesting(bid_ask_mid)
hourly_returns = getHourlyReturnTable(historical_data)
ticker_to_trade = args.ticker_to_trade
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Finding the most influential independent variables...')
thresh = 0.8
regression_model, top_indep_rets, top_coefs, r_sq = getTopIndependentVars(ticker_to_trade, hourly_returns, open_close, thresh, num_indep_vars=args.num_indep_vars)
top_model, r_sq, shuffle_r_sq = getLinearModelFromDepIndep(ticker_to_trade, top_indep_rets, hourly_returns, open_close)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting trading frame and fair prices...')
start_trade_time, end_trade_time, take_off_time = [dt.datetime.strptime(str_t, '%H%M').time() for str_t in args.start_end_take_off]
start_trade_time = dt.time(9,0)
end_trade_time = dt.time(20,0)
take_off_time = dt.time(1,0) # next day
fair_price_time = start_trade_time
backtesting_frame, valid_start_end_off_datetimes = getOpenTakeOffTradingFrame(backtesting_data, ticker_to_trade, top_indep_rets, start_trade_time, end_trade_time, take_off_time)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Simulating trading days...')
profit_required = args.profit_required
lean = args.lean
all_transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'size', 'position_before', 'position_after', 'modelled_price', 'max_bid', 'min_ask', 'return', 'expected_return', 'pl', 'expected_pl'])
for start_trade_datetime, end_trade_datetime, take_off_datetime in valid_start_end_off_datetimes:
	transactions = simulateDaysTrade(backtesting_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, fair_price_time, ticker_to_trade, top_indep_rets, top_model, profit_required, lean, args.size)
	all_transactions = pd.concat([all_transactions, transactions], ignore_index=True)

				