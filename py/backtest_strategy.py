"""
For testing a trading stratey or strategies on historical data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For making a "returns" table.')
parser.add_argument('-n', '--num_indep_vars', help='The number of independent variables to use for modelling.', type=int, default=5)
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

def getTradingFrame(bid_ask_mid, ticker_to_trade, indep_rets, start_trade_time, end_trade_time, take_off_time):
	"""
	For getting a dataframe containing the mid prices of the tikcer to trade and the independent variables during the 
	open times of the ticker to trade, and the take off times of the ticker to trade. Also returns the start, end, and
	take off datetimes for the ticker to trade on the days for which we have valid data.
	Arguments:	bid_ask_mid, pandas DataFrame
				ticker_to_trade, string,
				indep_rets, list of strings,
				start_trade_time, datetime time,
				end_trade_time, datetime time,
				take_off_time, datetime time,
	Returns:	List of 3 elements list of datetime datetime
	"""
	trade_dates = np.unique(bid_ask_mid.index.date)
	start_end_off_datetimes = []
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, indep_rets)
	trading_frame = pd.DataFrame()
	for date in trade_dates:
		start_trade_datetime = dt.datetime.combine(date, start_trade_time)
		end_trade_datetime = dt.datetime.combine(date, end_trade_time)
		next_date = date + dt.timedelta(days=1)
		take_off_datetime = dt.datetime.combine(next_date, take_off_time)
		if not take_off_datetime in bid_ask_mid.index:
			continue # if we don't have a take off time quote, we can't trade
		take_off_quote = bid_ask_mid.loc[take_off_datetime, required_cols]
		if take_off_quote.isna().any():
			continue # If we don't have a valid quote at the take off time, we cannot trade that day
		is_trading_time = (bid_ask_mid.index >= start_trade_datetime) & (bid_ask_mid.index <= end_trade_datetime)
		if not is_trading_time.any():
			continue # we don't have data for this date
		session_trading_frame = bid_ask_mid.loc[is_trading_time, required_cols]
		if not session_trading_frame.notna().all(axis=1).any():
			continue # if we don't have any valid quotes, we cannot trade
		trading_frame = pd.concat([trading_frame, session_trading_frame, take_off_quote.to_frame().T])
		start_end_off_datetimes += [[start_trade_datetime, end_trade_datetime, take_off_datetime]]
	return trading_frame, start_end_off_datetimes

def getRequiredTradingCols(ticker_to_trade, indep_rets):
	"""
	For getting the column names that we need for trading.
	Arguments:	ticker_to_trade, str
				indep_rets, list of str, returns columns
	Returns:	required_tickers,
				required_trading_cols, all the required cols for trading (excluding returns)
				bid_cols,
				ask_cols,
				mid_cols,
	"""
	indep_tickers = extractTickerNameFromColumns(indep_rets.tolist())
	required_tickers = np.hstack([ticker_to_trade, indep_tickers])
	required_cols = getTickerBidAskMidRetColNames(required_tickers)
	bid_cols = [c for c in required_cols if c.find('_Bid') > -1]
	ask_cols = [c for c in required_cols if c.find('_Ask') > -1]
	mid_cols = [c for c in required_cols if c.find('_Mid') > -1]
	required_cols = bid_cols + ask_cols + mid_cols
	return required_tickers, required_cols, bid_cols, ask_cols, mid_cols

def takeOffPosition(take_off_record, ticker, position):
	"""
	For entering the transactions to take off the positions at the end of the day. 
	Arguments:	take off record, pd.Series
				ticker, str
				position
	Returns:	the list of values for the columns of the transactions table			
	"""
	if position > 0:
		record_elements = [take_off_record.name, ticker, 's', take_off_record[ticker + '_Ask'], position, 0]
	elif position < 0:
		record_elements = [take_off_record.name, ticker, 'b', take_off_record[ticker + '_Bid'], position, 0]
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

def getMaxBidMinAsk(modelled_price, lean, position, profit_required):
	"""
	For getting the maximum price we are willing to pay (max bid) and minimum price for which we are willing to sell (min ask)
	given the modelled/fair price, a 'lean' amount (lean as in tilt, not lean as in not fat), and our current position.
	'Leaning' is integrating the market price into our model of the fair price by adjusting our modelled price scaled by our position.
	Arguments:	modelled_price, float
				lean, the amount by which we lean for every unit of our position
				position, our position in the ticker we are trading.
	Returns:	max_bid, the maximum price we are willing to pay
				min_ask, the minimum price for which we are willing to sell
	"""
	max_bid = modelled_price * (1 - (profit_required * (1+lean) * np.negative(position)))
	min_ask = modelled_price * (1 + (profit_required * (1+lean) * np.negative(position)))
	return max_bid, min_ask

def simulateDaysTrade(trading_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, fair_price_time, ticker_to_trade, indep_rets, returns_model, profit_required):
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
				profit_required
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
	transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'position_before', 'position_after'])
	current_position = 0
	indep_positions = np.zeros(5)
	for q_datetime, quote in day_frame.iterrows():
		if q_datetime == fair_price_datetime:
			continue 
		trade_executed = False
		modelled_price = getModelledPriceForTicker(reference_fair_quote[mid_to_trade], quote[indep_mids], reference_fair_quote[indep_mids], returns_model)
		# max_bid, min_ask = getMaxBidMinAsk(modelled_price, lean, current_position)
		max_bid = modelled_price * (1 - profit_required) # maximum that we are willing to pay
		min_ask = modelled_price * (1 + profit_required) # minimum at which we are willing to sell
		if (max_bid >= quote[ask_to_trade]) & (current_position < 3):
			trade_executed = True
			transaction_type = 'b'
			new_transactions = executeTradeAndHedge(quote, required_tickers, indep_coefs, current_position, indep_positions, transaction_type)
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
				transactions.loc[len(transactions)] = t
	take_off_record = trading_frame.loc[take_off_datetime]
	if transactions.shape[0] > 0:
		for ticker, position in zip(required_tickers, np.hstack([current_position, indep_positions])):
			if position != 0:
				transactions.loc[len(transactions)] = takeOffPosition(take_off_record, ticker, position)
	transactions = addProfitAndLossColumn(transactions, ticker_to_trade)
	return transactions

def chunker(seq, size):
    return (seq.iloc[pos:pos + size] for pos in range(0, len(seq), size))

def addProfitAndLossColumn(days_transactions, ticker_to_trade):
	"""
	Add a profit and loss column to the days transactions frame.
	Arguments:	days_transactions, pandas DataFrame
				ticker_to_trade, str
	Returns:	days_transactions, pandas DataFrame, with a profit and loss column
	"""
	buys = days_transactions.loc[(days_transactions.ticker == ticker_to_trade) & (days_transactions.transaction_type == 'b')]
	outgoings = (buys.price * (buys.position_after - buys.position_before)).cumsum()
	buys.insert(buys.shape[1], 'outgoings', outgoings, allow_duplicates=True)
	sells = days_transactions.loc[(days_transactions.ticker == ticker_to_trade) & (days_transactions.transaction_type == 's')]
	incomings = (sells.price * (sells.position_before - sells.position_after)).cumsum()
	sells.insert(sells.shape[1], 'incomings', incomings, allow_duplicates=True)
	days_transactions = days_transactions.merge(buys, how='left').merge(sells, how='left')
	days_transactions.loc[:,['outgoings', 'incomings']] = days_transactions.loc[:,['outgoings', 'incomings']].ffill()
	days_transactions.loc[:,['outgoings', 'incomings']] = days_transactions.loc[:,['outgoings', 'incomings']].fillna(value=0)
	days_transactions.loc[:,'pl'] = days_transactions.loc[:,'incomings'] - days_transactions.loc[:,'outgoings']
	return days_transactions

def getModelledPriceForTicker(fair_price_dep, quoted_price_indep, fair_price_indep, returns_model):
	"""
	For calculating the modelled price of a dependent variable given a model with independent variables, their quoted and 
	fair prices, and the dependent fair price. (NB: Which prices are 'fair' is decided by market knowledge, consult Dan and Stephen) 
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

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading in csvs...')
open_close = loadOpenCloseTimesCsv(csv_dir)
bid_ask_mid = loadBidAskMid(csv_dir)
historical_data, backtesting_data = divideDataForBacktesting(bid_ask_mid)
hourly_returns = getHourlyReturnTable(historical_data)
ticker_to_trade = 'KWN+1M BGN Curncy'
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Finding the most influential independent variables...')
thresh = 0.8
regression_model, top_indep_rets, top_coefs, r_sq = getTopIndependentVars(ticker_to_trade, hourly_returns, open_close, thresh, num_indep_vars=args.num_indep_vars)
top_model, r_sq, shuffle_r_sq = getLinearModelFromDepIndep(ticker_to_trade, top_indep_rets, hourly_returns, open_close)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting trading frame and fair prices...')
start_trade_time = dt.time(9,0)
end_trade_time = dt.time(20,0)
take_off_time = dt.time(1,0) # next day
fair_price_time = start_trade_time
backtesting_frame, valid_start_end_off_datetimes = getTradingFrame(backtesting_data, ticker_to_trade, top_indep_rets, start_trade_time, end_trade_time, take_off_time)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Simulating trading days...')
profit_required = 0.001 
transactions = pd.DataFrame()
for start_trade_datetime, end_trade_datetime, take_off_datetime in valid_start_end_off_datetimes:
	days_transactions = simulateDaysTrade(backtesting_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, fair_price_time, ticker_to_trade, top_indep_rets, top_model, profit_required)
	transactions = pd.concat([transactions, days_transactions])


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

				