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
parser.add_argument('-s', '--ref_start_end_take_off', help='The reference, start, end, and take off times for the ticker to trade.', nargs=4, type=str, default=['0900', '0900', '2100', '0100'])
parser.add_argument('-n', '--num_indep_vars', help='The number of independent variables to use for modelling.', type=int, default=5)
parser.add_argument('-e', '--edge_required', help='The expected profit that we require from each trade. Quoted in bps, but entered as %.', type=float, default=0.001)
parser.add_argument('-l', '--lean', help='The amount to lean after each trade. Quoted in bps, but entered as %.', type=float, default=0.001)
parser.add_argument('-v', '--size', help='The size of each trade.', type=float, default=1000000)
parser.add_argument('-p', '--price_type', help='Exact or indicative, depending on the data source.', type=str, choices=['exact', 'indicative'], default='indicative')
parser.add_argument('-i', '--investigate', help='Put together a day long example frame for investigating.', action='store_true', default=False)
parser.add_argument('-r', '--report_suffix', help='Report by saving down the trade summary table using the suffix supplied.', type=str, default='')
parser.add_argument('-f', '--fill_all', help='Fill all gaps in historical data using special columns and modelling.', action='store_true', default=False)
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

def getMarketBidAskFromQuoteWithPriceType(quote, ask_to_trade, bid_to_trade, mid_to_trade, price_type):
	"""
	For getting the market asking price and offering price from the market quote, with a classification of the price type.
	If the price is not exact, we use the mid price
	Arguments:	quote, pandas Series,
				ask_to_trade, str, the ask column name
				bid_to_trade, str, the bid column name
				mid_to_trade, str, the mid column name
				price_type, str, [indicative or exact], depends on data source
	Returns:	market_ask, float, the asking price of the market
				market_offer, float, the offering price of the market
	"""
	if price_type == 'exact':
		market_ask = quote[ask_to_trade]
		market_offer = quote[bid_to_trade]
	elif price_type == 'indicative':
		market_ask = quote[mid_to_trade]
		market_offer = quote[mid_to_trade]
	else:
		print(dt.datetime.now().isoformat() + ' ERROR: ' + 'Unrecognised price type!')
		raise ValueError
	return market_ask, market_offer

def takeOffPosition(take_off_quote, ticker, position, price_type):
	"""
	For entering the transactions to take off the positions at the end of the day. 
	Arguments:	take_off_quote, pd.Series
				ticker, str
				position,
				price_type, [indicative or exact], depends on data source
	Returns:	the list of values for the columns of the transactions table			
	"""
	ask_to_trade, bid_to_trade, mid_to_trade, _ = getTickerBidAskMidRetColNames(ticker)
	market_ask, market_offer = getMarketBidAskFromQuoteWithPriceType(take_off_quote, ask_to_trade, bid_to_trade, mid_to_trade, price_type)
	if position > 0:
		record_elements = [take_off_quote.name, ticker, 's', market_offer, position, 0, np.nan, np.nan, np.nan]
	elif position < 0:
		record_elements = [take_off_quote.name, ticker, 'b', market_ask, position, 0, np.nan, np.nan, np.nan]
	else:
		print(dt.datetime.now().isoformat() + ' ERR: ' + 'Position is 0!')
		record_elements = None
	return record_elements

def executeTradeAndHedge(quote, required_tickers, indep_to_coefs, current_position, indep_positions, transaction_type, price_type):
	"""
	For getting the transaction records resulting from a trade and the hedges associated with that trade.
	Arguments:	quote, the bid/ask record for that time, name is the datetime,
				required_tickers, [ticker_to_trade  and 5 independent tickers]
				indep_to_coefs, dict, indep_ticker => coef
				current_position, float
				indep_positions, array of floats,
				transaction_type, 's' or 'b'
				price_type, depending on the data source
	Returns:	a list of lists, each element is a record for the transactions frame
	"""
	is_buy = transaction_type == 'b'
	new_position = current_position + 1 if is_buy else current_position - 1
	ask_to_trade, bid_to_trade, mid_to_trade, _ = getTickerBidAskMidRetColNames(required_tickers[0])
	market_ask, market_offer = getMarketBidAskFromQuoteWithPriceType(quote, ask_to_trade, bid_to_trade, mid_to_trade, price_type)
	price = market_ask if is_buy else market_offer
	trade_record = [quote.name, required_tickers[0], transaction_type, price, current_position, new_position]
	hedge_records = []
	for ticker, position in zip(required_tickers[1:], indep_positions):
		coef = indep_to_coefs[ticker]
		[ask_col, bid_col, mid_col, _] = getTickerBidAskMidRetColNames(ticker)
		market_ask, market_offer = getMarketBidAskFromQuoteWithPriceType(quote, ask_col, bid_col, mid_col, price_type)
		if ((coef > 0) & is_buy) | ((coef < 0) & (not is_buy)):
			new_indep_position = position - np.abs(coef)
			hedge_type = 's'
			hedge_price = market_offer
		elif ((coef > 0) & (not is_buy)) | ((coef < 0) & is_buy):
			new_indep_position = position + np.abs(coef)
			hedge_type = 'b'
			hedge_price = market_ask
		else:
			continue
		hedge_record = [quote.name, ticker, hedge_type, hedge_price, position, new_indep_position]
		hedge_records.append(hedge_record)
	return [trade_record] + hedge_records

def getTransactionReturns(trans_record, take_off_quote, price_type):
	"""
	For calculating the return on the given transaction after it is take off at take off time.
	Arguments:	trans_record, pandas Series
				take_off_quote, pandas Series
				model_coef, model coef (defaulted to 1 for a ticker not used by the model i.e. the ticker to trade)
				price_type, [exact, indicative], depending on data source
	Returns:	trans_return, float
				trans_expected_return, float
	"""
	is_buy = trans_record.transaction_type == 'b'
	ask_col, bid_col, mid_col, ret_col = getTickerBidAskMidRetColNames(trans_record.ticker)
	take_off_ask, take_off_offer = getMarketBidAskFromQuoteWithPriceType(take_off_quote, ask_col, bid_col, mid_col, price_type)
	if is_buy:
		trans_return = (take_off_offer - trans_record.price) / trans_record.price
		trans_expected_return = (trans_record.modelled_price - trans_record.price) / trans_record.price
	else:
		trans_return = (trans_record.price - take_off_ask) / trans_record.price
		trans_expected_return = (trans_record.price - trans_record.modelled_price) / trans_record.price
	return trans_return, trans_expected_return

def addProfitAndLossColumn(transactions, ticker_to_trade, take_off_quote, take_off_datetime, indep_to_coefs, size, price_type):
	"""
	Add a profit and loss column to the days transactions frame. The column we add is actually a return
	rather than a profit and loss. This has the advantage of being a unitless measurement (currencyless 
	in our case).
	Arguments:	transactions, pandas DataFrame
				ticker_to_trade, str,
				take_off_quote, pandas Series, the bids, asks, and mids at the take off time,
				take_off_datetime, datetime datetime,
				indep_to_coefs, dict, independent ticker => model coefficient,
				size, the size of each trade in $,
				price_type, str, [exact, indicative], depending on data source,
	Returns:	transactions, pandas DataFrame, with a profit and loss column
	"""
	transactions['return'] = 0.0
	transactions['expected_return'] = 0.0
	transactions['size'] = size
	non_take_off_transactions = transactions[transactions.transaction_time != take_off_datetime]
	for t_ind, t in non_take_off_transactions.iterrows():
		ticker_coef = np.abs(indep_to_coefs.get(t.ticker, 1))
		transactions.loc[t_ind, 'size'] = size * ticker_coef
		transactions.loc[t_ind, ['return', 'expected_return']] = getTransactionReturns(t, take_off_quote, price_type)
	transactions['pl'] = transactions['return'] * transactions['size']
	transactions['expected_pl'] = transactions['expected_return'] * transactions['size']
	return transactions

def simulateDaysTrade(backtesting_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, reference_datetime, ticker_to_trade, indep_rets, returns_model, edge_required, lean, size, price_type):
	"""
	Simulate a days worth of trading, using a leaning strategy.
	Arguments:	backtesting_frame, pandas DataFrame,
				start_trade_datetime, datetime datetime.
				end_trade_datetime, datetime datetime,
				take_off_datetime, datetime datetime,
				reference_datetime, the time that we consider correct for fair prices
				ticker_to_trade, str
				indep_mids, list of str
				returns_model, a linear regression model that models a return for the ticker to trade given the returns for the independent variables
				edge_required, float, quoted in % so 10bps should be entered as 0.001 (consider changing this)
				lean, float, also quoted in %
				size, the size of the trade, set to $1000000 for now.
				price_type, str, [exact, indicative], depends on data source
	Returns:	a table of trades, with times, bids and asks, position, and take off
	"""
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, indep_rets)
	mid_to_trade, bid_to_trade, ask_to_trade = mid_cols[0], bid_cols[0], ask_cols[0]
	indep_mids = mid_cols[1:] # getting useful column names
	indep_to_coefs = dict(zip(required_tickers[1:], returns_model.coef_))
	reference_fair_quote, take_off_quote, is_valid = getReferenceAndTakeOffQuote(backtesting_frame, reference_datetime, take_off_datetime)
	reference_fair_quote = backtesting_frame.loc[reference_datetime] # in the future, there may be logic required here
	take_off_quote = backtesting_frame.loc[take_off_datetime]
	is_today = (backtesting_frame.index >= start_trade_datetime) & (backtesting_frame.index <= end_trade_datetime)
	day_frame = backtesting_frame.loc[is_today, required_cols] # quotes for today
	transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'position_before', 'position_after', 'modelled_price', 'max_bid', 'min_ask'])
	current_position = 0 # initialising transaction table and positions
	indep_positions = np.zeros(5)
	for q_datetime, quote in day_frame.iterrows():
		if q_datetime == reference_datetime:
			continue # I don't think we trade at the same time as the fair price is established.
		trade_executed = False
		modelled_price = getModelledPriceForTicker(reference_fair_quote[mid_to_trade], quote[indep_mids], reference_fair_quote[indep_mids], returns_model)
		max_bid, min_ask = getMaxBidMinAskLean(modelled_price, lean, current_position, edge_required)
		market_ask, market_offer = getMarketBidAskFromQuoteWithPriceType(quote, ask_to_trade, bid_to_trade, mid_to_trade, price_type)
		if (max_bid >= market_ask) & (current_position < 3):
			trade_executed = True
			transaction_type = 'b'
			new_transactions = executeTradeAndHedge(quote, required_tickers, indep_to_coefs, current_position, indep_positions, transaction_type, price_type)
		if (min_ask <= market_offer) * (current_position > -3):
			trade_executed = True
			transaction_type = 's'
			new_transactions = executeTradeAndHedge(quote, required_tickers, indep_to_coefs, current_position, indep_positions, transaction_type, price_type)
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
	if transactions.shape[0] > 0:
		for ticker, position in zip(required_tickers, np.hstack([current_position, indep_positions])):
			if position != 0:
				transactions.loc[len(transactions)] = takeOffPosition(take_off_quote, ticker, position, price_type)
	transactions = addProfitAndLossColumn(transactions, ticker_to_trade, take_off_quote, take_off_datetime, indep_to_coefs, size, price_type)
	return transactions

def getTradeSummaryFrame(transactions, start_trade_datetime, end_trade_datetime):
	"""
	For getting a table that aggregates the day together and shows net returns.
	Arguments:	transactions, a days worth of transactions, can include take off times but no needed
				start_trade_datetime, datetime datetime
				end_trade_datetime, datetime datetime
	Returns:	pandas DataFrame
	"""
	get_first = lambda x:x.iloc[0]
	non_take_off_transactions = transactions[(transactions.transaction_time >= start_trade_datetime) & (transactions.transaction_time <= end_trade_datetime)]
	agg_frame = non_take_off_transactions.groupby('transaction_time').agg({'ticker':get_first, 
		'transaction_type':get_first, 'price':get_first, 'position_before':get_first, 'position_after':get_first, 
		'modelled_price':get_first, 'max_bid':get_first, 'min_ask':get_first, 'return':get_first, 
		'expected_return':sum, 'size':get_first, 'pl':sum, 'expected_pl':sum})
	return agg_frame

def printNumDaysTradesVsNumDaysPossible(backtesting_frame, trade_summary):
	"""
	For printing the number of days on which we traded vs the number of days on which we could have traded.
	Arguments:	backtesting_frame, pandas DataFrame, days we have data
				trade_summary, pandas DataFrame, trades
	Returns:	nothing
	"""
	days_could_have_traded = np.unique(backtesting_frame.index.date)
	days_we_traded = np.unique(trade_summary.index.date)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Could have traded on ' + str(len(days_could_have_traded)) + ' days.')
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Traded on ' + str(len(days_we_traded)) + ' days.')
	return None

if not args.debug:
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading in csvs...')
	open_close = loadOpenCloseTimesCsv(csv_dir)
	bid_ask_mid = loadBidAskMid(csv_dir)
	ticker_to_trade = args.ticker_to_trade
	if args.fill_all:
		bid_ask_mid = fillWithSpecialModelExcludeDependent(csv_dir, bid_ask_mid, ticker_to_trade)
	historical_data, backtesting_data = divideDataForBacktesting(bid_ask_mid)
	hourly_returns = getHourlyReturnTable(historical_data)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Finding the most influential independent variables...')
	thresh = 0.8
	regression_model, top_indep_rets, top_coefs, reg_r_sq = getTopIndependentVars(ticker_to_trade, hourly_returns, open_close, thresh, num_indep_vars=args.num_indep_vars)
	top_model, r_sq, shuffle_r_sq = getLinearModelFromDepIndep(ticker_to_trade, top_indep_rets, hourly_returns, open_close)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting trading frame and fair prices...')
	reference_time, start_trade_time, end_trade_time, take_off_time = [dt.datetime.strptime(str_t, '%H%M').time() for str_t in args.ref_start_end_take_off]
	backtesting_frame, valid_start_end_off_datetimes = getOpenTakeOffTradingFrame(backtesting_data, ticker_to_trade, top_indep_rets, reference_time, start_trade_time, end_trade_time, take_off_time)
	print(dt.datetime.now().isoformat() + ' INFO: ' + 'Simulating trading days...')
	edge_required = args.edge_required
	lean = args.lean
	all_transactions = pd.DataFrame(columns=['transaction_time', 'ticker', 'transaction_type', 'price', 'size', 'position_before', 'position_after', 'modelled_price', 'max_bid', 'min_ask', 'return', 'expected_return', 'pl', 'expected_pl'])
	trade_summary = pd.DataFrame(columns=['ticker', 'transaction_type', 'price', 'size', 'position_before', 'position_after', 'modelled_price', 'max_bid', 'min_ask', 'return', 'expected_return', 'pl', 'expected_pl'])
	for reference_datetime, start_trade_datetime, end_trade_datetime, take_off_datetime in valid_start_end_off_datetimes:
		transactions = simulateDaysTrade(backtesting_frame, start_trade_datetime, end_trade_datetime, take_off_datetime, reference_datetime, ticker_to_trade, top_indep_rets, top_model, edge_required, lean, args.size, args.price_type)
		trade_summary = pd.concat([trade_summary, getTradeSummaryFrame(transactions, start_trade_datetime, end_trade_datetime)])
		all_transactions = pd.concat([all_transactions, transactions], ignore_index=True)
	printNumDaysTradesVsNumDaysPossible(backtesting_frame, trade_summary)
	if args.report_suffix != '':
		trade_summary_csv_file_name = os.path.join(csv_dir, 'trade_summaries', 'trade_summary_' + args.report_suffix + '.csv')
		trade_summary.index.name = 'transaction_time'
		trade_summary.to_csv(trade_summary_csv_file_name)
		print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saved: ' + trade_summary_csv_file_name)

if args.investigate & np.invert(args.debug):
	# Save a frame for Dan and I to investigate
	required_tickers, required_cols, bid_cols, ask_cols, mid_cols = getRequiredTradingCols(ticker_to_trade, top_indep_rets)
	is_required_quote = (backtesting_frame.index >= dt.datetime(2020,12,7,9,0,0)) & (backtesting_frame.index <= dt.datetime(2020,12,8,1,0,0))
	one_date_inv_frame = backtesting_frame.loc[is_required_quote, mid_cols]
	reference_quote = one_date_inv_frame.loc[dt.datetime(2020,12,7,9,0,0)]
	one_date_ret_frame = one_date_inv_frame/reference_quote - 1
	one_date_ret_frame.columns = [c.replace('_Mid', '_Ret') for c in one_date_ret_frame.columns]
	one_date_inv_frame = one_date_inv_frame.merge(one_date_ret_frame, how='left', left_index=True, right_index=True)
	indep_ret_cols = one_date_ret_frame.columns[1:]
	one_date_inv_frame['modelled_return'] = top_model.predict(one_date_ret_frame[indep_ret_cols])
	one_date_inv_frame['by_hand_return'] = (one_date_ret_frame[indep_ret_cols] * top_model.coef_).sum(axis=1)
	one_date_inv_frame['modelled_price'] = (1+one_date_inv_frame['modelled_return']) * reference_quote[mid_cols[0]]
	assert np.isclose(one_date_inv_frame['modelled_return'], one_date_inv_frame['by_hand_return']).all()
	one_date_inv_frame['edge_required'] = args.edge_required
	one_date_inv_frame['max_bid'] = one_date_inv_frame['modelled_price'] - one_date_inv_frame['edge_required'] * one_date_inv_frame['modelled_price']
	one_date_inv_frame['min_ask'] = one_date_inv_frame['modelled_price'] + one_date_inv_frame['edge_required'] * one_date_inv_frame['modelled_price']

	is_tradeable = (one_date_inv_frame[mid_cols[0]] < one_date_inv_frame['max_bid']) | (one_date_inv_frame[mid_cols[0]] > one_date_inv_frame['min_ask'])

