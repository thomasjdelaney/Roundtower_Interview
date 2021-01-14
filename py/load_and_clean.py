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

def plotInstrumentTimeProfile(bid_ask_counts, bid_col_name, ask_col_name, instrument_name, num_weekdays):
	"""
	For plotting the number of bids/asks we have per minute of the day, per weekday
	Arguments:	bid_ask_counts, pandas dataFrame
				bid_col_name, str
				ask_col_name, str
				instrument_name, str
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
	bid_ax.set_title(instrument_name + ', Total weekdays = ' + str(num_weekdays), fontsize='large')
	plt.tight_layout()

def getInstrumentTimeProfile(instrument_name, bid_ask_frame):
	"""
	For visualising the times at which we have bids and asks for a given instrument.
	Arguments:	instrument_name, str
				bid_ask_frame, pandas dataFrame
	Returns:	displays figures showing distributions of bid/ask times
	"""
	bid_col_name = instrument_name + '_Bid'
	ask_col_name = instrument_name + '_Ask'
	bid_ask_frame = bid_ask_frame[[bid_col_name, ask_col_name]].notna()
	bid_ask_frame['time_of_day'] = bid_ask_frame.index.time
	bid_ask_counts = bid_ask_frame.groupby('time_of_day').aggregate('sum')
	frame_dates = getDataDatesFromFrame(bid_ask_frame)
	num_weekdays = np.sum([date.weekday() < 5 for date in frame_dates])
	plotInstrumentTimeProfile(bid_ask_counts, bid_col_name, ask_col_name, instrument_name, num_weekdays)

#for csv_file in csv_files:
#	printFileInfo(csv_file)

csv_files = glob.glob(os.path.join(csv_dir,'*'))
csv_file = csv_files[0]
bid_ask_frame = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
contract_names = extractContractNameFromColumns(bid_ask_frame.columns)
[getInstrumentTimeProfile(con, bid_ask_frame) for con in contract_names[:3]]
plt.show() # TODO save instead of show