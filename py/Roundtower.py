"""
For keeping functions that may be used in multiple scripts
"""
import os, sys
import datetime as dt

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
		sys.exit(1)
	return tickers