"""
For creating a table of hourly returns from the cleaned bid/ask data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

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

all_clean = loadCleanBidAsk(csv_dir)