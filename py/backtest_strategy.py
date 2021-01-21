"""
For testing a trading stratey or strategies on historical data.
"""
import os, sys, glob, shutil, argparse
import datetime as dt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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

all_clean = loadCleanBidAsk(csv_dir)
full_mids = loadFullMids(csv_dir)
open_close = loadOpenClose(csv_dir)

# add the full mids to all clean
# pick a ticker with appropriate opening times
# start at position zero
# trade when there the difference between the ask or bid and 
#	our modelled mid is greater than 0.001
# limit the trading so we don't get out of position
# clear out at the end of the day

# check instructions again to see if I have missed anything.