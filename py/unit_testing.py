"""
For unit testing the python functions found in this project.
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

#### getMaxBidMinAsk, this function implements our 'leaning' ###
# getMaxBidMinAsk(modelled_price, profit_required, lean, position)
# leaning as position changes
try:
    assert getMaxBidMinAsk(1000, 0.001, 0.001, 3) == (996, 998)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, 2) == (997, 999)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, 1) == (998, 1000)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, 0) == (999, 1001)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, -1) == (1000, 1002)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, -2) == (1001, 1003)
    assert getMaxBidMinAsk(1000, 0.001, 0.001, -3) == (1002, 1004)
    # different leaning amount
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, 3) == (997.5, 999.5)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, 2) == (998, 1000)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, 1) == (998.5, 1000.5)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, 0) == (999, 1001)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, -1) == (999.5, 1001.5)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, -2) == (1000, 1002)
    assert getMaxBidMinAsk(1000, 0.001, 0.0005, -3) == (1000.5, 1002.5)
    # no leaning
    assert getMaxBidMinAsk(1000, 0.001, 0.0, 3) == (999.0, 1001)
except AssertionError:
    print(dt.datetime.now().isoformat() + ' ERROR: ' + 'getMaxBidMinAsk failed a unit test!')
except:
    print(dt.datetime.now().isoformat() + ' ERROR: ' + 'Error occurred during the getMaxBidMinAsk unit tests!')
else:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'getMaxBidMinAsk unit tests passed.')

print(dt.datetime.now().isoformat() + ' INFO: ' + 'All unit tests complete.')