# Notes on csv files

## all_clean.csv

- **Created by script:** py/load_and_clean.py
- **Contains:** The trading data after 'stage 1' cleaning (see README for more details). Basically the intra-day gaps are filled (except for those at the very start of the day), and out of hours trades are nulled. Holidays not yet implements.

## bid_ask_mid.csv

- **Created by script:** py/calc_returns.py
- **Contains:** Same as all_clean.csv, but with a mid column for every bid and ask pair. The mid column is just a simple average of the bid and the ask.

## coef_frame.csv

- **Created by script:** py/calc_returns.py
- **Contains:** Information on the linear models for tickers 'KWN+1M BGN Curncy', 'IHN+1M BGN Curncy', 'FXY1 KMS1 Index', and 'MXID Index'.

## full_mids.csv

- **Created by script:** py/calc_returns.py
- **Contains:** Similar to bid_ask_mid.csv, all columns fully filled, even outside of trading hours. Some columns (ES1, MES1, SGD) are just filled forward. The other columns are modelled based on those columns. The are nulls for early dates, but eveything else is full minute to minute. This is the end result of stage 3 data cleaning.

## open_close.csv

- **Created by script:** Dan downloaded these data and sent them to me.
- **Contains:** Opening and closing times for each of the tickers. Futures opening and closing hours overwrite the other hours when both are provided.
