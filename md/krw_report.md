
##### Abstract 

In this project we aimed to backtest a trading strategy for the Korean Wan / US Dollar one month forward. We began with approximately seven months of historical data for seventy two different instruments. We cleaned these data to the point where they could be reliably used to construct a linear regression model for KWN+1M. We trained a linear regression model for KWN+1M using all possible data, and examined this model to find the most correlated instruments. We then trained another linear regression model using only those most correlated instruments (R<sup>2</sup> = 0.56). We used this model to construct fair prices for KWN+1M during open hours. We then backtested a trading strategy based on these modelled prices using leaning and hedging in the most correlated instruments. The simulation of the strategy resulted in a return of 2% over approximately one month of trading. (This 2% is in relation to the amount invested in KWN+1M only, I'm ignoring hedging in this case).

# Introduction

## Short overview

Our ultimate aim in this project was to backtest, or simulate, a trading strategy based on modelled prices for KWN+1M. The model for these prices was a linear regression model of returns using other correlated currency forwards or USD exchange rates. That is, our modelled return for KWN+1M was a linear combination of the returns of other instruments over the same time period. We used these modelled returns together with a designated 'fair price' to model the price of KWN+1M at any time during its open hours. 

Based on this modelled price and a 'required edge' parameter we calculated our maximum bid and our minimum ask prices. We then made a decision to trade or not trade based on the market's asking price and offering price. In event that we had already traded and were holding a position in KWN+1M, we used 'leaning' to adjust our maximum bid and our minimum ask. Each day we 'closed out' or 'took off' our positions at a designated 'take off time' regardless of the prices of KWN+1M or the correlated independent instruments at that time.

We used the first six months of our seven months of data to train our linear model. We backtested using the last month. Only eleven days in the last month were tradeable [INVESTIGATE]. Of those eleven days, we traded on nine resulting in a 2% return on our investment over this time period. 

## More detailed summary

### Data cleaning

Our data consisted of minute by minute quotes for each ticker. If the bid or ask did not change over the course of a ticker, we would have a record of nulls for that ticker and that minute. 

In order to facilitate the construction of linear regression models from the data provided, our first task was to 'clean' the data. This consisted of finding the opening and closing times for each ticker then filling forward any null bid or ask prices, and subsequently removing any bid or ask prices quoted outside of the trading hours of the ticker. The opening and closing times were provided by Dan.

Further cleaning steps, such as fully filling certain special columns and modelling all other tickers based on these columns have been implemented. But these data have not been used yet.



Aims:
1. Clean the data provided until it can be used for modelling and backtesting.
1. Model fair prices for the KWN/USD 1 month forward.
1. Implement a trading strategy based on these fair prices that includes hedging and leaning.

# Results

# Data

Our data consisted of minute by minute quotes for each ticker. The index of the data was the datetime, the other columns were bid and ask prices for each ticker indicated by the column name ('AUD BGD_Ask', or 'AUD BGD_Bid' for example). If the bid or ask did not change over the course of a ticker, we would have a record of nulls for that ticker and that minute. 


# Methods

# Discussion

# Conclusion

# References

## Notes on this document