#### Email from Dan Needham

I cannot get data earlier than 1st June atm.

We will break the project into 4 parts.

1. Data cleaning
    - I have given you the data for every day
    - If there is no data on a day then this day is on holiday
    - You need to use average of bid and ask to get mid
    - You will need to use ffill for missing data intra day when a future or currency doesn’t trade much – but check hours of trading below as cannot use outside of those hours & for holidays & maybe if lots of missing data beware…
    - Expirations – for futures, they expire every 1 or 3 months – we can look into rolling futures at a later stage (rolling is when a future expires and we need to use the next future)
    - Currency trade from 1am until 9pm
    - FXY1 index is the same future as the KMS1 Index so you need to combine these futures
    - You will need to arrange so can be used for machine learning, etc.
    - You can break models into 3 time zones to get different models for different times of the day
      * 1am-6am
      * 9am-8pm
      * 1am-8pm
    - There is more below regarding data cleaning
1. Modelling
    - Use multiple regression as well as lots of other models to find the best model(s) for KWN+1M Curncy (KRW 1 month NDF), IHN+1M Curncy (IDR 1 month NDF), FXY1/KMS1 & MXID Index.
    - Start off with just multiple regression to keep it simple.
1. Backtesting
    - Initially, I will give you all of the parameters that you require
    - Start Time = 9am
    - Last tradeable time = 8pm
    - Next day time to take off open positions = 1am
    - Model from 9am (assuming 9am if fair) until 8pm
    - Assume need 0.1% (aka 10bps) profit, e.g. if fair value at 12pm is 1000, then we want to pay (aka bid) 999.9 (1000*.999) or sell (aka offer) 1000.1 (1000*1.001)
    - I have attached an example & more information below regarding leaning
    - Do up to a maximum 3 trades until can get out of position
    - Lets assume we trade $1m for each trade for now (so max position of long or short $3m)
    - Lets have a call about this once you have read it all so you can ask lots of questions
    - Parameter optimation (to be done at a later date after we are happy with the first 3 parts)
    - Amend all parameters to maximise p&l -  E.g.
      * start time
      * when to take off trade today and/or tomorrow
      * profit to look for
      * how much to lean after each trade
      * combining models
      * vi.     etc

# Data cleaning
For each variable. Using a51 index as example

#### Stage 1
My suggestion unless you have a better idea (which I am hoping!)
Fill in cells for times where each variable is open

1. have start time 1, end time 1, start time 2, end time 2 eg 0630, 1515, 1600, 2000
2. for each day, all cells are blank up to 630am
3. they are still blank until get tick
4. ffill down to next tick up to 1515
5. cells blank until 1600
6. they are still blank until get tick
7. ffill down to next tick up to 2000
8. cells blank until 0630 next day
9. repeat

#### Stage 2
You will need to use returns, either:

- A time on the 1st day eg today noon mid / 1st june noon mid – 1
- A daily return eg today noon mid / yesterday noon mid – 1
- An hourly return eg today noon mid / today 11am mid – 1

Regression for each variable using returns vs

- es1 index & mes1 index for index, equity & comdty
- sgd Curncy for all curncys

eg a51 coeffs = 0.5 for es1 & 0.4 for mes1

#### Stage 3
Fill in all blanks for the training dataset (only ffill for testing data)
Maybe start filling in es1, mes1 using ffill so they have no blanks

Eg a51
1. up to blanks until it opens around 0630, use last a51 tick from yesterday around 2000 eg 100, & es1 tick at same time eg 1000 and mes1 tick at same time eg 2000
2. 1am tick = 100* (1+ 0.5*(es1 1am tick/1000-1)+ 0.4*(mes1 1am tick/2000-1))
3. Basically, Wherever there is a blank, you take the last tick from a51 and model using last tick in a51 and same time ticks in es1 & mes1

You should now have all cells filled in
We will add rolls at a later date for equity index futures.

#### Leaning
I have attached the example for leaning.
Everytime you trade you need to hedge with the indept vars.
e.g. if buy $1m kwn, you sell $500k CNH, sell $250k NTN, etc using betas from indept vars.
Also, worth rerunning lasso & limiting to 3 indept vars & hedge with these 3 instead of the 5 you use to model kwn.
This can be 2nd iteration.

#### My own thoughts
* Clean product by product?
* Look at distribution of data existance over minutes of the day for an impression of when we _should_ have data.

TODO: 
  - function for extracting distribution of existing data. Complications include: holidays, weekends. The point is to identify missing data, and understand more about the data. Our ultimate goal is to clean up the data.